from __future__ import annotations

import os
import glob
import csv
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert, butter, filtfilt, periodogram
from scipy.ndimage import median_filter

import matplotlib as mpl
import matplotlib.pyplot as plt


_THD_NOISE_BAND_HZ: float = 1.0


def _apply_pyseafom_style():
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
    })


def _add_brand_box(ax, text: str = "pySEAFOM 0.1"):
    bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7)
    ax.text(
        0.98, 0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        bbox=bbox_props,
    )


def _add_metadata_box(ax, lines: List[str]):
    if not lines:
        return
    text = "Measurement Parameters\n" + "=" * 25 + "\n" + "\n".join(lines)
    metadata_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9)
    ax.text(
        0.98, 0.65,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
        family="monospace",
        bbox=metadata_props,
    )


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _median_safe(x: NDArray[np.floating], window_points: int) -> NDArray[np.floating]:
    x = np.asarray(x, float)
    if window_points is None or window_points <= 1 or x.size < 3:
        return x.copy()

    w = int(window_points)
    if w % 2 == 0:
        w += 1

    max_w = x.size if x.size % 2 == 1 else x.size - 1
    w = min(w, max_w)
    if w < 3:
        return x.copy()

    y = x.copy()
    finite = np.isfinite(y)
    if not finite.all():
        idx = np.where(finite)[0]
        if idx.size == 0:
            return y
        y[~finite] = np.interp(np.where(~finite)[0], idx, y[finite])

    return median_filter(y, size=w, mode="nearest").astype(float)


def _max_strain_last_cycle(
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    trigger_time: float,
    ref_freq_hz: float,
) -> float:
    time_s = np.asarray(time_s, float)
    sig = np.asarray(signal_microstrain, float)

    period = 1.0 / ref_freq_hz
    t_start = max(time_s[0], trigger_time - period)
    t_end = trigger_time

    mask = (time_s >= t_start) & (time_s <= t_end)
    if np.any(mask):
        segment = sig[mask]
    else:
        dt = np.mean(np.diff(time_s))
        n_samples = int(round(period / dt))
        idx_center = int(np.argmin(np.abs(time_s - trigger_time)))
        i0 = max(0, idx_center - n_samples + 1)
        i1 = idx_center + 1
        segment = sig[i0:i1]

    if segment.size == 0:
        return float("nan")
    return float(np.max(np.abs(segment)))


def _window_start_reference(time_s: NDArray[np.floating], time_start_s: Optional[float]) -> float:
    if time_start_s is not None:
        return float(time_start_s)
    return float(np.asarray(time_s, float)[0])


def _write_csv_row(csv_path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        out = {k: row.get(k, None) for k in fieldnames}
        writer.writerow(out)


def _append_hilbert_csv(results_dir: str, row: Dict[str, Any]) -> str:
    _ensure_dir(results_dir)
    csv_path = os.path.join(results_dir, "dynamic_range_hilbert.csv")

    fieldnames = [
        "mode",
        "delta_t_from_window_start [s]",
        "peak_last_cycle [µε]",
        "peak_over_basis [dB re radian/√Hz]",

        "trigger_time_abs [s]",
        "limit_envelope_strain [µε]",

        "save_results",
        "results_dir",
        "show_plot",
        "plot_metadata_box",

        "folder_or_file",
        "repetition_rate [Hz]",
        "delta_x [m]",
        "pos [m]",
        "time_start [s]",
        "duration [s]",
        "data_is_strain",
        "gauge_length [m]",
        "average_over_cols",
        "highpass [Hz]",

        "safezone [s]",

        "max_strain_theoretical [µε]",
        "ref_freq [Hz]",
    ]

    _write_csv_row(csv_path, fieldnames, row)
    return csv_path


def _append_thd_csv(results_dir: str, row: Dict[str, Any]) -> str:
    _ensure_dir(results_dir)
    csv_path = os.path.join(results_dir, "dynamic_range_thd.csv")

    fieldnames = [
        "mode",
        "delta_t_from_window_start [s]",
        "peak_last_cycle [µε]",
        "peak_over_basis [dB re radian/√Hz]",

        "trigger_time_abs [s]",
        "limit_strain_peak [µε]",
        "thd_hop_effective [s]",

        "save_results",
        "results_dir",
        "show_plot",
        "plot_metadata_box",

        "folder_or_file",
        "repetition_rate [Hz]",
        "delta_x [m]",
        "pos [m]",
        "time_start [s]",
        "duration [s]",
        "data_is_strain",
        "gauge_length [m]",
        "average_over_cols",
        "highpass [Hz]",

        "safezone [s]",

        "ref_freq [Hz]",
        "thd_window [s]",
        "thd_overlap [%]",
        "thd_threshold [%]",
        "thd_median_window [points]",
        "min_trigger_duration [s]",
    ]

    _write_csv_row(csv_path, fieldnames, row)
    return csv_path


def _phase_to_strain(
    phase_rad: NDArray[np.floating],
    *,
    wavelength_m: float = 1550e-9,
    n_eff: float = 1.4682,
    gauge_length: float = 6.38,
    xi: float = 0.78,
    output_in_microstrain: bool = True,
) -> NDArray[np.floating]:
    coef = wavelength_m / (4.0 * np.pi * n_eff * gauge_length * xi)
    strain = coef * phase_rad
    if output_in_microstrain:
        strain = strain * 1e6
    return strain


def _microstrain_to_phase(
    strain_microstrain: float | NDArray[np.floating],
    *,
    wavelength_m: float = 1550e-9,
    n_eff: float = 1.4682,
    gauge_length: float = 6.38,
    xi: float = 0.78,
) -> NDArray[np.floating]:

    coef = wavelength_m / (4.0 * np.pi * n_eff * gauge_length * xi)  
    strain = np.asarray(strain_microstrain, dtype=float) * 1e-6     
    phase_rad = strain / (coef + 1e-30)
    return phase_rad.astype(float)


def load_dynamic_range_data(
    folder_or_file: str,
    *,
    fs: float,
    delta_x_m: float,
    x1_m: float,
    x2_m: float,
    test_sections_channels: float,
    time_start_s: float = 0.0,
    duration: Optional[float] = None,
    average_over_cols: int = 1,
    matrix_layout: str = "time_space",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if os.path.isdir(folder_or_file):
        file_list = sorted(glob.glob(os.path.join(folder_or_file, "*.npy")))
    else:
        file_list = [folder_or_file]

    if not file_list:
        raise FileNotFoundError(f"No .npy files found in {folder_or_file!r}")

    matrices = [np.load(p) for p in file_list]
    data_full = np.concatenate(matrices, axis=0)

    if data_full.ndim != 2:
        raise ValueError("Expected a 2D matrix.")

    if matrix_layout not in {"time_space", "space_time", "auto"}:
        raise ValueError("matrix_layout must be 'time_space', 'space_time', or 'auto'.")

    data = data_full
    if matrix_layout == "space_time":
        data = data_full.T
    elif matrix_layout == "auto":
        if data_full.shape[0] < data_full.shape[1]:
            data = data_full.T

    ntime_full, nspace = data.shape

    frame_start = int(time_start_s * fs)
    if duration is None:
        frame_end = ntime_full
    else:
        frame_end = int((time_start_s + duration) * fs)
        frame_end = min(frame_end, ntime_full)

    if frame_start >= frame_end:
        raise ValueError("Invalid time window (frame_start >= frame_end).")

    data_time = data[frame_start:frame_end, :]
    ntime = data_time.shape[0]
    time_s_vec = np.arange(ntime) / fs + time_start_s

    idx_start = int(x1_m / delta_x_m)
    idx_end = int(x2_m / delta_x_m)
    idx_start = max(idx_start, 0)
    idx_end = min(idx_end, nspace)

    if idx_start >= idx_end:
        raise ValueError("Invalid spatial range.")

    data_slice = data_time[:, idx_start:idx_end]
    ncols = data_slice.shape[1]

    pos_idx = int(round(test_sections_channels / delta_x_m)) - idx_start
    if pos_idx < 0 or pos_idx >= ncols:
        raise ValueError("pos_m is outside the selected spatial window.")

    if average_over_cols < 1:
        average_over_cols = 1
    end_col = min(pos_idx + average_over_cols, ncols)
    trace = data_slice[:, pos_idx:end_col].mean(axis=1)

    return time_s_vec.astype(float), trace.astype(float)


def data_processing(
    trace: NDArray[np.floating],
    *,
    data_is_strain: bool,
    gauge_length: float = 6.38,
    wavelength_m: float = 1550e-9,
    n_eff: float = 1.4682,
    xi: float = 0.78,
    highpass_hz: Optional[float] = 5.0,
    fs: Optional[float] = None,
    interrogation_rate_hz: Optional[float] = None,
) -> NDArray[np.floating]:
    sig = np.asarray(trace, float)

    if not data_is_strain:
        sig = _phase_to_strain(
            sig,
            wavelength_m=wavelength_m,
            n_eff=n_eff,
            gauge_length=gauge_length,
            xi=xi,
            output_in_microstrain=True,
        )

    fs = interrogation_rate_hz if interrogation_rate_hz is not None else fs

    if highpass_hz is not None and highpass_hz > 0.0:
        if fs is None:
            raise ValueError("A sampling rate must be provided when using highpass_hz.")
        nyq = fs / 2.0
        wn = highpass_hz / nyq
        b, a = butter(5, wn, btype="high")
        sig = filtfilt(b, a, sig)

    return sig.astype(float)


def compute_dynamic_range_hilbert(
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    *,
    max_strain_microstrain: float,
    ref_freq_hz: float = 50.0,
    smooth_window_s: float = 0.5,
    error_threshold_frac: float = 0.3,
    safezone_s: float = 1.0,
) -> Dict[str, object]:
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal_microstrain, dtype=float)

    if time_s.ndim != 1 or signal.ndim != 1 or time_s.size != signal.size:
        raise ValueError("time_s and signal_microstrain must be 1D vectors of the same length.")

    dt = np.mean(np.diff(time_s))
    rep_rate = 1.0 / dt

    win = max(1, int(smooth_window_s * rep_rate))
    kernel = np.ones(win) / win

    env_raw = np.abs(hilbert(signal))
    env_measured = np.convolve(env_raw, kernel, mode="same")

    amp_ramp = np.linspace(0.0, max_strain_microstrain, signal.size)
    ref_signal = amp_ramp * np.sin(2 * np.pi * ref_freq_hz * time_s)
    ref_env_raw = np.abs(hilbert(ref_signal))
    env_theoretical = np.convolve(ref_env_raw, kernel, mode="same")

    abs_error = np.abs(env_measured - env_theoretical)
    rel_error = np.zeros_like(abs_error)
    valid = env_theoretical > 1e-12
    rel_error[valid] = abs_error[valid] / env_theoretical[valid]

    mask_violation = rel_error > error_threshold_frac

    safezone_end = time_s[0] + max(safezone_s, 0.0)
    mask_violation[time_s < safezone_end] = False

    segments: List[Tuple[float, float]] = []
    in_seg = False
    start_idx = 0
    for i, flag in enumerate(mask_violation):
        if flag and not in_seg:
            in_seg = True
            start_idx = i
        elif not flag and in_seg:
            in_seg = False
            segments.append((time_s[start_idx], time_s[i - 1]))
    if in_seg:
        segments.append((time_s[start_idx], time_s[-1]))

    if mask_violation.any():
        first_idx = int(np.argmax(mask_violation))
        dr_time = time_s[first_idx]
        dr_strain = env_theoretical[first_idx]
    else:
        dr_time = time_s[-1]
        dr_strain = env_theoretical[-1]

    return {
        "env_measured": env_measured,
        "env_theoretical": env_theoretical,
        "rel_error": rel_error,
        "mask_violation": mask_violation,
        "segments_violation": segments,
        "dynamic_range_limit_strain": float(dr_strain),
        "dynamic_range_limit_time": float(dr_time),
        "error_threshold_frac": float(error_threshold_frac),
        "safezone_s": float(safezone_s),
        "ref_freq_hz": float(ref_freq_hz),
        "max_strain_microstrain": float(max_strain_microstrain),
        "smooth_window_s": float(smooth_window_s),
    }


def plot_dynamic_range_hilbert(
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    dr_result: Dict[str, object],
    *,
    plot_metadata_box: bool = False,
    metadata_lines: Optional[List[str]] = None,
    title: str = "Hilbert Dynamic Range Analysis",
):
    _apply_pyseafom_style()

    env_measured = dr_result["env_measured"]
    env_theoretical = dr_result["env_theoretical"]
    rel_error = dr_result.get("rel_error")
    segments = dr_result["segments_violation"]
    thr_frac = dr_result.get("error_threshold_frac", 0.3)
    safezone_s = dr_result.get("safezone_s", 1.0)

    fig, ax_sig = plt.subplots(figsize=(16, 9))

    ax_sig.plot(time_s, signal_microstrain, color="gray", alpha=0.18, linewidth=0.8, label="DAS signal")
    ax_sig.plot(time_s, env_measured, color="r", linewidth=1.8, label="Measured envelope")
    ax_sig.plot(time_s, env_theoretical, color="b", linestyle="--", linewidth=1.8, label="Theoretical envelope")

    if safezone_s > 0.0:
        safe_end = time_s[0] + safezone_s
        ax_sig.axvspan(time_s[0], safe_end, color="green", alpha=0.10, label="Safe zone")

    for t0, t1 in segments:
        ax_sig.axvspan(t0, t1, color="orange", alpha=0.25, label="Error region")

    ax_sig.set_xlabel("Time [s]")
    ax_sig.set_ylabel("Strain [µε]")
    ax_sig.grid(True)

    ax_err = ax_sig.twinx()
    if rel_error is not None:
        err_pct = np.asarray(rel_error, float) * 100.0
        ax_err.plot(time_s, err_pct, color="k", linestyle=":", linewidth=1.5, label="Relative error")
        ax_err.axhline(thr_frac * 100.0, color="purple", linestyle="--", linewidth=2.0,
                       label=f"Error threshold ({thr_frac*100:.0f}%)")
        ax_err.set_ylabel("Relative error [%]")
        max_rel = np.nanmax(err_pct) if err_pct.size else 0.0
        ax_err.set_ylim(0, max(100, max_rel * 1.1))

    h1, lab1 = ax_sig.get_legend_handles_labels()
    h2, lab2 = ax_err.get_legend_handles_labels()
    uniq = dict(zip(lab1 + lab2, h1 + h2))
    ax_sig.legend(uniq.values(), uniq.keys(), loc="upper left")

    ax_sig.set_title(title)

    if plot_metadata_box and metadata_lines:
        _add_metadata_box(ax_sig, metadata_lines)

    _add_brand_box(ax_sig, text="pySEAFOM 0.1")

    plt.tight_layout()
    return fig, (ax_sig, ax_err)


def analyze_dynamic_range_hilbert(
    *,
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    max_strain_microstrain: float,
    ref_freq_hz: float = 50.0,
    smooth_window_s: float = 0.5,
    error_threshold_frac: float = 0.3,
    safezone_s: float = 1.0,
    time_start_s: Optional[float] = None,

    fs: Optional[float] = None,
    delta_x_m: Optional[float] = None,
    gauge_length: Optional[float] = None,
    highpass_hz: Optional[float] = None,

    radian_basis: Optional[float] = None,

    folder_or_file: Optional[str] = None,
    test_sections_channels: Optional[float] = None,
    duration: Optional[float] = None,
    data_is_strain: Optional[bool] = None,
    average_over_cols: Optional[int] = None,

    show_plot: bool = True,
    save_results: bool = False,
    results_dir: str = "results_dynamic_range",
    plot_metadata_box: bool = False,
):
    mode = "hilbert"

    dr = compute_dynamic_range_hilbert(
        time_s,
        signal_microstrain,
        max_strain_microstrain=max_strain_microstrain,
        ref_freq_hz=ref_freq_hz,
        smooth_window_s=smooth_window_s,
        error_threshold_frac=error_threshold_frac,
        safezone_s=safezone_s,
    )

    trigger_time = float(dr["dynamic_range_limit_time"])
    limit_env = float(dr["dynamic_range_limit_strain"])

    t0_ref = _window_start_reference(time_s, time_start_s)
    delta_t = trigger_time - t0_ref
    peak_last = _max_strain_last_cycle(time_s, signal_microstrain, trigger_time, ref_freq_hz)

    dr_dB = None
    if radian_basis is not None and gauge_length is not None:
        rb = float(radian_basis)
        gl = float(gauge_length)
        if np.isfinite(rb) and rb > 0 and np.isfinite(gl) and gl > 0 and np.isfinite(peak_last) and peak_last > 0:
            peak_rad = float(_microstrain_to_phase(peak_last, gauge_length=gl))
            if np.isfinite(peak_rad) and peak_rad > 0:
                dr_dB = float(20.0 * np.log10(peak_rad / rb))

    print("\n[HILBERT DYNAMIC RANGE]")
    print("+" + "-" * 60 + "+")
    print("| {:<35s} | {:>20s} |".format("Metric", "Value"))
    print("+" + "-" * 60 + "+")
    print("| {:<35s} | {:>20.3f} |".format("Trigger time (abs) [s]", trigger_time))
    print("| {:<35s} | {:>20.3f} |".format("Δt from window start [s]", delta_t))
    print("| {:<35s} | {:>20.1f} |".format("Limit envelope strain [µε]", limit_env))
    print("| {:<35s} | {:>20.1f} |".format("Peak strain (last cycle) [µε]", peak_last))
    if dr_dB is not None:
        print("| {:<35s} | {:>20.2f} |".format("Peak/Basis [dB re rad/√Hz]", dr_dB))
    print("+" + "-" * 60 + "+")
    print(f"[HILBERT] Safe zone = {safezone_s:.2f} s")

    metadata_lines: List[str] = []
    if fs is not None:
        metadata_lines.append(f"Repetition Rate: {fs:.1f} Hz")
    if gauge_length is not None:
        metadata_lines.append(f"Gauge Length: {gauge_length:.2f} m")
    if delta_x_m is not None:
        metadata_lines.append(f"Spatial Resolution: {delta_x_m:.2f} m")
    if highpass_hz is not None and highpass_hz > 0:
        metadata_lines.append(f"High-pass: {highpass_hz:.2f} Hz")

    metadata_lines.append(f"Ref. frequency: {ref_freq_hz:.2f} Hz")
    metadata_lines.append(f"Max strain: {max_strain_microstrain:.1f} µε")
    metadata_lines.append(f"Error threshold: {error_threshold_frac*100:.1f} %")
    metadata_lines.append(f"Safe zone: {safezone_s:.2f} s")
    metadata_lines.append(f"Error relative time trigger: {delta_t:.2f} s")
    metadata_lines.append(f"Error strain trigger: {peak_last:.2f} µε")
    if dr_dB is not None:
        metadata_lines.append(f"Peak/Basis: {dr_dB:.2f} dB re radian/√Hz")

    fig = None
    if show_plot or save_results:
        fig, _ = plot_dynamic_range_hilbert(
            time_s,
            signal_microstrain,
            dr,
            plot_metadata_box=plot_metadata_box,
            metadata_lines=metadata_lines,
            title="Hilbert Dynamic Range Analysis",
        )

    if save_results:
        _ensure_dir(results_dir)

        fig_path = os.path.join(results_dir, "dynamic_range_hilbert.png")
        if fig is not None:
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"[HILBERT] Figure saved to: {fig_path}")

        csv_path = _append_hilbert_csv(results_dir, {
            "mode": mode,
            "delta_t_from_window_start [s]": delta_t,
            "peak_last_cycle [µε]": peak_last,
            "peak_over_basis [dB re radian/√Hz]": dr_dB,

            "trigger_time_abs [s]": trigger_time,
            "limit_envelope_strain [µε]": limit_env,

            "save_results": save_results,
            "results_dir": results_dir,
            "show_plot": show_plot,
            "plot_metadata_box": plot_metadata_box,

            "folder_or_file": folder_or_file,
            "repetition_rate [Hz]": fs,
            "delta_x [m]": delta_x_m,
            "pos [m]": test_sections_channels,
            "time_start [s]": time_start_s,
            "duration [s]": duration,
            "data_is_strain": data_is_strain,
            "gauge_length [m]": gauge_length,
            "average_over_cols": average_over_cols,
            "highpass [Hz]": highpass_hz,

            "safezone [s]": safezone_s,

            "max_strain_theoretical [µε]": max_strain_microstrain,
            "ref_freq [Hz]": ref_freq_hz,
        })
        print(f"[HILBERT] CSV row appended to: {csv_path}")

    if show_plot and fig is not None:
        plt.show()
    elif not show_plot and fig is not None:
        plt.close(fig)


def compute_dynamic_range_thd(
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    *,
    ref_freq_hz: float = 50.0,
    window_s: float = 1.0,
    hop_frac: float = 0.25,
    overlap: Optional[float] = None,
    hop_s: Optional[float] = None,
    thd_threshold_frac: float = 0.15,
    safezone_s: float = 1.0,
    amp_gate_frac: float = 0.10,
    median_window_s: int = 0.005,
    min_trigger_duration: float = 1.0,
) -> Dict[str, object]:
    time_s = np.asarray(time_s, float)
    sig = np.asarray(signal_microstrain, float)

    if time_s.ndim != 1 or sig.ndim != 1 or time_s.size != sig.size:
        raise ValueError("time_s and signal_microstrain must be 1D vectors of the same length.")

    dt = np.mean(np.diff(time_s))
    fs = 1.0 / dt

    Nwin = int(round(window_s * fs))

    if overlap is not None:
        ov = float(overlap)
        if not np.isfinite(ov):
            ov = 0.75
        ov = max(0.0, min(0.99, ov))
        hop_frac = max(0.01, 1.0 - ov)

    if hop_s is None:
        hf = float(hop_frac)
        if not np.isfinite(hf):
            hf = 0.25
        hf = max(0.01, min(1.0, hf))
        hop_s_eff = max(1.0 / fs, hf * float(window_s))
    else:
        hop_s_eff = float(hop_s)

    Hop = int(round(hop_s_eff * fs))

    if Nwin <= 4:
        Nwin = int(round(1.0 * fs))
        Hop = max(1, Nwin // 4)

    idx0 = np.arange(0, len(sig) - Nwin + 1, Hop)
    if idx0.size == 0:
        raise ValueError("Signal is too short for the chosen window/hop.")

    center_idx = idx0 + Nwin // 2
    t_ctr = time_s[center_idx]

    A1_rms = np.zeros(idx0.size)
    thd_only = np.zeros(idx0.size)

    kmax = int((fs / 2) // ref_freq_hz)
    if kmax < 2:
        raise ValueError("Sampling frequency too low for THD analysis with this ref_freq_hz.")

    for k, i_start in enumerate(idx0):
        seg = sig[i_start: i_start + Nwin]
        t_loc = np.arange(Nwin) / fs

        s1 = np.sin(2 * np.pi * ref_freq_hz * t_loc)
        s2 = np.cos(2 * np.pi * ref_freq_hz * t_loc)
        M = np.column_stack([s1, s2])
        (a, b), *_ = np.linalg.lstsq(M, seg, rcond=None)
        A1 = np.hypot(a, b)
        A1_rms[k] = A1 / np.sqrt(2.0)

        f, Pxx = periodogram(seg, fs=fs, window="hann", scaling="density", detrend=False)

        sum_harm_pow = 0.0
        for h in range(2, kmax + 1):
            fh = h * ref_freq_hz
            band = (f >= fh - _THD_NOISE_BAND_HZ) & (f <= fh + _THD_NOISE_BAND_HZ)
            if np.any(band):
                sum_harm_pow += np.trapz(Pxx[band], f[band])

        thd_only[k] = np.sqrt(sum_harm_pow) / (A1_rms[k] + 1e-12)

    thd_frac = _median_safe(thd_only, int(median_window_s))

    amp_gate = amp_gate_frac * np.nanmax(A1_rms) if np.isfinite(A1_rms).any() else 0.0
    safezone_end = time_s[0] + max(safezone_s, 0.0)

    mask_raw = (
        (thd_frac >= thd_threshold_frac) &
        (A1_rms >= amp_gate) &
        (t_ctr >= safezone_end)
    )

    first_idx = None
    i = 0
    n = len(mask_raw)
    while i < n:
        if not mask_raw[i]:
            i += 1
            continue

        j = i
        while j + 1 < n and mask_raw[j + 1]:
            j += 1

        duration = t_ctr[j] - t_ctr[i] + (Hop / fs)
        if duration >= min_trigger_duration:
            first_idx = i
            break
        i = j + 1

    if first_idx is None:
        mask_violation = np.zeros_like(mask_raw, dtype=bool)
        segments: List[Tuple[float, float]] = []
        dr_time = t_ctr[-1]
        dr_strain = A1_rms[-1] * np.sqrt(2.0)
    else:
        mask_violation = np.zeros_like(mask_raw, dtype=bool)
        mask_violation[first_idx:] = True
        segments = [(t_ctr[first_idx], t_ctr[-1])]
        dr_time = t_ctr[first_idx]
        dr_strain = A1_rms[first_idx] * np.sqrt(2.0)

    hop_s_effective = float(Hop / fs)

    return {
        "thd_time": t_ctr,
        "thd_fraction": thd_frac,
        "amp_rms": A1_rms,
        "mask_violation": mask_violation,
        "segments_violation": segments,
        "dynamic_range_limit_strain": float(dr_strain),
        "dynamic_range_limit_time": float(dr_time),
        "thd_threshold_frac": float(thd_threshold_frac),
        "safezone_s": float(safezone_s),
        "min_trigger_duration": float(min_trigger_duration),
        "amp_gate_frac": float(amp_gate_frac),
        "ref_freq_hz": float(ref_freq_hz),
        "window_s": float(window_s),
        "hop_frac": float(hop_frac),
        "overlap": None if overlap is None else float(overlap),
        "hop_s_effective": hop_s_effective,
        "median_window_s": int(median_window_s),
        "noise_band_hz_fixed": float(_THD_NOISE_BAND_HZ),
    }


def plot_dynamic_range_thd(
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    thd_result: Dict[str, object],
    *,
    plot_metadata_box: bool = False,
    metadata_lines: Optional[List[str]] = None,
    title: str = "Sliding THD Analysis",
):
    _apply_pyseafom_style()

    t_ctr = thd_result["thd_time"]
    thd_frac = thd_result["thd_fraction"]
    segments = thd_result["segments_violation"]
    thr_frac = thd_result.get("thd_threshold_frac", 0.15)
    safezone_s = thd_result.get("safezone_s", 1.0)

    thd_pct = np.asarray(thd_frac, float) * 100.0

    fig, ax_sig = plt.subplots(figsize=(16, 9))

    ax_sig.plot(time_s, signal_microstrain, color="gray", alpha=0.18, linewidth=0.8, label="DAS signal")

    if safezone_s > 0.0:
        safe_end = time_s[0] + safezone_s
        ax_sig.axvspan(time_s[0], safe_end, color="green", alpha=0.10, label="Safe zone")

    for t0, t1 in segments:
        ax_sig.axvspan(t0, t1, color="orange", alpha=0.25, label="Error region")

    ax_sig.set_xlabel("Time [s]")
    ax_sig.set_ylabel("Strain [µε]")
    ax_sig.grid(True)

    ax_thd = ax_sig.twinx()
    ax_thd.plot(t_ctr, thd_pct, color="C0", linewidth=1.8, label="THD (sliding)")
    ax_thd.axhline(thr_frac * 100.0, color="purple", linestyle="--", linewidth=2.0,
                   label=f"THD threshold ({thr_frac*100:.0f}%)")
    ax_thd.set_ylabel("THD [%]")

    handles1, labels1 = ax_sig.get_legend_handles_labels()
    handles2, labels2 = ax_thd.get_legend_handles_labels()
    uniq = dict(zip(labels1 + labels2, handles1 + handles2))
    ax_sig.legend(uniq.values(), uniq.keys(), loc="upper left")

    ax_sig.set_title(title)

    if plot_metadata_box and metadata_lines:
        _add_metadata_box(ax_sig, metadata_lines)

    _add_brand_box(ax_sig, text="pySEAFOM 0.1")

    plt.tight_layout()
    return fig, (ax_sig, ax_thd)


def analyze_dynamic_range_thd(
    *,
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],

    ref_freq_hz: float = 50.0,
    window_s: float = 1.0,
    overlap: float = 0.75,
    hop_frac: Optional[float] = None,
    hop_s: Optional[float] = None,
    thd_threshold_frac: float = 0.15,
    safezone_s: float = 1.0,
    amp_gate_frac: float = 0.10,
    median_window_s: int = 0.005,
    min_trigger_duration: float = 1.0,

    time_start_s: Optional[float] = None,

    radian_basis: Optional[float] = None,

    fs: Optional[float] = None,
    delta_x_m: Optional[float] = None,
    gauge_length: Optional[float] = None,
    highpass_hz: Optional[float] = None,

    folder_or_file: Optional[str] = None,
    test_sections_channels: Optional[float] = None,
    duration: Optional[float] = None,
    data_is_strain: Optional[bool] = None,
    average_over_cols: Optional[int] = None,

    show_plot: bool = True,
    save_results: bool = False,
    results_dir: str = "results_dynamic_range",
    plot_metadata_box: bool = False,
):
    mode = "thd"

    if overlap is not None:
        ov = float(overlap)
        if not np.isfinite(ov):
            ov = 0.75
        ov = max(0.0, min(0.99, ov))
        hop_frac_eff = max(0.01, 1.0 - ov)
    else:
        hop_frac_eff = 0.25 if hop_frac is None else float(hop_frac)

    dr = compute_dynamic_range_thd(
        time_s,
        signal_microstrain,
        ref_freq_hz=ref_freq_hz,
        window_s=window_s,
        hop_frac=hop_frac_eff,
        overlap=None,
        hop_s=hop_s,
        thd_threshold_frac=thd_threshold_frac,
        safezone_s=safezone_s,
        amp_gate_frac=amp_gate_frac,
        median_window_s=median_window_s,
        min_trigger_duration=min_trigger_duration,
    )

    trigger_time = float(dr["dynamic_range_limit_time"])
    limit_peak = float(dr["dynamic_range_limit_strain"])

    t0_ref = _window_start_reference(time_s, time_start_s)
    delta_t = trigger_time - t0_ref
    peak_last = _max_strain_last_cycle(time_s, signal_microstrain, trigger_time, ref_freq_hz)

    dr_dB = None
    if radian_basis is not None and gauge_length is not None:
        rb = float(radian_basis)
        gl = float(gauge_length)
        if np.isfinite(rb) and rb > 0 and np.isfinite(gl) and gl > 0 and np.isfinite(peak_last) and peak_last > 0:
            peak_rad = float(_microstrain_to_phase(peak_last, gauge_length=gl))
            if np.isfinite(peak_rad) and peak_rad > 0:
                dr_dB = float(20.0 * np.log10(peak_rad / rb))

    print("\n[THD DYNAMIC RANGE]")
    print("+" + "-" * 60 + "+")
    print("| {:<35s} | {:>20s} |".format("Metric", "Value"))
    print("+" + "-" * 60 + "+")
    print("| {:<35s} | {:>20.3f} |".format("Trigger time (abs) [s]", trigger_time))
    print("| {:<35s} | {:>20.3f} |".format("Δt from window start [s]", delta_t))
    print("| {:<35s} | {:>20.1f} |".format("Limit strain (peak) [µε]", limit_peak))
    print("| {:<35s} | {:>20.1f} |".format("Peak strain (last cycle) [µε]", peak_last))
    if dr_dB is not None:
        print("| {:<35s} | {:>20.2f} |".format("Peak/Basis [dB re rad/√Hz]", dr_dB))
    print("+" + "-" * 60 + "+")
    print(f"[THD] Safe zone = {safezone_s:.2f} s | min duration = {min_trigger_duration:.2f} s")

    metadata_lines: List[str] = []
    if fs is not None:
        metadata_lines.append(f"Repetition Rate: {fs:.1f} Hz")
    if gauge_length is not None:
        metadata_lines.append(f"Gauge Length: {gauge_length:.2f} m")
    if delta_x_m is not None:
        metadata_lines.append(f"Spatial Resolution: {delta_x_m:.2f} m")
    if highpass_hz is not None and highpass_hz > 0:
        metadata_lines.append(f"High-pass: {highpass_hz:.2f} Hz")

    metadata_lines.append(f"Ref. frequency: {ref_freq_hz:.2f} Hz")
    metadata_lines.append(f"THD threshold: {thd_threshold_frac*100:.1f} %")
    metadata_lines.append(f"Window: {window_s:.2f} s | Overlap: {overlap*100:.1f} %")
    metadata_lines.append(f"Median window: {median_window_s:.3f} s")
    metadata_lines.append(f"Safe zone: {safezone_s:.2f} s")
    metadata_lines.append(f"Min. duration: {min_trigger_duration:.2f} s")
    metadata_lines.append(f"Error relative time trigger: {delta_t:.2f} s")
    metadata_lines.append(f"Error strain trigger: {peak_last:.2f} µε")
    if dr_dB is not None:
        metadata_lines.append(f"Peak/Basis: {dr_dB:.2f} dB re radian/√Hz")

    fig = None
    if show_plot or save_results:
        fig, _ = plot_dynamic_range_thd(
            time_s,
            signal_microstrain,
            dr,
            plot_metadata_box=plot_metadata_box,
            metadata_lines=metadata_lines,
            title="Sliding THD Dynamic Range Analysis",
        )

    if save_results:
        _ensure_dir(results_dir)

        fig_path = os.path.join(results_dir, "dynamic_range_thd.png")
        if fig is not None:
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"[THD] Figure saved to: {fig_path}")

        csv_path = _append_thd_csv(results_dir, {
            "mode": mode,
            "delta_t_from_window_start [s]": delta_t,
            "peak_last_cycle [µε]": peak_last,
            "peak_over_basis [dB re radian/√Hz]": dr_dB,

            "trigger_time_abs [s]": trigger_time,
            "limit_strain_peak [µε]": limit_peak,
            "thd_hop_effective [s]": dr.get("hop_s_effective", None),

            "save_results": save_results,
            "results_dir": results_dir,
            "show_plot": show_plot,
            "plot_metadata_box": plot_metadata_box,

            "folder_or_file": folder_or_file,
            "repetition_rate [Hz]": fs,
            "delta_x [m]": delta_x_m,
            "pos [m]": test_sections_channels,
            "time_start [s]": time_start_s,
            "duration [s]": duration,
            "data_is_strain": data_is_strain,
            "gauge_length [m]": gauge_length,
            "average_over_cols": average_over_cols,
            "highpass [Hz]": highpass_hz,

            "safezone [s]": safezone_s,

            "ref_freq [Hz]": ref_freq_hz,
            "thd_window [s]": window_s,
            "thd_overlap [%]": overlap * 100.0,
            "thd_threshold [%]": thd_threshold_frac * 100.0,
            "thd_median_window [points]": median_window_s,
            "min_trigger_duration [s]": min_trigger_duration,
        })
        print(f"[THD] CSV row appended to: {csv_path}")

    if show_plot and fig is not None:
        plt.show()
    elif not show_plot and fig is not None:
        plt.close(fig)
