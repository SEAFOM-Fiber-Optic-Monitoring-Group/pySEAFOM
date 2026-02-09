from __future__ import annotations

import os
import glob
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt


def _apply_pyseafom_style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
    })


def _add_brand_box(ax: plt.Axes, text: str = "pySEAFOM 0.1") -> None:
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


def _add_metadata_box(ax: plt.Axes, lines: List[str]) -> None:
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


def _stats_lines(freq_hz: NDArray[np.floating], db: NDArray[np.floating], *, label: str) -> List[str]:
    f = np.asarray(freq_hz, float).ravel()
    y = np.asarray(db, float).ravel()

    mask = np.isfinite(f) & np.isfinite(y)
    if not np.any(mask):
        return [
            f"{label} median: n/a",
            f"{label} min: n/a",
            f"{label} max: n/a",
        ]

    f = f[mask]
    y = y[mask]

    med = float(np.nanmedian(y))
    i_min = int(np.nanargmin(y))
    i_max = int(np.nanargmax(y))

    return [
        f"{label} median: {med:.2f} dB",
        f"{label} min: {float(y[i_min]):.2f} dB (f={float(f[i_min]):.2f} Hz)",
        f"{label} max: {float(y[i_max]):.2f} dB (f={float(f[i_max]):.2f} Hz)",
    ]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def phase_to_strain(
    phase_rad: NDArray[np.floating],
    *,
    wavelength_m: float = 1550e-9,
    n_eff: float = 1.4682,
    gauge_length: float = 6.38,
    xi: float = 0.78,
    output_in_microstrain: bool = True,
) -> NDArray[np.floating]:
    return _phase_to_strain(
        phase_rad,
        wavelength_m=wavelength_m,
        n_eff=n_eff,
        gauge_length=gauge_length,
        xi=xi,
        output_in_microstrain=output_in_microstrain,
    )


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


def _normalize_matrix_layout(
    data: NDArray[np.floating],
    *,
    matrix_layout: str = "auto",
    time_s: Optional[NDArray[np.floating]] = None,
    distance_m: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    if data.ndim != 2:
        raise ValueError("Expected a 2D matrix.")

    layout = matrix_layout.lower()
    if layout in {"time_space", "time_distance"}:
        return np.asarray(data, dtype=float)
    if layout in {"space_time", "distance_time"}:
        return np.asarray(data, dtype=float).T
    if layout != "auto":
        raise ValueError("matrix_layout must be 'time_distance', 'distance_time', or 'auto'.")

    out = np.asarray(data, dtype=float)

    if distance_m is not None:
        x = np.asarray(distance_m)
        if x.ndim == 1:
            if out.shape[1] == x.size:
                return out
            if out.shape[0] == x.size:
                return out.T

    if time_s is not None:
        t = np.asarray(time_s)
        if t.ndim == 1:
            if out.shape[0] == t.size:
                return out
            if out.shape[1] == t.size:
                return out.T

    return out


def extract_local_signal(
    *,
    data_microstrain: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    stretcher_start_m: float,
    stretcher_end_m: float,
    span_m: float,
    matrix_layout: str = "auto",
) -> Tuple[NDArray[np.floating], float, NDArray[np.int64]]:
    distance = np.asarray(distance_m, dtype=float)
    if distance.ndim != 1:
        raise ValueError("distance_m must be a 1D array.")

    data_td = _normalize_matrix_layout(
        np.asarray(data_microstrain, dtype=float),
        matrix_layout=matrix_layout,
        distance_m=distance,
    )

    if data_td.shape[1] != distance.size:
        raise ValueError(f"data has {data_td.shape[1]} distance points but distance_m has {distance.size}.")

    local_pos = 0.5 * (float(stretcher_start_m) + float(stretcher_end_m))
    mask = (distance >= local_pos - float(span_m)) & (distance <= local_pos + float(span_m))
    idx = np.where(mask)[0].astype(int)

    if idx.size == 0:
        raise ValueError(f"No points found within ±{span_m} m around {local_pos:.2f} m.")

    local_signal = data_td[:, idx].mean(axis=1)
    return local_signal.astype(float), float(local_pos), idx


def load_frequency_response_data(
    *,
    folder_or_file: str,
    fs: float,
    interrogation_rate_hz: float | None = None,
    delta_x_m: float,
    stretcher_start_m: float,
    stretcher_end_m: float,
    span_m: float = 10.0,
    matrix_layout: str = "auto",
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    p = os.path.expanduser(folder_or_file)

    if os.path.isdir(p):
        file_list = sorted(glob.glob(os.path.join(p, "*.npy")))
    else:
        file_list = [p]

    if not file_list:
        raise FileNotFoundError(f"No .npy files found in {folder_or_file!r}")

    matrices: List[NDArray[np.floating]] = []
    for fp in file_list:
        arr = np.asarray(np.load(fp)).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D matrix in {fp!r}, got shape={arr.shape}.")
        arr_td = _normalize_matrix_layout(arr, matrix_layout=matrix_layout)
        matrices.append(arr_td.astype(float))

    data_td = np.concatenate(matrices, axis=0) if len(matrices) > 1 else matrices[0]
    ntime, nspace = data_td.shape

    fs_use = interrogation_rate_hz if interrogation_rate_hz is not None else fs

    time_s = np.arange(ntime, dtype=float) / float(fs_use)
    distance_m = np.arange(nspace, dtype=float) * float(delta_x_m)

    trace_raw, local_pos, _idx = extract_local_signal(
        data_microstrain=data_td,
        distance_m=distance_m,
        stretcher_start_m=stretcher_start_m,
        stretcher_end_m=stretcher_end_m,
        span_m=span_m,
        matrix_layout="time_distance",
    )

    return time_s.astype(float), trace_raw.astype(float), distance_m.astype(float), float(local_pos)


def compute_single_sided_amplitude_spectrum(
    *,
    signal_microstrain: NDArray[np.floating],
    fs: float,
    interrogation_rate_hz: float | None = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    s = np.asarray(signal_microstrain, dtype=float)
    n = int(s.size)
    if n < 2:
        raise ValueError("signal_microstrain must have at least 2 samples.")

    fs_use = interrogation_rate_hz if interrogation_rate_hz is not None else fs

    spec = np.fft.rfft(s)
    amp = np.abs(spec) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_use))
    return freqs.astype(float), amp.astype(float)


def compute_frequency_response(
    *,
    signal_microstrain: NDArray[np.floating],
    interrogation_rate_hz: float,
    n_steps: int = 40,
    freq_min_frac_nyq: float = 0.02,
    freq_max_frac_nyq: float = 0.80,
) -> Dict[str, NDArray[np.floating]]:
    fs = float(interrogation_rate_hz)
    f, amp_ue = compute_single_sided_amplitude_spectrum(
        signal_microstrain=np.asarray(signal_microstrain, dtype=float),
        fs=fs,
    )

    amp_db = 20.0 * np.log10(amp_ue + 1e-20)

    nyq = fs / 2.0
    fmin = float(freq_min_frac_nyq) * nyq
    fmax = float(freq_max_frac_nyq) * nyq
    if not (0.0 < fmin < fmax <= nyq):
        raise ValueError("freq_min_frac_nyq/freq_max_frac_nyq must satisfy 0 < min < max <= 1.")
    if int(n_steps) < 2:
        raise ValueError("n_steps must be >= 2.")

    freq_points = np.linspace(fmin, fmax, int(n_steps))
    vals = np.interp(freq_points, f, amp_db)
    vals_norm = vals - float(np.mean(vals))

    return {
        "frequency_hz": f,
        "amplitude_ue": amp_ue,
        "amplitude_db_re_1ue": amp_db,
        "freq_points_hz": freq_points.astype(float),
        "normalized_db": vals_norm.astype(float),
    }


def plot_time_spectrogram_fft(
    *,
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    fs: float,
    interrogation_rate_hz: float | None = None,
    window_spectrogram_s: float = 0.5,
    overlap_spectrogram_frac: float = 0.5,
    show_plot: bool = True,
    save_results: bool = True,
    results_dir: str = "results_frequency_response",
    filename_prefix: str = "local_signal",
) -> Optional[str]:
    _apply_pyseafom_style()

    t = np.asarray(time_s, dtype=float)
    s = np.asarray(signal_microstrain, dtype=float)
    fs_use = interrogation_rate_hz if interrogation_rate_hz is not None else fs
    fs = float(fs_use)

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, s, color="navy")
    ax1.set_title("Local signal (time)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µε)")
    ax1.set_xlim(t[0], t[-1])
    ax1.grid(True)
    _add_brand_box(ax1)

    nperseg = max(8, int(round(window_spectrogram_s * fs)))
    noverlap = int(round(nperseg * overlap_spectrogram_frac))
    noverlap = min(noverlap, nperseg - 1)

    f, tt, Sxx = spectrogram(s, fs=fs, nperseg=nperseg, noverlap=noverlap)

    ax2 = fig.add_subplot(3, 1, 2)
    pcm = ax2.pcolormesh(tt, f, 10.0 * np.log10(Sxx + 1e-20), shading="gouraud", cmap="jet")
    ax2.set_ylim(0, fs / 2.0)
    ax2.set_title("Spectrogram")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    fig.colorbar(pcm, ax=ax2, label="Power (dB)")
    _add_brand_box(ax2)

    freqs_hz, amp_ue = compute_single_sided_amplitude_spectrum(signal_microstrain=s, fs=fs)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(freqs_hz, amp_ue, color="darkred")
    ax3.set_title("Signal spectrum (single-sided amplitude)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Normalized Amplitude")
    ax3.set_xlim(0, fs / 2.0)
    ax3.grid(True)
    _add_brand_box(ax3)

    fig.tight_layout()

    out_path: Optional[str] = None
    if save_results:
        _ensure_dir(results_dir)
        out_path = os.path.join(results_dir, f"{filename_prefix}_time_spectrogram_fft.png")
        fig.savefig(out_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def analyze_frequency_response(
    *,
    time_s: NDArray[np.floating],
    signal_microstrain: NDArray[np.floating],
    interrogation_rate_hz: float,
    n_steps: int = 40,
    freq_min_frac_nyq: float = 0.02,
    freq_max_frac_nyq: float = 0.80,
    window_spectrogram_s: float = 0.5,
    overlap_spectrogram_frac: float = 0.5,
    save_results: bool = False,
    results_dir: str = "results_frequency_response",
    show_plot: bool = True,
    plot_metadata_box: bool = False,
    plot_local_diagnostics: bool = True,
    folder_or_file: Optional[str] = None,
    local_analysis_position_m: Optional[float] = None,
    gauge_length: Optional[float] = 10.0,
    delta_x_m: Optional[float] = None,
) -> Dict[str, Any]:
    _apply_pyseafom_style()

    t = np.asarray(time_s, dtype=float)
    sig = np.asarray(signal_microstrain, dtype=float)

    if t.ndim != 1 or sig.ndim != 1 or t.size != sig.size:
        raise ValueError("time_s and signal_microstrain must be 1D vectors of the same length.")

    if plot_local_diagnostics:
        plot_time_spectrogram_fft(
            time_s=t,
            signal_microstrain=sig,
            fs=float(interrogation_rate_hz),
            window_spectrogram_s=float(window_spectrogram_s),
            overlap_spectrogram_frac=float(overlap_spectrogram_frac),
            show_plot=show_plot,
            save_results=save_results,
            results_dir=results_dir,
            filename_prefix="frequency_response_local",
        )

    fr = compute_frequency_response(
        signal_microstrain=sig,
        interrogation_rate_hz=float(interrogation_rate_hz),
        n_steps=int(n_steps),
        freq_min_frac_nyq=float(freq_min_frac_nyq),
        freq_max_frac_nyq=float(freq_max_frac_nyq),
    )

    meta_common: List[str] = []
    meta_common.append(f"Repetition Rate: {float(interrogation_rate_hz):.1f} Hz")
    if gauge_length is not None:
        meta_common.append(f"Gauge Length: {float(gauge_length):.2f} m")
    if delta_x_m is not None:
        meta_common.append(f"Spatial Resolution: {float(delta_x_m):.3f} m")
    if local_analysis_position_m is not None:
        meta_common.append(f"Local Position: {float(local_analysis_position_m):.2f} m")

    meta_resp = meta_common + _stats_lines(fr["frequency_hz"], fr["amplitude_db_re_1ue"], label="Response")
    meta_norm = meta_common + _stats_lines(fr["freq_points_hz"], fr["normalized_db"], label="Normalized")

    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(fr["frequency_hz"], fr["amplitude_db_re_1ue"], color="darkgreen")
    ax1.set_xlim(0, float(interrogation_rate_hz) / 2.0)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Strain (dB re 1 µε)")
    ax1.set_title("DAS Frequency Response")
    ax1.grid(True)
    _add_brand_box(ax1)
    if plot_metadata_box:
        _add_metadata_box(ax1, meta_resp)
    fig1.tight_layout()

    full_plot_path: Optional[str] = None
    if save_results:
        _ensure_dir(results_dir)
        full_plot_path = os.path.join(results_dir, "frequency_response.png")
        fig1.savefig(full_plot_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 9))
    ax2.plot(
        fr["freq_points_hz"], fr["normalized_db"],
        "-o", markersize=7, markerfacecolor="r", linewidth=1.2, color="k"
    )
    ymin_target, ymax_target = -6.0, 3.0
    data_min = float(np.min(fr["normalized_db"]))
    data_max = float(np.max(fr["normalized_db"]))

    ymin = min(ymin_target, data_min - 0.5)
    ymax = max(ymax_target, data_max + 0.5)

    ax2.set_ylim(ymin, ymax)


    ax2.set_xlim(0, float(interrogation_rate_hz) / 2.0 * 0.82)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude (dB)")
    ax2.set_title("Normalized Frequency Response")
    ax2.grid(True)
    _add_brand_box(ax2)
    if plot_metadata_box:
        _add_metadata_box(ax2, meta_norm)
    fig2.tight_layout()

    norm_plot_path: Optional[str] = None
    if save_results:
        _ensure_dir(results_dir)
        norm_plot_path = os.path.join(results_dir, "frequency_response_normalized.png")
        fig2.savefig(norm_plot_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig2)

    normalized_csv: Optional[str] = None
    if save_results:
        _ensure_dir(results_dir)
        normalized_csv = os.path.join(results_dir, "frequency_response_normalized.csv")
        with open(normalized_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frequency_hz", "normalized_db"])
            for fhz, ndb in zip(fr["freq_points_hz"], fr["normalized_db"]):
                w.writerow([format(float(fhz), ".2f"), format(float(ndb), ".2f")])



    return {
        "local_analysis_position_m": None if local_analysis_position_m is None else float(local_analysis_position_m),
        "signal_microstrain": sig,
        "frequency_hz": fr["frequency_hz"],
        "amplitude_db_re_1ue": fr["amplitude_db_re_1ue"],
        "freq_points_hz": fr["freq_points_hz"],
        "normalized_db": fr["normalized_db"],
        "plot_response_path": full_plot_path,
        "plot_normalized_path": norm_plot_path,
        "csv_normalized_path": normalized_csv,
    }
