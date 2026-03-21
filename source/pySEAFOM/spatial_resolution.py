from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft
from scipy.signal import butter, filtfilt, windows


# =============================================================================
# Plot style
# =============================================================================

def _apply_pyseafom_style() -> None:
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


# =============================================================================
# Signal processing
# =============================================================================

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
        if sig.ndim == 1:
            sig = filtfilt(b, a, sig)
        elif sig.ndim == 2:
            sig = filtfilt(b, a, sig, axis=1)
        else:
            raise ValueError("trace must be a 1D or 2D array.")

    return sig.astype(float)


# =============================================================================
# Load data
# =============================================================================

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


def load_spatial_resolution_data(
    *,
    folder_or_file: str,
    fs: float,
    interrogation_rate_hz: float | None = None,
    delta_x_m: float,
    x1_m: float,
    x2_m: float,
    time_start_s: float = 0.0,
    duration: Optional[float] = None,
    matrix_layout: str = "auto",
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
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
    ntime_full, nspace = data_td.shape

    fs_use = interrogation_rate_hz if interrogation_rate_hz is not None else fs

    frame_start = int(time_start_s * fs_use)
    if duration is None:
        frame_end = ntime_full
    else:
        frame_end = min(int((time_start_s + duration) * fs_use), ntime_full)

    if frame_start >= frame_end:
        raise ValueError("Invalid time window (frame_start >= frame_end).")

    data_td = data_td[frame_start:frame_end, :]
    time_s = np.arange(data_td.shape[0], dtype=float) / fs_use + time_start_s

    idx_start = int(x1_m / delta_x_m)
    idx_end = int(x2_m / delta_x_m)
    idx_start = max(idx_start, 0)
    idx_end = min(idx_end, nspace)

    if idx_start >= idx_end:
        raise ValueError("Invalid spatial range.")

    data_td = data_td[:, idx_start:idx_end]
    distance_m = np.arange(idx_start, idx_end, dtype=float) * delta_x_m
    section_data = data_td.T

    return time_s.astype(float), distance_m.astype(float), section_data.astype(float)


# =============================================================================
# Core analysis
# =============================================================================

def _validate_section_data(section_data: NDArray[np.floating]) -> NDArray[np.floating]:
    data = np.asarray(section_data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"section_data must be a 2D array with shape (n_ssl, n_samples). Got {data.shape}."
        )

    if data.shape[0] < 3:
        raise ValueError("section_data must contain at least 3 SSLs.")

    if data.shape[1] < 16:
        raise ValueError("section_data must contain at least 16 time samples.")

    return data


def _extract_tone_profile(
    section_data: NDArray[np.floating],
    *,
    fs: float,
    ref_freq_hz: float,
    fft_size: int,
) -> NDArray[np.floating]:
    data = _validate_section_data(section_data)
    n_ssl, n_samples = data.shape

    n_blocks = n_samples // fft_size
    if n_blocks == 0:
        raise ValueError(
            f"n_samples ({n_samples}) < fft_size ({fft_size}). "
            "Increase duration so at least one full FFT block is available."
        )

    window = windows.flattop(fft_size)
    freqs_hz = np.fft.fftfreq(fft_size, d=1.0 / fs)[: fft_size // 2]
    idx_ref = int(np.argmin(np.abs(freqs_hz - ref_freq_hz)))

    amp_profile = np.zeros(n_ssl, dtype=float)

    for i in range(n_ssl):
        sig = data[i]
        vals = []

        for blk in range(n_blocks):
            i0 = blk * fft_size
            i1 = (blk + 1) * fft_size
            x = sig[i0:i1]
            Y = np.abs(fft(x * window))[: fft_size // 2]
            vals.append(float(Y[idx_ref]))

        amp_profile[i] = float(np.mean(vals))

    return amp_profile.astype(float)


def _fit_slope_width(
    distance_m: NDArray[np.floating],
    amp_norm: NDArray[np.floating],
    *,
    side: str,
) -> Tuple[float, Optional[NDArray[np.floating]]]:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    x = np.asarray(distance_m, dtype=float)
    y = np.asarray(amp_norm, dtype=float)

    if not np.any(np.isfinite(y)):
        return np.nan, None

    peak_idx = int(np.nanargmax(y))
    mask = (y >= 0.05) & (y <= 0.95)

    if side == "left":
        mask[peak_idx:] = False
    else:
        mask[: peak_idx + 1] = False

    if np.count_nonzero(mask) < 2:
        return np.nan, None

    x_fit = x[mask]
    y_fit = y[mask]
    coeffs = np.polyfit(x_fit, y_fit, 1)

    a, b = coeffs
    if abs(a) < 1e-12:
        return np.nan, coeffs.astype(float)

    x0 = (0.0 - b) / a
    x1 = (1.0 - b) / a
    width_m = abs(x1 - x0)

    return float(width_m), coeffs.astype(float)



def _save_spatial_resolution_summary_csv(result: Dict[str, Any], results_dir: str) -> str:
    _ensure_dir(results_dir)

    csv_path = os.path.join(results_dir, "spatial_resolution_summary.csv")

    rows = [
        ("LL_m", result["LL_m"], "m"),
        ("LR_m", result["LR_m"], "m"),
        ("spatial_resolution_m", result["spatial_resolution_m"], "m"),
        ("peak_position_m", result["peak_pos_m"], "m"),
        ("snr_db", result["snr_db"], "dB"),
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("metric,value,unit\n")
        for metric, value, unit in rows:
            if np.isfinite(value):
                f.write(f"{metric},{value:.10g},{unit}\n")
            else:
                f.write(f"{metric},nan,{unit}\n")

    print(f"[OK] CSV summary saved: {os.path.abspath(csv_path)}")
    return csv_path






def calculate_spatial_resolution(
    *,
    section_data: NDArray[np.floating],
    fs: float,
    ref_freq_hz: float,
    target_pos_m: float,
    gauge_length: Optional[float] = None,
    delta_x_m: float = 1.0,
    fft_size: int = 16384,
    snr_threshold_db: float = 20.0,
    time_s: Optional[NDArray[np.floating]] = None,
    x_vec_m: Optional[NDArray[np.floating]] = None,
    stretcher_length_m: Optional[float] = None,
    highpass_hz: Optional[float] = None,
    save_results: bool = False,
    results_dir: str = "results_spatial_resolution",
    show_plot: bool = True,
    plot_metadata_box: bool = False,
    plot_spatiotemporal_map: bool = True,
    plot_profile: bool = True,
    print_report: bool = True,
) -> Dict[str, Any]:

    data = _validate_section_data(section_data)
    n_ssl = data.shape[0]

    if x_vec_m is None:
        distance_m = np.arange(n_ssl, dtype=float) * delta_x_m
    else:
        distance_m = np.asarray(x_vec_m, dtype=float)
        if distance_m.ndim != 1 or distance_m.size != n_ssl:
            raise ValueError(
                "x_vec_m must be a 1D array with the same length as section_data.shape[0]."
            )

    amp_profile = _extract_tone_profile(
        data,
        fs=fs,
        ref_freq_hz=ref_freq_hz,
        fft_size=fft_size,
    )

    peak_amp = float(np.nanmax(amp_profile))
    if peak_amp <= 0:
        raise ValueError("Amplitude profile peak is zero or invalid. Cannot normalize.")

    amp_norm = amp_profile / peak_amp

    noise_floor = float(np.nanpercentile(amp_profile, 5))
    snr_lin = peak_amp / max(noise_floor, 1e-12)
    snr_db = 20.0 * np.log10(snr_lin)

    if snr_db < snr_threshold_db:
        print(
            f"[WARN] SNR = {snr_db:.1f} dB below threshold "
            f"{snr_threshold_db:.1f} dB. Result may be invalid."
        )

    peak_idx = int(np.nanargmax(amp_profile))
    peak_pos_m = float(distance_m[peak_idx])

    print(f"[INFO] Target position : {target_pos_m:.2f} m")
    print(f"[INFO] Detected peak   : {peak_pos_m:.2f} m")
    print(f"[INFO] SNR            : {snr_db:.1f} dB")

    ll_m, coeffs_l = _fit_slope_width(distance_m, amp_norm, side="left")
    lr_m, coeffs_r = _fit_slope_width(distance_m, amp_norm, side="right")

    if np.isfinite(ll_m) and np.isfinite(lr_m):
        spatial_resolution_m = 0.5 * (ll_m + lr_m)
    else:
        spatial_resolution_m = np.nan
        print("[WARN] Could not estimate LL and/or LR.")

    result = {
        "x_vec_m": distance_m,
        "amp_profile": amp_profile,
        "amp_norm": amp_norm,
        "peak_idx": peak_idx,
        "peak_pos_m": peak_pos_m,
        "LL_m": float(ll_m) if np.isfinite(ll_m) else np.nan,
        "LR_m": float(lr_m) if np.isfinite(lr_m) else np.nan,
        "LL_ssl": float(ll_m / delta_x_m) if np.isfinite(ll_m) else np.nan,
        "LR_ssl": float(lr_m / delta_x_m) if np.isfinite(lr_m) else np.nan,
        "spatial_resolution_m": float(spatial_resolution_m) if np.isfinite(spatial_resolution_m) else np.nan,
        "snr_db": float(snr_db),
        "coeffs_L": coeffs_l,
        "coeffs_R": coeffs_r,
        "params": {
            "fs": float(fs),
            "delta_x_m": float(delta_x_m),
            "target_pos_m": float(target_pos_m),
            "gauge_length": float(gauge_length) if gauge_length is not None else None,
            "ref_freq_hz": float(ref_freq_hz),
            "fft_size": int(fft_size),
            "snr_threshold_db": float(snr_threshold_db),
            "n_ssl": int(n_ssl),
            "n_samples": int(data.shape[1]),
        },
    }

    if print_report:
        report_spatial_resolution(result)

    if save_results:
        _ensure_dir(results_dir)
        _save_spatial_resolution_summary_csv(result, results_dir)

    should_generate_plots = show_plot or save_results

    if plot_spatiotemporal_map and time_s is not None and should_generate_plots:
        title = f"Spatial Resolution - Spatiotemporal Map\nRef. freq = {ref_freq_hz:.0f} Hz"
        if gauge_length is not None:
            title += f" | GL = {gauge_length:.1f} m"
        title += f" | Target = {target_pos_m:.0f} m"

        fig1, _ = plot_spatiotemporal(
            data,
            distance_m,
            np.asarray(time_s, dtype=float),
            title=title,
            fs=fs,
            gauge_length=gauge_length,
            delta_x_m=delta_x_m,
            highpass_hz=highpass_hz,
            ref_freq_hz=ref_freq_hz,
            stretcher_center_m=target_pos_m,
            stretcher_length_m=stretcher_length_m,
            show_plot=show_plot,
            save_path=os.path.join(results_dir, "spatiotemporal_map.png") if save_results else None,
            plot_metadata_box=plot_metadata_box,
        )
        if not show_plot:
            plt.close(fig1)

    if plot_profile and should_generate_plots:
        title = "Spatial Resolution Profile"
        if gauge_length is not None:
            title += f"\nGL = {gauge_length:.1f} m | SR = {result['spatial_resolution_m']:.3f} m"
        else:
            title += f"\nSR = {result['spatial_resolution_m']:.3f} m"

        fig2, _ = plot_spatial_resolution_profile(
            result,
            title=title,
            ref_freq_hz=ref_freq_hz,
            gauge_length=gauge_length,
            delta_x_m=delta_x_m,
            show_plot=show_plot,
            save_path=os.path.join(results_dir, "spatial_resolution_profile.png") if save_results else None,
            plot_metadata_box=plot_metadata_box,
        )
        if not show_plot:
            plt.close(fig2)

    if save_results:
        print(f"[OK] Results saved in: {os.path.abspath(results_dir)}")

    return result


# =============================================================================
# Plot functions
# =============================================================================

def plot_spatiotemporal(
    section_data: NDArray[np.floating],
    x_vec_m: NDArray[np.floating],
    time_s: NDArray[np.floating],
    *,
    title: str = "Spatial Resolution - Spatiotemporal Map",
    fs: Optional[float] = None,
    gauge_length: Optional[float] = None,
    delta_x_m: Optional[float] = None,
    highpass_hz: Optional[float] = None,
    ref_freq_hz: Optional[float] = None,
    stretcher_center_m: Optional[float] = None,
    stretcher_length_m: Optional[float] = None,
    show_plot: bool = False,
    save_path: Optional[str] = None,
    plot_metadata_box: bool = True,
) -> Tuple[plt.Figure, Any]:
    _apply_pyseafom_style()

    data = _validate_section_data(section_data)
    distance_m = np.asarray(x_vec_m, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    max_time_s = min(0.5, float(time_s[-1]))
    time_mask = time_s <= max_time_s
    data_disp = data[:, time_mask]
    time_disp = time_s[time_mask]

    vmax = float(np.percentile(np.abs(data_disp), 99))
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax0 = axes[0]
    im = ax0.imshow(
        data_disp,
        aspect="auto",
        extent=[time_disp[0], time_disp[-1], distance_m[-1], distance_m[0]],
        cmap="jet",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax0, label="Strain [µε]", fraction=0.02, pad=0.01)

    if stretcher_center_m is not None and stretcher_length_m is not None:
        x_lo = stretcher_center_m - 0.5 * stretcher_length_m
        x_hi = stretcher_center_m + 0.5 * stretcher_length_m
        ax0.axhline(x_lo, color="white", lw=1.5, ls="--")
        ax0.axhline(x_hi, color="white", lw=1.5, ls="--")
        ax0.axhline(stretcher_center_m, color="yellow", lw=1.2, ls=":")

    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Position [m]")
    ax0.set_title(title, fontsize=16, fontweight="bold")

    if plot_metadata_box:
        metadata = []
        if ref_freq_hz is not None:
            metadata.append(f"Ref. frequency: {ref_freq_hz:.2f} Hz")
        if gauge_length is not None:
            metadata.append(f"Gauge Length: {gauge_length:.2f} m")
        if delta_x_m is not None:
            metadata.append(f"Spatial Sampling: {delta_x_m:.2f} m")
        if highpass_hz is not None and highpass_hz > 0.0:
            metadata.append(f"High-pass: {highpass_hz:.2f} Hz")
        if fs is not None:
            metadata.append(f"Repetition Rate: {fs:.1f} Hz")
        _add_metadata_box(ax0, metadata)

    _add_brand_box(ax0)

    ax1 = axes[1]
    idx_center = len(distance_m) // 2 if stretcher_center_m is None else int(np.argmin(np.abs(distance_m - stretcher_center_m)))
    idx_left = max(idx_center - 3, 0)
    idx_right = min(idx_center + 3, len(distance_m) - 1)

    for idx, label in [
        (idx_left, f"Left ({distance_m[idx_left]:.0f} m)"),
        (idx_center, f"Center ({distance_m[idx_center]:.0f} m)"),
        (idx_right, f"Right ({distance_m[idx_right]:.0f} m)"),
    ]:
        ax1.plot(time_disp, data_disp[idx], lw=1.0, label=label)

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Strain [µε]")
    ax1.set_title("Representative traces", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(os.path.abspath(save_path)) or ".")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Spatiotemporal figure saved: {os.path.abspath(save_path)}")

    if show_plot:
        plt.show()

    return fig, axes


def plot_spatial_resolution_profile(
    result: Dict[str, Any],
    *,
    title: str = "Spatial Resolution Profile",
    ref_freq_hz: Optional[float] = None,
    gauge_length: Optional[float] = None,
    delta_x_m: Optional[float] = None,
    show_plot: bool = False,
    save_path: Optional[str] = None,
    plot_metadata_box: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    _apply_pyseafom_style()

    x = np.asarray(result["x_vec_m"], dtype=float)
    y = np.asarray(result["amp_norm"], dtype=float)
    coeffs_L = result["coeffs_L"]
    coeffs_R = result["coeffs_R"]
    LL_m = result["LL_m"]
    LR_m = result["LR_m"]
    SR_m = result["spatial_resolution_m"]
    snr_db = result["snr_db"]
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(
        x,
        y,
        width=dx * 0.85,
        color="steelblue",
        alpha=0.60,
        label="Normalized amplitude per SSL",
    )

    def _slope_xy(coeffs: NDArray[np.floating]) -> List[float]:
        a, b = coeffs
        return sorted([(0.0 - b) / a, (1.0 - b) / a])

    if coeffs_L is not None and np.isfinite(LL_m):
        xl0, xl1 = _slope_xy(coeffs_L)
        yl = [0, 1] if coeffs_L[0] > 0 else [1, 0]
        ax.plot([xl0, xl1], yl, "r-", lw=2.5, label="Piecewise fit")

    if coeffs_R is not None and np.isfinite(LR_m):
        xr0, xr1 = _slope_xy(coeffs_R)
        yr = [1, 0] if coeffs_R[0] < 0 else [0, 1]
        ax.plot([xr0, xr1], yr, "r-", lw=2.5)

    if coeffs_L is not None and coeffs_R is not None and np.isfinite(LL_m) and np.isfinite(LR_m):
        x_top_l = (1.0 - coeffs_L[1]) / coeffs_L[0]
        x_top_r = (1.0 - coeffs_R[1]) / coeffs_R[0]
        ax.plot([x_top_l, x_top_r], [1.0, 1.0], "r-", lw=2.5)

    if np.isfinite(LL_m) and coeffs_L is not None:
        xl0, xl1 = _slope_xy(coeffs_L)
        xm = 0.5 * (xl0 + xl1)
        ax.annotate(
            "",
            xy=(xl1, 0.50),
            xytext=(xl0, 0.50),
            arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.8),
        )
        ax.text(
            xm,
            0.53,
            f"LL = {LL_m:.2f} m",
            ha="center",
            color="darkred",
            fontsize=10,
            fontweight="bold",
        )

    if np.isfinite(LR_m) and coeffs_R is not None:
        xr0, xr1 = _slope_xy(coeffs_R)
        xm = 0.5 * (xr0 + xr1)
        ax.annotate(
            "",
            xy=(xr1, 0.50),
            xytext=(xr0, 0.50),
            arrowprops=dict(arrowstyle="<->", color="darkorange", lw=1.8),
        )
        ax.text(
            xm,
            0.53,
            f"LR = {LR_m:.2f} m",
            ha="center",
            color="darkorange",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Position along fibre [m]")
    ax.set_ylabel("Normalized amplitude [-]")
    ax.set_ylim(-0.05, 1.18)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=10)

    if plot_metadata_box:
        metadata = []
        if ref_freq_hz is not None:
            metadata.append(f"Ref. frequency: {ref_freq_hz:.2f} Hz")
        if gauge_length is not None:
            metadata.append(f"Gauge Length: {gauge_length:.2f} m")
        if delta_x_m is not None:
            metadata.append(f"Spatial Sampling: {delta_x_m:.2f} m")
        metadata.extend([
            f"LL [m]       : {LL_m:.3f}",
            f"LR [m]       : {LR_m:.3f}",
            f"SR [m]       : {SR_m:.3f}",
            f"SNR [dB]     : {snr_db:.1f}",
            f"Peak pos [m] : {result['peak_pos_m']:.2f}",
        ])
        _add_metadata_box(ax, metadata)

    _add_brand_box(ax)

    plt.tight_layout()

    if save_path:
        _ensure_dir(os.path.dirname(os.path.abspath(save_path)) or ".")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] SR profile figure saved: {os.path.abspath(save_path)}")

    if show_plot:
        plt.show()

    return fig, ax


# =============================================================================
# Report
# =============================================================================

def report_spatial_resolution(result: Dict[str, Any]) -> None:
    p = result["params"]
    SR = result["spatial_resolution_m"]
    GL = p["gauge_length"]
    dev = SR - GL if GL is not None and np.isfinite(SR) else np.nan

    print("-" * 70)
    print(" Spatial Resolution Result")
    print("-" * 70)
    print(f"Target position        : {p['target_pos_m']:.2f} m")
    print(f"Detected peak          : {result['peak_pos_m']:.2f} m")
    print(f"Reference frequency    : {p['ref_freq_hz']:.2f} Hz")
    if GL is not None:
        print(f"Gauge length           : {GL:.3f} m")
    print(f"Spatial sampling       : {p['delta_x_m']:.3f} m")
    print(f"SNR                    : {result['snr_db']:.2f} dB")
    print("-" * 70)
    print(f"LL                     : {result['LL_m']:.3f} m  ({result['LL_ssl']:.2f} SSL)")
    print(f"LR                     : {result['LR_m']:.3f} m  ({result['LR_ssl']:.2f} SSL)")
    print(f"Spatial Resolution     : {SR:.3f} m")
    if GL is not None:
        print(f"Deviation from GL      : {dev:+.3f} m")
    print("-" * 70)


def plot_sr_profile(*args, **kwargs):
    """Backward-compatible alias for plot_spatial_resolution_profile()."""
    return plot_spatial_resolution_profile(*args, **kwargs)


def print_sr_report(*args, **kwargs):
    """Backward-compatible alias for report_spatial_resolution()."""
    return report_spatial_resolution(*args, **kwargs)
