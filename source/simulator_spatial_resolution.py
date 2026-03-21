"""
simulator_spatial_resolution.py
===================================
Synthetic data generator for the SEAFOM MSP-02 Spatial Resolution test.

This version was reorganized to stay closer to the style used in the
reference simulators of the project:
    - direct operational blocks
    - .npy output for the main matrix
    - optional quick-look plots
    - explicit [INFO] / [OK] messages

Output
------
synthetic_sr_data.npy
    2D matrix with shape (n_ssl, n_samples), in microstrain [ue]
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# CONTROL
# ======================================================================
SEED = 42

OUTPUT_DIR = "."
OUTPUT_FILE = "synthetic_sr_data.npy"

SAVE_FILE = True
SHOW_PLOTS = True
SAVE_COLORMAP = True
COLORMAP_FILE = "synthetic_sr_colormap.png"


# ======================================================================
# ACQUISITION
# ======================================================================
FS = 5000.0                  # [Hz]
TOTAL_DURATION_S = 7.0       # [s]
FFT_SIZE = 16384             # [samples]


# ======================================================================
# SPATIAL GRID
# ======================================================================
SECTION_LENGTH_M = 300.0     # [m]
CHANNEL_SPACING_M = 1.0      # [m/SSL]


# ======================================================================
# STIMULUS
# ======================================================================
STIMULUS_FREQ_HZ = 100.0     # [Hz]
NOISE_STD_UE = 0.005         # [microstrain]


# ======================================================================
# IU / STRETCHER
# ======================================================================
GAUGE_LENGTH_M = 10.0
STRETCHER_CENTER_M = 150.0
STRETCHER_LENGTH_M = 22.0


# ======================================================================
# SPATIAL PROFILE
# ======================================================================
RAMP_LEFT_M = 10.0
RAMP_RIGHT_M = 12.5
SHARPNESS = 1.0


# ======================================================================
# DISPLAY
# ======================================================================
DISPLAY_MAX_TIME_S = 0.5
DISPLAY_DOWNSAMPLE = 1
COLORMAP = "jet"


# ======================================================================
# HELPERS
# ======================================================================
def _derive_n_samples(total_duration_s: float, fs_hz: float, fft_size: int) -> int:
    """
    Round the number of samples up to the nearest FFT block multiple.
    """
    n_blocks = math.ceil(total_duration_s * fs_hz / fft_size)
    return int(n_blocks * fft_size)


def _derive_n_ssl(section_length_m: float, channel_spacing_m: float) -> int:
    """
    Derive number of SSLs from section length and channel spacing.
    """
    return int(math.ceil(section_length_m / channel_spacing_m) + 1)


def _get_stimulus_amplitude_ue(gauge_length_m: float) -> float:
    """
    MSP-02 Step 2:
        peak amplitude = 0.5 microstrain / gauge length
    """
    return 0.5 / gauge_length_m


def _build_spatial_profile(
    *,
    ssl_positions_m: np.ndarray,
    stretcher_center_m: float,
    stretcher_length_m: float,
    ramp_left_m: float,
    ramp_right_m: float,
    sharpness: float,
    amplitude_peak_ue: float,
) -> np.ndarray:
    """
    Build a generalized trapezoidal spatial profile.
    """
    if sharpness <= 0:
        raise ValueError(f"sharpness must be > 0, got {sharpness}.")
    if ramp_left_m <= 0 or ramp_right_m <= 0:
        raise ValueError("ramp_left_m and ramp_right_m must be > 0.")

    amp_profile = np.zeros_like(ssl_positions_m, dtype=float)

    half_len = 0.5 * stretcher_length_m
    x_left = stretcher_center_m - half_len
    x_right = stretcher_center_m + half_len

    x_left_ramp_start = x_left - ramp_left_m
    x_right_ramp_end = x_right + ramp_right_m

    for i, x in enumerate(ssl_positions_m):
        if x <= x_left_ramp_start or x >= x_right_ramp_end:
            amp_profile[i] = 0.0

        elif x_left <= x <= x_right:
            amp_profile[i] = amplitude_peak_ue

        elif x_left_ramp_start < x < x_left:
            u = (x - x_left_ramp_start) / ramp_left_m
            amp_profile[i] = amplitude_peak_ue * (u ** sharpness)

        else:
            u = (x_right_ramp_end - x) / ramp_right_m
            amp_profile[i] = amplitude_peak_ue * (u ** sharpness)

    return amp_profile


def _plot_colormap(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    ssl_positions_m: np.ndarray,
    stretcher_center_m: float,
    stretcher_length_m: float,
    max_time_s: float = DISPLAY_MAX_TIME_S,
    downsample: int = DISPLAY_DOWNSAMPLE,
    cmap: str = COLORMAP,
    save_path: str | None = None,
) -> None:
    """
    Plot a quick-look colormap of the simulated matrix.

    Matrix convention:
        data.shape = (n_ssl, n_samples)

    To match the visual style of the reference simulators, the image is shown as:
        x-axis -> time
        y-axis -> space
    """
    t_mask = time_s <= min(max_time_s, time_s[-1])
    time_disp = time_s[t_mask][::downsample]
    data_disp = data[:, t_mask][:, ::downsample]

    vmax = np.max(np.abs(data_disp))
    if vmax == 0:
        vmax = 1.0

    extent = [time_disp[0], time_disp[-1], ssl_positions_m[0], ssl_positions_m[-1]]

    s_lo = stretcher_center_m - 0.5 * stretcher_length_m
    s_hi = stretcher_center_m + 0.5 * stretcher_length_m

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        data_disp,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, label="Strain [ue]")
    plt.xlabel("Time [s]")
    plt.ylabel("Space [m]")
    plt.title("Simulated strain matrix (spatial resolution)")

    plt.axhline(s_lo, color="white", linestyle="--", linewidth=1.2)
    plt.axhline(s_hi, color="white", linestyle="--", linewidth=1.2)
    plt.axhline(stretcher_center_m, color="yellow", linestyle=":", linewidth=1.2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[OK] Colormap saved: {os.path.abspath(save_path)}")

    plt.show()


def simulate_spatial_resolution_data(
    *,
    total_duration_s: float = TOTAL_DURATION_S,
    fs_hz: float = FS,
    fft_size: int = FFT_SIZE,
    section_length_m: float = SECTION_LENGTH_M,
    channel_spacing_m: float = CHANNEL_SPACING_M,
    stimulus_freq_hz: float = STIMULUS_FREQ_HZ,
    gauge_length_m: float = GAUGE_LENGTH_M,
    stretcher_center_m: float = STRETCHER_CENTER_M,
    stretcher_length_m: float = STRETCHER_LENGTH_M,
    ramp_left_m: float = RAMP_LEFT_M,
    ramp_right_m: float = RAMP_RIGHT_M,
    sharpness: float = SHARPNESS,
    noise_std_ue: float = NOISE_STD_UE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate synthetic data for the Spatial Resolution test.

    Returns
    -------
    data : ndarray
        Simulated matrix with shape (n_ssl, n_samples) [ue]
    ssl_positions_m : ndarray
        Spatial vector [m]
    time_s : ndarray
        Time vector [s]
    amp_profile : ndarray
        Spatial amplitude profile [ue]
    meta : dict
        Generation metadata
    """
    if stimulus_freq_hz != 100.0:
        print(
            f"[WARN] stimulus_freq_hz = {stimulus_freq_hz:.2f} Hz. "
            "MSP-02 Step 2 specifies 100 Hz."
        )

    if stretcher_length_m < 2.0 * gauge_length_m:
        print(
            "[WARN] stretcher_length_m < 2 x gauge_length_m. "
            "MSP-02 recommends stretcher length >= 2 x GL."
        )

    rng = np.random.default_rng(seed)

    n_samples = _derive_n_samples(total_duration_s, fs_hz, fft_size)
    n_ssl = _derive_n_ssl(section_length_m, channel_spacing_m)
    actual_duration_s = n_samples / fs_hz

    time_s = np.arange(n_samples, dtype=float) / fs_hz
    ssl_positions_m = np.arange(n_ssl, dtype=float) * channel_spacing_m

    amplitude_peak_ue = _get_stimulus_amplitude_ue(gauge_length_m)

    amp_profile = _build_spatial_profile(
        ssl_positions_m=ssl_positions_m,
        stretcher_center_m=stretcher_center_m,
        stretcher_length_m=stretcher_length_m,
        ramp_left_m=ramp_left_m,
        ramp_right_m=ramp_right_m,
        sharpness=sharpness,
        amplitude_peak_ue=amplitude_peak_ue,
    )

    sine_wave = np.sin(2.0 * np.pi * stimulus_freq_hz * time_s)
    data = np.outer(amp_profile, sine_wave)
    data += noise_std_ue * rng.standard_normal(data.shape)

    meta = {
        "fs": float(fs_hz),
        "n_samples": int(n_samples),
        "n_ssl": int(n_ssl),
        "actual_duration_s": float(actual_duration_s),
        "section_length_m": float(section_length_m),
        "channel_spacing_m": float(channel_spacing_m),
        "stimulus_freq_hz": float(stimulus_freq_hz),
        "gauge_length_m": float(gauge_length_m),
        "stretcher_center_m": float(stretcher_center_m),
        "stretcher_length_m": float(stretcher_length_m),
        "ramp_left_m": float(ramp_left_m),
        "ramp_right_m": float(ramp_right_m),
        "sharpness": float(sharpness),
        "amplitude_peak_ue": float(amplitude_peak_ue),
        "noise_std_ue": float(noise_std_ue),
        "fft_size": int(fft_size),
        "seed": int(seed),
    }

    return data, ssl_positions_m, time_s, amp_profile, meta


def generate_spatial_resolution_data(*args, **kwargs):
    """Backward-compatible alias for simulate_spatial_resolution_data()."""
    return simulate_spatial_resolution_data(*args, **kwargs)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":

    print("=" * 70)
    print(" Synthetic Data Generation - Spatial Resolution (MSP-02 §7.5)")
    print("=" * 70)

    data, ssl_positions_m, time_s, amp_profile, meta = simulate_spatial_resolution_data()

    symmetric_tag = "symmetric" if np.isclose(RAMP_LEFT_M, RAMP_RIGHT_M) else "asymmetric"

    if np.isclose(SHARPNESS, 1.0):
        sharpness_tag = "linear"
    elif SHARPNESS > 1.0:
        sharpness_tag = "sharp"
    else:
        sharpness_tag = "spread"

    print(f"[INFO] seed={SEED}")
    print(
        f"[INFO] n_ssl={meta['n_ssl']}, n_samples={meta['n_samples']} | "
        f"fs={meta['fs']:.1f} Hz | duration={meta['actual_duration_s']:.4f} s"
    )
    print(
        f"[INFO] space=[{ssl_positions_m[0]:.1f}, {ssl_positions_m[-1]:.1f}] m | "
        f"dx={CHANNEL_SPACING_M:.3f} m"
    )
    print(
        f"[INFO] stretcher center={STRETCHER_CENTER_M:.2f} m | "
        f"length={STRETCHER_LENGTH_M:.2f} m | GL={GAUGE_LENGTH_M:.2f} m"
    )
    print(
        f"[INFO] ramps: left={RAMP_LEFT_M:.2f} m | right={RAMP_RIGHT_M:.2f} m | "
        f"shape={symmetric_tag}, {sharpness_tag}, sharpness={SHARPNESS:.3f}"
    )
    print(
        f"[INFO] stimulus={STIMULUS_FREQ_HZ:.2f} Hz | "
        f"peak amplitude={meta['amplitude_peak_ue']:.6f} ue | "
        f"noise std={NOISE_STD_UE:.6f} ue"
    )

    if SHOW_PLOTS:
        _plot_colormap(
            data=data,
            time_s=time_s,
            ssl_positions_m=ssl_positions_m,
            stretcher_center_m=STRETCHER_CENTER_M,
            stretcher_length_m=STRETCHER_LENGTH_M,
            max_time_s=DISPLAY_MAX_TIME_S,
            downsample=DISPLAY_DOWNSAMPLE,
            cmap=COLORMAP,
            save_path=COLORMAP_FILE if SAVE_COLORMAP else None,
        )

    if SAVE_FILE:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        np.save(output_path, data)

        print(f"[OK] Saved: {os.path.abspath(output_path)}")
        print(f"     Shape: {data.shape} (n_ssl, n_samples)")
        print(f"     Units: strain [ue] | seed={SEED}")
        print(f"     Amp range: [{amp_profile.min():.6f}, {amp_profile.max():.6f}] ue")
        print(f"     Time range: [{time_s[0]:.6f}, {time_s[-1]:.6f}] s")
        print(f"     Space: [{ssl_positions_m[0]:.2f}, {ssl_positions_m[-1]:.2f}] m")

    print("=" * 70)