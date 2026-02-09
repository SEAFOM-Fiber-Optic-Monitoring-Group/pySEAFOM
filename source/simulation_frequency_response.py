# This script generates synthetic DAS data directly in PHASE [rad] for the
# MSP-02 "Frequency Response" test and saves it as a 2D .npy array.
#
# data.shape = (time, space/channels)
#   - axis 0 (rows): time samples
#       * sampling rate: fs = REP_RATE_HZ  [Hz]
#       * time step:     dt = 1/fs                  [s]
#   - axis 1 (cols): spatial channels / distance
#       * spatial step resolution:  dx = DELTA_X_M  [m]
#       * channel i corresponds approximately to x = i * dx

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, filtfilt


# ======================================================================
# CONTROL
# ======================================================================
SEED = 12345                  # Fixed seed for deterministic results

# IMPORTANT:
# - frequency_response_test.py uses BASE_PATH="." and expects:
#     * dataDAS.npy
OUTPUT_DIR = "."              
OUTPUT_FILE = "dataDAS.npy"

SHOW_PLOTS = True
SAVE_FILE = True

# ======================================================================
# MATRIX (time × distance)
# ======================================================================
REP_RATE_HZ = 10_000.0   # [Hz]
TOTAL_TIME_S = 100.0              # [s]

DELTA_X_M = 1.0                   # [m]
LENGTH_M = 50.0                   # [m]

# ======================================================================
# MSP-02 STEP TEST (frequency program)
# ======================================================================
N_STEPS = 40                      # number of frequency steps
FREQ_MIN_FRAC_NYQ = 0.02          # min fraction of Nyquist
FREQ_MAX_FRAC_NYQ = 0.80          # max fraction of Nyquist

# ======================================================================
# PHYSICAL PARAMETERS (phase amplitude derived from strain target)
# ======================================================================
GAUGE_LENGTH_M = 10.0             # [m]
LAMBDA_LASER_M = 1550e-9          # [m]
XI = 0.78                         # [-]
N_EFF = 1.4682                    # [-]

TARGET_STRAIN_PEAK = 0.08e-6 / GAUGE_LENGTH_M  # [strain]

# ======================================================================
# NOISE (phase)
# ======================================================================
NOISE_STD_RAD = 0.05            # [rad]

# ======================================================================
# STRETCHER REGION (space)
# ======================================================================
STRETCHER_START_M = 20.0          # [m]
STRETCHER_END_M = 30.0            # [m]

# ======================================================================
# SPECTROGRAM DIAGNOSTICS
# ======================================================================
WINDOW_SPECTROGRAM_S = 0.5
OVERLAP_SPECTROGRAM_FRAC = 0.5


def _build_step_frequency_signal(
    *,
    fs_hz: float,
    total_time_s: float,
    n_steps: int,
    fmin_frac_nyq: float,
    fmax_frac_nyq: float,
    amplitude_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build MSP-02-like stepwise sine in phase [rad]."""

    fn_hz = fs_hz / 2.0
    d_step = total_time_s / n_steps

    F = np.linspace(fmin_frac_nyq, fmax_frac_nyq, n_steps) * fn_hz

    m = F * d_step
    F = np.round(m) / d_step

    ntime = int(total_time_s * fs_hz) + 1
    t = np.arange(ntime, dtype=float) / fs_hz

    j = np.minimum(np.floor(t / d_step).astype(int), n_steps - 1)
    S = amplitude_rad * np.sin(2.0 * np.pi * F[j] * t)

    # --- Optional filtering (for testing realism) ---------------------------------
    lowcut, highcut = 50, 1500
    order = 4
    nyquist = 0.5 * REP_RATE_HZ
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype="band")
    #S = filtfilt(b, a, S)

    return t, S


if __name__ == "__main__":

    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Build space/time axes
    # ------------------------------------------------------------------
    fs_hz = float(REP_RATE_HZ)

    distance_m = np.arange(0.0, float(LENGTH_M), float(DELTA_X_M))
    nspace = int(distance_m.size)

    t_s, S_rad = _build_step_frequency_signal(
        fs_hz=fs_hz,
        total_time_s=float(TOTAL_TIME_S),
        n_steps=int(N_STEPS),
        fmin_frac_nyq=float(FREQ_MIN_FRAC_NYQ),
        fmax_frac_nyq=float(FREQ_MAX_FRAC_NYQ),
        amplitude_rad=float(
            TARGET_STRAIN_PEAK
            * (4.0 * np.pi * N_EFF * GAUGE_LENGTH_M * XI / LAMBDA_LASER_M)
        ),
    )
    ntime = int(t_s.size)

    rad_amplitude = TARGET_STRAIN_PEAK * (4.0 * np.pi * N_EFF * GAUGE_LENGTH_M * XI / LAMBDA_LASER_M)

    print(f"[INFO] seed={SEED}")
    print(f"[INFO] fs={fs_hz:.1f} Hz | total_time={TOTAL_TIME_S:.2f} s | ntime={ntime}")
    print(f"[INFO] dx={DELTA_X_M:.3f} m | length={LENGTH_M:.1f} m | nspace={nspace}")
    print(f"[INFO] gauge_length={GAUGE_LENGTH_M:.2f} m | lambda={LAMBDA_LASER_M:.3e} m")
    print(f"[INFO] target_strain_peak={TARGET_STRAIN_PEAK:.3e} | derived_rad_amplitude={rad_amplitude:.3e} rad")

    # ------------------------------------------------------------------
    # Create spatio-temporal matrix (time × distance)
    # ------------------------------------------------------------------
    data = rng.normal(
        loc=0.0,
        scale=float(NOISE_STD_RAD),
        size=(ntime, nspace),
    ).astype(float)

    idx_stretcher = (distance_m >= float(STRETCHER_START_M)) & (distance_m <= float(STRETCHER_END_M))
    ncols = int(np.sum(idx_stretcher))
    if ncols <= 0:
        raise ValueError("Stretcher region has zero columns. Check STRETCHER_START_M/END_M and dx/length.")

    data[:, idx_stretcher] += S_rad[:, None]

    x_center = 0.5 * (float(STRETCHER_START_M) + float(STRETCHER_END_M))
    print(f"[INFO] stretcher=[{STRETCHER_START_M:.2f},{STRETCHER_END_M:.2f}] m | cols_injected={ncols} | center={x_center:.2f} m")
    print(f"[INFO] data shape: {data.shape} (time, distance) | units: phase [rad]")

    # ------------------------------------------------------------------
    # Diagnostics (optional)
    # ------------------------------------------------------------------
    if SHOW_PLOTS:

        inside_cols = np.where(idx_stretcher)[0]
        pos_idx = int(inside_cols[len(inside_cols) // 2])

        plt.figure(figsize=(10, 4))
        plt.plot(t_s, data[:, pos_idx], linewidth=1)
        plt.title(f"Phase at {distance_m[pos_idx]:.1f} m (col {pos_idx})")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase [rad]")
        plt.xlim(0, TOTAL_TIME_S)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t_s, S_rad)
        plt.title("Stepwise Frequency Sine Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, TOTAL_TIME_S)
        plt.grid(True)

        f, tt, Sxx = spectrogram(
            S_rad, fs=fs_hz,
            nperseg=int(WINDOW_SPECTROGRAM_S * fs_hz),
            noverlap=int(WINDOW_SPECTROGRAM_S * fs_hz * OVERLAP_SPECTROGRAM_FRAC),
        )
        plt.subplot(3, 1, 2)
        plt.pcolormesh(tt, f, 10*np.log10(Sxx + 1e-12), shading="gouraud", cmap="jet")
        plt.ylim(0, fs_hz/2)
        plt.xlim(0, TOTAL_TIME_S)
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram")
        plt.colorbar(label="Power (dB)")

        freqs = np.fft.rfftfreq(len(S_rad), d=1/fs_hz)
        fft_mag = np.abs(np.fft.rfft(S_rad)) / len(S_rad)
        plt.subplot(3, 1, 3)
        plt.plot(freqs, fft_mag, color="b")
        plt.title("Signal Frequency Spectrum (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            extent=[t_s[0], t_s[-1], distance_m[0], distance_m[-1]],
            cmap="jet",
        )
        plt.colorbar(label=r"$\delta \phi$ [rad]")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Simulated dataDAS (time × distance)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Save .npy
    # ------------------------------------------------------------------
    if SAVE_FILE:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        np.save(output_path, data)

        print(f"[OK] Saved: {output_path}")
        print(f"     Shape: {data.shape} (time, distance)")
        print(f"     Units: phase [rad] | seed={SEED}")
