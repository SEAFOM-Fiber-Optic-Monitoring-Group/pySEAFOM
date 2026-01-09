# This script generates synthetic DAS data directly in PHASE [rad] and saves it
# as a 2D .npy array.
#
# data.shape = (time, space/channels)
#   - axis 0 (rows): time samples
#       * sampling rate: fs = REPETITION_RATE_HZ  [Hz]
#       * time step:     dt = 1/fs               [s]
#   - axis 1 (cols): spatial channels / distance
#       * spatial step resolution:  dx = DELTA_X_M          [m]
#       * channel i corresponds approximately to x = i * dx

import os
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# CONTROL
# ======================================================================
SEED = 12345                  # Fixed seed for deterministic results
ENABLE_WRAP_UNWRAP = True     # If True: wrap to [-pi,pi) then unwrap along time

OUTPUT_DIR = "teste"
OUTPUT_FILE = "dynamic_range_sim_phase.npy"

SHOW_PLOTS = True

# ======================================================================
# MATRIX
# ======================================================================
REP_RATE_HZ = 2500.0          # [Hz]
TOTAL_TIME_S = 70.0           # [s]

DELTA_X_M = 0.2               # [m]
LENGTH_M = 400.0              # [m]

# ======================================================================
# NOISE (phase)
# ======================================================================
NOISE_STD_RAD = 0.05          # [rad] (phase noise std dev)

# ======================================================================
# EVENT (phase)
# ======================================================================
EVENT_START_S = 35.0          # [s]
EVENT_END_S   = 65.0          # [s]
EVENT_FREQ_HZ = 50.0          # [Hz]
EVENT_MAX_AMP_RAD = 30.0       # [rad] final sine amplitude (peak) at end of ramp

EVENT_X1_M = 150.0            # [m]
EVENT_X2_M = 250.0            # [m]
POS_PLOT_M = 200.0            # [m] channel to plot vs time

# Plot marker style for event start/end
EVENT_LINE_KW = dict(color="r", linestyle=":", linewidth=2.0)  # red dotted


if __name__ == "__main__":


    rng = np.random.default_rng(SEED)

    ntime = int(TOTAL_TIME_S * REP_RATE_HZ)
    nspace = int(LENGTH_M / DELTA_X_M)

    time_s = np.arange(ntime) / REP_RATE_HZ      # [s]
    space_m = np.arange(nspace) * DELTA_X_M      # [m]

    print(f"[INFO] ntime={ntime}, nspace={nspace} | seed={SEED}")
    print(f"[INFO] time=[{time_s[0]:.3f},{time_s[-1]:.3f}] s | space=[{space_m[0]:.1f},{space_m[-1]:.1f}] m")

    idx_x1 = int(EVENT_X1_M / DELTA_X_M)
    idx_x2 = int(EVENT_X2_M / DELTA_X_M)
    idx_x1 = max(0, idx_x1)
    idx_x2 = min(nspace, idx_x2)

    print(f"[INFO] Event span: x=[{EVENT_X1_M},{EVENT_X2_M}] m -> cols [{idx_x1},{idx_x2-1}]")

    data = np.zeros((ntime, nspace), dtype=float)

    data[:, idx_x1:idx_x2] += rng.normal(
        loc=0.0,
        scale=NOISE_STD_RAD,
        size=(ntime, idx_x2 - idx_x1),
    ).astype(float)

    print(f"[INFO] Noise applied only in cols [{idx_x1},{idx_x2-1}] (all times)")

    event_phase = np.zeros_like(time_s, dtype=float)

    mask_t = (time_s >= EVENT_START_S) & (time_s <= EVENT_END_S)

    tau = (time_s[mask_t] - EVENT_START_S) / (EVENT_END_S - EVENT_START_S)   
    amp_t = tau * EVENT_MAX_AMP_RAD                                         
    event_phase[mask_t] = amp_t * np.sin(2.0 * np.pi * EVENT_FREQ_HZ * time_s[mask_t])

    data[:, idx_x1:idx_x2] += event_phase[:, None]

    # ------------------------------------------------------------------
    # Optional wrap -> unwrap
    # ------------------------------------------------------------------
    if ENABLE_WRAP_UNWRAP:
        print("[INFO] Applying phase wrap [-pi,pi) then unwrap along time axis (axis=0).")

        phase_wrapped = (data + np.pi) % (2.0 * np.pi) - np.pi
        data = np.unwrap(phase_wrapped, axis=0)

    pos_idx = int(POS_PLOT_M / DELTA_X_M)

    if SHOW_PLOTS:
        plt.figure(figsize=(10, 4))
        plt.plot(time_s, data[:, pos_idx], linewidth=1)
        plt.axvline(EVENT_START_S, **EVENT_LINE_KW)
        plt.axvline(EVENT_END_S, **EVENT_LINE_KW)
        plt.title(f"Phase at {POS_PLOT_M:.1f} m (col {pos_idx})")
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [rad]")
        plt.xlim(30, 70)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        down = 10
        time_ds = time_s[::down]
        data_ds = data[::down, :]

        extent = [time_ds[0], time_ds[-1], space_m[0], space_m[-1]]

        plt.figure(figsize=(10, 6))
        im = plt.imshow(
            data_ds.T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap="jet",
            vmin=-EVENT_MAX_AMP_RAD,
            vmax=EVENT_MAX_AMP_RAD,
        )
        plt.colorbar(im, label="Phase [rad]")
        plt.xlabel("Time [s]")
        plt.ylabel("Space [m]")
        plt.title("Simulated phase matrix (noise + ramp sine)")

        plt.axvline(EVENT_START_S, **EVENT_LINE_KW)
        plt.axvline(EVENT_END_S, **EVENT_LINE_KW)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Save .npy
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    np.save(output_path, data)

    print(f"[OK] Saved: {output_path}")
    print(f"     Shape: {data.shape} (ntime, nspace)")
    print(f"     Units: phase [rad] | wrap/unwrap={ENABLE_WRAP_UNWRAP} | seed={SEED}")
