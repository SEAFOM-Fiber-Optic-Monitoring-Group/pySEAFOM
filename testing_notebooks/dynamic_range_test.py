# This script expects DAS data stored as a 2D array .npy file.
#
# data.shape = (time, space/channels)
#   - axis 0 (rows): time samples
#       * sampling rate: fs = REPETITION_RATE_HZ  [Hz]
#       * time step:     dt = 1/fs               [s]
#   - axis 1 (cols): spatial channels / distance
#       * spatial step resolution:  dx = DELTA_X_M          [m]
#       * channel i corresponds approximately to x = i * dx

import numpy as np

from pySEAFOM import dynamic_range

# ======================================================================
# CONTROL
# ======================================================================
ANALYSIS_TYPE = "both"          # "hilbert", "thd", or "both"
SAVE_RESULTS = True             # If True, saves figures + CSV files in OUTPUT_DIR
OUTPUT_DIR = "results_dynamic_range"
SHOW_PLOTS = True               # If True, displays figures (plt.show)
PLOT_METADATA_BOX = True        # If True, adds a measurement-parameter box to the plot

# ======================================================================
# INPUT PARAMETERS (data extraction)
# ======================================================================
FOLDER_OR_FILE = "teste"        # Folder with .npy files or a single .npy file
REPETITION_RATE_HZ = 2500.0     # Sampling / interrogator rate [Hz]
DELTA_X_M = 0.2                 # Spatial step between channels [m]

X1_M = 100.0                    # Spatial window start [m]
X2_M = 300.0                    # Spatial window end [m]
POS_M = 200.0                   # Central position inside the spatial window [m]

TIME_START_S = 35.0             # Analysis window start time [s]
DURATION_S = 30.0               # Analysis window duration [s]

AVERAGE_OVER_COLS = 5           # Number of adjacent channels to average starting at POS_M
DATA_IS_STRAIN = False           # True if the loaded trace is already strain [µε]
GAUGE_LENGTH_M = 6.38           # Gauge length used by interrogator [m]
HIGHPASS_HZ = 5.0               # High-pass cutoff [Hz] (set None to disable)

# ======================================================================
# COMMON PARAMETER (used by both Hilbert and THD)
# ======================================================================
SAFEZONE_S = 1.0                # Initial safe zone where triggering is ignored [s]

# ======================================================================
# HILBERT PARAMETERS
# ======================================================================
MAX_STRAIN_UE = 0.51          # Final theoretical envelope amplitude [µε]
REF_FREQ_HZ = 50.0              # Expected sine frequency [Hz]
HILBERT_SMOOTH_WINDOW_S = 0.5   # Envelope smoothing window [s]
HILBERT_ERROR_THRESHOLD_FRAC = 0.3   # Relative error threshold (0.3 = 30%)

# ======================================================================
# THD PARAMETERS
# ======================================================================
THD_WINDOW_S = 1.0              # Sliding window length [s]
THD_OVERLAP = 0.75              # Window overlap fraction (0.75 = 75% overlap)
THD_THRESHOLD_FRAC = 0.15       # THD threshold (0.15 = 15%)
THD_MEDIAN_WINDOW_S = 0.005    # Median filter window applied to THD curve [s]
MIN_TRIGGER_DURATION_S = 1.0    # Minimum continuous violation time to trigger [s]


if __name__ == "__main__":

    # Step 1: Load data and extract 1D trace (original units)
    time_s, trace_raw = dynamic_range.load_dynamic_range_data(
        folder_or_file=FOLDER_OR_FILE,
        fs=REPETITION_RATE_HZ,
        delta_x_m=DELTA_X_M,
        x1_m=X1_M,
        x2_m=X2_M,
        test_sections_channels=POS_M,
        time_start_s=TIME_START_S,
        duration=DURATION_S,
        average_over_cols=AVERAGE_OVER_COLS,
        matrix_layout="auto",
    )

    # Step 2: Convert to microstrain (if needed) + optional high-pass
    signal_ue = dynamic_range.data_processing(
        trace=trace_raw,
        data_is_strain=DATA_IS_STRAIN,
        gauge_length=GAUGE_LENGTH_M,
        highpass_hz=HIGHPASS_HZ,
        fs=REPETITION_RATE_HZ,
    )

    # Step 3: Hilbert dynamic range analysis
    if ANALYSIS_TYPE.lower() in {"hilbert", "both"}:
        dynamic_range.analyze_dynamic_range_hilbert(
            # 1) required data
            time_s=time_s,
            signal_microstrain=signal_ue,
            max_strain_microstrain=MAX_STRAIN_UE,

            # 2) analysis parameters
            ref_freq_hz=REF_FREQ_HZ,
            smooth_window_s=HILBERT_SMOOTH_WINDOW_S,
            error_threshold_frac=HILBERT_ERROR_THRESHOLD_FRAC,
            safezone_s=SAFEZONE_S,

            # 3) optional metadata / run context (plot box + CSV)
            time_start_s=TIME_START_S,
            duration=DURATION_S,
            folder_or_file=FOLDER_OR_FILE,
            test_sections_channels=POS_M,
            data_is_strain=DATA_IS_STRAIN,
            average_over_cols=AVERAGE_OVER_COLS,

            fs=REPETITION_RATE_HZ,
            delta_x_m=DELTA_X_M,
            gauge_length=GAUGE_LENGTH_M,
            highpass_hz=HIGHPASS_HZ,

            # 4) I/O options
            show_plot=SHOW_PLOTS,
            save_results=SAVE_RESULTS,
            results_dir=OUTPUT_DIR,
            plot_metadata_box=PLOT_METADATA_BOX,
        )

    # Step 4: THD dynamic range analysis
    if ANALYSIS_TYPE.lower() in {"thd", "both"}:
        dynamic_range.analyze_dynamic_range_thd(
            # 1) required data
            time_s=time_s,
            signal_microstrain=signal_ue,

            # 2) analysis parameters
            ref_freq_hz=REF_FREQ_HZ,
            window_s=THD_WINDOW_S,
            overlap=THD_OVERLAP,
            thd_threshold_frac=THD_THRESHOLD_FRAC,
            median_window_s= THD_MEDIAN_WINDOW_S,
            min_trigger_duration=MIN_TRIGGER_DURATION_S,
            safezone_s=SAFEZONE_S,

            # 3) optional metadata / run context (plot box + CSV)
            time_start_s=TIME_START_S,
            duration=DURATION_S,
            folder_or_file=FOLDER_OR_FILE,
            test_sections_channels=POS_M,
            data_is_strain=DATA_IS_STRAIN,
            average_over_cols=AVERAGE_OVER_COLS,

            fs=REPETITION_RATE_HZ,
            delta_x_m=DELTA_X_M,
            gauge_length=GAUGE_LENGTH_M,
            highpass_hz=HIGHPASS_HZ,

            # 4) I/O options
            show_plot=SHOW_PLOTS,
            save_results=SAVE_RESULTS,
            results_dir=OUTPUT_DIR,
            plot_metadata_box=PLOT_METADATA_BOX,
        )

    # ------------------------------------------------------------------
    # Examples (minimal calls)
    # ------------------------------------------------------------------
    dynamic_range.analyze_dynamic_range_thd(
        time_s=time_s,
        signal_microstrain=signal_ue,
        time_start_s=TIME_START_S,
    )

    dynamic_range.analyze_dynamic_range_thd(
        time_s=time_s,
        signal_microstrain=signal_ue,
        time_start_s=TIME_START_S,
        plot_metadata_box=PLOT_METADATA_BOX,
    )

    dynamic_range.analyze_dynamic_range_hilbert(
        time_s=time_s,
        signal_microstrain=signal_ue,
        max_strain_microstrain=0.51,
        time_start_s=TIME_START_S,
        save_results=True,
        results_dir=OUTPUT_DIR,
    )


