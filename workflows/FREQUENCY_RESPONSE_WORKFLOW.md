
## pySEAFOM Frequency Response Workflow 

> For detailed procedures and compliance requirements, refer to **SEAFOM MSP02**.

## Overview

- Purpose: estimate the **DAS frequency response** and the **normalized frequency response**  from a controlled stepwise-frequency excitation measured on a stretcher region.
- Implementation: `pySEAFOM.source.pySEAFOM.frequency_response` module (version 0.1.9).
- Key outputs: DAS response curve in dB re 1 µε, normalized response frequency points, and optional plots + CSV logs.

## Required Inputs: (load_frequency_response_data) and (data_processing) are used to extract data for Frequency Response

- `folder_or_file`: folder with `.npy` files or a single `.npy` matrix.
- 2D DAS matrix (expected 2D):
  - preferred: `(n_time, n_space)` = (time, space/channels)
  - supported: `(n_space, n_time)` via `matrix_layout="space_time"` or `"auto"`    
- `fs`: repetition rate in Hz.
- `delta_x_m`: spatial step between channels in meters.
- Spatial extraction:
- `stretcher_start_m`, `stretcher_end_m`: defines the stretcher region (used only to compute the center position).
- `span_m`: averaging around the stretcher center, ib **m** for each side.
- Trace units:
  - either **strain** (µε) OR **phase** (rad)
  - if phase is provided, conversion to strain uses `gauge_length` (and optical constants)
- Optional preprocessing:
  - `highpass_hz`: high-pass cutoff in Hz (requires `fs`)

## High-Level Pipeline

1. Load `.npy` matrix.
2. Apply layout normalization (`time_distance`, `distance_time`, `auto`).
3. Define stretcher center: `center_m = 0.5 * (stretcher_start_m + stretcher_end_m)`.
4. Extract 1D local trace by spatial averaging over the span window:
    - select channels in [center_m − span_m, center_m + span_m]  
    - average those columns to form `trace_raw(time)`
5. Optional conversion phase to strain and optional high-pass filtering.
6. Run frequency response analysis.
7. Save plots and report `frequency_response_normalized.csv` (frequency + normalized dB).

## Frequency Response Engine (analyze_frequency_response)

- Goal: estimate the **DAS frequency response** (in dB re 1 µε) and a **normalized response** over the step frequencies.
- Steps:
    - Compute FFT amplitude spectrum
    - Convert to dB (strain reference):
        - `resp_db = 20 * log10(amp + eps)` _(dB re 1 µε)_
    - Define requency window (Nyquist fractions):
        - `fmin = freq_min_frac_nyq * (fs/2)`
        - `fmax = freq_max_frac_nyq * (fs/2)`
        - `freq_points = linspace(fmin, fmax, n_steps)`
    - Sample response at those points:
        - `vals = interp(freq_points, freq, resp_db)`
    - Normalize (remove mean over sampled points):
        - `normalized_db = vals - mean(vals)`
- Outputs:
    - Normalized Frequency Response (`freq_points` vs `normalized_db`)

## Visualization Workflow

- Local diagnostics (3-panel):
    - Time trace (µε)
    - Spectrogram (dB)
    - FFT magnitude
- DAS Frequency Response:
    - Frequency (Hz) vs amplitude (dB re 1 µε)
- Normalized Frequency Response:
    - Step frequencies (Hz) vs normalized amplitude (dB)

## Reporting

- Console:
    - prints run parameters (repetition rate, gauge length, spatial resolution if provided, local position)
    - prints basic response stats (median dB, min/max dB with their frequencies) for:
        - DAS Frequency Response
        - Normalized Frequency Response
- CSV data:
    - `frequency_response_normalized.csv`
- Figure data:
    - `frequency_response_local_time_spectrogram_fft.png`
    - `frequency_response.png`
    - `frequency_response_normalized.png`

## Data Quality Considerations

- The method assumes a known excitation frequency (`ref_freq_hz`) and a clean ramped sine reference.
- `fs` must be high enough to resolve harmonics for THD (and avoid aliasing).
- Position of [center_m − span_m, center_m + span_m]  must be properly selected.

## Output Artifacts

- DAS frequency response curve (dB re 1 µε) over frequency.
- Normalized frequency response sampled at `N_STEPS` points between `FREQ_MIN_FRAC_NYQ` and `FREQ_MAX_FRAC_NYQ` in dB.
- Optional plots:
    - Local diagnostics (time trace + spectrogram + FFT magnitude)
    - DAS Frequency Response
    - Normalized Frequency Response
- Optional CSV:
    - `frequency_response_normalized.csv` (one row per frequency point)
        - `frequency_hz`: frequency of the MSP-02 evaluation points
        - `normalized_db`: normalized response in dB at each point


## Key Dependencies

- NumPy (array operations, FFT, interpolation)
- SciPy (spectrogram, optional high-pass filtering)
- Matplotlib (plots)


## Extension Points

- Support direct **phase-domain** analysis (skip strain conversion) if desired.
- Add alternative spectrogram analysis such as CWT.
- Add 3 stretchers analysis simultaneously.
