
## pySEAFOM Dynamic Range Workflow

> For detailed procedures and compliance requirements, refer to **SEAFOM MSP02**.

## Overview

- Purpose: estimate the **dynamic range limit** from a controlled input where amplitude and slew rate increases over time.
- Implementation: `pySEAFOM.source.pySEAFOM.dynamic_range` module (version 0.1.8).
- Key outputs: trigger time (absolute + relative), limit amplitude µε at trigger, and optional plots + CSV logs.
- Optional output: peak-over-basis in dB re rad/√Hz (requires radian_basis + gauge_length).

## Required Inputs: (load_dynamic_range_data) and (data_processing) are used to extract data for Dynamic Range

- `folder_or_file`: folder with `.npy` files or a single `.npy` matrix.
- 2D DAS matrix (expected 2D):
  - preferred: `(n_time, n_space)` = (time, space/channels)
  - supported: `(n_space, n_time)` via `matrix_layout="space_time"` or `"auto"`
- `fs`: repetition rate in Hz.
- `delta_x_m`: spatial step between channels in meters.
- Spatial extraction:
  - `x1_m`, `x2_m`: spatial window in meters
  - `test_sections_channels` (POS): position inside the window (meters)
  - `average_over_cols`: number of adjacent channels averaged to form a 1D trace
- Time extraction:
  - `time_start_s`, `duration`: analysis window in seconds
- Trace units:
  - either **strain** (µε) OR **phase** (rad)
  - if phase is provided, conversion to strain uses `gauge_length` (and optical constants)
- Optional preprocessing:
  - `highpass_hz`: high-pass cutoff in Hz (requires `fs`)

## High-Level Pipeline

1. Load one or more `.npy` matrices and build a 2D matrix (concatenate along time).
2. Apply layout normalization (`time_space`, `space_time`, `auto`).
3. Crop time window `[time_start_s, time_start_s + duration]`.
4. Crop spatial window `[x1_m, x2_m]`.
5. Extract 1D trace at `POS` (average `average_over_cols` channels).
6. Optional conversion phase to strain and optional high-pass filtering.
7. Run dynamic range detection using one (or both) engines:
   - Hilbert envelope error test
   - Sliding THD error test
1. Report trigger time + limit amplitude; optionally save plots + append CSV logs.

## Hilbert Envelope Engine (calculate_dynamic_range_hilbert)

- Goal: compare **measured envelope** vs **theoretical envelope** of a ramped sine.
- Steps:
  - Compute envelope: `env_measured = smooth(|hilbert(signal)|)`
  - Build a theoretical ramped sine (0 → `max_strain_microstrain`) at `ref_freq_hz`
  - Compute `env_theoretical` (Hilbert + smoothing)
  - Relative error: `rel_error = |env_measured - env_theoretical| / env_theoretical`
  - Trigger when `rel_error > error_threshold_frac`, ignoring the initial `safezone_s`
- Output decision:
  - If a violation exists: dynamic range limit is the **first violation** time and its theoretical envelope amplitude of the last stretcher cycle

## Sliding THD Engine (calculate_dynamic_range_thd)

- Goal: detect non-linearity by monitoring **Total Harmonic Distortion** over time.
- Steps (per sliding window):
  - Estimate fundamental RMS amplitude (`A1_rms`) of the signal
  - Compute PSD (`periodogram`) and integrate harmonic bands around `2f, 3f, ...`
  - THD fraction: `thd = sqrt(sum(harmonic_power)) / (A1_rms)`
  - Smooth THD with a median filter
- Trigger logic:
  - Apply a THD threshold (`thd_threshold_frac`)
  - Ignore initial `safezone_s`
  - Require continuous violation for at least `min_trigger_duration`
- Output decision:
  - If a valid violation exists: dynamic range limit is the **first violation** time and its theoretical amplitude of the last stretcher cycle

## Visualization Workflow

- Hilbert plot:
  - Raw signal + measured envelope + theoretical envelope
  - Relative error curve and threshold
  - Safe zone shading + violation regions
- THD plot:
  - Raw signal
  - THD curve (%) + threshold line
  - Safe zone shading + violation region

## Reporting 

- Console:
  - prints a formatted summary table (trigger time, Δt from window start, limit amplitude, peak in last cycle)
  - The peak_over_basis field is only computed when radian_basis is provided and gauge_length is available
- CSV data:
  - `dynamic_range_hilbert.csv`
  - `dynamic_range_thd.csv`
- Figure data:
  - `dynamic_range_hilbert.png`
  - `dynamic_range_thd.png`

## Data Quality Considerations

- The method assumes a known excitation frequency (`ref_freq_hz`) and a clean ramped sine reference.
- `fs` must be high enough to resolve harmonics for THD (and avoid aliasing).
- `window_s` controls THD time resolution vs frequency resolution (shorter windows react faster but blur harmonics).
- `smooth_window_s` affects Hilbert envelope stability (too small = noisy envelope, too large = delayed response).
- High-pass filtering can help remove drift that inflates low-frequency envelope error.

## Output Artifacts

- Trigger time (absolute) and dynamic range limit amplitude (strain peak/envelope).
- Optional: peak_over_basis [dB re rad/√Hz] (only if radian_basis is provided and gauge_length is available).
- Optional plots (Hilbert/THD).
- Optional CSV summary logs (one row per run).

## Key Dependencies

- NumPy (array operations)
- SciPy (Hilbert transform, filters, periodogram)
- Matplotlib (plots)

## Extension Points

- Support direct **phase-domain** analysis (skip strain conversion) if desired.
- Add alternative trigger rules (e.g., modify treshold or add Frequency Power treshold).
- Add 3 stretchers analysis simultaneously.
