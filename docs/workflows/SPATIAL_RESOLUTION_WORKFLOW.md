## pySEAFOM Spatial Resolution Workflow

> For detailed procedures and compliance requirements, refer to **SEAFOM MSP02**.

## Overview

- Purpose: estimate the **spatial resolution** of the DAS system from a controlled stimulation localized at a stretcher position.
- Implementation: `pySEAFOM.source.pySEAFOM.spatial_resolution` module (version 0.1.10).
- Key outputs: left slope width (`LL`), right slope width (`LR`), estimated spatial resolution in meters, detected peak position, SNR estimate, and optional plots + CSV summary.

## Required Inputs: `load_spatial_resolution_data` and `data_processing` are used to extract and prepare data for Spatial Resolution

- `folder_or_file`: folder with `.npy` files or a single `.npy` matrix.
- 2D DAS matrix (expected 2D):
  - preferred: `(n_time, n_space)` = (time, space/channels)
  - supported: `(n_space, n_time)` via `matrix_layout="space_time"` or `"auto"`
- `fs`: repetition rate in Hz.
- `delta_x_m`: spatial step between channels in meters.
- Spatial extraction:
  - `x1_m`, `x2_m`: spatial window in meters used to isolate the stimulated section
  - optional `x_vec_m`: explicit spatial coordinate vector for the extracted section
  - `target_pos_m`: expected stretcher center position in meters
  - `stretcher_length_m`: optional stretcher length used only for visualization overlays
- Time extraction:
  - `time_start_s`, `duration`: analysis window in seconds
-  Trace units:
  - either **strain** (µε) OR **phase** (rad)
  - if phase is provided, conversion to strain uses `gauge_length` (and optical constants)
- Analysis parameters:
  - `ref_freq_hz`: reference stimulus frequency in Hz
  - `fft_size`: FFT block size used to extract the tone amplitude per SSL
  - `snr_threshold_db`: minimum recommended SNR for a valid result
- Optional preprocessing:
  - `highpass_hz`: high-pass cutoff in Hz (requires `fs`)

## High-Level Pipeline

1. Load one or more `.npy` matrices and build a 2D matrix (concatenate along time).
2. Apply layout normalization (`time_space`, `space_time`, `auto`).
3. Crop time window `[time_start_s, time_start_s + duration]`.
4. Crop spatial window `[x1_m, x2_m]`.
5. Reformat the extracted section to `(n_ssl, n_samples)` for spatial analysis.
6. Optional conversion phase to strain and optional high-pass filtering.
7. Run spatial resolution analysis on the extracted section.
8. Report `LL`, `LR`, spatial resolution, peak position, and SNR; optionally save plots + CSV summary.

## Spatial Resolution Engine (`calculate_spatial_resolution`)

- Goal: calculate spatial resolution from the spatial amplitude profile of the response at the reference frequency.
- Steps:
  - Validate `section_data` as a 2D matrix `(n_ssl, n_samples)`.
  - Build the spatial coordinate vector from `x_vec_m` or from `delta_x_m`.
  - Extract the response amplitude at `ref_freq_hz` for each SSL:
    - split each SSL trace into consecutive FFT blocks of length `fft_size`
    - apply FlatTop window to each block
    - compute FFT magnitude and select the bin nearest `ref_freq_hz`
    - average the selected magnitudes across blocks
  - Form the spatial tone-amplitude profile `amp_profile`.
  - Normalize the profile by its peak value:
    - `amp_norm = amp_profile / max(amp_profile)`
  - Estimate noise floor and SNR:
    - `noise_floor = percentile(amp_profile, 5)`
    - `snr_db = 20 * log10(peak_amp / noise_floor)`
  - Detect the peak position from the maximum of `amp_profile`.
  - Fit the left and right transition slopes using the normalized interval between 5% and 95% amplitude.
  - Convert each fitted slope into an equivalent width:
    - `LL_m`: left slope width
    - `LR_m`: right slope width
  - Compute the final spatial resolution as the average of both sides:
    - `spatial_resolution_m = 0.5 * (LL_m + LR_m)`
- Output decision:
  - If both left and right fits are valid, the reported spatial resolution is the mean of `LL` and `LR`
  - If either fit fails, the spatial resolution is reported as `NaN` and a warning is printed

## Visualization Workflow

- Spatiotemporal map
  - upper panel: time vs position colormap of the extracted section
  - optional stretcher overlay: lower edge, upper edge, and center line
  - lower panel: representative traces around the target region
- Spatial profile:
  - normalized amplitude per SSL as bars
  - left and right fitted slope lines
  - `LL` and `LR` annotations
  - metadata box with `LL`, `LR`, spatial resolution, SNR, and detected peak position

## Reporting

- Console:
  - prints target position, detected peak position, and SNR before the final summary
  - prints includes:
    - target position
    - detected peak position
    - reference frequency
    - gauge length (if provided)
    - spatial sampling
    - SNR
    - `LL` in meters and SSL
    - `LR` in meters and SSL
    - spatial resolution in meters
    - deviation from gauge length (if `gauge_length` is provided)
- CSV data:
  - `spatial_resolution_summary.csv`
- Figure data:
  - `spatiotemporal_map.png`
  - `spatial_resolution_profile.png`

## Data Quality Considerations

- `fs` and `fft_size` must provide enough frequency resolution to isolate the reference tone.
- The selected spatial window must fully contain the transition slopes on both sides of the stimulated region.
- `snr_threshold_db` is used as a validity warning; results below threshold may be unstable.
- Spatial sampling (`delta_x_m`) should be fine enough to resolve the 5%–95% transition regions.

## Output Artifacts

- Spatial amplitude profile at the reference frequency.
- Normalized spatial profile across SSLs.
- `LL` and `LR` estimates in meters and SSL.
- Final spatial resolution estimate in meters.
- Peak position and SNR estimate.
- Optional plots:
  - Spatiotemporal map
  - Spatial resolution profile
- Optional CSV of summary logs.

## Key Dependencies

- NumPy (array operations).
- SciPy (FFT, filters).
- Matplotlib (plots).

## Extension Points

- The method assumes a localized stimulation at a known reference frequency. The current implementation could automatically determine the predominant frequency of the signal.
- Add alternative width estimators besides the current 5%–95% piecewise linear slope fit.
- Add automatic quality gates based on fit residuals in addition to SNR.
- Add support for analysis of multiple stretcher positions in a single run.
