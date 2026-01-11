# pySEAFOM Fidelity Workflow (THD)

> For detailed procedures and compliance requirements, refer to the applicable SEAFOM MSP fidelity / harmonic distortion guidance.

## Overview

- Purpose: quantify signal fidelity by measuring Total Harmonic Distortion (THD) at a known stimulus frequency.
- Implementation: `pySEAFOM.source.pySEAFOM.fidelity` module.
- Key outputs: per-section, per-level THD (%) plus relative harmonic levels (dB re fundamental).

## Required Inputs

- `time_series_data`: 2D NumPy array shaped `(n_channels_in_section, n_samples)` for a *single* pre-sliced section.
- `fs`: sampling frequency in Hz.
- `levels_time_steps`: either a single sample range `[start, end)` or a list of ranges `[[start, end), ...]` defining stimulus “levels” / time blocks.
- `stimulus_freq`: fundamental stimulus frequency in Hz.
- `snr_threshold_db`: SNR gate used to accept/reject FFT blocks.
- `section_name` (optional): label used for reporting.

## High-Level Pipeline (`calculate_fidelity_thd`)

1. Average across channels to form a 1D trace (`signal_mean`).
2. For each level time window (`levels_time_steps[level]`):
  - Slice the 1D trace to that time range.
  - Run `compute_thd()` to estimate THD.
3. Return a structured dictionary with a single section containing per-level THD and harmonics.

To compute THD for multiple spatial sections, slice sections in your caller code and run `calculate_fidelity_thd()` once per section.

## Block Screening (`is_good_quality_block`)

- Apply a FlatTop window.
- Compute FFT magnitude spectrum.
- Estimate a simple SNR metric:
  - **Signal power** = power at the stimulus bin.
  - **Noise power** = mean power of all other bins (excluding DC and the stimulus bin).
- Accept block if `snr_db >= snr_threshold_db`.

## THD Computation (`compute_thd`)

- Split the signal into consecutive `fft_size` blocks (default 16384).
- Keep only blocks passing `is_good_quality_block`.
- For each accepted block:
  - Measure magnitudes at the fundamental and harmonics (default harmonics 1..5).
- Average harmonic magnitudes across accepted blocks.
- Compute:
  $$\mathrm{THD}(\%) = \frac{\sqrt{\sum_{k=2}^{N} V_k^2}}{V_1} \times 100$$
- Report harmonics in dB relative to the fundamental:
  $$H_k\,(\mathrm{dB}) = 20\log_{10}(V_k / V_1)$$

## Outputs

- `results['sections'][i]['levels'][j]['thd_percent']`: THD (%) for section i, level j.
- `results['sections'][i]['levels'][j]['harmonics_db']`: array of harmonic levels (dB re fundamental).
- `results['sections'][i]['levels'][j]['n_good_blocks']`: number of accepted FFT blocks.

## Reporting

- `report_fidelity_thd(results)` prints a readable summary (per section, per level).

## Practical Notes

- The `levels_time_steps` end index is treated as exclusive (Python slicing semantics).
- Ensure `fs` is high enough to capture harmonics without aliasing.
