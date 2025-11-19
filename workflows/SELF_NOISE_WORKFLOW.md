# pySEAFOM Self-Noise Workflow

> For detailed procedures and compliance requirements, refer to **SEAFOM MSP02**.

## Overview

- Purpose: quantify DAS interrogator self-noise using multi-channel RMS averaging in the frequency domain.
- Implementation: `pySEAFOM.source.pySEAFOM.self_noise` module (version 0.1.5).
- Key outputs: per-section frequency spectra (`frequency_spectra`) expressed in the same engineering units as the input data (e.g., pε/√Hz).

## Required Inputs

- `data_sections`: list of 2D NumPy arrays, each shaped `(n_channels, n_samples)`.
- `interrogation_rate`: sampling frequency in Hz.
- `window_function`: FFT window identifier (`'blackman-harris'` default; `'flattop'`, `'hann'`, `'hamming'`, `'none'`).
- `data_type`: signal units (`'phase'`, `'rad'`, `'pε'`, `'nε'`, `'dphase'`).

## High-Level Pipeline

1. Iterate over each cable section supplied in `data_sections`.
2. For every channel in the section, transform the time-domain trace into an amplitude spectral density (ASD) using `calculate_asd()`.
3. Stack the individual ASDs and perform RMS averaging across channels to obtain the section-level spectrum.
4. Return a list of `(frequencies, rms_asd)` tuples, one per section.
5. Optional visualization/reporting via `plot_combined_self_noise_db()` and `report_self_noise()`.

## Channel-Level Processing (`calculate_asd`)

- **Detrend**: remove constant and linear trends (`scipy.signal.detrend`) unless `detrend_data=False`.
- **Windowing**: multiply by the chosen window; defaults to Blackman-Harris for sidelobe suppression.
- **FFT**: compute complex spectrum (`scipy.fft.fft`) and retain non-negative frequencies via `np.fft.fftfreq` mask.
- **Normalization**:
  - Divide magnitudes by the sum of the window to preserve amplitude.
  - Convert to single-sided ASD with `√2 / √(ENBW · interrogation_rate)`, where ENBW is computed by `calculate_enbw(window)`.

## Section-Level Aggregation (`calculate_self_noise`)

- Convert phase-rate inputs (`data_type='dphase'`) to phase via cumulative sum.
- Collect all channel ASDs into `individual_asds` (shape: `n_channels × n_freq_bins`).
- Compute RMS spectrum: `rms_asd = sqrt(mean(individual_asds**2, axis=0))`.
- Append `(frequencies, rms_asd)` to the output list `frequency_spectra`.

## Visualization Workflow (`plot_combined_self_noise_db`)

- Accepts `frequency_spectra`, section labels, and measurement metadata.
- Converts spectra to dB scale (`20 · log10(asd)`), applies log-spaced smoothing (`smooth_data`).
- Overlays seismic (1–100 Hz) and acoustic (100 Hz–Nyquist) bands and adds a metadata box.
- Supports sharing publication-quality plots for multiple sections on one figure.

## Reporting (`report_self_noise`)

- Formats tabular summaries at standard frequencies (10, 100, 1000, 10000 Hz).
- Provides optional band-averaged RMS values over user-defined frequency ranges.
- Supports linear and dB reporting modes.

## Data Quality Considerations

- Longer records (larger `n_samples`) improve low-frequency resolution (Δf = interrogation_rate / n_samples).
- Minimum ~10 channels per section recommended for stable RMS estimates; more channels reduce statistical variance by ~√N.
- Window choice trades spectral leakage for resolution: Blackman-Harris (default) gives ~2× ENBW versus rectangular.
- Detrending is essential for mitigating low-frequency leakage from slow thermal drifts.

## Output Artifacts

- `frequency_spectra`: list of `(frequencies, rms_asd)` tuples (Hz, unit/√Hz).
- Plot function returns matplotlib figure (no implicit save).
- Report function prints formatted text summaries to stdout.

## Key Dependencies

- NumPy (vector math, FFT frequencies).
- SciPy (FFT, window functions, detrend).
- Matplotlib (visualization).

## Extension Points

- Additional window functions can be wired into `calculate_asd` by extending the existing conditionals.
- Alternative aggregation schemes (e.g., median ASD) can wrap the channel ASD stacking stage.
- Frequency-band definitions and metadata layout can be customized within the plotting helper.
