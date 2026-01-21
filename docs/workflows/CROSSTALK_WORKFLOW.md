# pySEAFOM Crosstalk Workflow

> For detailed procedures and compliance requirements, refer to the applicable SEAFOM crosstalk procedure.

## Overview

- Purpose: quantify DAS spatial crosstalk around a stimulated point by measuring the response amplitude at the stimulus frequency versus distance.
- Implementation: `pySEAFOM.source.pySEAFOM.crosstalk` module.
- Key outputs:
  - `crosstalk_db`: per-channel (SSL) response in dB relative to a reference region.
  - `max_xt_db`: maximum crosstalk level in dB in an outer region.

## Required Inputs

- `section_data`: 2D array `(n_ssl, n_samples)` for one spatial segment centered on the stimulation point.
- `stimulus_freq`: stimulus frequency in Hz.
- `fs`: sampling frequency in Hz.
- `fft_size`: FFT block size (commonly 16384).
- `gauge_length`: gauge length [m].
- `stretcher_length`: length of the stretcher region [m].
- `channel_spacing`: distance per SSL [m].

## High-Level Pipeline (`calculate_crosstalk`)

1. Split each SSL time series into consecutive FFT blocks of size `fft_size`.
2. Apply FlatTop window to each block.
3. Compute FFT magnitude spectrum and extract the magnitude at `stimulus_freq`.
4. Average the extracted magnitudes across blocks to get one magnitude per SSL.
5. Compute a reference level as the mean magnitude within the stretcher region around the center SSL.
6. Convert to dB relative to reference:
   $$\mathrm{crosstalk\_dB}[i] = 20\log_{10}(M[i] / M_{ref})$$
7. Mask the center drive region (default: center ±2 GL) by setting to NaN.
8. Compute maximum crosstalk in the outer region (default: outside ±3 GL up to ±50 GL).

For typical usage, you can call `calculate_crosstalk(section_data=...)`, which runs `calculate_crosstalk(...)` for a single section and returns a single result dict.

## Plotting

- `plot_crosstalk(crosstalk_db, channel_spacing, title)`: single profile.
- `plot_crosstalk_sections(result, channel_spacing, title, section_label)`: plots a single profile from a single-section result.

## Outputs

- `crosstalk_db`: 1D array length `n_ssl`.
- `max_xt_db`: float.
- `magnitudes`: 1D array length `n_ssl` (linear).
- `reference_level`: float (linear).

## Practical Notes

- The stimulation point is assumed to be at the center SSL (`n_ssl//2`).
- Choose `channel_spacing` so `gauge_length/channel_spacing` is meaningful in SSL units.
- Ensure `fs` and `fft_size` provide adequate frequency resolution at `stimulus_freq`.
