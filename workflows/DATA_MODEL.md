# pySEAFOM Data Model

**Repository**: SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM

This document describes the data structures, parameters, and processing flow used in the pySEAFOM library for DAS (Distributed Acoustic Sensing) analysis.

---

## Overview

The pySEAFOM library provides analysis engines for processing DAS time-series data. The library uses standardized data structures and parameters across all analysis modules.
The current codebase exposes three main data-model families:

| Data Model Family | Used By | Canonical Input Shape | Canonical Output |
|------------------|---------|------------------------|------------------|
| `multi_section_spectra` | `self_noise.calculate_self_noise()` | `list[ndarray(n_channels, n_samples)]` | `list[(freqs, values)]` |
| `single_section_analysis` | `crosstalk.calculate_crosstalk()`, `fidelity.calculate_fidelity_thd()`, `spatial_resolution.calculate_spatial_resolution()` | `ndarray(n_channels_or_ssl, n_samples)` | `dict` |
| `single_trace_analysis` | `dynamic_range.*`, `frequency_response.calculate_frequency_response()` | `time_s: ndarray(n_samples)`, `signal_microstrain: ndarray(n_samples)` | `dict` or plotted/report side effects |


---

## Input Data Structure

### Primary Input Families

| Input Name | Type | Shape | Used By | Description |
|-----------|------|-------|---------|-------------|
| `data_sections` | `list[numpy.ndarray]` | each array `(n_channels, n_samples)` | `self_noise.calculate_self_noise()` | Multiple pre-sliced cable sections analyzed independently |
| `section_data` | `numpy.ndarray` | `(n_ssl, n_samples)` | `crosstalk.calculate_crosstalk()`, `spatial_resolution.calculate_spatial_resolution()` | One spatial section where each row is one SSL/channel |
| `time_series_data` | `numpy.ndarray` | `(n_channels, n_samples)` | `fidelity.calculate_fidelity_thd()` | One section used for per-channel THD across one or more time windows |
| `time_s` + `signal_microstrain` | `numpy.ndarray` + `numpy.ndarray` | both `(n_samples,)` | `dynamic_range.*`, `frequency_response.calculate_frequency_response()` | One extracted local trace aligned to a time vector |
| `data_td` / matrix loaders | `numpy.ndarray` | `(n_time, n_space)` after normalization | loader helpers | Raw `.npy` matrices are normalized to time-by-distance before local extraction |

### Multi-Section Example: `data_sections`

**Type**: `list of numpy.ndarray`

**Shape**: Each array is `(n_channels, n_samples)`
- `n_channels`: Number of spatial channels in the section
- `n_samples`: Number of time samples per channel

**Description**: List of 2D arrays, where each array represents a cable section to be analyzed independently by the analysis engines.

**Example**:
```python
# Two sections from a 100-channel DAS array
sections = [
    data[0:41, :],    # Section 1: channels 0-40 (41 channels)
    data[60:100, :]   # Section 2: channels 60-99 (40 channels)
]
```

### Single-Section Example: `section_data`

```python
# One section centered around the stimulation point
section_data = data[80:141, :]  # shape: (61 channels, n_samples)
```

### Single-Trace Example: `time_s` and `signal_microstrain`

```python
time_s, trace_raw = load_dynamic_range_data(...)
signal_microstrain = data_processing(trace_raw, data_is_strain=False, fs=10000)
```

---

## Input Parameters

### Measurement Configuration

| Parameter | Type | Units | Description | Example |
|-----------|------|-------|-------------|---------|
| `fs` (interrogation_rate) | `float` | Hz | DAS interrogation rate (sampling frequency) | `10000` (10 kHz) |
| `gauge_length` | `float` | meters | DAS gauge length | `10.0` |
| `delta_x_m` / `channel_spacing` | `float` | meters | Spatial spacing between adjacent channels/SSLs | `1.0` |

### Section Definition

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `test_sections_channels` | `list of [int, int]` | List of `[start, end]` channel ranges (inclusive) for each section | `[[0,40], [60,99]]` |
| `test_sections` | `list of str` | Names/labels for each test section | `['Section 1', 'Section 2']` |
| `section_data` | `ndarray(n_ssl, n_samples)` | One pre-sliced section passed directly to single-section engines | `data[80:141, :]` |
| `levels_time_steps` | `list[[int, int]]` | Time windows used by fidelity THD level analysis | `[[0, 200000], [200000, 400000]]` |

Note: channel ranges and section labels are commonly used in notebooks/caller code to slice `data` into per-section arrays. 

### Data Type and Processing

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `data_unit` | `str` | Physical unit of input data | `'pε'`, `'nε'`, `'rad'`, `'phase'`, `'dphase'` |
| `window_function` | `str` | FFT window function name | `'none'`, `'blackman-harris'`, `'hann'`, `'hamming'`, `'flattop'` |
| `data_is_strain` | `bool` | Whether a local trace is already in strain units or must be converted from phase | `True`, `False` |
| `matrix_layout` | `str` | Layout of raw 2D `.npy` matrices before normalization | `'time_space'`, `'space_time'`, `'time_distance'`, `'distance_time'`, `'auto'` |
| `highpass_hz` | `float or None` | Optional high-pass cutoff used in trace preprocessing | `5.0`, `None` |

---

## Computed Parameters

### Derived from Input Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `N` (FFT samples) | `n_samples` | Number of time samples used by an FFT on one channel or one extracted trace |
| `freq_resolution` | `fs / N` | Frequency bin width in Hz |
| `M` (fft_bins) | `N // 2 + 1` | Number of FFT frequency bins for a real FFT |
| `n_blocks` | `n_samples // fft_size` | Number of complete non-overlapping FFT blocks in block-based methods |
| `nyquist` | `fs / 2` | Maximum represented frequency |

**Example**:
```python
fs = 10000               # 10 kHz sampling
N = data.shape[1]        # Number of time samples in one channel/trace
freq_resolution = fs / N
M = N // 2 + 1
```

---

## Output Data Structure

### Output Families

| Output Name | Type | Used By | Structure |
|------------|------|---------|-----------|
| `frequency_spectra` | `list[tuple]` | `self_noise.calculate_self_noise()` | `[(frequencies, rms_asd), ...]` |
| `crosstalk_result` | `dict` | `crosstalk.calculate_crosstalk()` | keys: `crosstalk_db`, `max_xt_db`, `magnitudes`, `reference_level` |
| `fidelity_result` | `dict` | `fidelity.calculate_fidelity_thd()` | top-level keys: `fs`, `stimulus_freq`, `snr_threshold_db`, `fft_size`, `harmonics`, `sections` |
| `frequency_response_result` | `dict` | `frequency_response.calculate_frequency_response()` | keys: `signal_microstrain`, `frequency_hz`, `amplitude_db_re_1ue`, `freq_points_hz`, `normalized_db`, optional output paths |
| `spatial_resolution_result` | `dict` | `spatial_resolution.calculate_spatial_resolution()` | keys: `x_vec_m`, `amp_profile`, `amp_norm`, `LL_m`, `LR_m`, `spatial_resolution_m`, `snr_db`, `params` |
| `dynamic_range_result` | `dict` from `compute_*`; reporting side effects from `calculate_*` | `dynamic_range.compute_dynamic_range_hilbert()`, `dynamic_range.compute_dynamic_range_thd()` | keys include trigger time/strain, violation masks, and intermediate envelopes or THD arrays |

### Spectrum Output: `frequency_spectra`

**Type**: `list of tuple`

**Structure**: `[(frequencies_1, values_1), (frequencies_2, values_2), ...]`
- One tuple per input section
- `frequencies`: `numpy.ndarray` of frequency bins (Hz)
- `values`: `numpy.ndarray` of analysis results (units depend on analysis engine)

**Applies To**: `self_noise.calculate_self_noise()`

**Example**:
```python
frequency_spectra = calculate_self_noise(sections, interrogation_rate=fs, ...)

freqs, values = frequency_spectra[0]
# freqs: array([0.0, 0.0083, 0.0167, ..., 5000.0])
# values: array([...])  # RMS ASD in the same unit family as the input
```

### Dict Output Example: `crosstalk_result`

```python
result = calculate_crosstalk(section_data=section_data, stimulus_freq=1000, fs=10000)

# result keys
# - crosstalk_db: ndarray(n_ssl)
# - max_xt_db: float
# - magnitudes: ndarray(n_ssl)
# - reference_level: float
```

---

## Function-Level Data Contracts

| Module | Main Function | Primary Input | Primary Output | Notes |
|-------|---------------|---------------|----------------|-------|
| `self_noise` | `calculate_self_noise` | `list[ndarray(n_channels, n_samples)]` | `list[(freqs, rms_asd)]` | Multi-section spectral workflow |
| `crosstalk` | `calculate_crosstalk` | `section_data: ndarray(n_ssl, n_samples)` | `dict` | Assumes stimulation point is the center SSL |
| `fidelity` | `calculate_fidelity_thd` | `time_series_data: ndarray(n_channels, n_samples)` plus `levels_time_steps` | `dict` | Returns per-level and per-channel THD summary |
| `frequency_response` | `load_frequency_response_data` | raw `.npy` matrix or folder | `(time_s, trace_raw, distance_m, local_pos)` | Produces a local trace around the stretcher |
| `frequency_response` | `calculate_frequency_response` | `time_s`, `signal_microstrain` | `dict` | Returns full and normalized frequency response arrays |
| `dynamic_range` | `load_dynamic_range_data` | raw `.npy` matrix or folder | `(time_s, trace_raw)` | Extracts one local trace |
| `dynamic_range` | `compute_dynamic_range_hilbert` | `time_s`, `signal_microstrain` | `dict` | Returns envelopes and trigger point |
| `dynamic_range` | `compute_dynamic_range_thd` | `time_s`, `signal_microstrain` | `dict` | Returns sliding THD series and trigger point |
| `spatial_resolution` | `load_spatial_resolution_data` | raw `.npy` matrix or folder | `(time_s, distance_m, section_data)` | Returns a spatial section in `(n_ssl, n_samples)` form |
| `spatial_resolution` | `calculate_spatial_resolution` | `section_data: ndarray(n_ssl, n_samples)` | `dict` | Returns LL/LR/SR metrics and profile arrays |

---

## Frequency Array

**Range**: `0` to `fs/2` (Nyquist frequency)

**Spacing**: Uniform, `Δf = fs / N`

**Generation**: `numpy.fft.rfftfreq(N, 1.0/fs)`

Note: some block-FFT methods use `numpy.fft.fftfreq(fft_size, d=1.0/fs)[:fft_size//2]` instead of `rfftfreq`.

---

## Common Processing Steps

### 1. Raw Matrix Normalization
```python
# Loader functions normalize raw matrices to time-by-distance layout
data_td = _normalize_matrix_layout(data, matrix_layout='auto')
```

### 2. Section or Local-Trace Extraction
```python
# Extract channel ranges from full dataset
sections = [
    data[test_sections_channels[i][0] : test_sections_channels[i][1]+1, :]
    for i in range(len(test_sections))
]

# Or extract one local trace around a stretcher position
local_signal, local_pos, idx = extract_local_signal(
    data_microstrain=data_td,
    distance_m=distance_m,
    stretcher_start_m=90.0,
    stretcher_end_m=100.0,
    span_m=10.0,
)
```

### 3. Optional Phase-to-Strain Processing
```python
signal_microstrain = data_processing(
    trace_raw,
    data_is_strain=False,
    gauge_length=gauge_length,
    fs=fs,
    highpass_hz=5.0,
)
```

### 4. FFT or Windowed Analysis
```python
# Per-trace or per-channel FFT
fft_result = numpy.fft.rfft(signal_microstrain)

# Or block-based FFT / periodogram depending on the engine
```

### 5. Return Module-Specific Results
```python
# self_noise
frequency_spectra = [(frequencies, rms_asd), ...]

# crosstalk / frequency_response / spatial_resolution / fidelity
result = {"metric": value, "series": array_data, ...}
```

---

## Data Units and Conversions

### Supported Input Units

| Unit | Full Name | Typical Use |
|------|-----------|-------------|
| `'pε'` | Picostrain (10⁻¹² ε) | High-sensitivity strain measurements |
| `'nε'` | Nanostrain (10⁻⁹ ε) | Standard strain measurements |
| `'rad'` | Radians | Phase measurements |
| `'phase'` | Phase (radians) | Optical phase |
| `'dphase'` | Differential phase | Phase derivative |

### dB Conversion

For visualization, ASD values are converted to decibel scale:

```python
# Relative to 1 unit/√Hz
asd_db = 20 * log10(asd / 1.0)
```

**Example**:
- 10 pε/√Hz → 20 dB re 1 pε/√Hz
- 1 pε/√Hz → 0 dB re 1 pε/√Hz
- 0.1 pε/√Hz → -20 dB re 1 pε/√Hz

### Phase-to-Strain Conversion

Several modules share the same conversion model:

$$
\varepsilon = \frac{\lambda}{4 \pi n_{eff} L_g \xi} \cdot \phi
$$

where:

| Symbol | Parameter | Meaning |
|--------|-----------|---------|
| $\lambda$ | `wavelength_m` | Laser wavelength |
| $n_{eff}$ | `n_eff` | Effective refractive index |
| $L_g$ | `gauge_length` | Gauge length |
| $\xi$ | `xi` | Photoelastic scaling factor |
| $\phi$ | `phase_rad` | Phase input in radians |

When `output_in_microstrain=True`, the result is scaled by $10^6$.

---

## Frequency Bands

### Standard Analysis Bands

| Band Name | Frequency Range | Application |
|-----------|----------------|-------------|
| Seismic | 1 – 100 Hz | Earthquake monitoring, microseismic |
| Acoustic | 100 – 1000 Hz | Traffic, machinery, intrusion detection |
| Ultrasonic | 1000 – 5000 Hz | High-frequency vibrations |

---

## Example Workflow

```python
import numpy as np
from pySEAFOM import (
    calculate_self_noise,
    load_frequency_response_data,
    data_processing,
    calculate_frequency_response,
)

# === Input Parameters ===
fs = 10000                          # Sampling rate: 10 kHz
gauge_length = 10.0                 # Gauge length: 10 meters
test_sections_channels = [[0,40], [60,99]]

# === Load or Generate Data ===
data = np.load('das_measurements.npy')   # shape: (n_channels, n_samples)

# === Run A Multi-Section Engine ===
sections = [
    data[0:41, :],
    data[60:100, :]
]

frequency_spectra = calculate_self_noise(
    data_sections=sections,
    interrogation_rate=fs,
    window_function='blackman-harris',
    data_type='phase'
)

freqs_s1, values_s1 = frequency_spectra[0]
print(f"Section 1: {len(freqs_s1)} frequency bins")

# === Run A Single-Trace Engine ===
time_s, trace_raw, distance_m, local_pos = load_frequency_response_data(
    folder_or_file='example.npy',
    fs=fs,
    delta_x_m=1.0,
    stretcher_start_m=90.0,
    stretcher_end_m=100.0,
)

signal_microstrain = data_processing(
    trace_raw,
    data_is_strain=False,
    gauge_length=gauge_length,
    fs=fs,
)

fr_result = calculate_frequency_response(
    time_s=time_s,
    signal_microstrain=signal_microstrain,
    interrogation_rate_hz=fs,
    show_plot=False,
)

print(fr_result['freq_points_hz'][:5])
print(fr_result['normalized_db'][:5])
```

---

## Data Quality Considerations

### Frequency Resolution
- **Formula**: `Δf = fs / N` where `N` is the number of samples in the analyzed trace/block
- **Example**: 1,200,000 samples at 10 kHz → 0.0083 Hz resolution
- **Impact**: More samples provide better low-frequency resolution

### Channel Count per Section
- **Recommended**: 10+ channels for statistical stability when averaging across channels
- **Typical**: 20-50 channels per section
- **Impact**: More channels improve robustness for self-noise, fidelity, and some section-based workflows

### Window Functions
- **`'none'`**: No windowing, best frequency resolution, spectral leakage
- **`'blackman-harris'`**: Excellent sidelobe suppression, slightly reduced resolution
- **`'hann'`**: Good balance, industry standard
- **`'flattop'`**: Best amplitude accuracy, poor resolution

### Data-Model Caveat
- The package should be documented as a set of related workflows, not as one single analysis engine with one universal return type.
- Loader functions and analysis functions use different data contracts by design.

---

## Summary Table

| Category | Parameter | Type | Example |
|----------|-----------|------|---------|
| **Input Data** | `data_sections` | `list[ndarray(n_ch, n_samp)]` | Self-noise input |
| **Input Data** | `section_data` | `ndarray(n_ssl, n_samp)` | Crosstalk or spatial-resolution section |
| **Input Data** | `time_s`, `signal_microstrain` | `ndarray(n_samp)` | Dynamic-range or frequency-response trace |
| **Sampling** | `fs` | `float` | 10000 Hz |
| **Spatial** | `gauge_length` | `float` | 10.0 m |
| **Spatial** | `delta_x_m` / `channel_spacing` | `float` | 1.0 m |
| **Sections** | `test_sections_channels` | `list[[int,int]]` | `[[0,40],[60,99]]` |
| **Labels** | `test_sections` | `list[str]` | `['Section 1', 'Section 2']` |
| **Units** | `data_unit` | `str` | `'pε'` |
| **Processing** | `window_function` | `str` | `'blackman-harris'` |
| **Processing** | `data_is_strain` | `bool` | `False` |
| **Output** | `frequency_spectra` | `list[(freqs, values)]` | Self-noise output |
| **Output** | `result` | `dict` | Crosstalk, fidelity, frequency response, spatial resolution |

---

*Last Updated: March 21, 2026*
