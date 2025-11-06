# pySEAFOM Data Model

**Repository**: SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM  
**Version**: 0.1.3

This document describes the data structures, parameters, and processing flow used in the pySEAFOM library for DAS (Distributed Acoustic Sensing) self-noise analysis.

---

## Overview

The pySEAFOM library processes DAS time-series data to compute self-noise characteristics using RMS (Root Mean Square) averaging across multiple fiber channels. The analysis follows this flow:

```
Time-Domain Data → FFT → Amplitude Spectral Density → RMS Averaging → Self-Noise ASD
```

---

## Input Data Structure

### Primary Input: `data_sections`

**Type**: `list of numpy.ndarray`

**Shape**: Each array is `(n_channels, n_samples)`
- `n_channels`: Number of spatial channels in the section
- `n_samples`: Number of time samples per channel

**Description**: List of 2D arrays, where each array represents a cable section to be analyzed independently.

**Example**:
```python
# Two sections from a 100-channel DAS array
sections = [
    data[0:41, :],    # Section 1: channels 0-40 (41 channels)
    data[60:100, :]   # Section 2: channels 60-99 (40 channels)
]
```

---

## Input Parameters

### Measurement Configuration

| Parameter | Type | Units | Description | Example |
|-----------|------|-------|-------------|---------|
| `fs` (interrogation_rate) | `float` | Hz | DAS interrogation rate (sampling frequency) | `10000` (10 kHz) |
| `gauge_length` | `float` | meters | DAS gauge length (spatial resolution) | `10.0` |

### Section Definition

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `test_sections_channels` | `list of [int, int]` | List of `[start, end]` channel ranges (inclusive) for each section | `[[0,40], [60,99]]` |
| `test_sections` | `list of str` | Names/labels for each test section | `['Section 1', 'Section 2']` |

### Data Type and Processing

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `data_unit` | `str` | Physical unit of input data | `'pε'` (picostrain), `'nε'` (nanostrain) |
| `window_function` | `str` | FFT window function name | `'none'`, `'blackman-harris'`, `'hann'`, `'hamming'`, `'flattop'` |

---

## Computed Parameters

### Derived from Input Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `N` (total_samples) | `fs × duration` | Total number of time samples |
| `freq_resolution` | `fs / N` | Frequency bin width in Hz |
| `M` (fft_bins) | `N // 2 + 1` | Number of FFT frequency bins (real FFT) |

**Example**:
```python
fs = 10000              # 10 kHz sampling
N = fs * duration       # 1,200,000 samples
freq_resolution = fs / N # 0.0083 Hz
M = N // 2 + 1         # 600,001 FFT bins
```

---

## Output Data Structure

### Primary Output: `results`

**Type**: `list of tuple`

**Structure**: `[(frequencies_1, asd_1), (frequencies_2, asd_2), ...]`
- One tuple per input section
- `frequencies`: `numpy.ndarray` of frequency bins (Hz)
- `asd`: `numpy.ndarray` of amplitude spectral density values (data_unit/√Hz)

**Example**:
```python
results = calculate_self_noise(sections, interrogation_rate=fs, ...)
# results[0] = (freqs_section1, asd_section1)
# results[1] = (freqs_section2, asd_section2)

freqs, asd = results[0]  # First section
# freqs: array([0.0, 0.0083, 0.0167, ..., 5000.0])  # 0 to Nyquist
# asd: array([12.5, 18.3, 15.7, ..., 10.2])        # in pε/√Hz
```

### Frequency Array

**Range**: `0` to `fs/2` (Nyquist frequency)

**Spacing**: Uniform, `Δf = fs / N`

**Generation**: `numpy.fft.rfftfreq(N, 1.0/fs)`

---

## Processing Flow

### 1. Section Extraction
```python
# Extract channel ranges from full dataset
sections = [
    data[test_sections_channels[i][0] : test_sections_channels[i][1]+1, :] 
    for i in range(len(test_sections))
]
```

### 2. Per-Channel FFT
For each channel in each section:
```python
# Apply window function (if specified)
windowed_data = data[ch, :] * window

# Compute FFT
fft_result = numpy.fft.rfft(windowed_data)

# Compute amplitude spectral density
asd_ch = abs(fft_result) / normalization_factor
```

### 3. RMS Averaging Across Channels
```python
# For each frequency bin, compute RMS across all channels
asd_section = sqrt(mean(asd_ch[:, freq_bin]**2, axis=channels))
```

### 4. Return Results
```python
results = [
    (frequencies, asd_section_1),
    (frequencies, asd_section_2),
    ...
]
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

---

## Frequency Bands

### Standard Analysis Bands

| Band Name | Frequency Range | Application |
|-----------|----------------|-------------|
| Seismic | 1 – 100 Hz | Earthquake monitoring, microseismic |
| Acoustic | 100 – 1000 Hz | Traffic, machinery, intrusion detection |
| Ultrasonic | 1000 – 5000 Hz | High-frequency vibrations |

---

## Example Complete Workflow

```python
import numpy as np
from pySEAFOM import self_noise

# === Input Parameters ===
fs = 10000                          # Sampling rate: 10 kHz
duration = 120                      # Duration: 120 seconds
gauge_length = 10.0                 # Gauge length: 10 meters
n_channels = 100                    # Total channels: 100
test_sections_channels = [[0,40], [60,99]]  # Two sections
test_sections = ['Section 1', 'Section 2']
data_unit = 'pε'                    # Picostrain units

# === Load or Generate Data ===
# Shape: (100 channels, 1,200,000 samples)
data = np.load('das_measurements.npy')

# === Extract Sections ===
sections = [
    data[0:41, :],     # Section 1: 41 channels
    data[60:100, :]    # Section 2: 40 channels
]

# === Compute Self-Noise ===
results = self_noise.calculate_self_noise(
    data_sections=sections,
    interrogation_rate=fs,
    window_function='blackman-harris',
    data_type=data_unit
)

# === Access Results ===
freqs_s1, asd_s1 = results[0]  # Section 1
freqs_s2, asd_s2 = results[1]  # Section 2

print(f"Section 1: {len(freqs_s1)} frequency bins")
print(f"ASD at 100 Hz: {asd_s1[freqs_s1 == 100][0]:.2f} {data_unit}/√Hz")

# === Visualize ===
self_noise.plot_combined_self_noise_db(
    results=results,
    title='DAS Self-Noise Analysis',
    test_sections=test_sections,
    gauge_length=gauge_length,
    data_unit=data_unit,
    sampling_freq=fs,
    n_channels=n_channels,
    duration=duration
)

# === Generate Report ===
self_noise.report_self_noise(
    results=results,
    gauge_length=gauge_length,
    test_sections=test_sections,
    band_frequencies=[(1, 100), (100, 1000)],
    report_in_db=False,
    data_unit=data_unit
)
```

---

## Data Quality Considerations

### Frequency Resolution
- **Formula**: `Δf = fs / N = fs / (fs × duration) = 1 / duration`
- **Example**: 120 s duration → 0.0083 Hz resolution
- **Impact**: Longer duration provides better low-frequency resolution

### Channel Count per Section
- **Minimum**: ~10 channels recommended for stable RMS averaging
- **Typical**: 20-50 channels per section
- **Impact**: More channels reduce statistical variance in self-noise estimate

### Window Functions
- **`'none'`**: No windowing, best frequency resolution, spectral leakage
- **`'blackman-harris'`**: Excellent sidelobe suppression, slightly reduced resolution
- **`'hann'`**: Good balance, industry standard
- **`'flattop'`**: Best amplitude accuracy, poor resolution

---

## Validation Metrics

When testing with synthetic data (known ASD):

```python
# Compute relative error
rel_error = abs(estimated_asd - known_asd) / known_asd

# Typical performance
median_error < 5%      # Good agreement
mean_error < 10%       # Acceptable for most applications
```

---

## Summary Table

| Category | Parameter | Type | Example |
|----------|-----------|------|---------|
| **Input Data** | `data_sections` | `list[ndarray(n_ch, n_samp)]` | 2 sections, 40-41 channels each |
| **Sampling** | `fs` | `float` | 10000 Hz |
| **Duration** | `duration` | `float` | 120 s |
| **Spatial** | `gauge_length` | `float` | 10.0 m |
| **Channels** | `n_channels` | `int` | 100 |
| **Sections** | `test_sections_channels` | `list[[int,int]]` | `[[0,40],[60,99]]` |
| **Labels** | `test_sections` | `list[str]` | `['Section 1', 'Section 2']` |
| **Units** | `data_unit` | `str` | `'pε'` |
| **Processing** | `window_function` | `str` | `'blackman-harris'` |
| **Output** | `results` | `list[(freqs, asd)]` | 2 tuples, one per section |

---

*Last Updated: November 6, 2025*  
*pySEAFOM Version: 0.1.3*
