# pySEAFOM Data Model

**Repository**: SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM  

This document describes the data structures, parameters, and processing flow used in the pySEAFOM library for DAS (Distributed Acoustic Sensing) analysis.

---

## Overview

The pySEAFOM library provides analysis engines for processing DAS time-series data. The library uses standardized data structures and parameters across all analysis modules.

---

## Input Data Structure

### Primary Input: `data_sections`

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

Note: channel ranges and section labels are commonly used in notebooks/caller code to slice `data` into per-section arrays. Some engines (e.g., fidelity/crosstalk) operate on a single pre-sliced section array per call.

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
| `N` (total_samples) | `n_channels × n_samples` | Total number of time samples in data |
| `freq_resolution` | `fs / N` | Frequency bin width in Hz |
| `M` (fft_bins) | `N // 2 + 1` | Number of FFT frequency bins (real FFT) |

**Example**:
```python
fs = 10000              # 10 kHz sampling
# Given data array with shape (n_channels, n_samples)
N = data.shape[1]       # Number of samples per channel
freq_resolution = fs / N # Depends on data length
M = N // 2 + 1         # FFT bins
```

---

## Output Data Structure

### Output Format: `frequency_spectra`

**Type**: `list of tuple`

**Structure**: `[(frequencies_1, values_1), (frequencies_2, values_2), ...]`
- One tuple per input section
- `frequencies`: `numpy.ndarray` of frequency bins (Hz)
- `values`: `numpy.ndarray` of analysis results (units depend on analysis engine)

**Example**:
```python
# Generic analysis output
frequency_spectra = analysis_engine(sections, interrogation_rate=fs, ...)
# frequency_spectra[0] = (freqs_section1, values_section1)
# frequency_spectra[1] = (freqs_section2, values_section2)

freqs, values = frequency_spectra[0]  # First section
# freqs: array([0.0, 0.0083, 0.0167, ..., 5000.0])  # 0 to Nyquist
# values: array([12.5, 18.3, 15.7, ..., 10.2])      # Engine-specific units
```

### Frequency Array

**Range**: `0` to `fs/2` (Nyquist frequency)

**Spacing**: Uniform, `Δf = fs / N`

**Generation**: `numpy.fft.rfftfreq(N, 1.0/fs)`

---

## Common Processing Steps

### 1. Section Extraction
```python
# Extract channel ranges from full dataset
sections = [
    data[test_sections_channels[i][0] : test_sections_channels[i][1]+1, :] 
    for i in range(len(test_sections))
]
```

### 2. Per-Channel FFT (when applicable)
For frequency-domain analysis:
```python
# Apply window function (if specified)
windowed_data = data[ch, :] * window

# Compute FFT
fft_result = numpy.fft.rfft(windowed_data)

# Further processing depends on analysis engine
```

### 3. Multi-Channel Analysis
Analysis engines may aggregate across channels:
```python
# Example: RMS averaging, coherence analysis, etc.
# Implementation varies by engine
```

### 4. Return Results
```python
frequency_spectra = [
    (frequencies, values_section_1),
    (frequencies, values_section_2),
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

## Example Workflow

```python
import numpy as np
from pySEAFOM import analysis_engine  # Replace with specific engine

# === Input Parameters ===
fs = 10000                          # Sampling rate: 10 kHz
gauge_length = 10.0                 # Gauge length: 10 meters
test_sections_channels = [[0,40], [60,99]]  # Two sections
test_sections = ['Section 1', 'Section 2']
data_unit = 'pε'                    # Picostrain units

# === Load or Generate Data ===
# Shape: (n_channels, n_samples)
data = np.load('das_measurements.npy')

# === Extract Sections ===
sections = [
    data[0:41, :],     # Section 1: 41 channels
    data[60:100, :]    # Section 2: 40 channels
]

# === Run Analysis Engine ===
frequency_spectra = analysis_engine(
    data_sections=sections,
    interrogation_rate=fs,
    window_function='blackman-harris',
    data_type=data_unit
)

# === Access Results ===
freqs_s1, values_s1 = frequency_spectra[0]  # Section 1
freqs_s2, values_s2 = frequency_spectra[1]  # Section 2

print(f"Section 1: {len(freqs_s1)} frequency bins")
print(f"Value at 100 Hz: {values_s1[freqs_s1 == 100][0]:.2f} {data_unit}/√Hz")
```

---

## Data Quality Considerations

### Frequency Resolution
- **Formula**: `Δf = fs / N` where N is the number of samples per channel
- **Example**: 1,200,000 samples at 10 kHz → 0.0083 Hz resolution
- **Impact**: More samples provide better low-frequency resolution

### Channel Count per Section
- **Recommended**: 10+ channels for statistical stability
- **Typical**: 20-50 channels per section
- **Impact**: More channels improve statistical robustness of multi-channel analysis

### Window Functions
- **`'none'`**: No windowing, best frequency resolution, spectral leakage
- **`'blackman-harris'`**: Excellent sidelobe suppression, slightly reduced resolution
- **`'hann'`**: Good balance, industry standard
- **`'flattop'`**: Best amplitude accuracy, poor resolution

---

## Summary Table

| Category | Parameter | Type | Example |
|----------|-----------|------|---------|
| **Input Data** | `data_sections` | `list[ndarray(n_ch, n_samp)]` | 2 sections, 40-41 channels each |
| **Sampling** | `fs` | `float` | 10000 Hz |
| **Spatial** | `gauge_length` | `float` | 10.0 m |
| **Sections** | `test_sections_channels` | `list[[int,int]]` | `[[0,40],[60,99]]` |
| **Labels** | `test_sections` | `list[str]` | `['Section 1', 'Section 2']` |
| **Units** | `data_unit` | `str` | `'pε'` |
| **Processing** | `window_function` | `str` | `'blackman-harris'` |
| **Output** | `frequency_spectra` | `list[(freqs, values)]` | 2 tuples, one per section |

---

*Last Updated: November 6, 2025*  
