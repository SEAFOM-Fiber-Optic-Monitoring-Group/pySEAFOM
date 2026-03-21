<img src="docs/assets/logo.png" alt="pySEAFOM logo" width="280" style="max-width: 100%; height: auto;" />

  

# pySEAFOM

  
A Python library for performance analysis and testing of Distributed Acoustic Sensing (DAS) interrogators, developed by SEAFOM's Measuring Sensor Performance group. This package provides standardized tools for testing, benchmarking, and performance evaluation of DAS systems following SEAFOM recommended procedures.

  
## 🌐 Purpose


To promote transparency, consistency, and collaboration in the evaluation of DAS interrogator performance by providing open-source tools and standardized workflows.


## 📚 Documentation

https://seafom-fiber-optic-monitoring-group.github.io/pySEAFOM/

  

## ⚡ Quick Start

  

### Installation

  

```bash

pip install pySEAFOM

```

  

### Basic Usage

  
```python

from pySEAFOM import calculate_self_noise, plot_combined_self_noise_db
import numpy as np



# Load your DAS data (channels × time samples)
data = np.load('your_das_data.npy')  # Shape: (n_channels, n_samples)
 

# Define test sections (channel ranges to analyze)
sections = [data[0:50, :], data[100:150, :]]  # Two cable sections
section_names = ['Section A', 'Section B']
  
# Calculate self-noise for each section (using direct import)
results = calculate_self_noise(
    sections,
    interrogation_rate=10000,  # Hz
    gauge_length=10.0,         # meters
    window_function='blackman-harris',
    data_type='pε'             # picostrain
)
 

# OR using module import:
# results = pySEAFOM.self_noise.calculate_self_noise(
    sections,
    interrogation_rate=10000,  # Hz
    gauge_length=10.0,         # meters
    window_function='blackman-harris',
    data_type='pε'             # picostrain

)
 

# Visualize results
plot_combined_self_noise_db(
    results=results,
    test_sections=section_names,
    gauge_length=10.0,
    org_data_unit='pε',
    title='DAS Self-Noise Test Results'
)


```

  

## 📁 Features & Modules

  

### Current Modules

#### `pySEAFOM.self_noise`

Self-noise analysis

#### `pySEAFOM.dynamic_range`

Dynamic range analysis

#### `pySEAFOM.fidelity`

Fidelity (THD) analysis

#### `pySEAFOM.crosstalk`

Crosstalk analysis

#### `pySEAFOM.frequency_response`

Frequency response analysis

#### `pySEAFOM.spatial_resolution`

Spatial resolution analysis

### Future Modules (Planned)

- **Noise Floor**: System noise characterization


## Functions

### self_noise

#### `calculate_self_noise()`

Computes RMS amplitude spectral density across channels.

  

**Parameters:**

- `sections` (list): List of 2D arrays (channels × samples) for each test section
- `interrogation_rate` (float): Sampling frequency in Hz
- `gauge_length` (float): Gauge length in meters
- `window_function` (str): FFT window type ('blackman-harris', 'hann', 'none', etc.)
- `data_type` (str): Data unit ('pε', 'nε', 'rad', or custom)

  

**Returns:**

- List of tuples: `[(frequencies, asd), ...]` for each section

  

#### `plot_combined_self_noise_db()`

Creates publication-quality self-noise plots.

  

**Parameters:**

- `results`: Output from `calculate_self_noise()`
- `test_sections` (list): Section names
- `gauge_length` (float): Gauge length in meters
- `data_unit` (str): Display unit
- `title` (str): Plot title
- `sampling_freq` (float): Sampling rate (for metadata box)
- `n_channels` (int): Total channels (for metadata box)
- `duration` (float): Recording duration (for metadata box)

  

#### `report_self_noise()`

Prints formatted text report.




### dynamic_range
#### `load_dynamic_range_data()`

Loads one (or many) `.npy` files, builds a 2D matrix, and extracts a 1D trace at a chosen spatial position.

**Parameters:**

- `folder_or_file` (str): Folder with `.npy` files or a single `.npy` file
- `fs` (float): Sampling / interrogator rate in Hz
- `delta_x_m` (float): Spatial step between channels [m]
- `x1_m` (float): Spatial window start [m]
- `x2_m` (float): Spatial window end [m]
- `test_sections_channels` (float): Position inside the spatial window [m]
- `time_start_s` (float): Analysis window start time [s]
- `duration` (float | None): Analysis window duration [s]
- `average_over_cols` (int): Number of adjacent channels to average
- `matrix_layout` (str): `'time_space'`, `'space_time'`, or `'auto'`


**Returns:**

- `(time_s, trace)` where:
  - `time_s` is a 1D time vector [s]
  - `trace` is a 1D extracted signal

#### `data_processing()`

Optional unit conversion (phase to strain) and optional high-pass filtering for the extracted 1D trace.

**Parameters:**

- `trace` (1D array): Input trace (phase [rad] or strain)
- `data_is_strain` (bool): If False, converts phase [rad] to microstrain [µε]
- `gauge_length` (float): Gauge length [m] (used for converting)
- `highpass_hz` (float | None): High-pass cutoff [Hz] (set None to disable)
- `fs` (float): Sampling rate [Hz] (required when high-pass is enabled)


**Returns:**

- 1D array: processed signal (microstrain [µε] if conversion is enabled)

#### `calculate_dynamic_range_hilbert()
`
Hilbert envelope dynamic range test. Compares measured envelope vs theoretical envelope and triggers when the relative error exceeds a threshold.

**Parameters:**

- `time_s` (1D array): Time vector [s]
- `signal_microstrain` (1D array): Trace in microstrain [µε]
- `max_strain_microstrain` (float): Final theoretical envelope amplitude [µε]
- `ref_freq_hz` (float): Expected sine frequency [Hz]
- `smooth_window_s` (float): Envelope smoothing window [s]
- `error_threshold_frac` (float): Relative error threshold (e.g., 0.3 = 30%)
- `safezone_s` (float): Initial safe zone where triggering is ignored [s]
- `save_results` (bool): Save figure + append CSV row
- `radian_basis` (float | None): If provided with`gauge_length`, reports `peak_over_basis` as dB re rad/√Hz (computed from the peak of the last cycle converted from µε → rad). Otherwise the CSV field is empty and the metadata box omits it

- `results_dir` (str): Output directory

**Outputs:**

- Prints a formatted summary (trigger time, limit strain, etc.)
- Optional figure: `dynamic_range_hilbert.png`
- Optional CSV: `dynamic_range_hilbert.csv`

#### `calculate_dynamic_range_thd()`

Sliding THD dynamic range test. Computes THD in a moving window and triggers when THD exceeds a threshold for a minimum duration.

**Parameters:**

- `time_s` (1D array): Time vector [s]
- `signal_microstrain` (1D array): Trace in microstrain [µε]
- `ref_freq_hz` (float): Expected fundamental frequency [Hz]
- `window_s` (float): Sliding window length [s]
- `overlap` (float): Window overlap fraction (e.g., 0.75 = 75%)
- `thd_threshold_frac` (float): THD threshold (e.g., 0.15 = 15%)
- `median_window_s` (float): Median smoothing window applied to the THD curve
- `min_trigger_duration` (float): Minimum continuous violation time to trigger [s]
- `safezone_s` (float): Initial safe zone where triggering is ignored [s]
- `save_results` (bool): Save figure + append CSV row
- `radian_basis` (float | None): If provided with`gauge_length`, reports `peak_over_basis` as dB re rad/√Hz (computed from the peak of the last cycle converted from µε → rad). Otherwise the CSV field is empty and the metadata box omits it

- `results_dir` (str): Output directory

**Outputs:**

- Prints a formatted summary (trigger time, limit strain, etc.)
- Optional figure: `dynamic_range_thd.png`
- Optional CSV: `dynamic_range_thd.csv`


### fidelity

#### `calculate_fidelity_thd()`

Computes fidelity as THD (%) at a known stimulus frequency for a single pre-sliced spatial section, across one or more time “levels”.

**Inputs (typical):**

- `time_series_data` (2D array): section matrix (channels_in_section × samples)
- `fs` (float): Sampling frequency [Hz]
- `levels_time_steps` (list[[start,end]] | [start,end]): Sample index range(s) per stimulus level
- `stimulus_freq` (float): Fundamental frequency [Hz]
- `snr_threshold_db` (float): SNR gate used to accept FFT blocks
- `section_name` (str, optional): Label used in the report output

**Returns:**

- A structured dict with one section containing per-level THD and harmonic levels.

#### `report_fidelity_thd()`

Prints a compact text summary of `calculate_fidelity_thd()` results.


### crosstalk

#### `calculate_crosstalk()`

Computes a crosstalk profile and maximum crosstalk for a single spatial section centered on the stimulation point.

**Returns:**

- A result dict containing:
  - `crosstalk_db` (1D array): dB relative to reference region
  - `max_xt_db` (float): max crosstalk in the outer region
  - `magnitudes` (1D array): linear magnitudes at stimulus frequency
  - `reference_level` (float): linear reference magnitude

#### `plot_crosstalk()`

Plots a crosstalk profile (dB vs distance).

#### `report_crosstalk()`

Prints a compact text summary of `calculate_crosstalk()` results.

  


### frequency_response

#### `load_frequency_response_data()`

Loads one (or many) `.npy` files, builds a 2D matrix, and extracts a 1D trace at a chosen spatial position.

**Parameters:**

- `folder_or_file` (str): Folder with `.npy` files or a single `.npy` file
- `fs` (float): Sampling / interrogator rate in Hz
- `delta_x_m` (float): Spatial step between channels [m]
- `stretcher_start_m` (float): Spatial window start [m]
- `stretcher_end_m` (float): Spatial window end [m]
- `span_m` (int): Number of adjacent channels to average [m]
- `matrix_layout` (str): `'time_space'`, `'space_time'`, or `'auto'`


**Returns:**

- `(time_s, trace_raw, distance_m, local_pos_m)` where:
  - `time_s` is a 1D time vector [s]
  - `trace_raw` is a 1D extracted signal
  - `distance_m`is the size of the stretcher in [m]
  - `local_pos_m` is the central position of the stretcher in [m]

#### `data_processing()`

Optional unit conversion (phase to strain) and optional high-pass filtering for the extracted 1D trace.

**Parameters:**

- `trace` (1D array): Input trace (phase [rad] or strain)
- `data_is_strain` (bool): If False, converts phase [rad] to microstrain [µε]
- `gauge_length` (float): Gauge length [m] (used for converting)
- `highpass_hz` (float | None): High-pass cutoff [Hz] (set None to disable)
- `fs` (float): Sampling rate [Hz] (required when high-pass is enabled)


**Returns:**

- 1D array: processed signal (microstrain [µε] if conversion is enabled)

#### `calculate_frequency_response()
`
Frequency response test. Computes the DAS frequency response (FFT magnitude in dB re 1 µε) and the normalized frequency response over the step frequencies.

**Parameters:**

- `time_s` (1D array): Time vector [s]
- `signal_microstrain` (1D array): Local trace in microstrain [µε]
- `interrogation_rate_hz` (float): Repetition / sampling rate [Hz]
- `n_steps` (int): Number of frequency steps
- `freq_min_frac_nyq` (float): Minimum frequency as a fraction of Nyquist (e.g., 0.02)
- `freq_max_frac_nyq` (float): Maximum frequency as a fraction of Nyquist (e.g., 0.80)
- `window_spectrogram_s` (float): Spectrogram window length [s] (local diagnostics)
- `overlap_spectrogram_frac` (float): Spectrogram overlap fraction (0.5 = 50%) (local diagnostics)
- `save_results` (bool): If True, saves figures + CSV
- `results_dir` (str): Output directory

**Outputs:**

- Returns a dictionary with frequency arrays and dB curves
- Optional figure: `frequency_response_local_time_spectrogram_fft.png`,  `frequency_response.png`and `frequency_response_normalized.png`
- Optional CSV: `frequency_response_normalized.csv`




### spatial_resolution

#### `load_spatial_resolution_data()`

Loads one (or many) `.npy` files, builds a 2D matrix (concatenated along time), and extracts a spatial-temporal section for analysis.

**Parameters:**

- `folder_or_file` (str): Folder with `.npy` files or a single `.npy` file
- `fs` (float): Sampling / interrogator rate in Hz
- `delta_x_m` (float): Spatial step between channels [m]
- `x1_m` (float): Spatial window start [m]
- `x2_m` (float): Spatial window end [m]
- `time_start_s` (float): Analysis window start time [s]
- `duration` (float | None): Analysis window duration [s]
- `matrix_layout` (str): `'time_space'`, `'space_time'`, or `'auto'`

**Returns:**

- `(time_s, section_data)` where:
  - `time_s` is a 1D time vector [s]
  - `section_data` is a 2D matrix `(n_time, n_space)` representing the extracted section


#### `data_processing()`

Optional unit conversion (phase to strain) and optional high-pass filtering for the extracted data.

**Parameters:**

- `data` (2D array): Input data (phase [rad] or strain)
- `data_is_strain` (bool): If False, converts phase [rad] to microstrain [µε]
- `gauge_length` (float): Gauge length [m] (used for conversion)
- `highpass_hz` (float | None): High-pass cutoff [Hz] (set None to disable)
- `fs` (float): Sampling rate [Hz] (required when high-pass is enabled)

**Returns:**

- 2D array: processed data (microstrain [µε] if conversion is enabled)


#### `calculate_spatial_resolution()`

Estimates the spatial resolution from the spatial amplitude profile at a reference frequency.

**Parameters:**

- `section_data` (2D array): Input matrix `(n_ssl, n_samples)` (space × time)
- `fs` (float): Sampling rate [Hz]
- `delta_x_m` (float): Spatial step [m]
- `ref_freq_hz` (float): Reference stimulus frequency [Hz]
- `fft_size` (int): FFT block size
- `snr_threshold_db` (float): Recommended minimum SNR
- `target_pos_m` (float): Expected stretcher position [m]
- `save_results` (bool): Save figures + CSV summary
- `results_dir` (str): Output directory

**Outputs:**

- Returns a summary that includes:
  - detected peak position
  - SNR
  - `LL` (left slope width)
  - `LR` (right slope width)
  - spatial resolution (mean of `LL` and `LR`)
- Optional figures:
  - `spatiotemporal_map.png`
  - `spatial_resolution_profile.png`
- Optional CSV:
  - `spatial_resolution_summary.csv`



## 🤝 Contributing

  

We welcome contributions from researchers, engineers, and developers working in the fiber optic sensing space. Please see our [contribution guidelines](CONTRIBUTING.md) to get started.

  

## 📜 License

  

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

  

This repository follows the [SEAFOM Governance Policy](https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/governance/blob/main/GOVERNANCE.md).