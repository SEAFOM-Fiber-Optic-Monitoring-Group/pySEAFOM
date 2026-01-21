<img src="docs/assets/logo.png" alt="pySEAFOM logo" width="280" style="max-width: 100%; height: auto;" />

  

# pySEAFOM

  

A Python library for performance analysis and testing of Distributed Acoustic Sensing (DAS) interrogators, developed by SEAFOM's Measuring Sensor Performance group. This package provides standardized tools for testing, benchmarking, and performance evaluation of DAS systems following SEAFOM recommended procedures.

  

## 🌐 Purpose

  

To promote transparency, consistency, and collaboration in the evaluation of DAS interrogator performance by providing open-source tools and standardized workflows.

  

## ⚡ Quick Start

  

### Installation

  

```bash

pip install pySEAFOM

```

  

### Basic Usage

  

**Option 1: Import specific functions directly**

```python

from pySEAFOM import calculate_self_noise, plot_combined_self_noise_db

import numpy as np

```

  

**Option 2: Import modules (recommended when using multiple engines)**

```python

import pySEAFOM

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


# Fidelity (THD) example (single-section call; loop sections externally)
section_ranges = [[0, 49], [100, 149]]
section_names = ['Section A', 'Section B']

for name, (ch0, ch1) in zip(section_names, section_ranges):
  section = data[ch0:ch1 + 1, :]
  fidelity_results = pySEAFOM.fidelity.calculate_fidelity_thd(
    section,
    fs=10000,
    levels_time_steps=[[0, 600000], [660000, 1500000]],
    stimulus_freq=500,
    snr_threshold_db=-40,
    section_name=name,
  )
  pySEAFOM.fidelity.report_fidelity_thd(fidelity_results)

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

### Future Modules (Planned)

- **Frequency Response**: Frequency-dependent sensitivity

- **Spatial Resolution**: Gauge length verification

- **Noise Floor**: System noise characterization


## 📚 Documentation

Live site (GitHub Pages): https://seafom-fiber-optic-monitoring-group.github.io/pySEAFOM/


### Main Functions (self_noise)

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




### Main Functions (dynamic_range)

  

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


### Main Functions (fidelity)

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


### Main Functions (crosstalk)

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

  





  

## 🧪 Example Notebook

  

See `self_noise_test.ipynb` for a complete example using synthetic data:

- Generates known ASD synthetic signals

- Validates calculation accuracy

- Demonstrates all visualization options


See `dynamic_range_test.ipynb` for a complete example using synthetic data:

- Extract and process data from a npy DAS matrix

- Calculates dynamic range limit using Hilbert (`delta_t_from_window_start` [s], `peak_last_cycle` [µε], `peak_over_basis` [dB re rad/√Hz])

- Calculates dynamic range limit using THD (`delta_t_from_window_start` [s], `peak_last_cycle` [µε], `peak_over_basis` [dB re rad/√Hz])


See `fidelity_test.ipynb` for a complete example using synthetic data:

- Builds two time “levels” with different harmonic content
- Runs per-section THD using `calculate_fidelity_thd()`
- Prints a simple report via `report_fidelity_thd()`


See `crosstalk_test.ipynb` for a complete example using synthetic data:

- Generates synthetic stimulated data centered on a stimulation point
- Computes crosstalk using `calculate_crosstalk()`
- Plots the profile using `plot_crosstalk()` and prints a report via `report_crosstalk()`

  

## 📊 Typical Workflow


### Self-Noise Workflow

1. **Prepare Data**: Load DAS measurements (channels × samples)

2. **Define Sections**: Select channel ranges for analysis

3. **Calculate Self-Noise**: Use `calculate_self_noise()` with appropriate parameters

4. **Visualize**: Create plots with `plot_combined_self_noise_db()`

5. **Report**: Generate text summaries with `report_self_noise()`

### Dynamic Range Workflow

1. **Prepare Data**: Load DAS measurements (time × channels) from `.npy`

2. **Extract Trace**: Use `load_dynamic_range_data()` to pick `x1_m/x2_m`, select `POS`, and average channels

3. **Pre-process**: Use `data_processing()` for phase to strain (if needed) and high-pass (optional)

4. **Hilbert Test**: Run `calculate_dynamic_range_hilbert()` to detect envelope-error trigger

5. **THD Test**: Run `calculate_dynamic_range_thd()` to detect harmonic-distortion trigger

6. **Report / Save**: Store plots + CSV summaries for traceability


### Fidelity (THD) Workflow

1. **Prepare Data**: Load DAS measurements (channels × samples)
2. **Define Sections**: Select channel ranges for analysis
3. **Define Levels**: Select time windows (sample ranges) for each stimulus level
4. **Compute THD**: For each section, slice channels and run `calculate_fidelity_thd()` with `stimulus_freq` + `snr_threshold_db`
5. **Report**: Print summaries using `report_fidelity_thd()`


### Crosstalk Workflow

1. **Prepare Data**: Load one spatial section centered on the stimulation point (SSL × samples)
2. **Compute Crosstalk**: Run `calculate_crosstalk()` with `stimulus_freq`, `fs`, `gauge_length`, and `channel_spacing`
3. **Visualize**: Plot profiles with `plot_crosstalk()`
4. **Report**: Print summaries using `report_crosstalk()`


  

## 🔧 Development Setup

  

```bash

# Clone the repository

git clone https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM.git

cd pySEAFOM

  

# Install in development mode

pip install -e .

  

# Install development dependencies

pip install -e ".[dev]"

  

# Run tests (if available)

pytest tests/

```

  

## 📦 Package Structure

  

```

pySEAFOM/

├── source/

│   └── simulation_dynamic_range.py      # generate sythetic data for dynamic_range

│   └── pySEAFOM/

│       ├── __init__.py            # package exports

│       └── self_noise.py          # self-noise analysis engine

│       └── dynamic_range.py          # dynamic_range analysis engine

│       └── fidelity.py             # fidelity / THD analysis engine

├── testing_notebooks/

│   └── self_noise_test.ipynb      # synthetic validation notebook

│   └── dynamic_range_test.ipynb      # synthetic validation notebook

│   └── fidelity_test.ipynb         # synthetic validation notebook

├── workflows/

│   └── SELF_NOISE_WORKFLOW.md     # step-by-step processing summary

│   └── DYNAMIC_RANGE_WORKFLOW.md     # step-by-step processing summary

│   └── FIDELITY_WORKFLOW.md        # step-by-step processing summary

├── README.md

├── pyproject.toml

├── setup.py

├── MANIFEST.in

└── dist/                         # build artifacts (created on release)

```

  

## 🔌 Adding New Modules

  

To add a new analysis module:

  

1. Create `source/pySEAFOM/your_module.py` with your functions

2. Update `source/pySEAFOM/__init__.py`:

   ```python

   from . import self_noise, your_module

   ```

3. Add documentation to this README (and module docstrings)

4. Add or update an example notebook under `testing_notebooks/`

  

See the existing `self_noise.py` module as a template.

  

## 🤝 Contributing

  

We welcome contributions from researchers, engineers, and developers working in the fiber optic sensing space. Please see our [contribution guidelines](CONTRIBUTING.md) to get started.

  

## 📜 License

  

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

  

This repository follows the [SEAFOM Governance Policy](https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/governance/blob/main/GOVERNANCE.md).