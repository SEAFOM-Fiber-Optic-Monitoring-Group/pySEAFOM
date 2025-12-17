![SEAFOM Logo](https://seafom.otm-networks.com/wp-content/uploads/sites/20/2017-12-01_SEAFOM-Fiber-Optic-Monitoring-Group_450x124.png)

  

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

```

  

## 📁 Features & Modules

  

### Current Modules

  

#### `pySEAFOM.self_noise`

Self-noise analysis
#### `pySEAFOM.dynamic_range`

Dynamic range analysis

  

### Future Modules (Planned)

- **Frequency Response**: Frequency-dependent sensitivity

- **Spatial Resolution**: Gauge length verification

- **Noise Floor**: System noise characterization

  

## 📚 Documentation

  

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

#### `analyze_dynamic_range_hilbert()
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

- `results_dir` (str): Output directory

**Outputs:**

- Prints a formatted summary (trigger time, limit strain, etc.)
- Optional figure: `dynamic_range_hilbert.png`
- Optional CSV: `dynamic_range_hilbert.csv`

#### `analyze_dynamic_range_thd()`

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

- `results_dir` (str): Output directory

**Outputs:**

- Prints a formatted summary (trigger time, limit strain, etc.)
- Optional figure: `dynamic_range_thd.png`
- Optional CSV: `dynamic_range_thd.csv`

  





  

## 🧪 Example Notebook

  

See `self_noise_test.ipynb` for a complete example using synthetic data:

- Generates known ASD synthetic signals

- Validates calculation accuracy

- Demonstrates all visualization options


See `dynamic_range_test.ipynb` for a complete example using synthetic data:

- Extract and process data from a npy DAS matrix

- Calculates dynamic range limit using Hilbert

- Calculates dynamic range limit using THD

  

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

4. **Hilbert Test**: Run `analyze_dynamic_range_hilbert()` to detect envelope-error trigger

5. **THD Test**: Run `analyze_dynamic_range_thd()` to detect harmonic-distortion trigger

6. **Report / Save**: Store plots + CSV summaries for traceability


  

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

│   └── pySEAFOM/

│       ├── __init__.py            # package exports

│       └── self_noise.py          # self-noise analysis engine

│       └── dynamic_range.py          # dynamic_range analysis engine

├── testing_notebooks/

│   └── self_noise_test.ipynb      # synthetic validation notebook

│   └── dynamic_range_test.py      # synthetic validation notebook

│   └── simulation_dynamic_range.py      # generate sythetic data for dynamic_range

├── workflows/

│   └── SELF_NOISE_WORKFLOW.md     # step-by-step processing summary

│   └── DYNAMIC_RANGE_WORKFLOW.md     # step-by-step processing summary

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