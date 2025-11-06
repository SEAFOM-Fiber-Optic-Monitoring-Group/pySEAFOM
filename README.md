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
data = np.load('your_das_data.npy')  # Shape: (n_channels, n_samples)

# Define test sections (channel ranges to analyze)
sections = [data[0:50, :], data[100:150, :]]  # Two cable sections
section_names = ['Section A', 'Section B']

# Calculate self-noise for each section (using direct import)
results = calculate_self_noise(
    sections,
    interrogation_rate=10000,  # Hz
    gauge_length=10.0,         # meters
    window_function='blackman-harris',
    data_type='pε'             # picostrain
)

# OR using module import:
# results = pySEAFOM.self_noise.calculate_self_noise(
    sections,
    interrogation_rate=10000,  # Hz
    gauge_length=10.0,         # meters
    window_function='blackman-harris',
    data_type='pε'             # picostrain
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
Self-noise analysis tools for DAS interrogators:
- **Self-Noise Calculation**: RMS amplitude spectral density across multiple channels
- **Multi-Section Analysis**: Compare different cable sections or spatial locations
- **Publication-Ready Plots**: dB-scale visualization with metadata and frequency bands
- **Text Reports**: Generate formatted reports of key frequencies and band statistics
- **Flexible Units**: Support for picostrain (pε), nanostrain (nε), radians, and arbitrary units
- **Window Functions**: Multiple FFT window options (Blackman-Harris, Hann, Hamming, etc.)

### Future Modules (Planned)
- **Linearity Analysis**: Dynamic range and linearity testing
- **Frequency Response**: Frequency-dependent sensitivity
- **Spatial Resolution**: Gauge length verification
- **Noise Floor**: System noise characterization

## 📚 Documentation

### Main Functions

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
- `org_data_unit` (str): Display unit
- `title` (str): Plot title
- `sampling_freq` (float): Sampling rate (for metadata box)
- `n_channels` (int): Total channels (for metadata box)
- `duration` (float): Recording duration (for metadata box)

#### `report_self_noise()`
Prints formatted text report.

**Parameters:**
- `results`: Output from `calculate_self_noise()`
- `gauge_length` (float): Gauge length in meters
- `test_sections` (list): Section names
- `band_frequencies` (list): Frequency bands for averaging, e.g., `[(1, 100), (100, 1000)]`
- `report_in_db` (bool): Use dB scale or linear units
- `org_data_unit` (str): Display unit

## 🧪 Example Notebook

See `self_noise_test.ipynb` for a complete example using synthetic data:
- Generates known ASD synthetic signals
- Validates calculation accuracy
- Demonstrates all visualization options

## 📊 Typical Workflow

1. **Prepare Data**: Load DAS measurements (channels × samples)
2. **Define Sections**: Select channel ranges for analysis
3. **Calculate Self-Noise**: Use `calculate_self_noise()` with appropriate parameters
4. **Visualize**: Create plots with `plot_combined_self_noise_db()`
5. **Report**: Generate text summaries with `report_self_noise()`

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
├── self_noise              - Self-noise analysis module
│   ├── calculate_self_noise()
│   ├── plot_combined_self_noise_db()
│   └── report_self_noise()
├── (future modules)        - Additional analysis engines
└── examples/               - Example notebooks
    └── self_noise_test.ipynb
```

## 🔌 Adding New Modules

To add a new analysis module:

1. Create `source/your_module.py` with your functions
2. Update `source/__init__.py`:
   ```python
   from . import self_noise, your_module
   ```
3. Add documentation to README
4. Create example notebook in root directory

See the existing `self_noise.py` module as a template.

## 🤝 Contributing

We welcome contributions from researchers, engineers, and developers working in the fiber optic sensing space. Please see our [contribution guidelines](CONTRIBUTING.md) to get started.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

This repository follows the [SEAFOM Governance Policy](https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/governance/blob/main/GOVERNANCE.md).
