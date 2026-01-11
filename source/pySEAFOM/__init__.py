"""
pySEAFOM - Performance analysis tools for Distributed Acoustic Sensing (DAS) systems

This package provides standardized tools for testing and evaluating DAS interrogators,
following SEAFOM (Subsea Fibre Optic Monitoring) recommended procedures.

Modules:
- self_noise: Self-noise analysis and visualization tools
- dynamic_range: Dynamic range analysis (Hilbert envelope and sliding THD)
- fidelity: Fidelity / THD analysis tools
- crosstalk: Crosstalk analysis tools
- (more modules to be added)
"""

__version__ = "0.1.8"
__author__ = "SEAFOM Fiber Optic Monitoring Group"




# Import submodules
from . import self_noise
from . import dynamic_range
from . import fidelity
from . import crosstalk

# Convenience imports for commonly used functions
from .self_noise import (
    calculate_self_noise,
    plot_combined_self_noise_db,
    report_self_noise,
)

from .dynamic_range import (
    load_dynamic_range_data,
    data_processing,
    calculate_dynamic_range_hilbert,
    calculate_dynamic_range_thd,
)

from .fidelity import (
    calculate_fidelity_thd,
    report_fidelity_thd,
    compute_thd,
    is_good_quality_block,
)

from .crosstalk import (
    calculate_crosstalk,
    plot_crosstalk,
    plot_crosstalk_sections,
    report_crosstalk,
)

__all__ = [
    # submodules
    "self_noise",
    "dynamic_range",
    "fidelity",
    "crosstalk",

    # self_noise
    "calculate_self_noise",
    "plot_combined_self_noise_db",
    "report_self_noise",

    # dynamic_range
    "load_dynamic_range_data",
    "data_processing",
    "calculate_dynamic_range_hilbert",
    "calculate_dynamic_range_thd",

    # fidelity
    "calculate_fidelity_thd",
    "report_fidelity_thd",
    "compute_thd",
    "is_good_quality_block",

    # crosstalk
    "calculate_crosstalk",
    "plot_crosstalk",
    "plot_crosstalk_sections",
    "report_crosstalk",
]