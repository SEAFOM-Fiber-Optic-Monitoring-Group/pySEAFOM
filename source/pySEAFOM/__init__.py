"""
pySEAFOM - Performance analysis tools for Distributed Acoustic Sensing (DAS) systems

This package provides standardized tools for testing and evaluating DAS interrogators,
following SEAFOM (Subsea Fibre Optic Monitoring) recommended procedures.

Modules:
- self_noise: Self-noise analysis and visualization tools
- dynamic_range: Dynamic range analysis (Hilbert envelope and sliding THD)
- (more modules to be added)
"""

__version__ = "0.1.6"
__author__ = "SEAFOM Fiber Optic Monitoring Group"



# Import submodules
from . import self_noise

# Convenience imports for commonly used functions
from .self_noise import (
    calculate_self_noise,
    plot_combined_self_noise_db,
    report_self_noise,
)

__all__ = [
    "self_noise",
    "calculate_self_noise",
    "plot_combined_self_noise_db",
    "report_self_noise",
]



# Import submodules
from . import dynamic_range

# Convenience imports for commonly used functions (optional, but matches their style)
from .dynamic_range import (
    load_dynamic_range_data,
    data_processing,
    analyze_dynamic_range_hilbert,
    analyze_dynamic_range_thd,
)

__all__ = [
    "self_noise",
    "dynamic_range",
    "load_dynamic_range_data",
    "data_processing",
    "analyze_dynamic_range_hilbert",
    "analyze_dynamic_range_thd",
]