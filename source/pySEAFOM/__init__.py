"""
pySEAFOM - Performance analysis tools for Distributed Acoustic Sensing (DAS) systems

This package provides standardized tools for testing and evaluating DAS interrogators,
following SEAFOM (Subsea Fibre Optic Monitoring) recommended procedures.

Modules:
- self_noise: Self-noise analysis and visualization tools
- dynamic_range: Dynamic range analysis (Hilbert envelope and sliding THD)
- fidelity: Fidelity / THD analysis tools
- crosstalk: Crosstalk analysis tools
- frequency_response: MSP-02 frequency response analysis tools
"""

__version__ = "0.1.9"
__author__ = "SEAFOM Fiber Optic Monitoring Group"

# Import submodules
from . import self_noise
from . import dynamic_range
from . import fidelity
from . import crosstalk
from . import frequency_response

# Convenience imports for commonly used functions
from .self_noise import (
    calculate_self_noise,
    plot_combined_self_noise_db,
    report_self_noise,
)

from .dynamic_range import (
    load_dynamic_range_data,
    data_processing,
    analyze_dynamic_range_hilbert,
    analyze_dynamic_range_thd,
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

from .frequency_response import (
    load_frequency_response_data,
    phase_to_strain,
    extract_local_signal,
    compute_frequency_response,
    analyze_frequency_response,
)

__all__ = [
    # submodules
    "self_noise",
    "dynamic_range",
    "fidelity",
    "crosstalk",
    "frequency_response",

    # self_noise
    "calculate_self_noise",
    "plot_combined_self_noise_db",
    "report_self_noise",

    # dynamic_range
    "load_dynamic_range_data",
    "data_processing",
    "analyze_dynamic_range_hilbert",
    "analyze_dynamic_range_thd",

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

    # frequency_response
    "load_frequency_response_data",
    "phase_to_strain",
    "extract_local_signal",
    "compute_frequency_response",
    "analyze_frequency_response",
]
