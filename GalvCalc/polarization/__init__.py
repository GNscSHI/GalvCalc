# polarization/__init__.py
"""
GalvCalc.polarization
Polarization analysis module for corrosion studies
"""

from .curve_plot import (
    ElectrodeParameters,
    Composition,
    PolarizationCurvePlotter,
    create_mg_based_compositions,
    mg_second_phase_example,
    plot_mg_second_phases,
    plot_single_polarization,
    plot_comparison_polarization
)

from .area_ratio import (
    AreaRatioParameters,
    AreaRatioAnalyzer,
    analyze_optimal_area_ratio,
    create_example_parameters
)

from .content_scan import (
    CorrosionScanResult,
    corrosion_vs_anode_ratio,
    max_corrosion_in_domain,
    scan_corrosion_vs_content,
    wt_to_vol_fraction,
    mg3nd_vol_fraction,
    mg3nd_kinetics,
)
