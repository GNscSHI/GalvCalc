"""Package-level smoke tests: imports, metadata, CLI."""

import GalvCalc


def test_version():
    assert GalvCalc.__version__ == "1.0.0"


def test_metadata():
    assert GalvCalc.__author__
    assert "@" in GalvCalc.__email__


def test_top_level_import_is_lightweight():
    """import GalvCalc must not pull in the heavy ML stack."""
    import sys

    assert "torch" not in sys.modules
    assert "tabpfn" not in sys.modules


def test_subpackages_importable():
    from GalvCalc.core.structures import Bulk, Surface  # noqa: F401
    from GalvCalc.cathode import (  # noqa: F401
        AdsorptionManager,
        SurfaceProperties,
        facet_dependant_property,
        ic0,
        ic0_mg,
    )
    from GalvCalc.anode import DopingSite, SurfaceDopingManager  # noqa: F401
    from GalvCalc.polarization import (  # noqa: F401
        Composition,
        ElectrodeParameters,
        PolarizationCurvePlotter,
        corrosion_vs_anode_ratio,
        create_mg_based_compositions,
        plot_comparison_polarization,
        plot_single_polarization,
        scan_corrosion_vs_content,
    )
    from GalvCalc.polarization.area_ratio import (  # noqa: F401
        AreaRatioAnalyzer,
        AreaRatioParameters,
        analyze_optimal_area_ratio,
        create_example_parameters,
    )


def test_cli_modules(capsys):
    from GalvCalc.cli import main

    rc = main(["--modules"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "GalvCalc.core" in captured.out
