"""Command-line interface for GalvCalc.

Usage::

    galvcalc --version
    galvcalc --modules
"""

from __future__ import annotations

import argparse

import GalvCalc


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``galvcalc`` console script."""
    parser = argparse.ArgumentParser(
        prog="galvcalc",
        description=(
            "GalvCalc - a computational framework for modeling micro-galvanic "
            "corrosion of alloys."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"GalvCalc {GalvCalc.__version__}",
    )
    parser.add_argument(
        "--modules",
        action="store_true",
        help="list the available submodules of GalvCalc",
    )
    args = parser.parse_args(argv)

    if args.modules:
        modules = [
            "GalvCalc.core          - bulk/surface structures and equilibrium potentials",
            "GalvCalc.cathode       - surface properties, hydrogen adsorption, i_c0 estimation",
            "GalvCalc.anode         - surface doping and anodic dissolution descriptors",
            "GalvCalc.polarization  - polarization curves and anode/cathode area-ratio analysis",
            "GalvCalc.predictor     - ML predictors (CGCNN, TabPFN) for surface/adsorption properties",
        ]
        print("Available GalvCalc modules:")
        for module in modules:
            print(f"  {module}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
