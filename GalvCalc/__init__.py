"""
GalvCalc
A computational framework for modeling micro-galvanic corrosion of alloys with
coupled anodic dissolution and hydrogen evolution kinetics.

Subpackages are imported lazily to keep the top-level import lightweight:

- ``GalvCalc.core``: bulk/surface structures and equilibrium-potential engine
- ``GalvCalc.cathode``: surface properties, hydrogen adsorption and i_c0 estimation
- ``GalvCalc.anode``: surface doping and anodic dissolution descriptors
- ``GalvCalc.polarization``: polarization curves, area-ratio and alloy-content analysis
- ``GalvCalc.predictor``: machine-learning predictors (CGCNN / TabPFN)

Example
-------
>>> from GalvCalc.core.structures import Bulk
>>> bulk = Bulk.from_file("POSCAR")
"""

import sys

# The package prints unit symbols such as "A/cm^2" and "eV/A^2" in its
# console output.  On Windows the legacy GBK code page cannot encode them,
# so we switch the standard streams to UTF-8 when possible.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass
del _stream

__version__ = "1.0.0"
__author__ = "Gaoning Shi"
__email__ = "gaoning_shi@sjtu.edu.cn"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
