# cathode/__init__.py
"""
GalvCalc.cathode
Cathodic hydrogen evolution module for corrosion studies
"""

from .surfaces import SurfaceProperties
from .hydrogen import AdsorptionManager
from .estimate_ic0 import facet_dependant_property, ic0, ic0_mg