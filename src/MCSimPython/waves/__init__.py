"""
Waves (:mod:`MCSimPython.waves`)
================================

Waves package for simulations.

Contents
--------

 - wave_loads.py: Wave load calculations
 - wave_spectra.py: Wave spectra module

Examples
--------

>>> from MCSimPython.waves import JONSWAP, WaveLoad
"""

from MCSimPython.waves.wave_loads import WaveLoad, FluidMemory
from MCSimPython.waves.wave_spectra import JONSWAP, ModifiedPiersonMoskowitz