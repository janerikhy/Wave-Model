"""
Guidance (:mod:`MCSimPython.guidance`)
======================================

Guidance/reference models for DP systems.

Contents
--------

 - filter.py: Third-order reference filter
 - path_param.py: Hybrid way-point path parameterization

Description
-----------

The two different reference models are based on different
principles:

The third-order reference filter is a cascade of a low-pass
filter and a second-order damped harmonic oscillator.

The module `path_param` uses path parameterization to generate
a desired state used. 

Examples
--------

>>> from MCSimPython.guidance import WayPointsRefModel

Importing the reference filter

>>> from MCSimPython.guidance import ThrdOrderRefFilter

The models can also be imported using the full path

>>> from MCSimPython.guidance.filter import ThrdOrderRefFilter
>>> from MCSimPython.guidance.path_param import WayPointsRefModel
"""

from MCSimPython.guidance.path_param import WayPointsRefModel
from MCSimPython.guidance.filter import ThrdOrderRefFilter
