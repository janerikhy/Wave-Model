"""
Vessel Simulation Models (:mod:`MCSimPython.simulator`)
=======================================================

The MCSimPython simulator package contains a set of 
different simulation models.

Contents
---------
The following modules are found in the `MCSimPython.simulator`.

- vessel.py        --- Base class for vessel simulation models.

- csad.py          --- Simulation models for C/S Arctic Drillship.

- gunnerus.py      --- Simulation models for R/V Gunnerus.
"""

# Import all the models s.t we can access them from e.g. 
# from MCSimPython.simulator import CSAD_DP_6DOF
from MCSimPython.simulator.csad import CSAD_DP_6DOF, CSADMan3DOF, CSAD_DP_Seakeeping
from MCSimPython.simulator.gunnerus import GunnerusManeuvering3DoF, RVG_DP_6DOF