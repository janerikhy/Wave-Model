"""
Observer (:mod:`MCSimPython.observer`)
=======================================

The package consist of different observers
for state estimation.

Contents
--------

 - ekf.py: Extended Kalman Filter
 - nonlinobs.py: Nonlinear observers

Description
-----------

The different observers are implemented as 
classes, and are structured in different modules as
listed in `contents`.

Examples
--------

>>> from MCSimPython.observer import NonlinObs3dof

Alternative import

>>> from MCSimPython.observer.nonlinobs import NonlinObs3dof

Third option

>>> import MCSimPython.observer as obs
>>> estimator = obs.NonlinObs3dof(*args, **kwargs)
"""

from MCSimPython.observer.nonlinobs import NonlinObs3dof
from MCSimPython.observer.ekf import EKF