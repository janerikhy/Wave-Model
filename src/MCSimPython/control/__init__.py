"""
Control (:mod:`MCSimPython.control`)
====================================

Controller package for DP and maneuvering
control.

Contents
--------

 - backstepping.py: Controllers based on backstepping theory.
 - basic.py: Basic PD and PID controller.

Examples
--------

>>> from MCSimPython.control import PD, PID
"""

from MCSimPython.control.backstepping import BacksteppingController
from MCSimPython.control.basic import PD, PID