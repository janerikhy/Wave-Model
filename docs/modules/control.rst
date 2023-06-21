Control
=======

The ``MCSimPython.control`` package contains a set of DP and Maneuvering controllers. 

Basic Controllers
-----------------

The module contains simple PD and PID controllers. 

PD Controller
^^^^^^^^^^^^^

.. autoclass:: MCSimPython.control.basic.PD
    :members:


PID Controller
^^^^^^^^^^^^^^

.. autoclass:: MCSimPython.control.basic.PID
    :members:

PI Controller
^^^^^^^^^^^^^

.. autoclass:: MCSimPython.control.basic.PI
    :members:


Direct Bias Compensation Controller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MCSimPython.control.basic.DirectBiasCompensationController
    :members:


Adaptive Controller
-------------------

.. autoclass:: MCSimPython.control.adaptiveFS.AdaptiveFSController
    :members:

Backstepping Controllers
------------------------

In the ``MCSimPython.control.backstepping`` module there are controllers based on backstepping
theory. Currently, only a backstepping controller for maneuvering purposes has been implemented.

.. autoclass:: MCSimPython.control.backstepping.BacksteppingController
    :members:
    :no-undoc-members: