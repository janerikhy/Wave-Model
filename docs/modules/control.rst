Control
=======

The ``MCSimPython.control`` package contains a set of DP and Maneuvering controllers. 

Basic Controllers
-----------------

The module contains simple PD and PID controllers. 

.. autoclass:: MCSimPython.control.basic.PD
    :members:

.. autoclass:: MCSimPython.control.basic.PID
    :members:


Backstepping Controllers
------------------------

In the ``MCSimPython.control.backstepping`` module there are controllers based on backstepping
theory. 

.. autoclass:: MCSimPython.control.backstepping.BacksteppingController
    :members:
    :no-undoc-members: