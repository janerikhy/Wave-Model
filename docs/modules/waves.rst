=====
Waves
=====

Wave spectra and wave loads functionality can be found in the ``MCSimPython.waves`` subpackage.

Wave Spectra
------------

The ``MCSimPython.waves.wave_spectra`` module containts basic wave spectra functionality.

.. autoclass:: MCSimPython.waves.wave_spectra.BaseSpectrum
    :members:
    :special-members: __init__


Modified Pierson Moskowitz
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MCSimPython.waves.wave_spectra.ModifiedPiersonMoskowitz
    :members:
    :special-members: __call__

JONSWAP
^^^^^^^
.. autoclass:: MCSimPython.waves.wave_spectra.JONSWAP
    :special-members: __call__



Wave Loads
----------

.. autoclass:: MCSimPython.waves.wave_loads.WaveLoad
    :members: first_order_loads, second_order_loads, QTF_METHODS
    :private-members: _set_force_raos, _full_qtf_6dof
    :special-members: __call__, __init__
