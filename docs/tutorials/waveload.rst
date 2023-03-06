Simulations with Wave Loads
===========================

One of the main features of the ``MCSimPython`` library is to simulate vessels with wave loads. Using the
``MCSimPython.waves`` packages, we can generate sea states by from wave spectras, and calculate the loads
on the vessel. Here, laods refer to the first- and second-order wave loads. 

We will in this notebook show how to create a sea state, and use it to simulate a vessel in waves.

First, we will generate a sea state with 100 wave components. We will in this example use the ``JONSWAP`` wave spectra.

.. code-block:: python

    from MCSimPython.waves import JONSWAP
    import numpy as np

    hs = 2.5    # Signifcant wave height
    tp = 9.0    # Peak wave period
    gamma = 3.3 # Peakedness factor of wave spectra
    wp = 2*np.pi/tp

    wmin = 0.5*wp
    wmax = 3.0*wp

    N = 100     # Number of wave components

    wave_freqs = np.linspace(wmin, wmax, N) 

    jonswap = JONSWAP(wave_freqs)

    _, wave_spectra = jonswap(hs=hs, tp=tp, gamma=gamma)


Having defined the wave spectra, we can use compute the wave amplitudes using the simple relation:

.. math::
    \zeta_a = \sqrt{2S(\omega)d\omega}


