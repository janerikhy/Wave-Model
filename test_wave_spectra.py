import numpy as np
import pytest
from wave_spectra import ModifiedPiersonMoskowitz, JONSWAP


class TestSpectra:

    def test_realization_pm(self):
        # Test statistical output from time realization of wave MPM spectra.
        hs = 3
        tp = 12
        wp = 2*np.pi / tp
        
        freqs = np.arange(wp - wp/2, 3*wp, 0.01)
        time = np.arange(0, 1000, .1)

        spectrum = ModifiedPiersonMoskowitz(freqs, False)
        timeseries = spectrum.realization(time, hs=hs, tp=tp)

        hs_estimated = 4*np.sqrt(np.average(timeseries**2))

        assert np.isclose(hs, hs_estimated, rtol=1e-1)


    def test_realization_jonswap(self):
        # Test that estimated hs of time realization is close to that of the specified hs
        hs = 3
        tp = 12
        wp = 2*np.pi / tp
        gamma = 3.3

        freqs = np.arange(wp - wp/2, 3*wp, 0.01)
        time = np.arange(0, 1000, .1)

        spectrum = JONSWAP(freqs)
        timeseries = spectrum.realization(time, hs=hs, tp=tp, gamma=gamma)

        hs_estimated = 4*np.sqrt(np.average(timeseries**2))

        assert np.isclose(hs, hs_estimated, rtol=1e-1)
 

    def test_equal_spectra(self):
        hs = 3
        tp = 12
        wp = 2*np.pi / tp
        gamma = 1

        freqs = np.arange(wp - wp/2, 3*wp, 0.01)

        jonswap_spectrum = JONSWAP(freqs)
        pm_spectrum = ModifiedPiersonMoskowitz(freqs)

        assert np.all(np.isclose(jonswap_spectrum(hs, tp, gamma), pm_spectrum(hs, tp), rtol=1e-1))
