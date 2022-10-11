# Wave spectra
import numpy as np
from abc import ABC, abstractclassmethod


class BaseSpectrum(ABC):
    """
    Base class for 1-D Wave Spectra.

    Parameters
    ----------
    freq : 1-D array
        Frequencies of wave spectrum
    freq_hz : bool
        Boolean to define spectrum in either rad/s (false) or Hz (true)
    """

    def __init__(self, freq, freq_hz=False):
        self._freq = freq
        self._freq_hz = freq_hz

        if self._freq_hz:
            self._freq *= 2*np.pi
        

    def __call__(self, *args, freq_hz=None, **kwargs):
        freq = self._freq.copy()
        spectrum = self._spectrum(freq, *args, **kwargs)

        return freq, spectrum

    def moment(self, n, *args, **kwargs):
        freq, spec = self.__call__(*args, **kwargs)
        return np.trapz(freq**n * spec, freq)


    @abstractclassmethod
    def _spectrum(self, omega, *args, **kwargs):
        raise NotImplementedError


class BasePMSpectrum(BaseSpectrum):

    def __call__(self, A, B, freq_hz=None):
        return super().__call__(A, B, freq_hz=freq_hz)
    
    def _spectrum(self, omega, A, B):
        return A/omega**5 * np.exp(-B/omega**4)


class ModifiedPiersonMoskowitz(BasePMSpectrum):

    def __call__(self, hs, tp, freq_hz=None):
        A = self._A(hs, tp)
        B = self._B(tp)

        return super().__call__(A, B, freq_hz=freq_hz)

    def _A(self, hs, tp):
        wp = 2*np.pi / tp
        return (5.0/16.0) * hs**2 * wp**4

    def _B(self, tp):
        wp = 2*np.pi / tp
        return (5.0/4.0) * wp**4


class JONSWAP(ModifiedPiersonMoskowitz):

    def __call__(self, hs, tp, gamma=1, freq_hz=None):
        freq, pm_spectrum = super().__call__(hs, tp, freq_hz=freq_hz)

        alpha = self._alpha(gamma)
        b = self._b(tp)
        
        return freq, alpha * pm_spectrum * gamma**b

    def _alpha(self, gamma):
        return 1 - 0.287 * np.log(gamma)

    def _b(self, tp):
        wp = 2*np.pi / tp
        sigma = self._sigma(wp)
        return np.exp(-0.5 * (self._freq - wp)**2 / (sigma*wp))

    def _sigma(self, wp):
        arg = self._freq <= wp
        sigma = np.empty_like(self._freq)
        sigma[arg] = 0.07
        sigma[~arg] = 0.09
        return sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 14,
        'lines.linewidth': 1.4
    })

    hs = 2.5
    tp = 9.0
    wp = 2*np.pi / tp
    gamma = 3.3

    wmin = wp - wp/2
    wmax = 3*wp
    N = 100
    w = np.linspace(wmin, wmax, N)

    pm_spectrum = ModifiedPiersonMoskowitz(w)
    jonswap_spectrum = JONSWAP(w)

    print(4*np.sqrt(jonswap_spectrum.moment(0, hs=hs, tp=tp)))
    
    plt.plot(*pm_spectrum(hs, tp), label="PM")
    plt.plot(*jonswap_spectrum(hs, tp, gamma), label="JONSWAP")
    plt.grid()
    plt.legend()
    plt.show()
