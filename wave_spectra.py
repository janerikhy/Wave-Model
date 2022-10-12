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


    def realization(self, time, *args, **kwargs):
        """
        Generate a wave realization from wave spectrum at a fixed position.

        Parameters
        ----------
        time : 1D-array
            Array of time points used in realization
        *args : tuple
            Additional arguments should be passed as keyword arguments
        **kwargs : dict
            Wave spectrum parameters like hs and tp, include gamma for JONSWAP wave spectrum.
        
        Return
        ------
        timeseries : 1D-array
            Array of wave elevation at different time instances for a fixed point (x=0).
        """
        
        freq, spectrum = self.__call__(*args, **kwargs)
        dw = freq[1]-freq[0]
        amp = np.sqrt(2*spectrum*dw)
        eps = np.random.uniform(0, 2*np.pi, size=len(amp))
        return np.sum(amp * np.cos(freq*time[:, None] + eps), axis=1)


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
        """
        Generate a Modified Pierson-Moskowitz wave spectrum.

        Parameters
        ----------
        hs : float
            Significant wave height
        tp : float
            Peak period of wave spectrum
        freq_hz : bool
            Wave spectrum and frequencies in Hz or rad/s.

        Return
        ------
        freq : 1D-array
            Array of frequencies in wave spectrum
        spectrum : 1D-array
            1D Modified Pierson-Moskowitz wave spectrum
        """

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
        """
        Generate a JONSWAP wave spectrum.

        Parameters
        ----------
        hs : float
            Significant wave heihgt
        tp : float
            Peak period of wave spectrum
        gamma : float
            Steapness factor
        freq_hz : bool, default=False
            Frequency in Hz or rad/s

        Return
        ------
        freq : 1D-array
            Frequencies of wave spectrum
        spectrum : 1D-array
            1D JONSWAP wave spectrum
        """

        freq, pm_spectrum = super().__call__(hs, tp, freq_hz=freq_hz)

        alpha = self._alpha(gamma)
        b = self._b(tp)
        
        return freq, alpha * pm_spectrum * gamma**b

    def _alpha(self, gamma):
        return 1 - 0.287 * np.log(gamma)

    def _b(self, tp):
        wp = 2*np.pi / tp
        sigma = self._sigma(wp)
        return np.exp(-0.5 * ((self._freq - wp) / (sigma*wp))**2)

    def _sigma(self, wp):
        # Set conditional parameter sigma used in JONSWAP spectrum.
        arg = self._freq <= wp
        sigma = np.empty_like(self._freq)
        sigma[arg] = 0.07
        sigma[~arg] = 0.09
        return sigma


if __name__ == "__main__":
    # Simple How-To for wave module
    import doctest
    doctest.testmod()
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 14,
        'lines.linewidth': 1.4
    })

    hs = 2.5
    tp = 9.0
    wp = 2*np.pi / tp
    gamma = 2

    wmin = wp - wp/2
    wmax = 3*wp
    N = 1000
    w = np.linspace(wmin, wmax, N)

    pm_spectrum = ModifiedPiersonMoskowitz(w)       # Instantiate MPM spectrum object
    jonswap_spectrum = JONSWAP(w)                   # Instantiate Jonwswap spectrum object

    # Calculate the zero-moment for both spectra.
    m0_jonswap = jonswap_spectrum.moment(n=0, hs=hs, tp=tp, gamma=gamma)
    m0_pm = pm_spectrum.moment(n=0, hs=hs, tp=tp)

    print(f"JONSWAP: Hs = {4*np.sqrt(m0_jonswap):.2f} [m]")
    print(f"PM: Hs = {4*np.sqrt(m0_pm):.2f} [m]")
    
    plt.plot(*pm_spectrum(hs, tp), label="PM")
    plt.plot(*jonswap_spectrum(hs, tp, gamma), label="JONSWAP")
    plt.grid()
    plt.legend()
    plt.show()

    time = np.arange(0, 500, 0.1)

    jonswap_realizatin = jonswap_spectrum.realization(time, hs=hs, tp=tp, gamma=gamma)
    pm_realizatino = pm_spectrum.realization(time, hs=hs, tp=tp)

    plt.plot(time, pm_spectrum.realization(time, hs=hs, tp=tp))
    plt.plot(time, jonswap_spectrum.realization(time, hs=hs, tp=tp, gamma=gamma))
    plt.xlim(time[0], time[-1])
    plt.ylim(-6*np.sqrt(m0_jonswap), 6*np.sqrt(m0_jonswap))
    plt.show()