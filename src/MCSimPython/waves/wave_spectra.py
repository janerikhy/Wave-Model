# Wave spectra

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-10-11
# Revised: 2022-10-19 Jan-Erik Hygen    Add 2d spectrum functionality.
#          2022-10-27 Jan-Erik Hygen    Add spreading function.
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

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
        """Calculate n-th spectral moment."""
        freq, spec = self.__call__(*args, **kwargs)
        return np.trapz(freq**n * spec, freq)


    def realization(self, time, *args, **kwargs):
        """Generate a wave realization from wave spectrum at a fixed position.

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
        """Generate a Modified Pierson-Moskowitz wave spectrum.

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
        """Generate a JONSWAP wave spectrum.

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



class DirectionalSpectrum():
    
    """
    Directional Wave Spectrum class. 

    Attributes
    ----------
    _freq : 1D-array
        Array of frequencies for wave spectrum
    _angles : 1D-array
        Array of angles for spreading function of wave spectrum    
    _theta_p : float
        Peak angle for spreading function
    _spectrum : 1D-array
        Array of one-dimensional wave spectrum values for frequencies _freq.
    _s : int (default = 1)
        Peakness parameter of spreading functing. Must be large than or equal to 1.
    _spreading : 1D-array
        Spreading function for angles _angles.
    _dw : float
        Frequency steps
    _dtheta : float
        Angle step
    _g : float
        Gravitational acceleration constant
    _eps : 2D-array
        Array of random phases for wave component j, k. Where j <= len(freq) and k <= len(angles).
     
    Methods
    -------

    """

    def __init__(self, freqs, angles, theta_p, spectrum, s=1, seed=12345):
        self._freq = freqs
        self._angles = angles
        self._spectrum = spectrum
        self._theta_p = theta_p
        self._spreading = self._spread_func(s=s)
        self._dw = (freqs[-1] - freqs[0])/len(freqs)
        self._dtheta = (angles[-1] - angles[0])/len(angles)
        self._g = 9.81
        self._seed = seed
        np.random.seed(seed)
        self._eps = np.random.uniform(0, 2*np.pi, size=(len(freqs), len(angles)))

    
    def _spread_func(self, s=1):
        """
        Cosine spreading function.

        Parameters
        ----------
        theta : 1D-array
            Angles in spreading function.
        theta_p : float
            Peak angle for spreading function. (Dominant wave direction).
        s : int (default = 1)
            Steapness value to determine amount of spreading. Must be equal or large than 1.

        Return
        ------
        spreading : 1D-array
            Spreading function for the given angles, peak angles and steapness value.
        """
        d_theta = self._angles - self._theta_p
        state = np.abs(d_theta) < np.pi/2
        spreading = 2**(2*s - 1) * np.math.factorial(s)*np.math.factorial(s-1)/(np.pi*np.math.factorial(2*s-1)) * np.cos(d_theta)**(2*s)
        spreading[~state] = 0
        return spreading

    def spectrum2d(self):
        """
        Calculate the two-dimensional directional wave spectrum.

        Return
        ------
        F : 2D-array
            Frequencies in meshgrid
        T : 2D-array
            Theta angle in meshgrid
        spec2d : 2D-array
            Directional wave spectrum for frequencies and angles.
        """

        F, T = np.meshgrid(self._freq, self._angles)
        spec2d = self._spectrum*self._spreading[:, None]
        self._spectrum2d = spec2d
        return F, T, spec2d

    def wave_realization(self, time, x, y):
        """
        Generate a 2D Multidirectional wavefield

        Parameters
        ----------
        time : 1D-array
            Array of time instances
        x : 1D-array
            Array of x-positions
        y : 1D-array
            Array of y-positions
        
        Return
        ------
        X : 2D-array
            x values in meshgrid
        Y : 2D-array
            y values in meshgrid
        wave_elevation : 2D-array
            Wave elevation at position (x, y) at time t.
        """

        X, Y = np.meshgrid(x, y) # Create meshgrid
        wave_elevation = 0

        for j in range(len(self._freq)):
            k_j = self._freq[j]**2 / self._g
            for k in range(len(self._angles)):
                theta = self._angles[k]
                # eps_jk = np.random.uniform(0, 2*np.pi)
                amplitude = np.sqrt(2*self._spectrum2d[k, j] * self._dw * self._dtheta)
                wave_elevation += amplitude*np.sin(self._freq[j]*time - k_j*X*np.cos(theta) - k_j*Y*np.sin(theta) + self._eps[j, k])
                
        return X, Y, wave_elevation




if __name__ == "__main__":
    # Simple How-To for wave module
    import doctest
    doctest.testmod()
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 14,
        'lines.linewidth': 1.4
    })

    hs = 2.0
    tp = 9.5
    wp = 2*np.pi / tp
    gamma = 5

    wmin = wp - wp/2
    wmax = 2*wp
    N = 20
    w = np.linspace(wmin, wmax, N)

    pm_spectrum = ModifiedPiersonMoskowitz(w)       # Instantiate MPM spectrum object
    jonswap_spectrum = JONSWAP(w)                   # Instantiate Jonwswap spectrum object

    # Calculate the zero-moment for both spectra.
    m0_jonswap = jonswap_spectrum.moment(n=0, hs=hs, tp=tp, gamma=gamma)
    m0_pm = pm_spectrum.moment(n=0, hs=hs, tp=tp)

    print(f"JONSWAP: Hs = {4*np.sqrt(m0_jonswap):.2f} [m]")
    print(f"PM: Hs = {4*np.sqrt(m0_pm):.2f} [m]")
    
    # plt.plot(*pm_spectrum(hs, tp), label="PM")
    # plt.plot(*jonswap_spectrum(hs, tp, gamma), label="JONSWAP")
    # plt.grid()
    # plt.legend()
    # plt.show()

    freq, spectrum = jonswap_spectrum(hs, tp, gamma)

    time = np.arange(0, 500, 0.1)

    jonswap_realizatin = jonswap_spectrum.realization(time, hs=hs, tp=tp, gamma=gamma)
    pm_realizatino = pm_spectrum.realization(time, hs=hs, tp=tp)

    # plt.plot(time, pm_spectrum.realization(time, hs=hs, tp=tp))
    # plt.plot(time, jonswap_spectrum.realization(time, hs=hs, tp=tp, gamma=gamma))
    # plt.xlim(time[0], time[-1])
    # plt.ylim(-6*np.sqrt(m0_jonswap), 6*np.sqrt(m0_jonswap))
    # plt.show()

    theta0 = np.pi/36
    s=2
    psi = np.linspace(-np.pi, np.pi, 15)
    dir_spectra = DirectionalSpectrum(w, psi, theta_p=theta0, spectrum=spectrum, s=s)
    F, T, spectrum2d = dir_spectra.spectrum2d()

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection='3d')
    ax.view_init(35, 40)
    ax.plot_surface(F, T, spectrum2d, cmap=cm.coolwarm)
    ax.set_xlabel(r"$\omega \; [\frac{rad}{s}]$")
    ax.set_ylabel(r"$\psi \; [deg]$")
    plt.show()

    Nxy = 400
    x = np.linspace(-150, 150, Nxy)
    y = np.linspace(-150, 150, Nxy)
    X, Y, realization_2d = dir_spectra.wave_realization(0, x, y)

    def update3d(t, plot, x, y):
        # Callback function for animation.
        plot[0].remove()
        _, _, wave = dir_spectra.wave_realization(t, x, y)
        plot[0] = ax.plot_surface(X, Y, wave, cmap=cm.coolwarm)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(30, 40)
    # ax.plot_surface(X, Y, realization_2d, cmap=cm.coolwarm)
    ax.set_zlim([-2*hs, 2*hs])
    ax.set_xlabel("$x \, [m]$")
    ax.set_ylabel("$y \, [m]$")
    ax.set_zlabel("$\zeta \, [m]$")
    
    fps = 32

    plot = [ax.plot_surface(X, Y, realization_2d, cmap=cm.coolwarm)]

    from matplotlib.animation import FuncAnimation
    
    from datetime import datetime

    duration = 20
    t_start = datetime.now()
    w_anim = FuncAnimation(fig, update3d, frames=np.linspace(0, duration, (duration*fps)), fargs=(plot, x, y,))
    w_anim.save(f'wave_realization_disc{Nxy}_hs{hs:.2f}_tp{tp:.2f}_dir{np.rad2deg(theta0):.0f}_s{s:.0f}_{duration}_gamma{gamma:.1f}_sec.mp4', fps=fps, dpi=50)
    t_end = datetime.now()

    print(f"Execution time : {(t_end - t_start).total_seconds()} [s].")