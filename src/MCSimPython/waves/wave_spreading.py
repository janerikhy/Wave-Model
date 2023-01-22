# Spreading functions for unidirectional and multidirectional wave fields
import numpy as np
from abc import ABC, abstractclassmethod

class BaseSpreading(ABC):

    """
    Base class for spreading functions for wave fields.

    Parameters
    ----------
    angles : 1D-array
        Angles for spreading function in radians or degrees
    degrees : bool (default = False)
        Unit of angles radians if false degrees if true

    """

    def __init__(self, angles, degrees=False):
        self._angles = angles
        self._deg = degrees

    def __call__(self, *args, degrees=None, **kwargs):
        return self._spreading(*args, **kwargs)

    @abstractclassmethod
    def _spreading(self, *args, **kwargs):
        raise NotImplementedError

    
class Unidirectional(BaseSpreading):

    def _spreading(self, *args, **kwargs):
        pass


class MultiDirectional(BaseSpreading):

    def __call__(self):
        return super().__call__()

    def _spreading(self):
        in_pipi2 = np.abs(self._angles) < np.pi/2
        f_beta = (2/np.pi) * np.cos(self._angles)**2
        f_beta[~in_pipi2] = 0
        return self._angles, f_beta

    

class MultiDirectionalAlt(BaseSpreading):

    def __call__(self, theta0, s=1, degrees=None):
        if s < 1:
            raise ValueError(f"s={s}, s must be larger than or equal to 1.")
        return super().__call__(theta0, s, degrees=degrees)

    def _spreading(self, theta0, s):
        d_theta = self._angles - theta0
        in_pipi_half = np.abs(d_theta) < np.pi/2
        f_theta = 2**(2*s - 1) * np.math.factorial(s)*np.math.factorial(s-1)/(np.pi*np.math.factorial(2*s-1)) * np.cos(d_theta)**(2*s)
        f_theta[~in_pipi_half] = 0
        return self._angles, f_theta


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'font.size': 14,
        'axes.grid': True,
        'lines.linewidth': 1.4
    })

    N = 100
    theta_min, theta_max = -np.pi, np.pi
    theta = np.linspace(theta_min, theta_max, N)

    spreading_spectra = MultiDirectional(theta)
    spreading_nonzero = MultiDirectionalAlt(theta)
    angles, spreading = spreading_spectra()
    _, spreading2 = spreading_nonzero(np.pi/4, s=0)

    plt.plot(angles, spreading, label="$f(\theta)$")
    plt.plot(angles, spreading2, label="$f_2(\theta)$")
    plt.xlim(angles[0], angles[-1])

    plt.show()