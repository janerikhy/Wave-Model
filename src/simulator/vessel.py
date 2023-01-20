# vessel.py
import numpy as np
from abc import ABC, abstractclassmethod

from utils import Rz, Smat, J, timeit, pipi


class Vessel(ABC):
    """Base class for simulator vessels.

    """

    def __init__(self, dt, config_file, *args, dof=6, **kwargs):
        self._config_file = config_file
        self._dof = dof
        self._dt = dt
        self._M = np.zeros((dof, dof))
        self._D = np.zeros_like(self._M)
        self._G = np.zeros_like(self._D)
        self._eta = np.zeros(dof)
        self._nu = np.zeros(dof)
        self._x = np.zeros(2*dof)
        self._x_dot = np.zeros(2*dof)

    def __call__(self, *args, **kwargs):
        pass

    def get_eta(self):
        """Get vessel pose eta.

        Returns
        -------
        self._eta : 6 x 1 array.        
        """
        return self._eta

    def get_nu(self):
        """Get vessel velocity nu.
        
        Returns
        -------
        self._nu : 6 x 1 array.
        """
        return self._nu

    def reset(self):
        """Reset state vector to zeroes."""
        self._x = np.zeros(2*self._dof)
        self._x_dot = np.zeros_like(self._x)
        self._eta = np.zeros(self._dof)
        self._nu = np.zeros(self._dof)

    @abstractclassmethod
    def x_dot(self, *args, **kwargs):
        """Kinematic and kinetic equation of vessel. The method must be overwritten
        by inherting vessel classes."""
        raise NotImplementedError

    def integrate(self):
        """Integrate the state vector one forward, using forward Euler integration."""
        self._x = self._x + self._dt * self._x_dot
        self._eta = self._x[:self._dof]
        self._eta[self._dof//6+2:] = pipi(self._eta[self._dof//6+2:])
        self._nu = self._x[self._dof:]

