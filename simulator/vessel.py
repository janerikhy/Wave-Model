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
        return self._eta

    def get_nu(self):
        return self._nu

    def reset(self):
        self._x = np.zeros(2*self._dof)
        self._x_dot = np.zeros_like(self._x)
        self._eta = np.zeros(self._dof)
        self._nu = np.zeros(self._dof)

    @abstractclassmethod
    def x_dot(self, *args, **kwargs):
        raise NotImplementedError

    def integrate(self):
        self._x = self._x + self._dt * self._x_dot
        self._eta = self._x[:self._dof]
        self._eta[self._dof//6+2:] = pipi(self._eta[self._dof//6+2:])
        self._nu = self._x[self._dof:]

