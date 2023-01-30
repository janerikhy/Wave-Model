# vessel.py
import numpy as np
from abc import ABC, abstractclassmethod

from MCSimPython.utils import pipi


class Vessel(ABC):
    """Base class for simulator vessels.

    """

    def __init__(self, dt, method, config_file, *args, dof=6, **kwargs):
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
        if method == "Euler":
            self.int_method = self.forward_euler
        elif method == "RK4":
            self.int_method = self.RK4
        else:
            raise ValueError(f"{method} is not a valid integation method. Only 'Euler' and 'RK4' are exepted mehtods.")

    def __call__(self, *args, **kwargs):
        pass

    def set_eta(self, eta):
        """Set the pose of the vessel."""
        if eta.shape != self._eta.shape:
            raise ValueError(f"{eta.shape} does not correspond to the DOF. DOF = {self._dof}")
        self._eta = eta
        self._x[:self._dof] = self._eta

    def set_nu(self, nu):
        if nu.shape != self._nu.shape:
            raise ValueError(f"{nu.shape} does not correspond to the DOF. DOF = {self._dof}")
        self._nu = nu
        self._x[self._dof:2*self._dof] = self._nu

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

    def get_x(self):
        """Get vessel state vector x.
        
        Returns
        -------
        self._x : DOF x 1 array.
        """
        return self._x

    def reset(self):
        """Reset state vector to zeroes."""
        self._x = np.zeros(2*self._dof)
        self._x_dot = np.zeros_like(self._x)
        self._eta = np.zeros(self._dof)
        self._nu = np.zeros(self._dof)

    @abstractclassmethod
    def x_dot(self, *args, **kwargs):
        """Kinematic and kinetic equation of vessel. The method must be overwritten
        by inherting vessel classes. The method should return the result of f(x, u, ..).
        It should not modify any of the object attributes.
        The first argument of the x_dot function should be the state vector: x.
        """
        raise NotImplementedError

    def integrate(self, *args, **kwargs):
        """Integrate the state vector one forward, using the specified integration method."""
        # self._x = self._x + self._dt * self._x_dot
        x = self.get_x()
        self._x = self.int_method(x, *args, **kwargs)  # Compute new state vector through integration
        self._eta = self._x[:self._dof] # Set eta
        self._eta[self._dof//6+2:] = pipi(self._eta[self._dof//6+2:])   # Keep radians in (-pi, pi)
        self._nu = self._x[self._dof:]  # Set nu
        self._x = np.concatenate([self._eta, self._nu])

    def forward_euler(self, x, *args, **kwargs):
        return x + self._dt * self.x_dot(x, *args, **kwargs)

    def RK4(self, x, *args, **kwargs):
        """Runge-Kutta 4 integration method."""
        k1 = self.x_dot(x, *args, **kwargs)
        k2 = self.x_dot(x + k1*self._dt/2, *args, **kwargs)
        k3 = self.x_dot(x + k2*self._dt/2, *args, **kwargs)
        k4 = self.x_dot(x + k3*self._dt, *args, **kwargs)
        
        return self._x + (k1 + 2*k2 + 2*k3 + k4)*self._dt/6
        