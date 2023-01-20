# base_observer.py 
import numpy as np
from abc import ABC, abstractclassmethod

# Base class for observers

class BaseObserver(ABC):
    
    """Base Observer
    Base class for observers.
    Assumed observers are for horizontal plane.

    Attributes
    ----------

    self._x_hat : array_like
        Observer states
    self._eta_hat : array_like
        Observer pose estimate
    self._nu_hat : array_like
        Observer velocity estimate
    self._b_hat : array_like
        Observer bias estimate
    self._xi_hat : array_like
        Observer wave filter
    
    Methods
    -------

    self.integrate()
    self.get_states()
    self.get_eta()
    self.get_nu()
    self.get_b()
    self.get_xi()
    """
    def __init__(self, dt, dof=3):
        self._dof = dof
        self.dt = dt
        self._x_hat = np.zeros(6+3*dof)
        self._x_hat_dot = np.zeros_like(self._x_hat)
        self._eta_hat = np.zeros(dof)
        self._nu_hat = np.zeros(dof)
        self._xi_hat = np.zeros(6)
        self._b_hat = np.zeros(dof)
        self._L1 = np.zeros((6, dof))
        self._L2 = np.zeros((dof, dof))
        self._L3 = np.zeros((dof, dof))
        self._L4 = np.zeros((dof, dof))

    def integrate(self):
        self._x_hat = self._x_hat + self.dt * self._x_hat_dot
        self._xi_hat = self._x_hat[:6]
        self._eta_hat = self._x_hat[6:self._dof]
        self._b_hat = self._x_hat[6+self._dof:6+2*self._dof]
        self._nu_hat = self._x_hat[-self._dof:]

    
        

