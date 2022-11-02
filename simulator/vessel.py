# vessel.py
import numpy as np
from abc import ABC, abstractclassmethod
import os
import pickle
import json

from utils import Rz, Smat, J, timeit, pipi

BASE_DIR = os.path.join(os.getcwd(), os.pardir)
GUNNERUS_DATA = os.path.join(BASE_DIR, 'vessel_data/gunnerus')
CSAD_DATA = os.path.join(BASE_DIR, 'vessel_data/CSAD')


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
        print(self._x[:self._dof])
        self._eta = self._x[:self._dof]
        self._nu = self._x[self._dof:]


class GunnerusDP3Dof(Vessel):
    """
    3DOF DP Model for R/V Gunnerus. The model is valid for stationkeeping and low-speed
    maneuvering up to approx. 2 m/s. 

    eta_dot = R(psi)*nu
    M*nu_dot + C_rb(nu) + N(nu_r)nu_r = tau + tau_wind + tau_wave

    where N(nu_r)nu_r := C_A(nu_r)nu_r + D(nu_r)nu_r

    states: 
        nu = [u, v, r]^T
        eta = [N, E, psi]^T

    M = np.array([
        [m - X_udot, 0, 0],
        [0, m - Y_vdot, m*x_g - Y_rdot],
        [0, m*x_g - Y_rdot, Iz - N_rdot]
    ]),

    C_rb(nu) = np.array([
        [0, 0, -m(x_g*r + v)],
        [0, 0, m*u],
        [m(x_g*r + v), -m*u, 0]
    ])

    """
    # CONFIG_FILE = "parV_RVG3DOF.pkl"

    # def __init__(self, *args, **kwargs):
    #     with open(CONFIG_FILE, 'rb') as f:
    #         self.data = pickle.read(f)
    #     self._Mrb = 
    

# What is general for all of the vessels?
# 
# Each vessel simulator model has a specific DOF
#
# They have a set of matrices:
# - mass matrix
# - added mass matrix
# - coriolis and centripetal matrix
# - damping matrix
# - nonlinear damping matrix
# - Restoring force matrix
# 
# 
# Each model should use a specific time step (h or dt)
# 
# Each model has the same kinematic equation
# 
# Each model has a specific self defined kinetic equation (abstract class method witch must be overwritten)
# 
# Each model has a state vector (we will only have eta and nu I guess)