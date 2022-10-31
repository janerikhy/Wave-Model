# vessel.py
import numpy as np
from abc import ABC, abstractclassmethod
import os
import pickle

from utils import Rx, Ry, Rz, Rzyx, Smat

BASE_DIR = os.path.join(os.getcwd(), os.pardir)
GUNNERUS_DATA = os.path.join(BASE_DIR, 'vessel_data/gunnerus')
CSAD_DATA = os.path.join(BASE_DIR, 'vessel_data/CSAD')


class Vessel(ABC):
    """Base class for simulator vessels.

    """

    def __init__(self, config_file, *args, dof=3, **kwargs):
        self._config_file = config_file
        self._dof = dof
        self._M = np.zeros((dof, dof))
        self._D = np.zeros_like(self._M)
        self._G = np.zeros_like(self._D)
        self._eta = np.zeros(dof)
        self._nu = np.zeros(dof)

    def __call__(self, *args, **kwargs):
        pass

    def get_eta(self):
        return self._eta

    def get_nu(self):
        return self._nu

    @abstractclassmethod
    def x_dot(self, *args, **kwargs):
        raise NotImplementedError



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


class GunnerusManeuvering3DoF(Vessel):
    """
    3DOF Manuevering model for R/V Gunnerus. The model is based on maneuvering theory.
    Zero-Frequency model.    

    References
    ----------
    Fossen 20--. Handbook of marine craft hydrodynamics and motion control
    """
    DOF = 3

    def __init__(self, *args, dt=0.1, config_file="parV_RVG3DOF.pkl", **kwargs):
        self._config = os.path.join(GUNNERUS_DATA, config_file)
        with open(self._config, 'rb') as f:
            self.data = pickle.load(f)
        self._dt = dt
        self._Mrb = self.data['Mrb']
        self._Ma = self.data['Ma']
        self._Minv = np.linalg.inv(self._Mrb + self._Ma)
        self._D = np.zeros((3, 3))
        self._Dl = self.data['Dl']
        self._Du = self.data['Du']
        self._Dv = self.data['Dv']
        self._Dr = self.data['Dr']
        self._ref_vel = self.data['reference_velocity']
        self._eta = np.zeros(3)
        self._nu = np.zeros(3)
        self._x = np.zeros(6)

    def x_dot(self, U_c, beta_c, tau):
        nu_c_n = U_c * np.array([np.cos(beta_c), np.sin(beta_c), 0])    # Current in NED-frame
        nu_c = Rz(self._eta[2]).T@nu_c_n                                # Current in body-frame
        S = Smat([0, 0, self._nu[2]])
        dnu_c = (S@Rz(self._eta[2])).T@nu_c_n
        nu_r = self._nu - nu_c

        self._D = self._Dl + self._Du*np.abs(nu_r[0]) + self._Dv*np.abs(nu_r[1]) + self._Dr*np.abs(self._nu[2])
        self._Ca = self.Cor3(nu_r, self._Ma)
        self._Crb = self.Cor3(self._nu, self._Mrb)

        eta_dot = Rz(self._eta[2])@self._nu
        nu_dot = self._Minv@(tau - self._D@nu_r - self._Crb@self._nu - self._Ca@nu_r + self._Ma@dnu_c)
        self._x_dot = np.concatenate([eta_dot, nu_dot])
        return eta_dot, nu_dot

    def integrate(self, *args, **kwargs):
        self._x = self._x + self._dt * self._x_dot
        self._eta = self._x[:3]
        self._nu = self._x[3:]


    def Cor3(self, nu, M):
        return np.array([
            [0, 0, -M[1, 1]*nu[1] - .5*(M[1, 2] + M[2, 1])*nu[2]],
            [0, 0, M[0, 0]*nu[0]],
            [M[1, 1]*nu[1] + .5*(M[1, 2] + M[2, 1])*nu[2], -M[0, 0]*nu[0], 0]
        ])







class GunnerusManeuvering6DoF(Vessel):
    """
    6DOF Maneuvering model for R/V Gunnerus. The model is based on maneuvering theory.
    Zero-Frequency model is assumed for surge, sway and yaw. 
    Natural Frequency models for heave, roll and pitch. 
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    


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