# gunnerus.py
from simulator.vessel import Vessel
from utils import J, Rz, Smat, pipi

import numpy as np
import pickle


import os

GUNNERUS_DATA = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        'vessel_data',
        'gunnerus'
    )
)


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
        super().__init__(dt, config_file=config_file, dof=GunnerusManeuvering3DoF.DOF)
        with open(self._config, 'rb') as f:
            self.data = pickle.load(f)
        self._dt = dt
        self._dof = 3
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


    def Cor3(self, nu, M):
        return np.array([
            [0, 0, -M[1, 1]*nu[1] - .5*(M[1, 2] + M[2, 1])*nu[2]],
            [0, 0, M[0, 0]*nu[0]],
            [M[1, 1]*nu[1] + .5*(M[1, 2] + M[2, 1])*nu[2], -M[0, 0]*nu[0], 0]
        ])
