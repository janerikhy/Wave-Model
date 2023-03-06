# gunnerus.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2022-11-02
# Revised: 2023-02-13 Jan-Erik Hygen    Add 6DOF DP model for RVG
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

from MCSimPython.simulator.vessel import Vessel
from MCSimPython.utils import J, Rz, Smat, pipi

import numpy as np
import pickle
import json

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
    """3DOF Manuevering model for R/V Gunnerus. The model is based on maneuvering theory.
    Zero-Frequency model.    

    References
    ----------
    Fossen 20--. Handbook of marine craft hydrodynamics and motion control
    """
    DOF = 3

    def __init__(self, dt, *args, method="Euler", config_file="parV_RVG3DOF.pkl", **kwargs):
        self._config = os.path.join(GUNNERUS_DATA, config_file)
        super().__init__(dt, method=method, config_file=config_file, dof=GunnerusManeuvering3DoF.DOF)
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

    def x_dot(self, x, U_c, beta_c, tau):
        eta = x[:self._dof]
        nu = x[self._dof:]
        nu_c_n = U_c * np.array([np.cos(beta_c), np.sin(beta_c), 0])    # Current in NED-frame
        nu_c = Rz(eta[2]).T@nu_c_n                                # Current in body-frame
        S = Smat([0, 0, nu[2]])
        dnu_c = (S@Rz(eta[2])).T@nu_c_n
        nu_r = nu - nu_c

        self._D = self._Dl + self._Du*np.abs(nu_r[0]) + self._Dv*np.abs(nu_r[1]) + self._Dr*np.abs(nu[2])
        self._Ca = self.Cor3(nu_r, self._Ma)
        self._Crb = self.Cor3(nu, self._Mrb)

        eta_dot = Rz(eta[2])@self._nu
        nu_dot = self._Minv@(tau - self._D@nu_r - self._Crb@nu - self._Ca@nu_r + self._Ma@dnu_c)
        return np.concatenate([eta_dot, nu_dot])
        #return eta_dot, nu_dot


    def Cor3(self, nu, M):
        return np.array([
            [0, 0, -M[1, 1]*nu[1] - .5*(M[1, 2] + M[2, 1])*nu[2]],
            [0, 0, M[0, 0]*nu[0]],
            [M[1, 1]*nu[1] + .5*(M[1, 2] + M[2, 1])*nu[2], -M[0, 0]*nu[0], 0]
        ])


class RVG_DP_6DOF(Vessel):
    """6 Degree of Freedom simulation model of R/V Gunnerus.

    The model is created simply by using Veres data (ShipX).
    """

    DOF = 6

    def __init__(self, dt, method="Euler", config_file="vessel_2.json", dof=6):
        config_file = os.path.join(GUNNERUS_DATA, config_file)
        super().__init__(dt, config_file=config_file, method=method, dof=6)
        with open(config_file,'r') as f:
            data = json.load(f)
        
        # Mass
        self._Mrb = np.asarray(data['MRB'])
        self._Ma = np.asarray(data['A'])[:, :, 30, 0]
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)

        # Damping
        self._Dp = np.asarray(data['B'])[:, :, 30, 0]
        self._Dv = np.asarray(data['Bv'])
        self._D = self._Dp + self._Dv

        # Restoring coefficients
        self._G = np.asarray(data['C'])[:, :, 0, 0]

    def x_dot(self, x, Uc, betac, tau):
        """Kinematic and kinetic equations.
        
        Parameters
        ----------
        x : array_like
            State vector with dimensions 12x1
        Uc : float
            Current velocity in earth-fixed frame
        betac : float
            Current direction in earth-fixed frame [rad]
        tau : array_like
            External loads (e.g wind, thrusters, ice, etc). Must be a 6x1 vector.

        Returns
        -------
        x_dot : array_like
            The derivative of the state vector.
        """
        eta = x[:self._dof]
        nu = x[self._dof:]

        nu_cn = Uc*np.array([np.cos(betac), np.sin(betac), 0])
        # nu_cn = np.concatenate([nu_cn, np.zeros(4)])
        Jinv = np.linalg.inv(J(eta))
        nu_c = Rz(eta[-1]).T@nu_cn
        nu_c = np.insert(nu_c, [3, 3, 3], 0)
        nu_r = nu - nu_c
        # Calculate the time derivative of nu_c_b
        dnu_cb = -Smat([0., 0., nu[-1]])@Rz(eta[-1]).T@nu_cn
        dnu_cb = np.insert(dnu_cb, [2, 2, 2], 0)

        eta_dot = J(eta)@nu

        nu_dot = self._Minv@(tau - self._D@nu_r - self._G@eta + self._Ma@dnu_cb)
        return np.concatenate([eta_dot, nu_dot])
    
    def set_hydrod_parameters(self, freq):
        """Set the hydrodynamic added mass and damping for a given frequency.
        
        Parameters
        ----------
        freq : array_like
            Frequency in rad/s. Can either be a single frequency, or 
            multiple frequencies with dimension n = DOF. 

        Examples
        --------
        
        Set a hydrodynamic parameters for one frequency
        
        >>> dt = 0.01
        >>> model = CSAD_DP_6DOF(dt)
        >>> frequency = 2*np.pi
        >>> model.set_hydrod_parameters(frequency)

        Set frequency for individual components

        >>> freqs = [0., 0., 2*np.pi, 2*np.pi, 2*np.pi, 0.]
        >>> model.set_hydrod_parameters(freqs)
        """
        
        if type(freq) not in [list, np.ndarray]:
            freq = [freq]
        freq = np.asarray(freq)
        print(freq.shape)
        if (freq.shape[0] > 1) and (freq.shape[0] != self._dof):
            raise ValueError(f"Argument freq: {freq} must either be a float or have shape n = {self._dof}. \
                             freq.shape = {freq.shape} != {self._dof}.")
        with open(self._config_file, 'r') as f:
            param = json.load(f)

        freqs = np.asarray(param['freqs'])
        if freq.shape[0] == 1:
            freq_indx = np.argmin(np.abs(freqs - freq))
        else:
            freq_indx = np.argmin(np.abs(freqs - freq[:, None]), axis=1)
        all_dof = np.arange(6)
        self._Ma = np.asarray(param['A'])[:, all_dof, freq_indx, 0]
        self._Dp = np.asarray(param['B'])[:, all_dof, freq_indx, 0]
        self._M = self._Mrb + self._Ma
        self._Minv = np.linalg.inv(self._M)
        self._D = self._Dv + self._Dp