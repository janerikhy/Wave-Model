# basic.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2023-01-30
# Revised: 2023-02 - Added Direct Bias Compensation Controller - Harald Mo
# Tested:
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from MCSimPython.utils import Rz, pipi


class PD:
    """Proportional-Derivative controller."""
    def __init__(self, kp: list, kd: list):
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)
        self.tau_cmd = np.zeros(3)

    def get_tau(self, eta, eta_d, nu, nu_d):
        """Calculate control loads.
        
        Parameters
        ----------
        eta : array_like
            Vessel pose in surge, sway and yaw.
        eta_d : array_like
            Desired vessel pose in surge, sway and yaw (NED-frame).
        nu : array_like
            Vessel velocity in surge, sway, and yaw (body-frame).
        nu_d : array_like
            Desired vessel velocity in surge, sway and yaw (body-frame).
        
        Returns
        -------
        tau : array_like
            Controller load in surge, sway, and yaw (body-frame).
        """
        psi = eta[-1]
        z1 = Rz(psi).T@(eta-eta_d)
        z1[2] = pipi(eta[2] - eta_d[2])
        z2 = nu - nu_d
        return -self.Kp@z1 - self.Kd@z2
    
    def set_kd(self, kd: list):
        """Set the derivative gain coefficients."""
        self.Kd = np.diag(kd)
    
    def set_kp(self, kp: list):
        """Set the proportional gain coefficients."""
        self.Kp = np.diag(kp)

class PID:
    """Proportional-Derivative control with integral action."""
    
    def __init__(self, kp: list, kd: list, ki: list, dt: float = 0.01):
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)
        self.Ki = np.diag(ki)
        self.zi = np.zeros(3)
        self.dt = dt

    def get_tau(self, eta, eta_d, nu, nu_d):
        """Calculate control loads.
        
        Parameters
        ----------
        eta : array_like
            Vessel pose in surge, sway and yaw.
        eta_d : array_like
            Desired vessel pose in surge, sway and yaw (NED-frame).
        nu : array_like
            Vessel velocity in surge, sway, and yaw (body-frame).
        nu_d : array_like
            Desired vessel velocity in surge, sway and yaw (body-frame).
        
        Returns
        -------
        tau : array_like
            Controller load in surge, sway, and yaw (body-frame).
        """
        psi = eta[-1]
        z1 = Rz(psi).T@(eta - eta_d)
        z1[2] = pipi(eta[2] - eta_d[2])
        z2 = nu - nu_d

        self.zi[:2] += self.dt*(eta[:2] - eta_d[:2])
        self.zi[2] += self.dt*pipi(eta[2] - eta_d[2])
        return -self.Kp@z1 - self.Kd@z2 - Rz(psi).T@self.Ki@self.zi



class DirectBiasCompensationController():
    '''
    Bias estimate provided from the observer as direct compensation in a nominal PD control law.
    '''
    def __init__(self, kp: list, kd: list):
        
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)

    def get_tau(self, eta, eta_d, nu, nu_d, b):
        '''
        Parameters
        ----------
        eta : array_like
            Vessel pose in surge, sway and yaw.
        eta_d : array_like
            Desired vessel pose in surge, sway and yaw (NED-frame).
        nu : array_like
            Vessel velocity in surge, sway, and yaw (body-frame).
        nu_d : array_like
            Desired vessel velocity in surge, sway and yaw (body-frame).
        b : array_like
            Estimated bias in surge, sway and yaw (body-frame)
        '''
        psi = eta[-1]
        z1 = Rz(psi).T@(eta-eta_d)              # P
        z1[2] = pipi(eta[2] - eta_d[2])
        z2 = nu - nu_d                          # D
        zb = Rz(psi).T@b                        # bias
        return -self.Kp@z1 - self.Kd@z2 - zb
    
    def set_kd(self, kd: list):
        self.Kd = np.diag(kd)
    
    def set_kp(self, kp: list):
        self.Kp = np.diag(kp)
