# backstepping.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2023-02-01
# Revised: 
# Tested:
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from MCSimPython.utils import Rz, pipi

def Smat(r):
    return np.array([
        [0., -r, 0.],
        [r, 0., 0.],
        [0., 0., 0.]
    ])



class BacksteppingController:
    """Backsteppipng controller for maneuvering.
    
    The controller should be used together with 
    a path parameterization guidance/reference model.

    See Also
    --------
    MCSimPython.guidance.path_parm.WayPointPathParam :
        Way-points path parameterization.
    """

    def __init__(self, M, D, K1, K2):
        self.M = M
        self.D = D
        self.K1 = K1
        self.K2 = K2

    def lyapunov_grad_s(self, z1, eta, eta_d_s):
        """Lyapunov function gradient."""
        return -z1@Rz(eta[2]).T@eta_d_s

    def error_pos(self, eta, eta_d):
        """Deviation from desired pose in body-frame."""
        return Rz(eta[2]).T@(eta - eta_d)

    def error_vel(self, nu, alpha):
        """Deviation from virtual control law."""
        return nu - alpha

    def virtual_control_law(self, z1, eta, eta_d_s, u_s):
        """Virtual control law (alpha)."""
        return -self.K1@z1 + Rz(eta[2]).T@eta_d_s*u_s

    def virtual_control_dot(self, z1, z2, alpha, eta, eta_d, eta_d_s, eta_d_s2, u_s, u_s_dot, nu, s_dot, ddt_u_s):
        """Time derivative of virtual control law (alpha_dot)."""
        line1 = self.K1@Smat(nu[2])@z1 - self.K1@nu - \
            Smat(nu[2])@Rz(eta[2]).T@eta_d_s*u_s
        line2 = Rz(eta[2]).T@eta_d_s*ddt_u_s
        line3 = (self.K1@Rz(eta[2]).T@eta_d_s + Rz(eta[2]).T @
                 eta_d_s2*u_s + Rz(eta[2]).T@eta_d_s*u_s_dot)*s_dot
        return line1 + line2 + line3

    def control_law(self, z1, z2, alpha, alpha_dot, nu):
        """Complete control law."""
        return self.D@alpha + self.M@alpha_dot - z1 - self.K2@z2

    def unit_gradient_update_law(self, u_s, mu, eta_d_s, lyapunov_gradient):
        """Unit-tangent gradient update law for path parameter (s)."""
        return u_s - mu/np.linalg.norm(eta_d_s)*lyapunov_gradient

    def u(self, eta, nu, eta_d, eta_d_s, eta_d_s2, mu, u_s, u_s_dot, ddt_u_s):
        """Control command."""
        z1 = self.error_pos(eta, eta_d)
        z1[2] = np.mod(z1[2] + np.pi, 2*np.pi) - np.pi
        a = self.virtual_control_law(z1, eta, eta_d_s, u_s)
        z2 = self.error_vel(nu, a)
        v_d_s = self.lyapunov_grad_s(z1, eta, eta_d_s)
        s_dot = self.unit_gradient_update_law(u_s, mu, eta_d_s, v_d_s)
        #s_dot = u_s
        self._s_dot = s_dot
        a_dot = self.virtual_control_dot(
            z1, z2, a, eta, eta_d, eta_d_s, eta_d_s2, u_s, u_s_dot, nu, s_dot, ddt_u_s)

        return self.control_law(z1, z2, a, a_dot, nu)