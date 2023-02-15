# nonlinobs.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2023-01-23
# Revised: 
# Tested:
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from MCSimPython.utils import Rz, pipi


class NonlinObs3dof:

    def __init__(self, dt, wc, wo, lambd, T, M, D, lambda_w=0.03):
        self._dt = dt
        lambda_w = lambda_w
        self.zeta = 1.0
        self.K1 = np.block([
            [-2*(self.zeta-lambd)*(wc/wo) * np.eye(3)],
            [2*wo*(self.zeta-lambd)*np.eye(3)]
        ])
        self.K2 = np.diag([wc, wc, wc])
        self.K4 = np.diag([1.0, 1.0, 0.1])
        self.K3 = .1*self.K4
        self.Aw = np.block([
            [np.zeros((3,3)), np.eye(3)],
            [-wo**2*np.eye(3), -2*lambd*wo*np.eye(3)],
        ])
        self.Tinv = np.linalg.inv(T*np.eye(3))
        self._x_hat = np.zeros((15))
        self._y_hat = np.zeros(3)
        self.M = M
        self.D = D
        self.Minv = np.linalg.inv(M)

    @property
    def xi(self):
        return self._x_hat[:6]

    @property
    def eta(self):
        return self._x_hat[6:9]
    
    @property
    def bias(self):
        return self._x_hat[9:12]

    @property
    def nu(self):
        return self._x_hat[12:]
    
    def dynamics(self, tau, y):
        y_tilde = y - self._y_hat
        # Ensure that the smallest angle is used
        y_tilde[2] = pipi(np.copy(y_tilde[2]))
        xi_hat_dot = self.Aw@self.xi + self.K1@y_tilde
        eta_hat_dot = Rz(y[2])@self.nu + self.K2@y_tilde
        b_hat_dot = -self.Tinv@self.bias + self.K3@y_tilde
        nu_hat_dot = self.Minv@(-self.D@self.nu + tau + Rz(y[2]).T@self.bias + Rz(y[2]).T@self.K4@y_tilde)
        return np.concatenate((xi_hat_dot, eta_hat_dot, b_hat_dot, nu_hat_dot))

    def update(self, y, tau):
        self._x_hat = self._x_hat + self._dt * self.dynamics(tau, y)
        self._y_hat = self.xi[3:] + self.eta