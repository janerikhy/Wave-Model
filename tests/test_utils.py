import numpy as np
import pytest

import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

from utils import Rz, Ry, Rx, Rzyx, J


class TestKinematicUtils:

    def test_Rz(self):
        # Transformation from body frame to NED frame for rotation about z-axis
        psi = np.deg2rad(90)
        nu = np.array([1, 0, 0])

        # Expected true value
        eta_dot = np.array([0, 1, 0])

        assert np.all(np.isclose(eta_dot, Rz(psi)@nu, rtol=1e-6))


    def test_Rzyx(self):
        psi, theta, phi = np.pi / 2, 0, 0
        nu = np.array([1., 0., 0.])

        # Expected true value
        eta_dot = np.array([0., 1., 0.])

        assert np.all(np.isclose(eta_dot, Rzyx(phi, theta, psi)@nu, rtol=1e-6))


    def test_J_dim(self):
        psi, theta, phi = 0, 0, 0
        rotmat_6dof = J(psi, theta, phi)

        assert np.all(np.equal((6, 6), rotmat_6dof.shape))

    def test_J(self):
        nu = np.array([1., 0., 0., 0., 0., 0.])
        phi, theta, psi = 0., 0., np.pi/2
        eta_dot = np.array([0., 1., 0., 0., 0., 0.])

        print(J(phi, theta, psi)@nu)

        assert np.all(np.isclose(eta_dot, J(phi, theta, psi)@nu, rtol=1e-6))