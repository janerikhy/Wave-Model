import numpy as np
import pytest

import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

from MCSimPython.utils import Rz, Ry, Rx, Rzyx, J, to_positive_angle, rigid_body_transform


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
        eta = np.array([0, 0, 0, phi, theta, psi])

        # Expected true value
        eta_dot = np.array([0., 1., 0.])

        assert np.all(np.isclose(eta_dot, Rzyx(eta)@nu, rtol=1e-6))


    def test_J_dim(self):
        psi, theta, phi = 0, 0, 0
        eta = np.array([0, 0, 0, phi, theta, psi])

        rotmat_6dof = J(eta)

        assert np.all(np.equal((6, 6), rotmat_6dof.shape))

    def test_J(self):
        nu = np.array([1., 0., 0., 0., 0., 0.])
        phi, theta, psi = 0., 0., np.pi/2
        eta = np.array([0, 0, 0, phi, theta, psi])

        eta_dot = np.array([0., 1., 0., 0., 0., 0.])
        
        assert np.all(np.isclose(eta_dot, J(eta)@nu, rtol=1e-6))

    def test_positive_angle_map(self):
        angle = -np.pi / 2  # corresponding to 270 degrees (3pi/2)
        pos_angle = to_positive_angle(angle)
        true_angle = 3*np.pi / 2
        assert np.equal(pos_angle, true_angle)

    def test_array_positive_angle_map(self):
        angles = np.array([-np.pi, -np.pi/2])
        pos_angles = to_positive_angle(angles)
        true_angles = np.array([np.pi, 3*np.pi/2])
        assert np.all(np.isclose(pos_angles, true_angles, rtol=1e-6))

    def test_rigid_body_transform_translation(self):
        r = np.array([1., 0., -1.]) # Defined in the body-frame
        eta_1 = np.concatenate((np.array([1., 0., 0.]), np.zeros(3)))
        eta_2 = np.concatenate((np.array([0., 1., 0]), np.zeros(3)))
        eta_3 = np.concatenate((np.array([0., 0., 1.0]), np.zeros(3)))

        s_true_1 = np.array([1., 0., 0.])
        s_true_2 = np.array([0., 1., 0.])
        s_true_3 = np.array([0., 0., 1.])

        assert np.all(np.isclose(rigid_body_transform(r, eta_1, in_ned=False), s_true_1, rtol=1e-5))
        assert np.all(np.isclose(rigid_body_transform(r, eta_2, in_ned=False), s_true_2, rtol=1e-5))
        assert np.all(np.isclose(rigid_body_transform(r, eta_3, in_ned=False), s_true_3, rtol=1e-5))

    def test_rigid_body_transform_rotations(self):
        r = np.array([1., 0., -1]) # Define the lever arm in body-frame
        eta13 = np.zeros(3) # No translational motions
        eta36_1 = np.array([np.pi/180*2, 0.0, 0.])  # 2 degree roll
        eta36_2 = np.array([0., 3*np.pi/180, 0.0])  # 3 degree pitch
        eta36_3 = np.array([0., 0., 5*np.pi/180])   # 5 degree yaw

        eta_1 = np.concatenate([eta13, eta36_1])
        eta_2 = np.concatenate([eta13, eta36_2])
        eta_3 = np.concatenate([eta13, eta36_3])

        s_1 = np.array([0.0, 2*np.pi/180, 0.0]) # Only a change in the y-axis
        s_2 = np.array([-1*3*np.pi/180, 0.0, -3*np.pi/180])
        s_3 = np.array([0., 5*np.pi/180, 0.0])

        assert np.allclose(rigid_body_transform(r, eta_1, in_ned=False), s_1, rtol=1e-5)
        assert np.allclose(rigid_body_transform(r, eta_2, in_ned=False), s_2, rtol=1e-5)
        assert np.allclose(rigid_body_transform(r, eta_3, in_ned=False), s_3, rtol=1e-5)

    def test_rigid_transform_ref_frames(self):
        r = np.array([1., 0., 0.])  # Lever arm defined in body-frame

        # Vessel pose in NED-frame
        eta = np.array([
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            np.pi/4
        ])

        s_body = np.array([1., np.pi/4, 0.0])    # Lever arm change in body-frame
        s_ned = np.array([np.cos(np.pi/4), -np.sin(np.pi/4) + np.pi/4, 0.]) # Lever arm change in ned-frame

        print(rigid_body_transform(r, eta, in_ned=True))
        
        assert np.allclose(rigid_body_transform(r, eta, in_ned=False), s_body)
        assert np.allclose(rigid_body_transform(r, eta, in_ned=True), s_ned)

