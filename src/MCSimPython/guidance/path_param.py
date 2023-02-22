# path_param.py

# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Jan-Erik Hygen
# Created Date: 2023-01-30
# Revised: 
# Tested:
#
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from MCSimPython.utils import Rz, pipi2cont


class WayPointsRefModel:
    """Way-point path parametrization model. 
    
    Attributes
    ----------
    pd : array_like
        Array of desired vessel positions (2D).
    pd_s : array_like
        Array of the parital derivative of pd w.r.t path parameter s.
    pd_s2 : array_like
        Array of the two times partial derivative of pd w.r.t s
    pd_s3 : array_like
        The three time partial derivative of pd w.r.t s.
    
    """

    r = 3
    A = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 2, 6, 12, 20, 30, 42],
        [0, 0, 0, 6, 24, 60, 120, 210]
    ])
    Ainv = np.linalg.pinv(A)
    THETA = np.arange(0, 1, 0.01)

    def __init__(self, way_points, slope, ds):
        self.n = len(way_points)    # Number of way-points
        self.I = self.n-1           # Number of line-segments
        self.k = 2*self.r + 1       # Number of coefficients we must solve for
        self.slope = slope
        self.wps = way_points

        pd = np.zeros((self.I, int(1/ds), 2))
        pd_s = np.zeros_like(pd)
        pd_s2 = np.zeros_like(pd)
        pd_s3 = np.zeros_like(pd)

        # Set initial values
        pd[:, 0] = way_points[0:-1]
        pd[:, -1] = way_points[1:]

        pd_s[0, 0] = way_points[1] - way_points[0]
        pd_s[-1, -1] = way_points[self.I] - way_points[self.I-1]

        pd_s[1:self.I, 0] = slope * \
            (way_points[2:] - way_points[0:self.I-1])
        pd_s[0:self.I-1, -1] = slope * \
            (way_points[2:] - way_points[0:self.I-1])

        pd_s2[0:self.I, 0] = 0
        pd_s3[0:self.I, 0] = 0
        pd_s2[0:self.I, 1] = 0
        pd_s3[0:self.I, 1] = 0

        # Calculate the coefficients
        self.bn = np.zeros((self.I, self.k+1, 2))
        for j in range(self.I):
            self.bn[j, :] = np.array([
                pd[j, 0],
                pd_s[j, 0],
                pd_s2[j, 0],
                pd_s3[j, 0],
                pd[j, -1],
                pd_s[j, -1],
                pd_s2[j, -1],
                pd_s3[j, -1]
            ])
        self.xn = np.zeros_like(self.bn)
        for j in range(self.I):
            self.xn[j, :] = WayPointsRefModel.Ainv@self.bn[j]

    def T(self, theta):
        return np.array([1, theta, theta**2, theta**3, theta **
                         4, theta**5, theta**6, theta**7])

    def T2(self, theta):
        return np.array([0, 1, 2*theta, 3*theta**2, 4*theta **
                         3, 5*theta**4, 6*theta**5, 7*theta**6])

    def T3(self, theta):
        return np.array([0, 0, 2, 6*theta, 12*theta**2, 20*theta**3, 30*theta**4, 42*theta**5])

    def T4(self, theta):
        return np.array([0, 0, 0, 6, 24*theta, 60*theta**2, 120*theta**3, 210*theta**4])

    def pd(self, theta):
        s = theta - np.floor(theta)
        i = int(np.floor(theta))
        if i >= self.I:
            i = 0
            s = 0
        point = self.T(s)@self.xn[i]
        return np.array([point[0], point[1]]).T

    def pd_s(self, theta):
        s = theta - np.floor(theta)
        i = int(np.floor(theta))
        if i >= self.I:
            i = 0
            s = 0
        point = self.T2(s)@self.xn[i]
        return np.array([point[0], point[1]]).T

    def pd_s2(self, theta):
        s = theta - np.floor(theta)
        i = int(np.floor(theta))
        if i >= self.I:
            i = 0
            s = 0
        point = self.T3(s)@self.xn[i]
        return np.array([point[0], point[1]]).T

    def pd_s3(self, theta):
        s = theta - np.floor(theta)
        i = int(np.floor(theta))
        if i >= self.I:
            i = 0
            s = 0
        point = self.T4(s)@self.xn[i]
        return np.array([point[0], point[1]]).T

    def eta_d(self, theta, psi_prev):
        pd = self.pd(theta)
        pd_s = self.pd_s(theta)
        psi_d = np.arctan2(pd_s[1], pd_s[0])
        return np.array([pd[0], pd[1], pipi2cont(psi_d, psi_prev)])

    def eta_d_s(self, theta):
        pd_s = self.pd_s(theta)
        pd_s2 = self.pd_s2(theta)
        psi_d_s = (pd_s2[1]*pd_s[0] - pd_s[1]*pd_s2[0]) / \
            (pd_s[0]**2 + pd_s[1]**2)
        return np.array([pd_s[0], pd_s[1], psi_d_s])

    def eta_d_s2(self, theta):
        pd_s = self.pd_s(theta)
        pd_s2 = self.pd_s2(theta)
        pd_s3 = self.pd_s3(theta)
        psi_d_s2 = ((pd_s3[1]*pd_s[0] - pd_s[1]*pd_s3[0])*(pd_s[0]**2 + pd_s[1]**2) -
                    2*(pd_s2[1]*pd_s[0] - pd_s[1]*pd_s2[0])*(pd_s[0]*pd_s2[0] + pd_s[1]*pd_s2[1])) / (pd_s[0]**2 + pd_s[1]**2)**2
        return np.array([pd_s2[0], pd_s2[1], psi_d_s2])

    def full_path(self, theta):
        path = np.zeros((len(theta), 3))
        for i in range(len(theta)):
            path[i] = self.eta_d(theta[i], path[i-1, -1])
        return path

    def path_speed(self, theta):
        speed = np.zeros((len(theta), 2))
        for i in range(len(theta)):
            speed[i] = self.pd_s(theta[i])
        return speed