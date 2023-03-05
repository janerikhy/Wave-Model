# ltv_kf.py
# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Harald Mo
# Created Date: 2023-05-03
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import numpy as np
from MCSimPython.utils import Rz, pipi, six2threeDOF, three2sixDOF


class LTVKF():
    '''
    Implementation of a Linear time-varying (LTV) Kalman filter for CSAD

    Assume that accurate compass and gyro measurements (psi and r) are available throughout the entire simulation
    
    Parameters
    -----------
        - dt (float): Time step
        - M (numpy array (6, 6)): Inertia matrix of system (including added mass) 6DOF
        - D (numpy array (6, 6)): Full damping matrix of system 6DOF
        - Tp (float): 
            
    '''

    def __init__(self, dt, M, D, Tp, x0=np.zeros(15), P0 = np.zeros((15,15))):
        '''
        Initialization
        '''
        self._dt = dt

        M = six2threeDOF(M)
        self._Minv = np.linalg.inv(M)
        self._D = six2threeDOF(D)


        # Initial values
        self.xhat = np.zeros(15)
        self.xbar = x0

        self.Phat = np.zeros((15,15))
        self.Pbar = P0


        # Initial tuning
        self.Qd = np.eye(5)
        self.Rd = np.eye(3)

        # Define constant system parameters Aw, Bd, Ed and H
        '''
        A_w = [
            0_(3x3)     I_3             \n
            -omega^2    -2*zeta*omega ]

        Bd  = dt * [
            0_(12x3)    \n
            M_inv ]

        Ed  = dt * [
            0_(3x3)     0_(3x3)   \n
            E_w         0_(3x3)   \n 
            0_(3x3)     0_(3x3)   \n
            0_(3x3)     E_b       \n
            0_(3x3)     0_(3x3)  ]
        
        H = [0_(3x3)     I_3     I_3     0_(3x6)]
        '''
        omega = 2 * np.pi / Tp
        zeta = .05
        self._Aw = np.block([
            [np.zeros((3,3)),   np.eye(3)],
            [-omega**2*np.eye(3),   -2*zeta*omega*np.eye(3)]
        ])
        
        self._Bd = NotImplementedError

        self._Ed = NotImplementedError

        self._H = np.zeros((3,15))
        self._H[0:3,3:6], self._H[0:3,6:9] = np.eye(3), np.eye(3)


    def update(self, tau, y, psi_m):
        '''
        Update:
        Update function to be called at every timestep during a simulation. Calls the predictor and corrector.

        Parameters
        ----------
            - tau: Control Parameters (3DOF)
            - y: Measured position (3DOF)
            - psi_m: Vessel heading, used to find R(t)
        
        Returns
        -----------
            - N/A
        '''
        # Predict        
        self.predictor(tau, psi_m)
        # Correct
        self.corrector(y)
        
    
    def predictor(self, tau, psi_m):
        Ad =  self.Ad(psi_m)   # Get the time-varying state vector

        self.xbar = Ad@self.xhat + self._Bd@tau
        self.Pbar = Ad@self.Phat@Ad.T + self._Ed@self.Qd@self._Ed.T

    
    def corrector(self, y):
        if np.any(np.isnan(y)) == True:    # If no new measurements: Set corrector equal to predictor (Dead reckoning)
            self.Phat = self.Pbar
            self.xhat = self.xbar
        else:
            K = self.KF_gain
            prediction_error = y - (self._H@self.xbar)
            prediction_error[2] = pipi(prediction_error[2])   # Smallest signed angle modification

            parenthesis = np.eye(15) - K @ self._H

            self.Phat = parenthesis @ self.Pbar @ parenthesis.T + K @ self.Rd @ K.T
            self.xhat = self.xbar + K @ prediction_error


    @property
    def KF_gain(self):
        '''
        Kalman gain:
        Used to balance the contributions of the predicted state estimate and the measurement data in the updated estimate.

        Parameters
        -----------
            - N/A

        Output
        -----------
            - K: Kalman gain (Dim = 15x3)
        '''
        parenthesis = self._H @ self.Pbar @ self._H.T + self.Rd
        K = self.Pbar @ self._H.T @ np.linalg.inv(parenthesis)
        return K

    def state_function(self, x, tau, psi_m, noise = np.zeros(6)):
        '''
        x_dot = A(t)*x + B*u + E*w
        where:

        A(t)= [Aw       0_(6x3)     0_(6x3)         0_(6x3)    \n
              0_(3x6)   0_(3x3)     0_(3x3)         R(t)        \n
              0_(3x6)   0_(3x3)     0_(3x3)         0_(3x3)     \n
              0_(3x6)   0_(3x3)     M_inv*R(t).T    -M_inv*D ]
        
        B =  [0_(12x3)  \n
              M_inv ]
        
        E  = [0_(3x3)     0_(3x3)   \n
              E_w         0_(3x3)   \n 
              0_(3x3)     0_(3x3)   \n
              0_(3x3)     E_b       \n
              0_(3x3)     0_(3x3)  ]

        Parameters
        ----------
            - x: state vector (Dim = 15)
            - tau: Control vector [X  Y  N]
            - noise: Modelled white noise, set to zero in a deterministic KF. (Dim=6)
            - psi_m: Measured heading

        Output
        ----------
            - f: x_dot (Dim = 15)

        Tod0:
            - Implement Aw
            - Discretize
        '''
        xi = x[0:6]
        eta = x[6:9]
        psi = eta[2]
        b = x[9:12]
        nu = x[12:15]
        

        # Define countinous system matrices
        A = np.block([
                [self._Aw,          np.zeros((6,3)),    np.zeros((6,3)),      np.zeros((6,3))],
                [np.zeros((3,6)),   np.zeros((3,3)),    np.zeros((3,3)),      Rz(psi_m)],
                [np.zeros((3,6)),   np.zeros((3,3)),    np.zeros((3,3)),      np.zeros((3,3)),],
                [np.zeros((3,6)),   np.zeros((3,3)),    self._Minv@Rz(psi_m).T, -self._Minv@self._D]
        ])

        # Discretize
        Ad = np.exp(A*self._dt)
        Bd = 0
        Ed = 0
        
        # Calculate state function
        x_dot = Ad @ x + Bd @ tau + Ed @ noise

        return x_dot

    def Ad(self, psi_m):
        '''
        A(t)= [Aw       0_(6x3)     0_(6x3)         0_(6x3)    \n
              0_(3x6)   0_(3x3)     0_(3x3)         R(t)        \n
              0_(3x6)   0_(3x3)     0_(3x3)         0_(3x3)     \n
              0_(3x6)   0_(3x3)     M_inv*R(t).T    -M_inv*D ]
        and 
        Ad = e^(dt*A)   OR   Ad = I + dt * A
        '''
        A = np.block([
            [self._Aw,          np.zeros((6,3)),    np.zeros((6,3)),      np.zeros((6,3))],
            [np.zeros((3,6)),   np.zeros((3,3)),    np.zeros((3,3)),      Rz(psi_m)],
            [np.zeros((3,6)),   np.zeros((3,3)),    np.zeros((3,3)),      np.zeros((3,3)),],
            [np.zeros((3,6)),   np.zeros((3,3)),    self._Minv@Rz(psi_m).T, -self._Minv@self._D]
        ])
        return np.exp(A*self._dt)
    


    def set_tuning_matrices(self, Q, R):
        '''
        set_tuning_matrices
        Customize tuning matrices in the LTVKF

        Parameters
        ----------
        Q : array_like
            (6,6)-dim array stating covariance in model uncertainty
        R : array_like
            (6,6)-dim array stating covariance in measurement uncertainty

        Return
        ------
        N/A
        '''
        self.Qd = Q
        self.Rd = R


    # Get functions:
    def get_x_hat(self):
        return self.xhat
    
    def get_P_hat(self):
        return self.Phat
    
    def get_eta_hat(self):
        return self.xhat[6:9]
    
    def get_nu_hat(self):
        return self.xhat[12:15] 