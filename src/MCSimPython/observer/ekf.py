
# ekf.py
# ----------------------------------------------------------------------------
# This code is part of the MCSimPython toolbox and repository.
# Created By: Harald Mo
# Created Date: 2023-02-13

# Revised: N/A

# Tested:  See demos\observer\ekd3dof.py
# 
# Copyright (C) 2023: NTNU, Trondheim
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

# Imports
import numpy as np
from MCSimPython.utils import Rz, pipi, six2threeDOF


class EKF():
    '''
    Implementation of Extended Kalman Filter (EKF) for CSAD.

    x_hat = [
        zeta    (6DOF)
        eta     (3DOF)
        bias    (3DOF)
        nu      (3DOF)
    ]

    Reference:
        - Fossen (2021), Marine craft hydrodynamics and motion control, ch 13.
    '''
    def __init__(self, dt, M, D, Tp, x0=np.zeros(15), P0 = np.zeros((15,15))):
        '''
        Initialization:

        Parameters
        -----------
            - dt (float): Time step
            - M (numpy array (6, 6)): Inertia matrix of system (including added mass) 6DOF
            - D (numpy array (6, 6)): Full damping matrix of system 6DOF
            - Tp (int): Peak period of wave spectre
        '''
        self._dt = dt

        # Tuning - Done manually (as of now)
        self._Qd = np.array([
            [1e-3,0,0,0,0,0],
            [0,1e-3,0,0,0,0],
            [0,0,.2*np.pi/180,0,0,0],
            [0,0,0,1e2,0,0],
            [0,0,0,0,1e2,0],
            [0,0,0,0,0,1e1]
        ])
    
        self._Rd = np.array([
            [100,0,0],
            [0,100,0],
            [0,0,500*np.pi/180]
        ])
        #self._Qd = np.diag([4.57666453e-04, 6.54424838e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.35349136e-01])
        #self._Rd = np.eye(3)

        # Constant matrices
        M = six2threeDOF(M)
        self._Minv = np.linalg.inv(M)                                       # 3DOF inverted mass matrix
        self._D = six2threeDOF(D)                                           # 3DOF damping matrix
        
        self._H, self._B, self._E, self._Aw, self._gamma = 0,0,0,0,0
        self.initialize_constant_matrices(Tp)
        
        
        # Initial values
        self._xhat = np.zeros(15)
        self._xbar = x0

        self._Pbar = P0
        self._Phat = np.zeros((15,15))
    
    
      
    def update(self, tau, y):
        '''
        Update:
        Update function to be called at every timestep during a simulation. Calls the predictor and corrector.

        Parameters
        ----------
            - tau: Control Parameters (3DOF)
            - y: Measured position (3DOF)

        To be implemented / Improvements
        -----------
            - Error checks?
            - Asynchronous measurements?
            - Set y to nan if no measurement
        '''
        # Correct
        self.corrector(y)
        # Predict        
        self.predictor(tau)
        

    
    def predictor(self, tau):
        '''
        Predictor:
        Used to estimate the state of the system at the next time step based on the current state estimate and the system dynamics model.
        
        Parameters
        ----------
            - tau = [X, Y, N]',  Control input (3DOF)
             
        '''
        phi = self.state_function_jacobian()
        f = self.state_function(self._xhat, tau, np.zeros(6))

        self._Pbar = phi@self._Phat@(phi.T) + self._gamma@self._Qd@(self._gamma.T)
        self._xbar = self._xhat + self._dt * f


    def corrector(self, y):
        '''
        Corrector:
        Used to refine the predicted state estimate from the predictor, based on the obtained measurements. The function checks for signal loss (no measurement).

        Parameters
        -----------
            - y = [eta1  eta2  eta6]', Measurements (3DOF) 

        To be implemented / Improvements
        ----------
            - Check if measurement is given. Can also be made with modulus operator if measurement freq is different from dt.
            - 
        '''

        if np.any(np.isnan(y)) == True:    # If no new measurements: Set corrector equal to predictor (Dead reckoning)
            self._Phat = self._Pbar
            self._xhat = self._xbar
        else:
            K = self.EKF_gain

            parenthesis = np.eye(15) - K@self._H

            prediction_error = y - (self._H@self._xbar)
            prediction_error[2] = pipi(prediction_error[2])   # Smallest signed angle modification

            self._Phat = parenthesis@self._Pbar@(parenthesis.T) + K@self._Rd@(K.T)
            self._xhat = self._xbar + K@prediction_error
        

    @property
    def EKF_gain(self):
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
        parenthesis = self._H@self._Pbar@(self._H.T) + self._Rd
        K = self._Pbar@(self._H.T)@np.linalg.inv(parenthesis)
        return K


    def state_function(self, x, tau, noise = np.zeros(6)):                    # f(x,u,w)
        '''
        x_dot = A(x) + B*tau + E*w      = f(x,u,w)

        where A(x) = [
            Aw * xi
            R(psi) * nu
            0_(3x1)
            -M_inv*D * nu + M_inv*R(psi).T * b
        ]

        Parameters
        ----------
            - x: state vector (Dim = 15)
            - tau: Control vector [X  Y  N]
            - noise: Modelled white noise, set to zero in a deterministic EKF. (Dim=6)

        Output
        ----------
            - f: x_dot (Dim = 15)
        '''
        xi = x[0:6]
        eta = x[6:9]
        psi = eta[2]
        b = x[9:12]
        nu = x[12:15]

        Ax = np.array(
            [
            (self._Aw@xi),
            (Rz(psi)@nu),
            np.zeros(3),
            (-self._Minv@self._D@nu + self._Minv@(Rz(psi).T)@b)     # plus or minus in 2nd term?
            ], dtype=object
        )

        Ax = np.concatenate(Ax)                                     # Convert to 15x1

        f = Ax + self._B@tau + self._E@noise
        return f


    def state_function_jacobian(self):       # df/dx
        '''
        phi = eye(15) + dt * del(f)/del(x)

        where

        del(f)/del(x) = [
            row (1):        A_w         0_(6x3)     0_(6x3)         0_(6x3)                 
            row (2):        0_(3x6)     del_f2      0_(3x3)         R(psi)                  
            row (3):        0_(3x6)     0_(3x3)     0_(3x3)         0_(3x3)                
            row (4):        0_(3x6)     del_f4      M_inv*R(psi).T  -M_inv*D                
        ]   

        Parameters
        ----------
            - N/A

        Output
        -----------
            - phi: Discretized jacobian of system dynamics (Dim = 15x15)

        '''
        # Extract states
        psi = self._xhat[8]
        b1 = self._xhat[9]
        b2 = self._xhat[10]
        u = self._xhat[12]
        v = self._xhat[13]

        del_f2 = np.array([
                            [0,0,-u*np.sin(psi)-v*np.cos(psi)],
                            [0,0,u*np.cos(psi)-v*np.sin(psi)],
                            [0,0,0]
        ])
        del_f4 = np.array([
                            [0,0,self._Minv[0,0]*(b2*np.cos(psi) - b1*np.sin(psi))],
                            [0,0,-self._Minv[1,1]*(b1*np.cos(psi) + b2*np.sin(psi))],
                            [0,0,-self._Minv[2,1]*(b1*np.cos(psi) + b2*np.sin(psi))]
        ])

        row1 = np.concatenate((self._Aw, np.zeros((6,9))), axis=1)
        row2 = np.concatenate((np.zeros((3,6)), del_f2, np.zeros((3,3)), Rz(psi)), axis=1)
        row3 = np.zeros((3,15))
        row4 = np.concatenate((np.zeros((3,6)), del_f4, self._Minv@(Rz(psi).T), -self._Minv@self._D), axis=1)

        phi = np.eye(15) + self._dt * np.concatenate((row1,row2,row3,row4), axis = 0)

        return phi



    def initialize_constant_matrices(self, Tp):
        '''
        Initialize following matrices:

        A_w = [
            0_(3x3)     I_3
            -omega^2    -2*zeta*omega
        ]

        E = [
            0_(3x3)     0_(3x3)
            E_w         0_(3x3)
            0_(3x3)     0_(3x3)
            0_(3x3)     E_b
            0_(3x3)     0_(3x3)
        ]

        gamma = dt * E

        H = [
            0_(3x3)     I_3     I_3     0_(3x6)
        ]

        B = [
            0_(12x3)
            M_inv
        ]

        Parameters
        ----------
            - Tp: Sea state peak period

        To be implemented / Improvements
        ----------
            - Tuning must be done
            - Can add wave spectrum properties as input in tuning? Tp, Damping, Kw_i
            - Use np.block() in _Aw
        '''

        # A_w
        omega = 2*np.pi/Tp                                      # Should approx be peak freq in wave spectrum
        zeta = .05                                              # Damping coeff
        '''Define A_w = [
                Aw11   Aw12
                Aw21   Aw22  ]'''
        Aw11 = np.zeros((3,3))
        Aw12 = np.eye(3)
        Aw21 = -omega**2*np.eye(3)
        Aw22 = -2*zeta*omega*np.eye(3)
        self._Aw = np.block([[Aw11, Aw12], [Aw21, Aw22]])

        # E
        self._E = np.zeros((15,6))
        Ew = np.diag([1,1,1])                               # Multiplied with .01 because no waves
        Eb = np.eye(3)
        self._E[3:6,0:3] = Ew 
        self._E[9:12,3:6] = Eb

        # Gamma
        self._gamma = self._dt * self._E

        # H
        self._H = np.zeros((3,15))
        self._H[0:3,3:6] = np.eye(3)
        self._H[0:3,6:9] = np.eye(3)

        # B
        self._B = np.zeros((15,3))
        self._B[12:15, 0:3] = self._Minv
    

    def set_tuning_matrices(self, Q, R):
        '''
        set_tuning_matrices
        Customize tuning matrices in the EKF

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
        self._Qd = Q
        self._Rd = R
    

    
    def get_x_hat(self):
        return self._xhat
    
    def get_P_hat(self):
        return self._Phat
    
    def get_eta_hat(self):
        return self._xhat[6:9]
    
    def get_nu_hat(self):
        return self._xhat[12:15] 
        

'''
"Tuning factors":
    - Q and R
    - Tp in A_w
    - zeta in A_w
    - diag(Kw,Kw,Kw) in E
    - Eb in E (njæ)
    - Adjustment of noise
    - Time step
    - Initial values
'''