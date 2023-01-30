import numpy as np
from observer import Observer

from utils import J, Rz, Smat, pipi


class EKF(Observer):
    '''
    Implementation of Extended Kalman Filter (EKF).

    x_hat = [
        zeta    (6DOF)
        eta     (3DOF)
        bias    (3DOF)
        nu      (3DOF)
    ]
    '''
    def __init__(self, dt, M, D, x0=np.zeros(15), P0 = np.zeros((15,15))):
        '''
        Initialization:

        Input:
            - dt: Time step
            - M: Inertia matrix of system (including added mass)
            - D: Full damping matrix of system
        '''
        # Tuning
        self._Qd = np.eye(6)
        self._Rd = np.eye(3)

        # Constant matrices
        i = np.ix_([0,1,5],[0,1,5])     # Extract surge-sway-yaw DOFs
        M = M[i]
        self._Minv = np.linalg.inv(M)
        D = D[i]
        self._D = D
        self._H, self._B, self._E, self._Aw = 0,0,0,0
        self.initialize_constant_matrices()
        
        

        self._dt = dt

        self._DOF = 3

        self._xhat = np.zeros(15)
        self._xbar = x0

        self._Pbar = P0
        self._Phat = np.zeros((15,15))

        


    def predictor(self, tau):
        '''
        Documentation
        '''
        phi = self.state_function_jacobian()
        gamma = self.state_function_noise_jacobian()
        f = self.state_function(self._xhat, tau, np.zeros(6))

        self._Pbar = phi@self._Phat@phi.T + gamma@self._Qd@gamma.T
        self._xbar = self._xhat + self._dt * f


    def corrector(self, y):
        '''
        Documentation
        '''
        H = self.measurement_function_jacobian()
        K = self.EKF_gain(H)

        parenthesis = np.eye(15) - K@H

        self._Phat = parenthesis@self._Pbar@parenthesis.T + K@self._Rd@K.T
        self._xhat = self._xbar + K @(y - H@self._xbar)
        



    def EKF_gain(self, H):
        '''
        Documentation
        '''
        a = H@self._Pbar@H.T + self._Rd
        K = self._Pbar@H.T@np.linalg.inv(a)
        return K


    def state_function(self, x, tau, noise):                    # f(x,u,w)
        '''
        Documentation
        '''
        xi = x[0:6]
        eta = x[6:9]
        psi = eta[2]
        b = x[9:12]
        nu = x[12:15]

        Ax1 = np.array(
            [
            (self._Aw@xi).A1,
            (Rz(psi)@nu),
            np.zeros(3),
            (-self._Minv@self._D@nu - self._Minv@Rz(psi).T@b)
            ]
        )
        Ax = np.concatenate(Ax1)
        f = Ax + self._B@tau + self._E@noise
        return f


    def state_function_jacobian(self):       # df/dx
        '''
        Documentation
        phi = eye(15) + dt * del(f)/del(x)

        where

        del(f)/del(x) = [
            Aw          0_(6x3)     0_(6x3)         0_(6x3)                 row (1)
            0_(3x6)     del_f2      0_(3x3)         R(psi)                  row (2)
            0_(3x6)     0_(3x3)     0_(3x3)         0_(3x3)                 row (3)
            0_(3x6)     del_f4      M_inv*R(psi).T  -M_inv*D                row (4)
        ]   
        '''
        phi = np.zeros((15,15))

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



    def state_function_noise_jacobian(self):
        '''
        Documentation
        '''
        gamma = self._dt * self._E
        return gamma

    def measurement_function(self, x):                          # h(x,u)
        '''
        y = h(x,u)
        '''
        h = self._H @ x
        raise h

    def measurement_function_jacobian(self):                    # dh/dx
        '''
        H = [ dh(x,u) / dx ]_(x=x_hat) = h

        !!! NOT NECESSARY TO USE !!!
        '''
        H = self._H
        return H

    def initialize_constant_matrices(self):
        '''
        x_dot = f(x,u,w) 
        '''
        # Aw
        omega = 2*np.pi/1 # afjdfasdfdasfsfasdf!!!!!!!!!!!!!!!!!!!!!!!!!
        zeta = 0.05
        Aw1 = np.zeros((3,3))
        Aw2 = np.eye(3)
        Aw3 = -omega**2*np.eye(3)
        Aw4 = -2*zeta*omega*np.eye(3)
        self._Aw = np.bmat([[Aw1, Aw2],[Aw3, Aw4]])

        # E
        self._E = np.zeros((15,6))
        Ew = np.diag([1,1,1]) 
        Eb = np.eye(3)
        self._E[3:6,0:3] = Ew 
        self._E[9:12,3:6] = Eb
 
        # H
        self._H = np.zeros((3,15))
        self._H[0:3,3:6] = np.eye(3)
        self._H[0:3,6:9] = np.eye(3)

        # B
        self._B = np.zeros((15,3))
        self._B[12:15, 0:3] = self._Minv
        
    def get_xhat(self):
        return super().get_xhat()

'''
Yet to do:
    - Implement state function
    - Implement jacobians
    - Discretize (phi and gamma)
    - Vessel object as input?

'''