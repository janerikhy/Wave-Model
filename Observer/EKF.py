import numpy as np
from observer import Observer

from utils import J, Rz, Smat, pipi


class EKF(Observer):
    '''
    EKF

    x_hat = [
        zeta    (6DOF)
        eta     (3DOF)
        nu      (3DOF)
        bias    (3DOF)
    ]


    '''
    def __init__(self, dt, Minv, D, x0=np.zeros(1), P0 = np.zeros((1,1))):
        # Tuning
        self._Qd = 0
        self._Rd = 0

        # Constant matrices
        self._Minv = Minv
        self._D = D
        self._H, self._B, self._E = self.initialize_constant_matrices()
        

        self._dt = dt

        self._DOF = 3

        self._xhat = np.zeros(15)
        self._xbar = x0

        self._Pbar = P0
        self._Phat = np.zeros((1,1))

        


    def predictor(self, tau):
        phi = self.transition_function_jacobian()
        gamma = self.transition_function_noise()
        f = self.transition_function()

        self._Pbar = phi@self._Phat@phi.T + gamma@self._Qd@gamma.T
        self._xbar = self._xhat + self._dt * f


    def corrector(self, y):
        H = self.measurement_function_jacobian()
        K = self.EKF_gain(self, H)

        parenthesis = np.eye(1) - K@H

        self._Phat = parenthesis@self._Pbar@parenthesis.T + K@self._Rd@K.T
        self._xhat = self._xbar + K @(y - H@self._xbar)
        pass


    



    def EKF_gain(self, H):
        a = H@self._Pbar@H.T + self._Rd
        K = self._Pbar@H.T@np.linalg.inv(a)
        return K


    def state_function(self, x, tau, noise):                    # f(x,u,w)
        # f = Ax + Bu + Ew
        raise NotImplementedError

    def state_function_jacobian(self, state, tau, noise):       # df/dx
         raise NotImplementedError

    def state_function_noise_jacobian(self):
        raise NotImplementedError                               # df/dw

    def measurement_function(self, x):                          # h(x,u)
        h = self._H @ x
        raise h

    def measurement_function_jacobian(self):                    # dh/dx
        raise NotImplementedError


    def initialize_constant_matrices():
        H = np.zeros((3,15))
        B = np.zeros((15,3))
        E = np.zeros((15,6))
        # Aw = 
        # Ew = 
        # Eb = 
        return H, B, E


'''
Yet to do:

    - Implement constant matrices
    - Implement state function
    - Implement jacobians
    - Discretize (phi and gamma)

'''