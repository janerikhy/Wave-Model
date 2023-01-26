import numpy as np
from observer import Observer

from utils import J, Rz, Smat, pipi


class EKF(Observer):
    '''
    EKF
    '''
    def __init__(self, x0=np.zeros(1), P0 = np.zeros((1,1)), dt=0.1):
        self._Qd = 0
        self._Rd = 0

        self._phi = 0
        self._gamma = 0
        self._H = 0

        self._dt = dt

        self._xhat = np.zeros(1)
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


    def transition_function(self, state, tau, noise):              # f(x,u,w)
        pass
    def transition_function_jacobian(self, state, tau, noise):     # df/dx
        pass
    def measurement_function(self):             # h(x,u)
        pass
    def measurement_function_jacobian(self):    # dh/dx
        pass


