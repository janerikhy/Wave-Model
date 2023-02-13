# imports
import numpy as np
import matplotlib.pyplot as plt
from MCSimPython.utils import Rz, six2threeDOF, three2sixDOF, pipi, Smat
from MCSimPython.simulator.csad import CSAD_DP_6DOF



class AdaptiveFSController():
    '''
    Adaptive controller using a Fourier series-based internal disturbance model in a MRAC
    '''
    def __init__(self, dt, M, D, N=15):
        '''
        Parameters:
            - 
            - 

        '''
        self._dt = dt
        self._M = six2threeDOF(M)
        self._D = six2threeDOF(D)
        
        self.theta_hat = np.zeros(2*N + 1)
        
        self._N = N
        w_min = 2*np.pi/20
        w_max = 2*np.pi/2
        dw = (w_max-w_min)/self._N
        self._freqs = np.arange(w_min, w_max, dw)

        # Tuning:
        self._K1 = np.diag([10., 1., 1.])
        self._K2 = np.diag([10., 1., 1.])
        self._gamma = np.eye(2*self._N +1) * 5
        self._kappa = 1



    def get_tau(self, eta, eta_d, nu, eta_d_dot, eta_d_dotdot, t):

        R = Rz(eta[-1])
        S = Smat(np.array([0,0,nu[-1]]))

        # eta_d_dot = R@nu_d
        regressor_transpose = self.get_regressor(t)

        Phi_transpose = np.array([regressor_transpose, regressor_transpose, regressor_transpose])
        Phi = Phi_transpose.T
        
        z1 = R.T@(eta-eta_d)
        z1[-1] = pipi(z1[-1])


        alpha0 = -self._kappa*np.eye(3)@z1
        alpha = -self._K1@z1 + R.T@eta_d_dot + alpha0

        z2 = nu - alpha 

        z1_dot = -S@z1 + z2 -(self._K1 + self._kappa*np.eye(3))@z1

        alpha_dot = -(self._K1 + self._kappa*np.eye(3))@z1_dot + S@R.T@eta_d_dot + R.T@eta_d_dotdot         # + eller - i andre ledd

        theta_hat_dot = self._gamma@Phi@z2
        self.theta_hat  += (theta_hat_dot * self._dt)


        tau = -self._K2@z2 + self._D@alpha + self._M@alpha_dot - Phi.T@self.theta_hat

        b_hat = Phi.T@self.theta_hat

        return tau

    def get_regressor(self, t):
        regressor = np.zeros(2*self._N + 1)
        regressor[0] = 1
        for i in range(self._N):
            regressor[2*i + 1] = np.cos(self._freqs[i]*t)
            regressor[2*i + 2] = np.sin(self._freqs[i]*t)
        
        return regressor

    

