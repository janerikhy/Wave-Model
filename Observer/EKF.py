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
    def __init__(self, dt, M, D, x0=np.zeros(15), P0 = np.zeros((15,15))):
        # Tuning
        self._Qd = 0
        self._Rd = 0

        # Constant matrices
        i = np.ix_([0,1,5],[0,1,5])
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
        phi = NotImplementedError
        gamma = NotImplementedError
        f = self.state_function(self._xhat, tau, np.zeros(6))

        self._Pbar = phi@self._Phat@phi.T + gamma@self._Qd@gamma.T
        self._xbar = self._xhat + self._dt * f


    def corrector(self, y):
        '''
        Documentation
        '''
        H = self.measurement_function_jacobian()
        K = self.EKF_gain(self, H)

        parenthesis = np.eye(1) - K@H

        self._Phat = parenthesis@self._Pbar@parenthesis.T + K@self._Rd@K.T
        self._xhat = self._xbar + K @(y - H@self._xbar)
        pass



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


    def state_function_jacobian(self, state, tau, noise):       # df/dx
        '''
        Documentation
        '''
        raise NotImplementedError

    def state_function_noise_jacobian(self):
        '''
        Documentation
        '''

        raise NotImplementedError                               # df/dw

    def measurement_function(self, x):                          # h(x,u)
        '''
        Documentation
        '''
        h = self._H @ x
        raise h

    def measurement_function_jacobian(self):                    # dh/dx
        '''
        Documentation
        '''
        raise NotImplementedError


    def initialize_constant_matrices(self):
        '''
        Documentation
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
        


'''
Yet to do:
    - Implement state function
    - Implement jacobians
    - Discretize (phi and gamma)
    - Vessel object as input?

'''