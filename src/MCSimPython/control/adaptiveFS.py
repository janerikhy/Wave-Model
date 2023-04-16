# imports
import numpy as np
from MCSimPython.utils import Rz, six2threeDOF, three2sixDOF, pipi, Smat



class AdaptiveFSController():
    '''
    Adaptive controller using a truncated Fourier series-based internal disturbance model in a "Model Reference Adaptive Control" (MRAC).
    The control law is determined by using LgV-backstepping, based on calculations and procedures done in (Fossen, 2021) and (Brørby, 2022).
    
    To be implemented:
        - Improved tuning.
        - 
    '''
    def __init__(self, dt, M, D, N=15):
        '''
        Initialize

        Parameters
        ------------
            - dt: timestep
            - M: Mass-matrix (6x6) of vessel (inertia + added mass)
            - D: Full damping matrix (6x6) of vessel
            - N: Number of wave frequency components. Set to 15 as default
        '''
        self._dt = dt
        self._M = six2threeDOF(M)
        self._D = six2threeDOF(D)
        
        self.theta_hat = np.zeros((2*N + 1)*3)
        
        # Frequencies to be used in disturbance model 
        w_min = 2*np.pi/20                          # Lower bound
        w_max = 2*np.pi/2                           # Upper bound
        self.set_freqs(w_min, w_max, N)

        # Tuning:
        self._K1 = np.diag([.1, 1., .1])
        self._K2 = np.diag([.1, 1., .1])*10
        self._gamma = np.eye((2*self._N +1)*3) * 5
        self._kappa = 1                             # Must be positive



    def get_tau(self, eta, eta_d, nu, eta_d_dot, eta_d_ddot, t, calculate_bias=False):
        '''
        Calculate controller output based on the adaptive update law. 

        Parameters
        -----------
            - eta (3DOF): [surge, sway, yaw] expressed in NED
            - eta_d (3DOF): Desired position expressed in NED
            - nu (3DOF): [u, v, r] expressed in body
            - eta_d_dot (3DOF): Desired velocity expressed in NED
            - eta_d_ddot (3DOF): Desired acceleration expressed in NED
            - t (float): Time
            - Calculate_bias (bool): Function will also return estimated bias if True. Set to False as default.

        Output
        ------------
            - tau (3DOF): Calculated control forces
            - b_hat (3DOF): Estimated bias in surge, sway, yaw. Only returned if "calculate_bias == True"
        
        '''
        R = Rz(eta[-1])                                 # Rotation matrix
        S = Smat(np.array([0,0,nu[-1]]))                # Skew-symmetric matrix

        # Use get_regressor() to calculate Phi
        regressor_transpose = self.get_regressor(t)
        zeros = np.zeros(len(regressor_transpose))

        Phi_transpose = np.block([[regressor_transpose, zeros, zeros], 
                                  [zeros, regressor_transpose, zeros], 
                                  [zeros, zeros, regressor_transpose]])

        Phi = Phi_transpose.T
       
        # Introduce the first new state, which shall converge towards zero
        z1 = R.T@(eta-eta_d)
        z1[-1] = pipi(z1[-1])

        # Virtual input function
        alpha0 = -self._kappa*np.eye(3)@z1
        alpha = -self._K1@z1 + R.T@eta_d_dot + alpha0

        # The second introduced state, functioning as a virtual input to the first control system
        z2 = nu - alpha 

        # Differentiate
        z1_dot = -S@z1 + z2 -(self._K1 + self._kappa*np.eye(3))@z1
        alpha_dot = -(self._K1 + self._kappa*np.eye(3))@z1_dot - S@R.T@eta_d_dot + R.T@eta_d_ddot        

        # Adaptive update law
        theta_hat_dot = self._gamma@Phi@z2                                                                  # @z1?
        self.theta_hat  += (theta_hat_dot * self._dt)

        # Control law
        tau = -self._K2@z2 + self._D@alpha + self._M@alpha_dot - Phi.T@self.theta_hat
        # tau = R@tau                              # Ref frame of tau?

        # Calculate bias
        if calculate_bias:
            b_hat = Phi.T@self.theta_hat
            return tau, b_hat
        
        return tau

    def get_regressor(self, t):
        '''
        get_regressor:
        Extract a (2*N+1)-dimensional regressor defined as [1   cos(w1*t)   sin(w1*t)   cos(w2*t)   ...   sin(wN*t)].
        Used in get_tau() to calculate control forces.

        Parameters
        ----------
            - t: time (float)

        Return
        ----------
            - regressor: (2N + 1)-dim, time-dependent vector as defined above
        '''
        regressor = np.zeros(2*self._N + 1)
        regressor[0] = 1
        for i in range(self._N):
            regressor[2*i + 1] = np.cos(self._freqs[i]*t)
            regressor[2*i + 2] = np.sin(self._freqs[i]*t)
        return regressor

    
    def set_freqs(self, w_min, w_max, N):
        '''
        set_freqs:
        Customize the number and magnitude of wave frequencies to be included in the internal disturbance model.

        UNIT IN RAD/S OR HZ?

        Parameters
        -----------
            - w_min: Lower bound frequency
            - w_max: Upper bound frequency
            - N: Number of components
        '''
        self._N = N
        dw = (w_max-w_min)/ self._N
        self._freqs = np.arange(w_min, w_max, dw)

