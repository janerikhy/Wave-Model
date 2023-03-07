import numpy as np
from MCSimPython.utils import Rz


class PD:
    
    def __init__(self, kp: list, kd: list):
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)
        self.tau_cmd = np.zeros(3)

    def get_tau(self, eta, eta_d, nu, nu_d):
        psi = eta[-1]
        z1 = Rz(psi).T@(eta-eta_d)
        z2 = nu - nu_d
        return -self.Kp@z1 - self.Kd@z2
    
    def set_kd(self, kd: list):
        self.Kd = np.diag(kd)
    
    def set_kp(self, kp: list):
        self.Kp = np.diag(kp)

class PID:
    
    def __init__(self, kp: list, kd: list, ki: list, dt: float = 0.01):
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)
        self.Ki = np.diag(ki)
        self.zi = np.zeros(3)
        self.dt = dt

    def get_tau(self, eta, eta_d, nu, nu_d):
        psi = eta[-1]
        z1 = Rz(psi).T@(eta - eta_d)
        z2 = nu - nu_d

        self.zi += self.dt*(eta - eta_d)
        return -self.Kp@z1 - self.Kd@z2 - Rz(psi).T@self.Ki@self.zi



class DirectBiasCompensationController():
    '''
    Bias estimate provided from the observer as direct compensation in a nominal PD control law.
    '''
    def __init__(self, kp: list, kd: list):
        
        self.Kp = np.diag(kp)
        self.Kd = np.diag(kd)

    def get_tau(self, eta, eta_d, nu, nu_d, b):
        psi = eta[-1]
        z1 = Rz(psi).T@(eta-eta_d)              # P
        z2 = nu - nu_d                          # D
        zb = Rz(psi).T@b                        # bias
        return -self.Kp@z1 - self.Kd@z2 - zb
    
    def set_kd(self, kd: list):
        self.Kd = np.diag(kd)
    
    def set_kp(self, kp: list):
        self.Kp = np.diag(kp)
