import numpy as np

propellerDiameter = 0.03    #[m]
n_dot_max = 5.0             #[1/s^2]
alpha_dot_max = 2.0         #[1/s]
thrust_max = 1.5            #[N]

class ThrusterDynamics:

    def __init__(self, u, K):
        self._u = u
        self._K = K


    def saturate(self, signal, min, max):
        return np.clip(signal, min, max)

    def propeller_rev(self):
        n = np.sign(self._u) * np.sqrt(np.abs(self._u))
        return n
    
    def compute_tau(self):
        n = self.propeller_rev()
        tau = self._K @ np.abs(n) @ n
        return tau


