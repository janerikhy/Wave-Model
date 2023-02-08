import numpy as np

propellerDiameter = 0.03    #[m]
n_dot_max = 5.0             #[1/s^2]
alpha_dot_max = 2.0         #[1/s]
thrust_max = 1.5            #[N]

class ThrusterDynamics:

    def __init__(self, u, K, dt):
        self._u = u
        self._K = K
        self._dt = dt

    def saturate(self, signal, min, max):
        return np.clip(signal, min, max)

    def limit_rate(self, signal_curr, signal_prev, max):
        if (np.abs(signal_curr-signal_prev)/self._dt > max):
            signal_curr = signal_prev + abs(max * self._dt)
        return signal_curr

    def propeller_rev(self):
        n = np.sign(self._u) * np.sqrt(np.abs(self._u))
        return n
    
    def compute_tau(self, n):
        tau = self._K @ np.abs(n) @ n
        return tau
    
    def get_tau(self):
        n = self.propeller_rev()



