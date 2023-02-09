import numpy as np
import MCSimPython.vessel_data.CSAD.thruster_data as data 


class ThrusterDynamics:

    def __init__(self, u, alpha, dt):
        self._u = u
        self._alpha = alpha
        self._dt = dt

        self._K = np.diag(data.K)
        self._lx = data.lx
        self._ly = data.ly
        self._n_thrusters = len(data.lx)

    def saturate(self, signal, min, max):

        """
        Saturates a given signal with both an upper and lower bound.

        Parameters
        -----------
        signal : array_like or scalar
            To be saturated.

        Returns
        --------
        array_like or scalar
            Saturated signal
        """

        return np.clip(signal, min, max)

    def limit_rate(self, signal_curr, signal_prev, max):

        """
        Limits the rate of change of the signal with respect to a max rate.
        Signal is a vector with the same dimension as number of thrusters.

        POSSIBLE EXTENTIONS:
            - at the time being, each thruster will need to have the same rate limit,
              better to exclude the loop?

        Parameters
        -----------
        signal_curr : array_like
            Current signal.
        signal_prev : array_like
            Signal prom frevious time step.
        max : scalar
            Max rate of change for signal

        Returns
        --------
        array_like
            Rate limited signal
        """

        for i in range(self._n_thrusters):
            if (np.abs(signal_curr[i]-signal_prev[i])/self._dt > max):
                signal_curr[i] = signal_prev[i] + abs(max * self._dt)

        return signal_curr

    def compute_n(self, u):

        """
        Calculates the propeller revolution number from the control signal.
        To be used if n needs to be saturated/rate limited.

        CHECK:
            - is this the correct way to multiply
        
        Parameters
        -----------
        u : array_like
            Control signal.

        Returns
        --------
        n : array_like
            propeller revolution number.
        """

        n = np.sign(u) * np.sqrt(np.abs(u))

        return n
    
    def compute_u(self, n):

        """
        Calculates the the control input from the propeller revolution number.
        Use this if you have been saturating n.

        CHECK:
            - is this the correct way to multiply

        Parameters
        -----------
        n : array_like
            propeller revolution number.

        Returns
        --------
        u : array_like
            Control signal.
        """

        u = n * np.abs(n)

        return u 

    def compute_actuator_loads(self, u):

        """
        Computes load on each actuator from the control inputs.

        Parameters
        -----------
        u : array_like
            Control signal.

        Returns
        --------
         : array_like
            Actuator loads.

        """
        
        return self._K @ u

    def get_tau(self, actual_actuator_loads, actual_alpha):

        """
        Computes resulting thrust and moments in surge, sway and yaw from the actuators.
        
        Parameters
        -----------
        actual_actuator_loads : array_like
            The value of the actuator loads after rate limitation and saturation.
        alpha_actual : array_like
            The angle of the azimuth thrusters after rate limitations and zone restriction.

        Returns
        --------
        tau : array_like
            Thrust and moments in surge, sway and yaw.
        """

        
        F_x = np.sum(actual_actuator_loads * np.cos(actual_alpha))
        F_y = np.sum(actual_actuator_loads * np.sin(actual_alpha))
        M_r = np.sum(self._lx * actual_actuator_loads * np.sin(actual_alpha)) - np.sum(self._ly * actual_actuator_loads * np.cos(actual_alpha))
        
        tau = np.array([F_x, F_y, M_r])

        return tau



        



