import numpy as np
import MCSimPython.vessel_data.CSAD.thruster_data as data 


class ThrusterDynamics:

    def __init__(self):
        #self._u = u
        #self._alpha = alpha
        #self._dt = dt

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

    def propeller_revolution(self, u):

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
    
    def control_input(self, n):

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

    def actuator_loads(self, u):

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
    
    def thruster_configuration(self, alpha):

        """
        Sets up the thrust configuration matrix.
        
        Parameters
        -----------
        alpha : array_like
            Azimuth angles.

        Returns
        --------
         : array_like
            Thrust configuration matrix.
        """

        return np.array([
            np.cos(alpha),
            np.sin(alpha),
            self._lx * np.sin(alpha) - self._ly * np.cos(alpha)
        ])

    def get_tau(self, u, alpha):

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
         : array_like
            Thrust and moments in surge, sway and yaw.
        """

        return self.thruster_configuration(alpha) @ self.actuator_loads(u)



        



