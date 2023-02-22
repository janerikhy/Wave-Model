import numpy as np
 
class Thruster:

    def __init__(self, pos, K):
        self._r = np.array(pos)
        self._K = K
        self._u = np.array([0, 0])

    @property
    def pos_x(self):
        """
        x-position of thruster
        """
        return self._r[0]

    @property
    def pos_y(self):
        """
        y-position of thruster
        """
        return self._r[1]
    
    @property
    def K(self):
        """
        thrust coefficient of thruster
        """
        return self._K


class TunnelThruster(Thruster):

    def __init__(self, pos, max_thrust, angle):
        super().__init__(pos)

        self._max_thrust = max_thrust
        self._angle = angle

class AzimuthThruster(Thruster):

    def __init__(self, pos, max_thrust, rotation):
        super().__init__(pos)

        self._max_thrust = max_thrust
        self._rotation = rotation

