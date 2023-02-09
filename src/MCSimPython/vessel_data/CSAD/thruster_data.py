import numpy as np

# ---THRUSTER DATA FOR CSAD---

# ---AZIMUTH THRUSTERS---
propellerDiameter = 0.03    #[m]
n_dot_max = 5.0             #[1/s^2]
alpha_dot_max = 2.0         #[1/s]
thrust_max = 1.5            #[N]

# ---THRUST CONFIGURATION---
lx = np.array([1.0678, 0.9344, 0.9344, -1.1644, -0.9911, -0.9911])
ly = np.array([0.0, 0.11, -0.11, 0.0, -0.1644, 0.1644])

# ---THRUST COEFFICIENTS---
K = np.array([0.3763, 0.3901, 0.3776, 0.5641, 0.4799, 0.5588])