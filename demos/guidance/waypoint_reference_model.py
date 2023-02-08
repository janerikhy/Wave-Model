from MCSimPython.guidance.path_param import WayPointsRefModel
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
})

# Way-points
wps = np.array([
    [0, 0],
    [50, 0],
    [50, 50],
    [0, 50],
    [0, 0]
])

# Slope
L = 0.2

ds = 0.1

ref_model = WayPointsRefModel(wps, L, ds)
theta = np.arange(0, ref_model.I, 0.001)
path = ref_model.full_path(theta)
path_speed = ref_model.path_speed(theta)

plt.plot(path[:, 1], path[:, 0], label="Path")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

fig, ax = plt.subplots(3, 1, sharex=True)
plt.sca(ax[0])
plt.plot(theta, np.rad2deg(path[:, 2]), label="$\psi$")
plt.legend()
plt.sca(ax[1])
plt.plot(theta, path[:, 0], label="N")
plt.legend()
plt.sca(ax[2])
plt.plot(theta, path[:, 1], label="$E$")
plt.legend()
plt.xlim(theta[0], theta[-1])
plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)
plt.sca(ax[0])
plt.plot(theta, path[:, 0], label="$p_{d,x}$")
plt.plot(theta, path_speed[:, 0], label="$p^s_{d, x}$")
plt.legend()

plt.sca(ax[1])
plt.plot(theta, path[:, 1], label="$p_{d,y}$")
plt.plot(theta, path_speed[:, 1], label="$p^s_{d, y}$")
plt.xlabel(r"$\theta$")
plt.legend(frameon=True)
plt.show()