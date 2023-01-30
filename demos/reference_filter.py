from MCSimPython.guidance.filter import ThrdOrderRefFilter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})


# Simulation settings
sim_time = 600
dt = 0.01
t = np.arange(0, sim_time, dt)

# Velocity coefficients
omega = np.array([0.02, 0.02, 0.02])

ref_model = ThrdOrderRefFilter(dt)

# Set points
set_points = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.0, 50.0, 0.0]),
    np.array([50.0, 50.0, 0.0]),
    np.array([50.0, 50.0, -np.pi/4]),
    np.array([50.0, 0.0, -np.pi/4]),
    np.array([0.0, 0.0, 0.0])
]



x = np.zeros((len(t), 9))
for i in range(1, len(t)):
    if t[i] > 500:
        ref_model.set_eta_r(set_points[5])
    elif t[i] > 400:
        ref_model.set_eta_r(set_points[4])
    elif t[i] > 300:
        ref_model.set_eta_r(set_points[3])
    elif t[i] > 200:
        ref_model.set_eta_r(set_points[2])
    elif t[i] > 100:
        ref_model.set_eta_r(set_points[1])
    else:
        ref_model.set_eta_r(set_points[0])
    ref_model.update()
    x[i] = ref_model._x

plt.figure(figsize=(6, 6))
plt.title("Reference Path (NED)")
plt.plot(x[:, 1], x[:, 0])
plt.xlabel("E")
plt.ylabel("N")


# Plot the reference path
fig, ax = plt.subplots(3, 1, sharex=True)
plt.suptitle("Reference Path (NED-frame)")
plt.sca(ax[0])
plt.plot(t, x[:, 0], 'k-', label="$\eta_1$")
plt.ylabel("$N \; [m]$")
plt.legend()

plt.sca(ax[1])
plt.plot(t, x[:, 1], 'r-', label="$\eta_2$")
plt.ylabel("$E \; [m]$")
plt.legend()

plt.sca(ax[2])
plt.plot(t, np.rad2deg(x[:, 2]), 'g-', label="$\eta_3$")
plt.ylabel("$\psi \; [deg]$")
plt.legend()

# Plot reference velocity
fig, ax = plt.subplots(3, 1, sharex=True)
plt.suptitle("Reference velocity (NED-frame)")
plt.sca(ax[0])
plt.plot(t, x[:, 3], 'k-', label="$\dot{\eta}_1$")
plt.ylabel("$N \; [m/s]$")
plt.legend()

plt.sca(ax[1])
plt.plot(t, x[:, 4], 'r-', label="$\dot{\eta}_2$")
plt.ylabel("$E \; [m/s]$")
plt.legend()

plt.sca(ax[2])
plt.plot(t, np.rad2deg(x[:, 5]), 'g-', label="$\dot{\eta}_3$")
plt.ylabel("$\psi \; [deg/s]$")
plt.legend()

# Plot acceleration
fig, ax = plt.subplots(3, 1, sharex=True)
plt.suptitle("Reference Acceleration (NED-frame)")
plt.sca(ax[0])
plt.plot(t, x[:, 6], 'k-', label="$\ddot{\eta}_1$")
plt.ylabel("$N \; [m/s^2]$")
plt.legend()

plt.sca(ax[1])
plt.plot(t, x[:, 7], 'r-', label="$\ddot{\eta}_2$")
plt.ylabel("$E \; [m/s^2]$")
plt.legend()

plt.sca(ax[2])
plt.plot(t, np.rad2deg(x[:, 8]), 'g-', label="$\ddot{\eta}_3$")
plt.ylabel("$\psi \; [deg/s^2]$")
plt.legend()
plt.show()