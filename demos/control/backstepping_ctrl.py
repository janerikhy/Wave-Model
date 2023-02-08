import numpy as np
import matplotlib.pyplot as plt

from MCSimPython.control.backstepping import BacksteppingController
from MCSimPython.simulator.csad import CSADMan3DOF
from MCSimPython.guidance.path_param import WayPointsRefModel
from MCSimPython.utils import pipi

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})

# Simulation parameters
simtime = 700
dt = 0.01
t = np.arange(0, simtime, dt)

# Reference model

wps = np.array([
    [0, 0],
    [5, 0],
    [5, 5],
    [0, 5],
    [0, 0]
])

ref_model = WayPointsRefModel(wps, slope=0.8, ds=0.1)

# Vessel simulation model with Runge-Kutta 4 integration method.
vessel = CSADMan3DOF(dt, method="RK4")

# Vessel control model matrices
Mobs = vessel._M.diagonal()
Dobs = vessel._D.diagonal()

# Backstepping controller

K1 = np.diag([.3, .3, .2])
K2 = np.diag([15, 15, 10])
control = BacksteppingController(Mobs, Dobs, K1, K2)

# Path parametrization variable
theta = 0

ss_dot = np.zeros(len(t))
uss = np.zeros_like(ss_dot)

u_ref = .05     # Reference speed

# Create a sinus desired speed
u_d = u_ref*np.sin(2*np.pi*1/(2*simtime)*t)

#u_d_dot = (u_d[1] - u_d[0])/dt
mu = 2e-6

# Store the desired path
desired_path = np.zeros((3, len(t)))
desired_path[:, 0] = eta_d = np.array([ref_model.pd(0)[0], ref_model.pd(0)[1], np.arctan2(ref_model.pd_s(0)[1], ref_model.pd_s(0)[0])])

eta = np.zeros((len(t), 3))
nu = np.zeros((len(t), 3))

for i in range(1, len(t)):
    desired_path[:, i] = eta_d = ref_model.eta_d(theta, eta_d[-1])
    eta_d_s = ref_model.eta_d_s(theta)
    eta_d_s2 = ref_model.eta_d_s2(theta)
    uss[i] = u_s = u_d[i]/(np.linalg.norm(eta_d_s[:2]))
    u_d_dot = u_d[i] - u_d[i-1]
    ddt_u_s = u_d_dot/(np.linalg.norm(eta_d_s[:2]))
    u_s_dot = -(eta_d_s[0]*eta_d_s2[0] + eta_d_s[1] *
                eta_d_s2[1])/(eta_d_s[0]**2 + eta_d_s[1]**2)**(3/2)*u_ref
    tau = control.u(vessel.get_eta(), vessel.get_nu(), eta_d,
                    eta_d_s, eta_d_s2, mu, u_s, u_s_dot, ddt_u_s)
    tau = np.clip(tau, -40, 40) # saturate the allowed commaned load
    ss_dot[i] = control._s_dot
    vessel.integrate(0.0, 0.0, tau)
    eta[i] = vessel.get_eta()
    nu[i] = vessel.get_nu()

    theta += ss_dot[i]*dt


error_1 = eta[:, 0] - desired_path[0, :]
error_2 = eta[:, 1] - desired_path[1, :]
error_3 = pipi(eta[:, 2] - desired_path[2, :])

plt.figure()
plt.title("N-E Plot")
plt.plot(eta[:, 1], eta[:, 0], label="$\eta$")
plt.plot(desired_path[1, :], desired_path[0, :], label="$\eta_d$")
plt.xlabel("E")
plt.ylabel("N")
plt.legend()

fig, ax = plt.subplots(3, 1, sharex=True)
plt.sca(ax[0])
plt.plot(t, nu[:, 0], label=r"$\nu_1$")
plt.legend()

plt.sca(ax[1])
plt.plot(t, nu[:, 1], label=r"$\nu_2$")
plt.legend()

plt.sca(ax[2])
plt.plot(t, nu[:, 2], label=r"$\nu_3$")
plt.legend()

plt.xlabel("$t \; [s]$")

plt.figure()
plt.title("Reference velocity")
plt.plot(t, u_d)


fig, ax = plt.subplots(3, 1, sharex=True)
plt.sca(ax[0])
plt.plot(t, error_1, label=r"$\bar{\eta}_1$")
plt.legend()
plt.sca(ax[1])
plt.plot(t, error_2, label=r"$\bar{\eta}_2$")
plt.legend()
plt.sca(ax[2])
plt.plot(t, np.rad2deg(error_3), label=r"$\bar{\eta}_3$")
plt.legend()
plt.xlabel("$t \; [s]$")
plt.show()