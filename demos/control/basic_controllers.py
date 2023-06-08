import matplotlib.pyplot as plt
import numpy as np

#from /home/hydrolab/dev/Wave-Model/src/MCSimPython/simulator/

# spør jan om hvordan vi får tak i mc sim

from MCSimPython.simulator.csad import CSADMan3DOF
from MCSimPython.guidance.filter import ThrdOrderRefFilter
from MCSimPython.control.basic import PD, PID
from MCSimPython.utils import Rz

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})

# Simulation parameters
simtime = 800
dt = 0.01
t = np.arange(0, simtime, dt)

# Vessel
vessel = CSADMan3DOF(dt, method="RK4")
vessel_pid = CSADMan3DOF(dt, method="RK4")
control = PD(kp=[10, 10, 50.0], kd=[100.5, 100.5, 200])
control_i = PID(kp=[10, 10, 50.], kd=[100.5, 100.5, 200.], ki=[.3, .3, .4], dt=dt)
ref_model = ThrdOrderRefFilter(dt, omega=[0.2, 0.05, .02])

# Current 
uc = 0.05                   # Current velocity [m/s]
beta_c = np.deg2rad(0)      # Current direction [rad]

eta = np.zeros((len(t), 3))
nu = np.zeros((len(t), 3))
eta_pid = np.zeros_like(eta)
nu_pid = np.zeros_like(eta)
xd = np.zeros((len(t), 9))

# Set reference points for a fore corner test
set_points = np.zeros((len(t), 3))
distance = 3
t_1 = t > 50
t_2 = t > 150
t_3 = t > 300
t_4 = t > 400
t_5 = t > 600

set_points[t_1, 0] = distance
set_points[t_2, 0] = distance
set_points[t_2, 1] = distance
set_points[t_4, 0] = 0
set_points[t_4, 1] = distance
set_points[t_5, 1] = 0

# Simulate response: compare PD with PID control

for i in range(1, len(t)):
    ref_model.set_eta_r(set_points[i])
    ref_model.update()
    eta_d = ref_model.get_eta_d()
    eta_d_dot = ref_model.get_eta_d_dot()
    nu_d = Rz(vessel.get_eta()[-1]).T@eta_d_dot
    nu_d2 = Rz(vessel_pid.get_eta()[-1]).T@eta_d_dot
    xd[i] = ref_model._x

    tau_cmd = control.get_tau(vessel.get_eta(), eta_d, vessel.get_nu(), nu_d)
    tau_cmd_pid = control_i.get_tau(vessel_pid.get_eta(), eta_d, vessel_pid.get_nu(), nu_d2)

    vessel.integrate(uc, beta_c, tau_cmd)
    vessel_pid.integrate(uc, beta_c, tau_cmd_pid)
    
    eta[i] = vessel.get_eta()
    nu[i] = vessel.get_nu()
    eta_pid[i] = vessel_pid.get_eta()
    nu_pid[i] = vessel_pid.get_nu()

plt.figure(figsize=(6, 6))
plt.axis("equal")
plt.plot(xd[:, 1], xd[:, 0], 'r-', label="$\eta_d$")
plt.plot(eta[:, 1], eta[:, 0], 'k-', label="PD")
plt.plot(eta_pid[:, 1], eta_pid[:, 0], 'b--', label="PID")
#plt.plot(xd[:, 1], xd[:, 0], 'r-', label="$\eta_d$")
plt.xlabel("E [m]")
plt.ylabel("N [m]")
plt.legend()

fig, ax = plt.subplots(3, 1, sharex=True)
for i in range(3):
    plt.sca(ax[i])
    plt.plot(t, eta[:, i], label="PD")
    plt.plot(t, eta_pid[:, i], label="PID")
    plt.plot(t, xd[:, i], label="$\eta_d$")
    plt.legend()
    
fig, ax = plt.subplots(3, 1, sharex=True)
for i in range(3):
    plt.sca(ax[i])
    plt.plot(t, xd[:, i+3], label=r"$\nu_d$")
    plt.plot(t, nu[:, i], 'k-', label="PD")
    plt.plot(t, nu_pid[:, i], 'm--', label="PID")
    #plt.plot(t, xd[:, i+3], label=r"$\nu_d$")
    plt.legend()

plt.show()