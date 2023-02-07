import numpy as np
import matplotlib.pyplot as plt

from MCSimPython.simulator.csad import CSADMan3DOF
from MCSimPython.control.basic import PD, PID
from MCSimPython.guidance.filter import ThrdOrderRefFilter
from MCSimPython.observer.nonlinobs import NonlinObs3dof
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP

from MCSimPython.utils import dof3_array, Rz

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})

# Simulation parameters

simtime = 50
dt = 0.01
t = np.arange(0, simtime, dt)

# Define the sea state
N = 25              # Number of wave components
hs = 0.03           # Significant wave height
tp = 1.0            # Peak period
wp = 2*np.pi/tp     # Peak frequency
wmin = wp/2
wmax = 2.5*wp
dw = (wmax-wmin)/N

w = np.linspace(wmin, wmax, N)


jonswap = JONSWAP(w)
freq, spec = jonswap(hs, tp, gamma=1.8)

wave_amps = np.sqrt(2*spec*dw)
eps = np.random.uniform(0, 2*np.pi, size=N)
wave_dir = np.deg2rad(np.random.normal(25, 10/3, size=N))
wave_dir = np.ones(N)*np.deg2rad(170)

# Vessel
vessel = CSADMan3DOF(dt, method="RK4")

# Observer
Mobs = np.diag(vessel._M.diagonal())        # Observer mass matrix
Dobs = np.diag(vessel._D.diagonal())        # Observer damping matrix
T = 100                                   # Time constant for bias term

w0 = wp
wc = 1.1*wp

observer = NonlinObs3dof(dt, wc, w0, 0.05, T, Mobs, Dobs)

# Wave Loads
waveload = WaveLoad(wave_amps, w, eps, wave_dir, config_file=vessel._config_file)

# Ocean Current
Uc = 0.02
betac = 4/5*np.pi

# Set up a very simple reference model 
eta_ref = np.zeros((len(t), 3))
t_cond = t > 50
t_cond2 = t > 200
eta_ref[t_cond, 0] = 5
eta_ref[t_cond2, 1] = 2

ref_model = ThrdOrderRefFilter(dt, omega=[.01, .01, .01])

# Set up controller
controller = PD(kp=[10., 10., 50.], kd=[100., 100., 160.])
controller = PID(kp=[10., 10., 50.], kd=[100., 100., 160.], ki=[.2, .2, .6])


# Simulate the vessel:
eta = np.zeros((len(t), 3))   # Store vessel position
nu = np.zeros_like(eta)

eta_hat = np.zeros_like(eta)
nu_hat = np.zeros_like(eta)
b_hat = np.zeros_like(eta)
y_hat = np.zeros_like(eta)

x_r = np.zeros((len(t), 9))
error = np.zeros_like(eta)

tau_cmd = np.zeros(3)

for i in range(1, len(t)):
    # Vessel heading
    psi = vessel.get_eta()[-1]

    # Get the reference
    ref_model.set_eta_r(eta_ref[i])
    ref_model.update()
    eta_d = ref_model.get_eta_d()
    eta_d_dot = ref_model.get_eta_d_dot()
    nu_d = Rz(psi).T@eta_d_dot      # convert desired velocity to body frame
    x_r[i] = ref_model._x

    # Calculate the first order wave loads
    tau_wf = waveload.first_order_loads(t[i], vessel.get_eta())[dof3_array]
    tau_sv = waveload.second_order_loads(t[i], vessel.get_eta()[-1])

    # tau_wf = np.zeros(3)
    # Compute the tau_cmd
    # tau_cmd = controller.get_tau(vessel.get_eta(), eta_d, vessel.get_nu(), nu_d)
    tau_cmd = controller.get_tau(observer.eta, eta_d, observer.nu, nu_d)
    # Integrate / update the vessel dynamic
    vessel.integrate(Uc, betac, tau_cmd + tau_wf)
    eta[i] = vessel.get_eta()
    nu[i] = vessel.get_nu()

    # Observer update
    observer.update(vessel.get_eta(), tau_cmd)

    eta_hat[i] = observer.eta
    nu_hat[i] = observer.nu
    b_hat[i] = observer.bias
    y_hat[i] = observer._y_hat
    error[i] = vessel.get_eta() - observer._y_hat

# Plot the response 
plt.figure()
plt.title("Vessel Position (NED)")
plt.plot(eta[:, 1], eta[:, 0], label="$\eta$")
plt.plot(eta_hat[:, 1], eta_hat[:, 0], label="$\hat{\eta}$")
plt.plot(x_r[:, 1], x_r[:, 0], label="$\eta_d$")
plt.legend()
plt.xlabel("$E \; [m]$")
plt.ylabel("$N \; [m]$")


fig, ax = plt.subplots(2, 3, sharex=True)
plt.sca(ax[0, 0])
plt.plot(t, eta[:, 0], label="$\eta_1$")
plt.plot(t, eta_hat[:, 0], label="$\hat{\eta}_1$")
plt.plot(t, y_hat[:, 0], label="$\hat{y}$")
plt.plot(t, x_r[:, 0], label="$\eta_{d, 1}$")
plt.legend()

plt.sca(ax[0, 1])
plt.plot(t, eta[:, 1], label="$\eta_2$")
plt.plot(t, eta_hat[:, 1], label="$\hat{\eta}_2$")
plt.plot(t, x_r[:, 1], label="$\eta_{d, 2}$")
plt.legend()

plt.sca(ax[0, 2])
plt.plot(t, eta[:, 2], label="$\eta_6$")
plt.plot(t, eta_hat[:, 2], label="$\hat{\eta}_6$")
plt.plot(t, x_r[:, 2], label="$\eta_{d, 6}$")
plt.legend()

plt.sca(ax[1, 0])
plt.plot(t, nu[:, 0], label=r"$\nu_1$")
plt.plot(t, nu_hat[:, 0], label=r"$\hat{\nu}_1$")
plt.legend()

plt.sca(ax[1, 1])
plt.plot(t, nu[:, 1], label=r"$\nu_2$")
plt.plot(t, nu_hat[:, 1], label=r"$\hat{\nu}_2$")
plt.legend()

plt.sca(ax[1, 2])
plt.plot(t, nu[:, 2], label=r"$\nu_6$")
plt.plot(t, nu_hat[:, 2], label=r"$\hat{\nu}_6$")
plt.legend()

plt.xlabel("$t \; [s]$")

plt.figure()
plt.plot(t, b_hat[:, 0], label="$b_1$")
plt.plot(t, b_hat[:, 1], label="$b_2$")
plt.plot(t, b_hat[:, 2], label="$b_3$")
plt.legend()


plt.figure()
plt.plot(t, error[:, 0], label="$e_1$")
plt.plot(t, error[:, 1], label="$e_2$")
plt.plot(t, error[:, 2], label="$e_3$")
plt.legend()

plt.show()