import numpy as np
import matplotlib.pyplot as plt

from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.simulator.csad import CSAD_DP_6DOF

width = 426.8       # Latex document width in pts
inch_pr_pt = 1/72.27        # Ratio between pts and inches

golden_ratio = (np.sqrt(5) - 1)/2
fig_width = width*inch_pr_pt
fig_height = fig_width*golden_ratio
fig_size = [fig_width, fig_height]

params = {#'backend': 'PS',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.grid': True,
          'text.usetex': True,
          'figure.figsize': fig_size} 

plt.rcParams.update(params)

vessel = CSAD_DP_6DOF(0.01)

N = 3       # Number of wave components
heading = 0
np.random.seed(1)

if N == 1:
    wave_amps = np.array([1.0])
    wave_freqs = np.array([0.8])

elif N == 2:
    wave_amps = np.array([1.0, 0.8])
    wave_freqs = np.array([0.8, 0.7])

elif N == 3:
    wave_amps = np.array([0.5, 1.0, 0.8])
    wave_freqs = np.array([0.4, 0.25, 0.7])

eps = np.random.uniform(0, 2*np.pi, size=N)
wave_angles = np.ones(N)*(-np.pi/2)

wave_load = WaveLoad(
    wave_amps,
    wave_freqs,
    eps,
    wave_angles,
    config_file=vessel._config_file,
    depth=1.5,
    deep_water=False
)

wl2 = WaveLoad(
    wave_amps,
    wave_freqs,
    eps,
    wave_angles,
    config_file=vessel._config_file,
    depth=1.5*1e3,
    deep_water=True
)

print(wave_load._k)
print(wl2._k)

dt = 0.01
time = np.arange(0, 200, dt)

tau_wf_1 = np.zeros((len(time), 6))
tau_wf_2 = np.zeros_like(tau_wf_1)


for i in range(len(time)):
    tau_wf_1[i] = wave_load.first_order_loads(time[i], vessel.get_eta())
    tau_wf_2[i] = wave_load.second_order_loads(time[i], vessel.get_eta()[-1])



# Plot wave elevation as function of time
# plt.figure(figsize=(12, 4))
# plt.plot(time, np.sum(wave_amps*np.cos(wave_freqs*time[:, None] + eps), axis=1), label=r"$\zeta$")
# plt.xlabel("$t \; [s]$")
# plt.ylabel("$\zeta (t, x=0, y=0) \; [m]$")
# plt.xlim(time[0], time[-1])
# plt.show()

# Plot 1st order wave loads
fig, axs = plt.subplots(3, 1, constrained_layout=True)
fig.suptitle("1st Order Wave Loads" + r", $\beta " + f"= {np.rad2deg(wave_angles[0])}$")

plt.sca(axs[0])
plt.title("Surge")
plt.plot(time, tau_wf_1[:, 0])
plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^1 \; [N]$")

plt.sca(axs[1])
plt.title("Sway")
plt.plot(time, tau_wf_1[:, 1])
plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^2 \; [N]$")

plt.sca(axs[2])
plt.title("Yaw")
plt.plot(time, tau_wf_1[:, 5])
plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^6 \; [Nm]$")

#plt.show()

# Plot 2nd Order Wave Loads
fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle("2nd Order Wave Loads" + r", $\beta " + f"= {np.rad2deg(wave_angles[0])}$")

plt.sca(axs[0])
plt.plot(time, tau_wf_2[:, 0])
#plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^1 \; [N]$")
#plt.ylim(np.min(tau_wf_2), np.max(tau_wf_2))
plt.ylim(-.05, .05)

plt.sca(axs[1])
#plt.title("Sway")
plt.plot(time, tau_wf_2[:, 1])
# plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^2 \; [N]$")
#plt.ylim(np.min(tau_wf_2), np.max(tau_wf_2))
plt.ylim(-.05, .05)

plt.sca(axs[2])
#plt.title("Yaw")
plt.plot(time, tau_wf_2[:, 5])
plt.xlabel("$t \; [s]$")
plt.xlim(time[0], time[-1])
plt.ylabel(r"$\tau_{sv}^6 \; [Nm]$")
#plt.ylim(np.min(tau_wf_2), np.max(tau_wf_2))
plt.ylim(-.05, .05)

#plt.savefig(f'second_order_wave_loads_N{N}.eps')
#plt.show()

fig, axs = plt.subplots(2, 1, sharex=True)
plt.sca(axs[0])
plt.plot(time, np.sum(wave_amps*np.cos(wave_freqs*time[:, None] + eps), axis=1))
plt.ylabel("$\zeta(t) \; [m]$")

plt.sca(axs[1])
plt.plot(time, tau_wf_2[:, 0], label=r"$\tau_{sv}^1$")
plt.plot(time, tau_wf_2[:, 1], label=r"$\tau_{sv}^2$")
plt.plot(time, tau_wf_2[:, 5], label=r"$\tau_{sv}^6$")
plt.ylabel("Load")
plt.xlabel("$t \; [s]$")
plt.legend(ncol=3)
plt.xlim(time[0], time[-1])
plt.ylim(-0.05, 0.05)
#plt.savefig(f"2nd_wave_loads_and_realization_N{N}.eps")
plt.show()