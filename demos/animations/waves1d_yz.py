import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from MCSimPython.waves import JONSWAP, WaveLoad
from MCSimPython.simulator import RVG_DP_6DOF
from MCSimPython.utils import Rx

fps = 10
dt = 1/fps
simtime = 60
time = np.arange(0, simtime + dt, dt)

np.random.seed(1235)

# Set vessel parameter

B = 9.6
T = 2.786

points = np.array([
    [-B/2, -T],
    [-B/2, T],
    [B/2, T],
    [B/2, -T],
    [-B/2, -T]
])

points[:, 1] = -points[:, 1]

hs = 3.0
tp = 20.
wp = 2*np.pi/tp

g = 9.81

wmin = wp/2
wmax = 3.*wp

N = 10
w = np.linspace(wmin, wmax, N)
k = w**2/g

dw = (wmax-wmin)/N

Nx = 1000
ymin = -30
ymax = 30
y = np.linspace(ymin, ymax, Nx)

jonswap = JONSWAP(w)
_, spectrum = jonswap(hs, tp, gamma=3.3)


wave_amp = np.sqrt(2*spectrum*dw)
eps = np.random.uniform(0, 2*np.pi, size=N)
beta = np.ones(N)*(-np.pi/2)     # Waves moving East (positive y-axis)

vessel = RVG_DP_6DOF(dt, config_file="rvg_config.json", method="RK4")

wl = WaveLoad(
    wave_amps=wave_amp,
    freqs=w,
    eps=eps,
    angles=beta,
    config_file=vessel._config_file,
    interpolate=True
)

# Perform a pre-simulation to remove transient start.
time_prior = np.arange(-150, 0, dt)
for i, t in enumerate(time_prior):
    tau_wf = wl.first_order_loads(t, vessel.get_eta())
    vessel.integrate(0., 0., tau_wf)

metadata = dict(title="Wave", artist="Hygen")
writer = PillowWriter(fps=fps, metadata=metadata)

def wave(t):
    return np.sum(wave_amp*np.cos(w*t - k*np.sin(beta)*y[:, None]-eps), axis=1)


fig = plt.figure(figsize=(20, 6))
l, = plt.plot([], [], 'b-')
l1, = plt.plot([], [], 'r-')
plt.xlim(ymin, ymax)
plt.ylim(-3.*T, 3.*T)
plt.gca().invert_yaxis()

hps = np.ix_([1, 2], [1, 2])
yz = np.ix_([1, 2])


with writer.saving(fig, os.path.join(os.path.dirname(__file__),"wave_motion1d_yz_wavdir90.gif"), 100):
    for j, t in enumerate(time):
        print(f"Time:= {t:.2f}")
        zeta = wave(t)
        plt.title(f"t = {t:2f} [s]")

        tau_wf = wl(t, vessel.get_eta())
        tau_wf[0] = 0
        tau_wf[4] = 0

        vessel.integrate(0., 0., tau_wf)
        eta = vessel.get_eta()
        vessel_points = np.array([eta[yz] + Rx(eta[3])[hps]@p for p in points])

        l.set_data(y, zeta)
        l1.set_data(vessel_points[:, 0], vessel_points[:, 1])

        writer.grab_frame()