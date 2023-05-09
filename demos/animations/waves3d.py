import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from matplotlib.colors import LightSource

from MCSimPython.waves import JONSWAP, WaveLoad
from MCSimPython.simulator import CSAD_DP_6DOF, RVG_DP_6DOF

from MCSimPython.utils import Rzyx, J

import os

fullscale = True

# Simulation and visualization params
if fullscale:
    fps = 20
else:
    fps = 45

dt = 1/fps

tmax = 50
tmin = 0
time = np.arange(tmin, tmax, dt)

metadata = dict(title="Wave", artist="Hygen")
writer = PillowWriter(fps=fps, metadata=metadata)



# Vessel coordinates
if fullscale:
    Lpp = 33
    B = 9.6
    T = 2.786
else:
    Lpp = 2.578
    B = 0.3
    T = 0.2

# Points in xz-coordinates
# points = np.array([
#     [Lpp/2, T],
#     [10, T],
#     [7, T+2],
#     [3, T+2],
#     [3, T+3],
#     [5, T+5],
#     [2, T+5.3],
#     [-4.5, T + 5.3],
#     [-4., T+3],
#     [-4, T],
#     [-Lpp/2, T],
#     [-Lpp/2, -T + 0.7],
#     [-Lpp/2 + 1.3, -T+.5],
#     [-Lpp/2 + 2, -T],
#     [-Lpp/2+6, -T - 0.1],
#     [0, -T-0.1],
#     [Lpp/2 - 6, -T],
#     [Lpp/2-3, -T],
#     [Lpp/2 -2, -T + 0.1],
#     [Lpp/2 - 1, -T + 0.25],
#     [Lpp/2 - 0.8, -T + 0.6],
#     [Lpp/2 -1, -T + 0.95],
#     [Lpp/2 - 2, -T + 1.0],
#     # [Lpp/2 - 2, -T + 1.8],
#     # [Lpp/2, -T],
#     [Lpp/2, T]
# ])

L = Lpp/2
H = T
scale = 3/7
# 3d Vessel points
points = np.array([
    # Four points in the back
    [-L, -B, -H],
    [-L, -B, H],
    [-L, B, H],
    [-L, B, -H],
    # Four points at 2/3 of lenght
    [L*scale, B, -H],
    [L*scale, B, H],
    [L*scale, -B, H],
    [L*scale, -B, -H],
    # Front of vessel
    [L, 0, -H],
    [L, 0, H],
    # Now go back to complete the figure
    [L*scale, B, H],
    [L*scale, B, -H],
    [L, 0, -H],
    [L*scale, -B, -H],
    [-L, -B, -H],
    [-L, -B, H],
    [L*scale, -B, H],
    [L, 0, H],
    [L*scale, B, H],
    [-L, B, H]
])
points[:, 2] = -points[:, 2]

# Vessel and sea state
if fullscale:
    vessel = RVG_DP_6DOF(dt, method="RK4", config_file="rvg_config.json")
    hs = 2.5
    tp = 12.0
else:
    vessel = CSAD_DP_6DOF(dt, method="RK4")
    hs = 0.06
    tp = 1.9

wp = 2*np.pi/tp

N = 100

wmin = wp/2.
wmax = wp*3.

dw = (wmax - wmin)/N

w = np.linspace(wmin, wmax, N, endpoint=True)
k = w**2/9.81

Nx = 250
Ny = 250

xlim, ylim = 3.5*L, 2.5*B
ylim = xlim
x = np.linspace(-xlim, xlim, Nx)
y = np.linspace(-ylim, ylim, Ny)

X, Y = np.meshgrid(x, y)

jonswap = JONSWAP(w)
_, spectrum = jonswap(hs, tp, gamma=3.3)

wave_amps = np.sqrt(2*spectrum*dw)
wave_angle = -np.pi*np.ones(N)
# wave_angle = np.random.normal(np.pi/2, np.pi/8/3, size=N)  # Waves with some spreading, going West
eps = np.random.uniform(0, 2*np.pi, size=N)

waveload = WaveLoad(
    wave_amps,
    freqs=w,
    eps=eps,
    angles=wave_angle,
    config_file=vessel._config_file,
    interpolate=True,
    qtf_method="geo-mean"
)

def wave_elevation(t):
    zeta = 0
    for i in range(N):
        zeta += wave_amps[i]*np.cos(w[i]*t - k[i]*(X*np.cos(wave_angle[i])+Y*np.sin(wave_angle[i])) - eps[i])
    return zeta

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(12, 12))

hps = np.ix_([0, 1, 2], [0, 1, 2])
ls = LightSource(azdeg=0, altdeg=60)

vessel.set_eta(np.array([17., 0., 0., 0., 0., 0.]))

# Simulate vessel for some seconds to remove transients
time_prior = np.arange(-100, 0, dt)
for i in range(time_prior.size):
    tau_wf = waveload.first_order_loads(time_prior[i], vessel.get_eta())
    vessel.integrate(0., 0., tau_wf)

with writer.saving(fig, os.path.join(os.path.dirname(__file__),"vessel_motion3d__rvg_waveangle_180.gif"), 100):
    for j, t in enumerate(time):

        # tau_wf = waveload.first_order_loads(t, vessel.get_eta())
        tau_wf = waveload(t, vessel.get_eta())
        print(f"t={t:.2f}")
        plt.suptitle(f"t={t:.2f}")
        vessel.integrate(0., 0., tau_wf)
        eta = vessel.get_eta()
        vessel_points = np.array([eta[:3] + (J(eta)[hps])@point for point in points])
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_zlim(-4.1*H, 2.1*H)
        # ax.set_zlim(-.05, .05)
        ax.invert_zaxis()
        ax.view_init(30, 30)
        wave = wave_elevation(t)
        rgb = ls.shade(wave, cmap=cm.Blues)
        ax.plot_surface(X, Y, wave, alpha=0.7, facecolors=rgb, vmin=-.7, vmax=7)
        # ax.plot_surface(X[100:150, 100:150], Y[100:150, 100:150], wave[100:150, 100:150], alpha=0.72, facecolors=rgb, vmin=-.7, vmax=7)
        ax.plot(vessel_points[:, 0], vessel_points[:, 1], vessel_points[:, 2], 'r-', linewidth=3.5)
        ax.plot(eta[0], eta[1], eta[2], 'ro', linewidth=10)

        writer.grab_frame()
        plt.cla()
