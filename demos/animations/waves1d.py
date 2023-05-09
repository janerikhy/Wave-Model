import os
import json

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.animation import FuncAnimation, PillowWriter

from MCSimPython.waves import JONSWAP, WaveLoad
from MCSimPython.simulator import RVG_DP_6DOF
from MCSimPython.control.basic import PID
from MCSimPython.utils import pipi, J, Rz, Ry, Rx, Rzyx, generate_config_file

# plt.style.use('science')

fps = 50
dt = 1/fps

np.random.seed(1235)

Lpp = 33
T = 2.786

# Points in xz-coordinates
points = np.array([
    [Lpp/2, T],
    [10, T],
    [7, T+2],
    [3, T+2],
    [3, T+3],
    [5, T+5],
    [2, T+5.3],
    [-4.5, T + 5.3],
    [-4., T+3],
    [-4, T],
    [-Lpp/2, T],
    [-Lpp/2, -T + 0.7],
    [-Lpp/2 + 1.3, -T+.5],
    [-Lpp/2 + 2, -T],
    [-Lpp/2+6, -T - 0.1],
    [0, -T-0.1],
    [Lpp/2 - 6, -T],
    [Lpp/2-3, -T],
    [Lpp/2 -2, -T + 0.1],
    [Lpp/2 - 1, -T + 0.25],
    [Lpp/2 - 0.8, -T + 0.6],
    [Lpp/2 -1, -T + 0.95],
    [Lpp/2 - 2, -T + 1.0],
    # [Lpp/2 - 2, -T + 1.8],
    # [Lpp/2, -T],
    [Lpp/2, T]
])

points[:, 1] = -points[:, 1]

hs = 2.9
tp = 9.0
wp = 2*np.pi/tp

g = 9.81

wmin = wp/2
wmax = 3.*wp

N = 4
w = np.linspace(wmin, wmax, N)
k = w**2/g

dw = (wmax-wmin)/N

Nx = 1000
xmin = -24
xmax = 24
x = np.linspace(xmin, xmax, Nx)

jonswap = JONSWAP(w)
_, spectrum = jonswap(hs, tp, gamma=3.3)


wave_amp = np.sqrt(2*spectrum*dw)
eps = np.random.uniform(0, 2*np.pi, size=N)
beta = np.ones(N)*np.pi     # Waves moving south (positive x-direction)

vessel = RVG_DP_6DOF(dt, config_file="rvg_config.json", method="RK4")

wl = WaveLoad(
    wave_amps=wave_amp,
    freqs=w,
    eps=eps,
    angles=beta,
    config_file=vessel._config_file,
    interpolate=True
)

motionRAOamp = np.asarray(wl._params['motionRAO']['amp'])[:, :, :, 0]
motionRAOphase = np.asarray(wl._params['motionRAO']['phase'])[:, :, :, 0]
rao_freqs = np.asarray(wl._params['motionRAO']['w'])

rao_amp_1 = np.interp(w, rao_freqs, -motionRAOamp[0, :, 0])
rao_phase_1 = np.interp(w, rao_freqs, pipi(np.abs(motionRAOphase[0, :, 0])))

rao_amp_3 = np.interp(w, rao_freqs, -motionRAOamp[2, :, 0])
rao_phase_3 = pipi(np.abs(np.interp(w, rao_freqs, motionRAOphase[2, :, 0])))

rao_amp_5 = np.interp(w, rao_freqs, motionRAOamp[4, :, 0])
rao_phase_5 = pipi(np.abs(np.interp(w, rao_freqs, motionRAOphase[4, :, 0])))

plt.figure()
plt.plot(w, rao_amp_1)
plt.plot(w, wl._forceRAOamp[0, :, 0]/np.max(wl._forceRAOamp[0, :, 0]), 'x--')
plt.plot(w, rao_amp_3)
plt.plot(w, wl._forceRAOamp[2, :, 0]/np.max(wl._forceRAOamp[2, :, 0]), 'x--')
plt.plot(w, rao_amp_5)
plt.plot(w, wl._forceRAOamp[4, :, 0]/np.max(wl._forceRAOamp[4, :, 0]), 'x--')

plt.figure()
plt.plot(w, rao_phase_1)
plt.plot(w, wl._forceRAOphase[0, :, 0], 'x--')
plt.plot(w, rao_phase_3)
plt.plot(w, wl._forceRAOphase[2, :, 0], 'x--')
plt.plot(w, rao_phase_5)
plt.plot(w, wl._forceRAOphase[4, :, 0], 'x--')

plt.show()


fig = plt.figure(figsize=(20, 6))
l, = plt.plot([], [], 'b-')
# l2, = plt.plot([], [], 'o--', lw=1.5)
# l3, = plt.plot([], [], 'x-', lw=1.2)
l4, = plt.plot([], [], 'r-', lw=1.4)
l5, = plt.plot([], [], 'o--', lw=1.5)

plt.xlim(xmin, xmax)
plt.ylim(-3.5*T, 2.*T)
plt.xlabel("$x \; [m]$")
plt.ylabel("$\zeta (t, x) \; [m]$")
plt.gca().invert_yaxis()

def wave(t):
    return np.sum(wave_amp*np.cos(w*t - k*np.cos(beta)*x[:, None]-eps), axis=1)

def motion(t):
    surge = np.sum(rao_amp_1*wave_amp*np.cos(w*t - k*np.cos(beta)*0 - eps - rao_phase_1)) 
    heave = np.sum(rao_amp_3*wave_amp*np.cos(w*t - k*np.cos(beta)*0 - eps - rao_phase_3))
    pitch = np.sum(rao_amp_5*wave_amp*np.cos(w*t - k*np.cos(beta)*0 - eps - rao_phase_5))
    return surge, heave, pitch

def transform(x, z, eta_1, eta_3, eta_5):
    return np.array([eta_1 + z*eta_5, eta_3 - x*eta_5])


metadata = dict(title="Wave", artist="Hygen")
writer = PillowWriter(fps=fps, metadata=metadata)

tmax = 30
time = np.arange(0, tmax, 1/fps)


sim_eta = [points]
x_cg_eta = [np.array([-1.38, 0.])]

hp = np.ix_([0,2],[0,2])
for i, t in enumerate(time):
    pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, vessel.get_eta()[-1]])
    tau_wf = wl.first_order_loads(t, vessel.get_eta())
    tau_wf[1] = 0.
    tau_wf[3] = 0.
    tau_wf[5] = 0.

    vessel.integrate(0.0, 0.0, tau_wf)
    eta = vessel.get_eta()

    # Try to transform from NED frame to classical xyz-plane (with z positive up)
    # sim_eta.append(points + [transform(points[i, 0], points[i, 1], eta[0], eta[2], -eta[4]) for i in range(len(points))])
    sim_eta.append(np.array([eta[0], eta[2]]) + [Ry(eta[4])[hp]@point for point in points])
    x_cg_eta.append(transform(x_cg_eta[0][0], x_cg_eta[0][1], eta[0], eta[2], -eta[4]))


with writer.saving(fig, os.path.join(os.path.dirname(__file__),"wave_motion1d_head_sea.gif"), 100):
    for j, t in enumerate(time):
        print(f"Time:= {t:.2f}")
        zeta = wave(t)
        surge, heave, pitch = motion(t) 
        plt.title(f"t = {t:2f} [s]")

        s = [transform(points[i, 0], points[i, 1], surge, heave, pitch) for i in range(len(points))]
        cg = transform(0, 0, surge, heave, pitch)
        l.set_data(x, zeta)
        # l2.set_data(cg[0], cg[1])
        # l3.set_data((points + s)[:, 0], (points + s)[:, 1])
        l4.set_data(sim_eta[j][:, 0], sim_eta[j][:, 1])
        l5.set_data(x_cg_eta[j][0], x_cg_eta[j][1])

        writer.grab_frame()