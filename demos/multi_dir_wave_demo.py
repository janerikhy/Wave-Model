# Multidirectional wave field demo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

print(cwd)
print(sys.path)

from MCSimPython.waves.wave_spectra import DirectionalSpectrum, JONSWAP

# Set plot parameters
width = 426.8       # Latex document width in pts
inch_pr_pt = 1/72.27        # Ratio between pts and inches

golden_ratio = (np.sqrt(5) - 1)/2
fig_width = width*inch_pr_pt
fig_height = fig_width*golden_ratio
fig_size = [fig_width, fig_height]

params = {'backend': 'PS',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size} 

plt.rcParams.update(params)

# Set sea-state parameters
hs = 2.
tp = 9.
gamma = 3.3
theta_p = 0     # Dominant wave direction

# Frequencies of wave spectra
wp = 2*np.pi/tp
wmin = .6*wp
wmax = 2.5*wp
N = 200         # Number of wave components

w = np.linspace(wmin, wmax, N)

# Generate 1D wave spectrum

jonswap = JONSWAP(w)
freq, spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)

# Angles in directional spreading function 
N_theta = 100
angles = np.linspace(-np.pi, np.pi, N_theta)
s = 1       # Peakness of spreading function

rand_seed = 12345   # Random seed

dir_spectrum = DirectionalSpectrum(freq, angles, theta_p, spectrum, s=s, seed=rand_seed)
FREQ, THETA, spectrum_2d = dir_spectrum.spectrum2d()


spherical = True
if spherical:
    X1, Y1 = FREQ*np.cos(THETA), FREQ*np.sin(THETA)
else:
    X1, Y1 = FREQ, THETA
# Create a full wave field for a mesh grid
Nx = 100    # Number of discrete x locations
Ny = 100    # Number of discrete y locations

x = np.linspace(-100, 100, Nx)
y = np.linspace(-100, 100, Ny)

X, Y, wave_elevation = dir_spectrum.wave_realization(0, x=x, y=y)

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.view_init(40, 40)
ax.plot_surface(X1, Y1, spectrum_2d, cmap=cm.coolwarm)

# plt.show()

# fig = plt.figure()
ax2 = fig.add_subplot(122, projection="3d")
ax2.view_init(40, 40)
ax2.plot_surface(X, Y, wave_elevation, cmap=cm.coolwarm)
ax2.set_zlim(-1.5*hs, 1.5*hs)
#ax2.set_zlabel("$\zeta_a \; [m]$")
#ax2.set_ylabel("$y \; [m]$")
#ax2.set_xlabel("$x \; [m]$")
plt.savefig("multi_dir_wave_realization.eps")
plt.show()
