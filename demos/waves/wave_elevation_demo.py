# Wave elevation demo for fixed point in unidirectional sea.
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)

from MCSimPython.waves.wave_spectra import JONSWAP

# Set plot parameters
plt.rcParams.update({
    'figure.figsize': (12, 12),
    'font.size': 14,
    'lines.linewidth': 1.5,
    'axes.grid': True    
})

# Sea state parameters
hs = 2.     # Significant wave heihgt
tp = 9.     # Peak frequency of wave spectrum
N = 100      # Number of wave components

# Frequency for wave spectra
wp = 2*np.pi/tp
wmin = wp/2
wmax = 2*wp
dw = (wmax - wmin)/N

w = np.linspace(wmin, wmax, N)

# Initialize wave spectrum object
jonswap = JONSWAP(w)

# Generate a spectrum for the sea state
freq, spectrum = jonswap(hs, tp, gamma=3.3)

# Create wave realization
t = np.arange(0, 1000, 0.1)
wave = jonswap.realization(t, hs=hs, tp=tp, gamma=3.3)


# Plot the wave spectrum and realization
fig, ax = plt.subplots(3, 1)

plt.sca(ax[0])
plt.plot(freq, spectrum, label="$JONSWAP$")
plt.xlabel(r"$\omega \; \left[\frac{rad}{s}\right]$")
plt.ylabel(r"$S(\omega) \; \left[\frac{m^2}{s}\right]$")
plt.vlines(x=freq, ymin=np.zeros_like(freq), ymax=spectrum, color="red", linestyle='--')
plt.xlim(wmin, wmax)
plt.ylim(0, np.max(spectrum)+.5)
plt.legend()

plt.sca(ax[1])
plt.plot(t, wave)
plt.ylabel(r"$\zeta(t, x=0) \; [m]$")
plt.xlabel(r"$t \; [s]$")
plt.xlim(t[0], t[-1])
plt.ylim(-1.5*hs, 1.5*hs)


plt.sca(ax[2])
plt.plot(t[t<100], wave[t < 100], label=r"$\zeta(t) \;, \; t < 100 \, [s]$")
plt.xlim(0, 100)
plt.ylim(-1.5*hs, 1.5*hs)
plt.ylabel(r"$\zeta(t, x=0) \; [m]$")
plt.xlabel(r"$t \; [s]$")
plt.legend()
plt.show()

