# computational_speed.py
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.grid': True
})

from simulator.csad import CSAD_DP_6DOF
from waves.wave_loads import WaveLoad

import time
from utils import timeit


def time_func(function, *args, **kwargs):
    # Return excecution time of function
    t_start = time.time()
    function(*args, **kwargs)
    t_end = time.time()
    return t_end - t_start

# Compare the computational speed of second order loads between Brørby method versus MHJE method.

# Brørby method

@timeit
def brorby_wave_load(t, wave_freqs, wave_amp, wave_phase, driftForceFreq, driftForceAmpX, driftForceAmpY, driftForceAmpPsi):
    # First calculating the drift loads
    head_indx = 0
    count = 0
    driftLoads = np.zeros(3)
    N = len(wave_freqs)
    for w in wave_freqs:
        freq_indx = np.argmin(np.abs(driftForceFreq - w))

        driftCoeff = np.array([
            driftForceAmpX[freq_indx, head_indx],
            driftForceAmpY[freq_indx, head_indx],
            driftForceAmpPsi[freq_indx, head_indx]
        ])

        driftLoads = np.add(driftLoads, np.abs(driftCoeff)*np.power(wave_amp[count], 2))
        count += 1

    # Calculating Slowly Varying loads
    xElement = 0
    yElement = 0
    head_indx = 0
    psiElement = 0

    realValue = np.zeros([3, 1])
    #for k in range(int(0.2*N), int(0.7*N), 1):
    for k in range(1, N, 1):
        freq_indx_k = np.argmin(np.abs(driftForceFreq - wave_freqs[k]))
        Q_k = np.array([
            driftForceAmpX[freq_indx_k, head_indx],
            driftForceAmpY[freq_indx_k, head_indx],
            driftForceAmpPsi[freq_indx_k, head_indx]
        ])

        for i in range(k):
            freq_indx_i = np.argmin(np.abs(driftForceFreq - wave_freqs[i]))
            Q_i = np.array([
                driftForceAmpX[freq_indx_i, head_indx],
                driftForceAmpY[freq_indx_i, head_indx],
                driftForceAmpPsi[freq_indx_i, head_indx]
            ])
            Q_ki = 0.5*(np.abs(Q_k) + np.abs(Q_i))
            element = Q_ki*wave_amp[k]*wave_amp[i]*np.exp(1j*(wave_freqs[k]-wave_freqs[i])*t - (wave_phase[k] - wave_phase[i]))
            element = np.resize(element, (3, 1))

            realValue += np.real(element)

    slowlyVarying = 2*0*realValue

    return slowlyVarying + driftLoads


if __name__ == "__main__":
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

    n = np.arange(1, 251, 1, dtype=int)
    t_brorby = np.zeros(n.shape[0])
    t_mjeh = np.zeros(n.shape[0])

    np.random.seed(1024)
    
    dt = 0.01
    vessel = CSAD_DP_6DOF(dt)

    for i in range(len(n)):
    
        wave_amps = np.random.uniform(0, 1, size=n[i])
        wave_freqs = np.linspace(0.01, 8, n[i])
        wave_angles = np.zeros(n[i])
        eps = np.random.uniform(0, 2*np.pi, size=n[i])
        
        wave_load = WaveLoad(wave_amps, wave_freqs, eps, wave_angles, vessel._config_file)
        vessel_params = wave_load._params
        
        # Prepare data for Brorby method
        driftForceFreqs = np.array(vessel_params['freqs'])
        driftForceAmps = np.array(vessel_params['driftfrc']['amp'])[:, :, :, 0]
        driftForceAmpX = driftForceAmps[0, :, :]
        driftForceAmpY = driftForceAmps[1, :, :]
        driftForceAmpPsi = driftForceAmps[2, :, :]

        M = 50  # Number of trials to average the execution time in case of errors.
        t_b_avg = np.zeros(M)
        #t_mjeh_avg = np.zeros(M)

        for j in range(M):
            # Compute and average over 10 times

            t_b_avg[j] = time_func(brorby_wave_load, *(0, wave_freqs, wave_amps, eps, driftForceFreqs, driftForceAmpX, driftForceAmpY, driftForceAmpPsi,))
            #t_mjeh_avg[j] = time_func(wave_load.second_order_loads, *(0, 0))

        t_brorby[i] = np.mean(t_b_avg)
        #t_mjeh[i] = np.mean(t_mjeh_avg)

    #np.savetxt(f't_brorby_m{M}_n{n[-1]}_new', t_brorby)
    #np.savetxt(f't_mjeh_m{M}_n{n[-1]}_new', t_mjeh)
    #plt.plot(n, t_mjeh*1e3, label="MJEH")
    plt.plot(n, t_brorby*1e3, label="Brorby")
    #plt.axhline(y=10, xmin=1, xmax=np.max(n), linestyle="--", color="r", label="simulation time step")
    plt.axhline(y=dt*1e3, linestyle="--", color="r")#, label="simulation time step")
    plt.xlabel("Number of wave components")
    plt.ylabel("Excecution time [ms]")
    plt.legend()
    plt.savefig(f"computational_time_n_brorby_{n.shape[0]}_avgOf{M}_new.eps")
    plt.show()

