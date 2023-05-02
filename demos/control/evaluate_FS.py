# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
# MCSimPython library
from MCSimPython import simulator as sim
from MCSimPython import control as ctrl
from MCSimPython import guidance as ref
from MCSimPython import observer as obs
from MCSimPython import thrust_allocation as ta
from MCSimPython import waves as wave
from MCSimPython.utils import six2threeDOF, three2sixDOF, timeit, Rz
# Not yet pushed to init.py
from MCSimPython.observer.ltv_kf import LTVKF 
from MCSimPython.control.adaptiveFS import AdaptiveFSController
from MCSimPython.thrust_allocation.allocation import fixed_angle_allocator
from MCSimPython.simulator.thruster_dynamics import ThrusterDynamics
from MCSimPython.thrust_allocation.thruster import Thruster
from MCSimPython.vessel_data.CSAD.thruster_data import lx, ly, K

# Sim parameters --------------------------------------------------------
dt = 0.02
N = 200
np.random.seed(1234)
# Plot white noise and observe


# Vessel simulator --------------------------------------------------
vessel = sim.CSAD_DP_6DOF(dt = dt)

# Define sea states from Monet ---------------------------------------
sea_states = np.array([[0.03, 0.7, 3.3], [0.04, 0.8, 3.3], [0.04, 0.9, 1.8], [0.05, 1, 1.8], [0.05, 1.1, 1.0], [0.06, 1.2, 1.0], [0.06, 1.3, 1.0], [0.06, 1.4, 1.0], [0.06, 1.5, 1.0], [0.03, 1.5, 1.0]])

# Define sea states and run simulations for controllers for all sea states -------------------

for i in range(len(sea_states)):
    # External forces -----------------------------------------------------
    # Current
    U = 0
    beta_u = np.deg2rad(0)

    # Waves
    N_w = 25                                            # Number of wave components
    hs = sea_states[i][0]                               # Significant wave height
    tp = sea_states[i][1]                               # Peak period
    gamma = sea_states[i][2]
    wp = 2*np.pi/tp                                     # Peak frequency
    wmin = wp/2
    wmax = 2.5*wp
    dw = (wmax-wmin)/N_w
    w_wave = np.linspace(wmin, wmax, N_w)               # Frequency range

    # Spectrum
    jonswap = wave.JONSWAP(w_wave)
    freq, spec = jonswap(hs, tp, gamma=gamma)              # PM spectrum

    wave_amps = np.sqrt(2*spec*dw)                      # Wave amplitudes
    eps = np.random.uniform(0, 2*np.pi, size=N_w)       # Phase 
    wave_dir = np.deg2rad(180) * np.ones(N_w)           # Direction

    # Wave loads
    waveload = wave.WaveLoad(wave_amps, w_wave, eps, wave_dir, 
                                config_file=vessel._config_file) 

    # Initialize and run simulation for different number of frequency components------------------

    N_adap_lst = [1, 5, 10, 15, 20]
    for j in range(len(N_adap_lst)):
        # Observer ----------------------------------------------------------
        observer = LTVKF(dt, vessel._M, vessel._D, Tp=tp)
        observer.set_tuning_matrices(
                    np.array([
                        [1e8,0,0,0,0,0],
                        [0,1e4,0,0,0,0],
                        [0,0,10*np.pi/180,0,0,0],
                        [0,0,0,1e3,0,0],
                        [0,0,0,0,1e3,0],
                        [0,0,0,0,0,1e2]]), 
                    np.array([
                        [1,0,0],
                        [0,1,0],
                        [0,0,50*np.pi/180]]))

        # Reference model -----------------------------------------------------
        ref_model = ref.ThrdOrderRefFilter(dt, omega = [.25, .2, .2])   #
        eta_ref = np.zeros((N, 3))                                      # Stationkeeping



        # Controller ----------------------------------------------------------
        N_adap = N_adap_lst[j]
        controller = AdaptiveFSController(dt, vessel._M, vessel._D, N = N_adap)

        w_min_adap = 1
        w_max_adap = 2
        #controller.set_freqs(w_min_adap, w_max_adap, N_adap)
        #controller.set_tuning_params()



        # Thrust allocation --------------------------------------------------
        thrust_dynamics = ThrusterDynamics()
        thrust_allocation = fixed_angle_allocator()
        for i in range(6):
            thrust_allocation.add_thruster(Thruster([lx[i],ly[i]],K[i]))

        wave_realization = jonswap.realization(time=np.arange(0, N*dt, dt), hs = hs, tp=tp)

        # Simulation ========================================================
        N_theta = (6*N_adap + 3)
        storage = np.zeros((N, 78 + N_theta))


        for k in tqdm(range(N)):
            t = (k+1)*dt
            
            zeta= wave_realization[k]
            
            # Accurate heading measurement
            psi = vessel.get_eta()[-1]

            # Ref. model
            eta_d, eta_d_dot, eta_d_ddot = np.zeros(3), np.zeros(3) , np.zeros(3)                                                     # 3 DOF
            nu_d = Rz(psi).T@eta_d_dot

            # Wave forces
            tau_w_first = waveload.first_order_loads(t, vessel.get_eta())
            tau_w_second = waveload.second_order_loads(t, vessel.get_eta()[-1])
            tau_w = tau_w_first + tau_w_second

            # Controller
            time_ctrl = time()
            tau_cmd, bias_ctrl = controller.get_tau(observer.get_eta_hat(), eta_d,  observer.get_nu_hat(), eta_d_dot, eta_d_ddot, t, calculate_bias = True)
            time_ctrl2 = time() - time_ctrl

            theta_hat = controller.theta_hat

            # Thrust allocation - not used - SATURATION
            u, alpha = thrust_allocation.allocate(tau_cmd)
            tau_ctrl = thrust_dynamics.get_tau(u, alpha)

            # Measurement
            noise = np.concatenate((np.random.normal(0,.001,size=3),np.random.normal(0,.0002,size=3)))
            y = np.array(vessel.get_eta() + noise)

            # Observer
            observer.update(tau_ctrl, six2threeDOF(y), psi)

            # Calculate x_dot and integrate
            tau = three2sixDOF(tau_ctrl) + tau_w
            vessel.integrate(U, beta_u, tau)

            storage[k] = np.concatenate([t, vessel.get_eta(), vessel.get_nu(), eta_d, nu_d, y, tau_cmd, tau_w, 
                                        observer.get_x_hat(), eta_ref[k], bias_ctrl, tau_ctrl, time_ctrl2, tau_w_first, 
                                        tau_w_second, u, zeta, theta_hat], axis=None)
                                        # OBSOBS: Legg inn ny data i storage FØR theta_hat


        headers = ['time', 'eta1', 'eta2', 'eta3', 'eta4', 'eta5', 'eta6', 'nu1','nu2','nu3','nu4','nu5','nu6','eta_d_1', 'eta_d_2', 'eta_d_6', 'nu_d_1','nu_d_2','nu_d_6', 'y1','y2','y3','y4','y5','y6',
                'tau_cmd_1','tau_cmd_2','tau_cmd_6', 'tau_w_1', 'tau_w_2','tau_w_3','tau_w_4','tau_w_5','tau_w_6', 'xi_hat_1', 'xi_hat_2','xi_hat_3','xi_hat_4','xi_hat_5','xi_hat_6', 
                'eta_hat_1', 'eta_hat_2', 'eta_hat_6', 'bias_hat_1', 'bias_hat_2', 'bias_hat_6', 'nu_hat_1', 'nu_hat_2', 'nu_hat_6', 'eta_ref_1','eta_ref_2','eta_ref_6', 
                'bias_ctrl_1','bias_ctrl_2','bias_ctrl_6', 'tau_ctrl_1', 'tau_ctrl_2', 'tau_ctrl_6', 'time_ctrl', 'tau_w_first_1','tau_w_first_2','tau_w_first_3','tau_w_first_4',
                'tau_w_first_5','tau_w_first_6', 'tau_w_second_1','tau_w_second_2','tau_w_second_3','tau_w_second_4', 'tau_w_second_5','tau_w_second_6', 'u1','u2','u3','u4','u5',
                'u6','zeta']
        
        for k in range(N_theta):
            headers.append('theta_hat_'+str(k+1))


        # Create dataframe
        df = pd.DataFrame(storage, columns=headers)
        # Convert to csv
        df.to_csv('Hs'+str(hs)+'_Tp'+str(tp)+'_gamma'+str(gamma)+'_N'+str(N_adap)+'.csv')





'''

# Post processing 
# ============================================================================================
t = storage[:,0]
eta = storage[:,1:7]
nu = storage[:, 7:13]
eta_d = storage[:, 13:16]
nu_d = storage[:, 16:19]
y = storage[:,19:25]
tau_cmd = storage[:, 25:28]
tau_w = storage[:, 28:34]
xi_hat = storage[:, 34:40]
eta_hat = storage[:, 40:43]
bias_hat = storage[:, 43:46]
nu_hat = storage[:, 46:49]
eta_ref = storage[:, 49:52]
bias = storage[:, 52:55]
tau_ctrl = storage[:, 55:58]
t_ctrl = storage[:, 58]
tau_w_first = storage[:, 59:65]
tau_w_second = storage[:, 65:71]
u = storage[:, 71:77]
zeta = storage[:,77]

fig, axs = plt.subplots(2, 3)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        axs[i,j].plot(t, y[:, DOF-1], label=r'$y$'+str(DOF))
        axs[i,j].plot(t, eta[:, DOF-1], label=r'$\eta$'+str(DOF))

        axs[i,j].grid()
        axs[i,j].set_xlim([0,dt*N])
        axs[i,j].set_xlabel('t [s]')
        axs[i,j].set_title(r'$\eta $'  + str(DOF))
        if i == 1:
            axs[i,j].set_ylabel('Angle [rad]')
        else:
            axs[i,j].set_ylabel('Disp [m]')  
        if DOF in [1,2,6]: 
            axs[i,j].plot(t, eta_hat[:,i_obs], label=r'$\eta_{obs} $ '+str(DOF))
            axs[i,j].plot(t, eta_ref[:, i_obs], '--', label=r'$\eta_{ref}$'+str(DOF))
            axs[i,j].plot(t, eta_d[:, i_obs], label=r'$\eta_{d}$'+str(DOF))
            i_obs+=1
        
        axs[i,j].legend( edgecolor="k")
plt.tight_layout()
plt.suptitle('Response', fontsize=32)
plt.show()
# North-East plot ==============================================================================
plt.figure(figsize=(8,8))
plt.plot(eta[:,1], eta[:,0], label='Trajectory', linewidth=2)
plt.plot(eta_d[:,1], eta_d[:,0], '-.', label='eta_d')
plt.grid()
plt.title('XY-plot')
plt.legend(loc="upper right", edgecolor="k")
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')
plt.show()
'''