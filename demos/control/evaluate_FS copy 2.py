# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import scienceplots
plt.style.use(['science','grid', 'no-latex'])
from scipy.signal import csd 
from scipy.linalg import eig

width = 1000            # Latex document width in pts
inch_pr_pt = 1/72.27        # Ratio between pts and inches
golden_ratio = (np.sqrt(5) - 1)/2
fig_width = width*inch_pr_pt
fig_height = fig_width*golden_ratio
fig_size = [fig_width, fig_height]

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
N = 70000
np.random.seed(1234)
# Plot white noise and observe


# Vessel simulator --------------------------------------------------
vessel = sim.CSAD_DP_6DOF(dt = dt)


# External forces -----------------------------------------------------
# Current
U = 0.02
beta_u = np.deg2rad(180)
nu_cn = U*np.array([np.cos(beta_u), np.sin(beta_u), 0])

# Waves
N_w = 25                                            # Number of wave components
hs = 0.03                                           # Significant wave height
tp = 1.5                                            # Peak period
wp = 2*np.pi/tp                                     # Peak frequency
wmin = wp/2
wmax = 2.5*wp
dw = (wmax-wmin)/N_w
w_wave = np.linspace(wmin, wmax, N_w)               # Frequency range

# Spectrum
jonswap = wave.JONSWAP(w_wave)
freq, spec = jonswap(hs, tp, gamma=1.)              # PM spectrum

wave_amps = np.sqrt(2*spec*dw)                      # Wave amplitudes
eps = np.random.uniform(0, 2*np.pi, size=N_w)       # Phase 
wave_dir = np.deg2rad(180) * np.ones(N_w)           # Direction

# Wave loads
waveload = wave.WaveLoad(wave_amps, w_wave, eps, wave_dir, 
                            config_file=vessel._config_file) 



# Observer ----------------------------------------------------------
observer = LTVKF(dt, vessel._M, vessel._D, Tp=tp)
observer.set_tuning_matrices(
            np.array([
                [1e7,0,0,0,0,0],
                [0,1e7,0,0,0,0],
                [0,0,1e2*np.pi/180,0,0,0],
                [0,0,0,1e3,0,0],
                [0,0,0,0,1e3,0],
                [0,0,0,0,0,1e1]]), 
            np.array([
                [1e-2,0,0],
                [0,1e-2,0],
                [0,0,1e2*np.pi/180]]))

# Reference model -----------------------------------------------------
ref_model = ref.ThrdOrderRefFilter(dt, omega = [.25, .2, .2])   
eta_ref = np.zeros((N, 3))   
t = np.arange(0, dt*N, dt)

cond1, cond2, cond3, cond4, cond5 = 200, 150, 200, 250, 350
t_cond1 = t > cond1 

# Yaw test
eta_ref[t_cond1, 2] = np.linspace(0, np.deg2rad(360), int(N - 1  - (cond1/dt))) 


# Controllers ----------------------------------------------------------
N_adap = 10
controller_adap = AdaptiveFSController(dt, vessel._M, vessel._D, N = N_adap)

w_min_adap = 2*np.pi/20 
w_max_adap = 2*np.pi/2                           # Upper bound
controller_adap.set_freqs(w_min_adap, w_max_adap, N_adap)
K1 = [20, 20, .1]
K2 = [60, 60, 1]
gamma_adap = np.ones((2*N_adap+1)*3)*2
controller_adap.set_tuning_params(K1, K2, gamma=gamma_adap)

controller_pid = ctrl.PID(kp=[60., 60., 60.], kd=[50., 50., 50.], ki=[2, 2, 2], dt=dt, returnIntegral=True)

M_eig = six2threeDOF(vessel._M)
K_eig = six2threeDOF(vessel._G + three2sixDOF(np.diag([60., 60., 60.])))
values, vectors = eig(a = M_eig, b = K_eig)
print(np.sqrt(values))


# Thrust allocation --------------------------------------------------
thrust_dynamics = ThrusterDynamics()
thrust_allocation = fixed_angle_allocator()
for i in range(6):
    thrust_allocation.add_thruster(Thruster([lx[i],ly[i]],K[i]))

wave_realization = jonswap.realization(time=np.arange(0,N*dt,dt), hs = hs, tp=tp)

# Simulation ========================================================
N_theta = (6*N_adap + 3)
storage = np.zeros((N, 91 + N_theta))

t_global = time()
for i in tqdm(range(N)):
    t = (i+1)*dt
    
    zeta= wave_realization[i]
    
    # Accurate heading measurement
    psi = vessel.get_eta()[-1]
    nu_cb = Rz(psi).T@nu_cn

    # Ref. model
    #eta_d, eta_d_dot, eta_d_ddot = np.zeros(3), np.zeros(3) , np.zeros(3)                                                     # 3 DOF
    
    ref_model.set_eta_r(eta_ref[i])
    ref_model.update()
    eta_d = ref_model.get_eta_d()                                                           # 3 DOF
    eta_d_dot = ref_model.get_eta_d_dot()
    eta_d_ddot = ref_model.get_eta_d_ddot()
    nu_d = Rz(psi).T@eta_d_dot

    # Wave forces
    tau_w_first = waveload.first_order_loads(t, vessel.get_eta())
    tau_w_second = waveload.second_order_loads(t, vessel.get_eta()[-1])
    tau_w = tau_w_first + tau_w_second
    #tau_w_second = np.zeros(6)
    #tau_w_first = np.zeros(6)
    #tau_w = np.zeros(6)

    # Controller
    time_ctrl = time()
    #tau_cmd, bias_ctrl = controller_adap.get_tau(observer.get_eta_hat(), eta_d,  observer.get_nu_hat(), eta_d_dot, eta_d_ddot, t, calculate_bias = True)
    time_ctrl2 = time() - time_ctrl
    #theta_hat = controller_adap.theta_hat

    tau_cmd, bias_ctrl = controller_pid.get_tau(observer.get_eta_hat(), eta_d, observer.get_nu_hat(), nu_d)
    theta_hat = np.zeros(N_theta)

    # Thrust allocation - not used - SATURATION
    u, alpha = thrust_allocation.allocate(tau_cmd)
    tau_ctrl = thrust_dynamics.get_tau(u, alpha)
    tau_ctrl = tau_cmd

    # Measurement
    #noise = np.concatenate((np.random.normal(0,.001,size=3),np.random.normal(0,.0002,size=3)))
    y = np.array(vessel.get_eta())  # + noise)

    # Observer
    observer.update(tau_ctrl, six2threeDOF(y), psi)

    # Calculate x_dot and integrate
    tau = three2sixDOF(tau_ctrl) + tau_w
    vessel.integrate(U, beta_u, tau)

    # Calculate simulator bias
    nu_cb_ext = np.concatenate((nu_cb, np.zeros(3)), axis=None)
    b_optimal = six2threeDOF((vessel._D)@nu_cb_ext + tau_w_second)
    b_optimal_ned = Rz(psi)@b_optimal
    gamma_adap = 0

    storage[i] = np.concatenate([t, vessel.get_eta(), vessel.get_nu(), eta_d, nu_d, y, tau_cmd, tau_w, 
                                 observer.get_x_hat(), eta_ref[i], bias_ctrl, tau_ctrl, time_ctrl2, tau_w_first, 
                                 tau_w_second, u, zeta, K1, K2, gamma_adap, b_optimal, b_optimal_ned, theta_hat], axis=None)
                                # OBSOBS: Legg inn ny data i storage FØR theta_hat
t_global = time() - t_global

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
k1 = storage[:,78:81]
k2 = storage[:, 81:84]
gamma_adap = storage[:, 84]
b_optimal_body = storage[:, 85:88]
b_optimal_ned = storage[:, 88:91]
theta_hat = storage[:,91:]

headers = ['time', 'eta1', 'eta2', 'eta3', 'eta4', 'eta5', 'eta6', 'nu1','nu2','nu3','nu4','nu5','nu6','eta_d_1', 'eta_d_2', 'eta_d_6', 'nu_d_1','nu_d_2','nu_d_6', 'y1','y2','y3','y4','y5','y6',
           'tau_cmd_1','tau_cmd_2','tau_cmd_6', 'tau_w_1', 'tau_w_2','tau_w_3','tau_w_4','tau_w_5','tau_w_6', 'xi_hat_1', 'xi_hat_2','xi_hat_3','xi_hat_4','xi_hat_5','xi_hat_6', 
           'eta_hat_1', 'eta_hat_2', 'eta_hat_6', 'bias_hat_1', 'bias_hat_2', 'bias_hat_6', 'nu_hat_1', 'nu_hat_2', 'nu_hat_6', 'eta_ref_1','eta_ref_2','eta_ref_6', 
           'bias_ctrl_1','bias_ctrl_2','bias_ctrl_6', 'tau_ctrl_1', 'tau_ctrl_2', 'tau_ctrl_6', 'time_ctrl', 'tau_w_first_1','tau_w_first_2','tau_w_first_3','tau_w_first_4',
           'tau_w_first_5','tau_w_first_6', 'tau_w_second_1','tau_w_second_2','tau_w_second_3','tau_w_second_4', 'tau_w_second_5','tau_w_second_6', 'u1','u2','u3','u4','u5',
           'u6','zeta', 'K1_1', 'K1_2', 'K1_3', 'K2_1', 'K2_2', 'K2_3', 'gamma_adap', 'bias_optimal_1','bias_optimal_2','bias_optimal_6', 'bias_optimal_ned_1','bias_optimal_ned_2',
           'bias_optimal_ned_6']
for i in range(N_theta):
    headers.append('theta_hat_'+str(i+1))

omega = np.linspace(w_min_adap, w_max_adap, N_adap) 

omega_plot = np.ones(2*N_adap+1)
omega_plot[0] = 0

for i in range(N_adap):
    omega_plot[2*i +1] = omega[i]
    omega_plot[2*(i+1)] = omega[i]

omega = np.concatenate((np.zeros(1), omega, omega), axis=None)

# Create dataframe
#df = pd.DataFrame(storage, columns=headers)
# Convert to csv
#name = 'TEST_EvaluateFS__HeadSea_Hs_' + str(hs) + '_Tp_' + str(tp) + '_N_' + str(N_adap) + '.csv'
#df.to_csv(name)



fig, axs = plt.subplots(2, 3, figsize=fig_size)
plt.suptitle('Response', fontsize=32)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        #axs[i,j].plot(t, y[:, DOF-1], label=r'$y$'+str(DOF))
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
            axs[i,j].plot(t, eta_d[:, i_obs], label=r'$\eta_{d}$'+str(DOF))
            i_obs+=1
        axs[i,j].legend( edgecolor="k")


fig, axs = plt.subplots(2, 3, figsize=fig_size)
plt.suptitle('Velocity', fontsize=32)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        axs[i,j].plot(t, nu[:, DOF-1], label=r'$\nu$'+str(DOF))
        axs[i,j].grid()
        axs[i,j].set_xlim([0,dt*N])
        axs[i,j].set_xlabel('t [s]')
        axs[i,j].set_title(r'$\nu $'  + str(DOF))
        if i == 1:
            axs[i,j].set_ylabel('Angle vel. [rad/s]')
        else:
            axs[i,j].set_ylabel('Vel [m/s]')  
        if DOF in [1,2,6]: 
            axs[i,j].plot(t, nu_hat[:,i_obs], label=r'$\nu_{obs} $ '+str(DOF))
            axs[i,j].plot(t, nu_d[:,i_obs], label=r'$\nu_{d} $ '+str(DOF))
            i_obs+=1
        axs[i,j].legend(edgecolor="k")

# North-East plot ==============================================================================
plt.figure(figsize=(8,8))
plt.plot(eta[:,1], eta[:,0], label='Trajectory', linewidth=2)
plt.plot(eta_d[:,1], eta_d[:,0], 'o', label='eta_d')
plt.plot(eta_hat[:,1], eta_hat[:,0], label='Observer trajectory', linewidth=1)
plt.grid()
plt.title('XY-plot')
plt.legend(loc="upper right", edgecolor="k")
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')

'''
fig, axs = plt.subplots(3, 1)
plt.suptitle('Control Forces')
for i in range(3):
    axs[i].plot(t, tau_cmd[:,i], label=r'$\tau_{cmd}$ '+ str(i+1))
    axs[i].plot(t, tau_ctrl[:,i], label=r'$\tau_{ctrl}$ '+ str(i+1))
    axs[i].plot(t, bias[:,i], label='Disturbance estimation')
    axs[i].plot(t, bias_hat[:,i], label='Observer estimation')
    axs[i].legend()
    axs[i].set_title(r'$\tau$' + str(i+1))
plt.legend()
'''

fig, axs = plt.subplots(3, 1)
plt.suptitle('Residual loads')
for i in range(3):
    #axs[i].plot(t, bias[:,i], label='Bias from controller '+ str(i+1))

    axs[i].plot(t, bias[:,i], label='Integral action '+ str(i+1))
    axs[i].plot(t, bias_hat[:,i], label='Bias from observer '+ str(i+1))
    axs[i].plot(t, b_optimal_body[:,i], label='Residual load from simulator - BODY')
    axs[i].plot(t, b_optimal_ned[:,i], label='Residual load from simulator, NED')
    axs[i].legend()
    axs[i].set_title(r'$\tau$' + str(i+1))
plt.legend()

'''
fig, axs = plt.subplots(3, 1)
plt.suptitle('Low freq and wave freq motion')
for i in range(3):
    j = i if i in [0,1] else 5
    axs[i].plot(t, eta[:,j], label='eta simulator '+ str(i+1))
    axs[i].plot(t, eta_hat[:,i], label='Low frequency observer '+ str(i+1))
    axs[i].plot(t, xi_hat[:,3+i], label='Wave frequency observer')
    axs[i].plot(t, xi_hat[:,3+i] + eta_hat[:,i], label='Wave frequency + low frequency observer')
    #axs[i].plot(t, bias_hat[:,i], label='bias observer')
    axs[i].legend()
plt.legend()



fig, axs = plt.subplots(2, 1)
plt.suptitle('Wave loads')
for i in range(2):
    axs[i].plot(t, tau_w_first[:,i], label='1st order wave '+ str(i+1))
    axs[i].plot(t, tau_w_second[:,i], label='2nd order wave '+ str(i+1))
    axs[i].plot(t, bias_hat[:,i], label='Bias from observer '+ str(i+1))

    axs[i].legend()
plt.legend()
'''


N_theta_plot = 2*N_adap + 1
theta_surge = theta_hat[-1, 0:N_theta_plot]
theta_sway = theta_hat[-1, N_theta_plot:2*N_theta_plot]
theta_yaw = theta_hat[-1, 2*N_theta_plot:]
theta_list = [theta_surge, theta_sway, theta_yaw]
'''
fig, axs = plt.subplots(1, 1)
plt.suptitle('Adaptive gains')
for i in range(3):
    axs.plot(theta_hat[-1, 0+i*(N_theta_plot):(N_theta_plot)+i*(N_theta_plot)], label='DOF '+str(i+1))
    axs.plot(theta_list[i], label='check')
    axs.legend()
'''
theta_surge_split = np.zeros((2, N_adap+1))
theta_surge_split[:,0] = np.concatenate((theta_surge[0], theta_surge[0]), axis=None)


for i in range(N_adap-1):
    theta_surge_split[0, i + 1] = theta_surge[2*i + 1]
    theta_surge_split[1, i + 1] = theta_surge[2*i + 2]

omega = np.linspace(w_min_adap, w_max_adap, N_adap) / (2*np.pi)
omega = np.concatenate((np.zeros(1), omega), axis=None)
print(omega)


plt.figure()
plt.scatter(omega, theta_surge_split[0])
plt.scatter(omega, theta_surge_split[1])






# Residual loads frequency
freq1, psd1 = csd(b_optimal_body[:,0], b_optimal_body[:,0], fs = 50, nperseg=2**10)
freq2, psd2 = csd(bias[:,0], bias[:,0], fs = 50, nperseg=2**10)
freq21, psd21 = csd(bias[int(-N/5):,0], bias[int(-N/5):,0], fs = 50, nperseg=2**10)

freq_test = np.fft.fftfreq(t.shape[-1])
psd_test = np.fft.fft(bias[:,0])

plt.figure()
plt.plot(freq1, psd1, label='Residual actual bias')
plt.plot(freq2, psd2, label='Residual controller bias')
plt.plot(freq21, psd21, label='Residual controller bias 2')

plt.legend()
plt.show()
