import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.observer.ltv_kf import LTVKF
from MCSimPython.control.basic import PD, PID
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.guidance.filter import ThrdOrderRefFilter

from MCSimPython.utils import six2threeDOF, three2sixDOF, Rz


plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})

# Sim parameters
dt = 0.01
N = 50000
t = np.arange(0, dt*N, dt)
np.random.seed(1234)


# Current parameters
U = 0.0            # m/s
beta_u = 45         # deg


# Sea state parameters
N_w = 25                # Number of wave components
hs = 0.03               # Significant wave height
tp = 1.0                # Peak period
wp = 2*np.pi/tp         # Peak frequency
wmin = wp/2
wmax = 2.5*wp
dw = (wmax-wmin)/N_w
w = np.linspace(wmin, wmax, N_w)


# Create wave spectrum
jonswap = JONSWAP(w)
freq, spec = jonswap(hs, tp, gamma=1.8)

wave_amps = np.sqrt(2*spec*dw)
eps = np.random.uniform(0, 2*np.pi, size=N_w)
wave_dir = np.deg2rad(np.random.normal(25, 10/3, size=N_w))
wave_dir = np.ones(N_w)*np.deg2rad(170)


# Create vessel, observer and wave loads objects
vessel = CSAD_DP_6DOF(dt)                                                           # Vessel
KalmanFilter = LTVKF(dt, vessel._M, vessel._D, Tp=tp)                               # Observer
waveload = WaveLoad(wave_amps, w, eps, wave_dir, config_file=vessel._config_file)   # Wave loads

# Set up a very simple reference model and reference points
eta_ref = np.zeros((N, 3))
t_cond = t > 50
t_cond2 = t > 150
t_cond3 = t > 350
eta_ref[t_cond, 0] = 5
eta_ref[t_cond2, 1] = 2
eta_ref[t_cond3, 0] = 0
eta_ref[t_cond3, 1] = 0

ref_model = ThrdOrderRefFilter(dt, omega=[.1, .1, .01])

# Initialize controller
controller = PD(kp=[10., 10., 50.], kd=[100., 100., 160.])

# Allocate memory to plot variables
storage_state = np.zeros((N,19))
storage_observer = np.zeros((N, 75))
storage_env = np.zeros((N, 6))


# Simulation: =============================================================================
for i in tqdm(range(N)):

    t = (i+1)*vessel._dt

    # Ref. model
    ref_model.set_eta_r(eta_ref[i])
    ref_model.update()
    eta_d = ref_model.get_eta_d()                                                           # 3 DOF
    eta_d_dot = ref_model.get_eta_d_dot()                                                   # 3 DOF

    psi = vessel.get_eta()[-1]                  # Extract heading (can be taken from observer)
    nu_d = Rz(psi).T@eta_d_dot                  # Convert desired velocity to body frame
    
    # Thruster forces
    tau_cmd = controller.get_tau(KalmanFilter.get_eta_hat(), eta_d, KalmanFilter.get_nu_hat(), nu_d)    # 3 DOF

    # Wave forces
    tau_wf = waveload.first_order_loads(t, vessel.get_eta())
    tau_sv = waveload.second_order_loads(t, vessel.get_eta()[-1])
    tau_w = tau_wf + tau_sv                                                                 # 6 DOF

    # Calculate x_dot and integrate
    tau = three2sixDOF(tau_cmd) + tau_w
    vessel.integrate(U, np.deg2rad(beta_u), tau)

    # Measurement
    noise = np.concatenate((np.random.normal(0,.1,size=3),np.random.normal(0,.017,size=3)))
    y = np.array(vessel.get_eta() + noise)                                                  # 6 DOF

    # Observer
    psi_measured = vessel.get_eta()[-1]
    KalmanFilter.update(tau_cmd, six2threeDOF(y), psi_measured)
    
    # Save for plotting
    K = KalmanFilter.KF_gain    
    storage_state[i] = np.concatenate([t, vessel.get_eta(),y, vessel.get_nu()], axis=None)
    storage_observer[i] = np.concatenate([KalmanFilter.get_x_hat(), np.diag(KalmanFilter.get_P_hat()), K[:,0], K[:,1], K[:,2]], axis=None)
    storage_env[i] = np.concatenate([eta_d, nu_d], axis=None)

# end =======================================================================================


# Store data in variables for plotting --------------------------------------
t = storage_state[:,0]
eta = storage_state[:,1:7]
y = storage_state[:,7:13]
nu = storage_state[:, 13:19]

xi_hat = storage_observer[:,0:6]
eta_hat = storage_observer[:,6:9]
b_hat = storage_observer[:,9:12]
nu_hat = storage_observer[:,12:15]

var = storage_observer[:,15:30]         # Variance in uncertainty in estimates
K_surge = storage_observer[:, 30:45]    # Kalman gain
K_sway = storage_observer[:, 45:60]     # Kalman gain
K_yaw = storage_observer[:, 60:75]      # Kalman gain

eta_d = storage_env[:, 0:3]
nu_d = storage_env[:, 3:6]

# Plot response --------------------------------------------------------------
width = 426.8            # Latex document width in pts
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
          #'text.usetex': True,
          'figure.constrained_layout.use': True,
          'figure.figsize': fig_size
          } 
plt.rcParams.update(params)

fig, axs = plt.subplots(2, 3)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        axs[i,j].plot(t, y[:,DOF-1], label=r'$y$'+str(DOF))
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
            i_obs+=1
        
        axs[i,j].legend(loc="upper right", edgecolor="k")
plt.tight_layout()
plt.suptitle('Response', fontsize=32)


fig, axs = plt.subplots(2, 3)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        axs[i,j].plot(t, nu[:, DOF-1], label=r'$\eta$'+str(DOF))
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
            i_obs+=1
            # axs[i,j].set_ylim([min(y[:, DOF-1]),max(y[:,DOF-1])])
        
        axs[i,j].legend(loc="upper right", edgecolor="k")
plt.tight_layout()
plt.suptitle('Velocity', fontsize=32)

plt.figure(figsize=(8,4))
plt.suptitle('XY-plot', fontsize=32)
plt.subplot(121) 
plt.plot(eta[:,1], eta[:,0], label='Trajectory', linewidth=2)
plt.plot(eta_ref[:,1], eta_ref[:,0], '--', label='eta_ref')
plt.plot(eta_d[:,1], eta_d[:,0], '-.', label='eta_d')
plt.grid()
plt.title('XY-plot')
plt.legend(loc="upper right", edgecolor="k")
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')
plt.subplot(122)
plt.plot(eta_hat[:,1], eta_hat[:,0], label='Trajectory')
plt.grid()
plt.title('Observer Estimates')
plt.legend(loc="upper right", edgecolor="k")
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')
plt.show()


# Plot observer data --------------------------------------------------------------
observer_legends = [r'$\xi1$', r'$\xi2$', r'$\xi3$', r'$\xi4$', r'$\xi5$', r'$\xi6$', r'$\eta1$', r'$\eta2$', r'$\eta3$', 'b1', 'b2', 'b3', r'$\nu1$',r'$\nu2$',r'$\nu3$']
plt.subplot(311)
plt.suptitle('Observer Estimates', fontsize=32)
plt.title('Bias estimate')
for i in range(3):
    plt.plot(t, b_hat[:, i], label='b'+str(i+1))
plt.legend()
plt.subplot(312)
plt.title('Wave estimate I')
for i in range(3):
    plt.plot(t, xi_hat[:,i], label=r'$\xi$'+ str(i+1))
plt.legend()
plt.subplot(313)
plt.title('Wave estimate II')
for i in range(3):
    plt.plot(t, xi_hat[:,i+3], label=r'$\xi$'+ str(i+4))
plt.legend()
plt.xlabel('time [s]')

plt.figure()
plt.subplot(111)
for i in range(len(var[0])):
    plt.plot(t, var[:,i], label= 'P' + str(i+1) + ' - ' + observer_legends[i])
plt.legend()
plt.title('Covariance matrix trace')
plt.ylim([0,100])
plt.show()


for i in range(15):
    if var[:,i][-1]-var[:,i][-5000]>5:
        print(i+1)