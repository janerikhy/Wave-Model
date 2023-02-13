import MCSimPython.thrust_allocation.allocation as al
import MCSimPython.simulator.thruster_dynamics as dynamics
import MCSimPython.thrust_allocation.thruster as thruster
import MCSimPython.vessel_data.CSAD.thruster_data as data 
import numpy as np

import matplotlib.pyplot as plt


allocator = al.AllocatorCSAD()


# Set thruster configuration

for i in range(6):
    allocator.add_thruster(thruster.Thruster((data.lx[i], data.ly[i]), data.K[i]))

# Check that all thrusters are added - in this case there should be 6 thrusters and 12 problems (thrusters * 2)

print(allocator.n_thrusters)
print(allocator.n_problem)

# Set random desired force (tau_d) - here tested with two "time steps"

tau_d = np.array([[10, 15, 5], [4, 6, 9], [8, 4, 7]])

Fx = []
Fy = []
Mz = []


for i in range(len(tau_d)):
    u, alpha = allocator.allocate(tau_d[i])
    ThrustDyn = dynamics.ThrusterDynamics(u, alpha, dt=0.2)
    Fx.append(ThrustDyn.get_tau(u, alpha)[0])
    #print(Fx[i])
    Fy.append(ThrustDyn.get_tau(u, alpha)[1])
    Mz.append(ThrustDyn.get_tau(u, alpha)[2])

fig, axs = plt.subplots(2)
fig.suptitle('Thrust allocation')
axs[0].plot(Fx, label='Fx')
axs[0].plot(Fy, label='Fy')
axs[0].plot(Mz, label='Fz')
axs[1].plot(Fy)
axs[0].legend()
plt.show()