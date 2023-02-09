import MCSimPython.thrust_allocation.allocation as al
import MCSimPython.simulator.thruster_dynamics as dynamics
import MCSimPython.thrust_allocation.thruster as thruster
import MCSimPython.vessel_data.CSAD.thruster_data as data 

import matplotlib.pyplot as plt


allocator = al.AllocatorCSAD()


# Set thruster configuration

for i in range(6):
    allocator.add_thruster(thruster.Thruster((data.lx[i], data.ly[i]), data.K[i]))

# Check that all thrusters are added - in this case there should be 6 thrusters and 12 problems (thrusters * 2)

print(allocator.n_thrusters)
print(allocator.n_problem)

# Set random desired force (tau_d)

tau_d = [10, 20, 5]

u, alpha = allocator.allocate(tau_d)

print(u)

ThrustDyn = dynamics.ThrusterDynamics(u, alpha, dt=0.2)

tau = ThrustDyn.get_tau(u, alpha)

print(tau)

#figure, axis = plt.subplots(2, 1)
#axis[0, 0].plot(tau)
#plt.show()