import MCSimPython.thrust_allocation.allocation as al
import MCSimPython.simulator.thruster_dynamics as dynamics
import MCSimPython.thrust_allocation.thruster as thruster
import MCSimPython.vessel_data.CSAD.thruster_data as data 
import numpy as np

import matplotlib.pyplot as plt


allocator = al.fixed_angle_allocator()

# Set thruster configuration

for i in range(6):
    allocator.add_thruster(thruster.Thruster((data.lx[i], data.ly[i]), data.K[i]))

# Check that all thrusters are added - in this case there should be 6 thrusters

print(allocator.n_thrusters)

# Set random desired force (tau_d)

tau_d = np.array([[1, 0.3, 0.5], [0, 0.2, 0.7], [1, 3, 1]])

# First check that tau = tau_d when there are no constraints


for i in range(len(tau_d)):
    u, alpha = allocator.allocate(tau_d[i])
    ThrustDyn = dynamics.ThrusterDynamics(u, alpha, dt=0.2)
    tau = ThrustDyn.get_tau(u, alpha)

# Get tau when u set constraints for u

for i in range(len(tau_d)):
    u, alpha = allocator.allocate(tau_d[i])
    ThrustDyn = dynamics.ThrusterDynamics(u, alpha, dt=0.2)
    u = ThrustDyn.saturate(u, -1, 1) 
    tau_sat = ThrustDyn.get_tau(u, alpha)
    