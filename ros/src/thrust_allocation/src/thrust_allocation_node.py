import rospy
import numpy as np
import math
from std_msgs.msg import Float64MultiArray

import MCSimPython.thrust_allocation.allocation as al
import MCSimPython.simulator.thruster_dynamics as dynamics
import MCSimPython.thrust_allocation.thruster as thruster
import MCSimPython.vessel_data.CSAD.thruster_data as data 

# Actuators
allocator = al.fixed_angle_allocator()

# Set thruster configuration
for i in range(6):
    allocator.add_thruster(thruster.Thruster((data.lx[i], data.ly[i]), data.K[i]))

def callback(msg_in, endSim):
    msg_out = Float64MultiArray()
    tau = msg_in.data

    u, alpha = allocator.allocate(tau)

    #u = thrustAllocation(tau,endSim)
    msg_out.data = u
    pub.publish(msg_out)

if __name__ == '__main__':
    rospy.set_param('endSim', 0)
    global pub, node
    node = rospy.init_node('thrust_allocation')
    pub = rospy.Publisher('CSAD/u', Float64MultiArray, queue_size=1)
    rospy.Subscriber("CSAD/tau", Float64MultiArray, callback, rospy.get_param('endSim')) # har rekkefølgen noe å si?
    r = rospy.Rate(100)

    rospy.spin()

if __name__ == '__main__':

    thrusterNodeInit()
    r = rospy.Rate(params["runfrequency"]) # Usually set to 100 Hz
    rospy.sleep(0.23)
    print("ALLOCATION time", tic.time()-t0)
    while not rospy.is_shutdown():
        
        loop()
        r.sleep()
        
    nodeEnd()