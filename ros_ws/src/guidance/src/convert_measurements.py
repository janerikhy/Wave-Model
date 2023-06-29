#!/usr/bin/env python3
import rospy
import numpy as np
from datetime import datetime

from std_msgs.msg import Float64MultiArray, Bool, Int64
from nav_msgs.msg import Odometry

from MCSimPython.utils import Rx, pipi, Rz

import sys
sys.path.append(r'/home/hydrolab/dev/Wave-Model/')
import src.MCSimPython.observer.ltv_kf as ltv
from src.MCSimPython.utils import quat2eul

class Measurements(object):
    def __init__(self) -> None:

        # Initialize measurements
        self.eta = np.zeros(3)
        self.nu = np.zeros(3)

        # Subscriber
        self.odom_sub = rospy.Subscriber(f"/qualisys/{vessel_name}/odom", Odometry, self.odomCallback, queue_size=1)

        # Publisher
        self.pub = rospy.Publisher(f"/{vessel_name}/measurements", Float64MultiArray, queue_size=1)
        self.measurements_msg = Float64MultiArray()

        self.pub_dead_reckoning = rospy.Publisher(f"/{vessel_name}/dead_reck", Bool, queue_size=1)
        self.dead_reck_msg = Bool()

        # Initialize check for no signal received
        self.dr_check = 0
        self.last_msg_time = datetime.now()

    def rotate(self):
        '''
        Rotate from global basin frame to global NED and local body frame.
        '''
        self.eta = Rx(np.pi)@self.eta
        self.nu = Rz(-self.eta[-1])@Rx(np.pi)@self.nu

    def odomCallback(self, msg):
        """
            Callback function for odometry message. Updating position and attitude of vessel.
        """
        self.last_msg_time = datetime.now()

        self.odom_msg = msg

        # Position
        eta_x = msg.pose.pose.position.x
        eta_y = msg.pose.pose.position.y

        # Quaternions (attitude)
        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z

        euler_angles = quat2eul(q_w, q_x, q_y, q_z)
        eta_psi = (euler_angles[0])                    # Edited to work with Qualisys

        # Velocity
        nu_x = msg.twist.twist.linear.x
        nu_y = msg.twist.twist.linear.y
        nu_psi = msg.twist.twist.angular.x
        
        self.eta = np.array([eta_x, eta_y, eta_psi])
        self.nu = np.array([nu_x, nu_y, nu_psi])

        self.rotate()

    def check_message_time(self):
        time_diff = datetime.now() - self.last_msg_time
        if time_diff.total_seconds() >= 1.0:
            rospy.logwarn('No odom message received for 1 second')
            self.dr_check = 1
        elif self.dr_check:
            self.dr_check = 0
        


    def publish(self):
        self.measurements_msg.data[0:3] = self.eta
        self.measurements_msg.data[3:6] = self.nu

        self.dead_reck_msg.data = True if self.dr_check else False

        self.pub.publish(self.measurements_msg)
        self.pub_dead_reckoning.publish(self.dead_reck_msg)
        

if __name__ == '__main__':
    vessel_name = "CSAD"
    rospy.init_node(f"{vessel_name}_measurements")
    rospy.loginfo(f"INITIALIZING {vessel_name} measurement NODE")
    r = rospy.Rate(100)
    
    # Initialize
    y = Measurements()
    
    while not rospy.is_shutdown():
        y.check_message_time()
        
        # Publish message
        y.publish()


        r.sleep()

    rospy.spin()