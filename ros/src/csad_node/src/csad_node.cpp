#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "csad_actuator_driver/csad_actuator.h"

# define M_PI 3.14159265358979323846  /* pi */

CSAD_Actuator ship;

void uCallback(const std_msgs::Float64MultiArray &msg){
    double buffer[6];
    for (int i = 0; i<6; i++){
        buffer[i] = (double)msg.data[i];
    }
    ship.setAllMotorPower(buffer);
    for (int i = 6; i<12; i++){
        buffer[i - 6] = (double)msg.data[i];
    }
    ROS_INFO("%f", buffer);    
    ship.setAllServoPositions(buffer);
    
}

int main(int argc, char* argv[])
{

    //init
    ros::init(argc, argv, "CSAD_Actuator");
    //message code
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("CSAD/u",1 ,uCallback);
    ros::Publisher pub = n.advertise<std_msgs::Float64MultiArray>("CSAD/alpha",1);
    ros::Rate loop_rate(100);
    std_msgs::Float64MultiArray msg;
    double servoPos[6];

    while (ros::ok())
    {
        ship.getAllServoPresentPositions(servoPos);
        msg.data.clear();
        for (int i = 0; i < 6; i++){
            msg.data.push_back(servoPos[i]);
        }
        pub.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }
    ros::shutdown();
}