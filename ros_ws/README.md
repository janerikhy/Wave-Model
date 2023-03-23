# Running the system
This README explains how to run the ROS system. 

## Sourcing 
In order to have open acccess to ROS commands you have to run the setup.bash script of your distro. In our case we are using noetic.
```bash
source /opt/ros/noetic/setup.bash
```
You will need to run this on every new shell that you want to access ROS commands from.
Therefore it is recommended to put this on the bottom of your computers .bashrc file.

## Building the system
Change directory to the workspace directory
```bash
cd ~/Wave-Model/ros_ws
```

Build the system with catkin
```bash
catkin_make
```

If faced with build difficulties try to remove the devel/ and build/ directories and build again
```bash
rm -rf devel/ build/
catkin_make
```

## Using launch files
To run the system you first need to source the workspace in order to get acces to the ROS functionalities of the workspace. 
Note that you should be located in the workspace directory: ~/Wave-Model/ros_ws when sourcing.
```bash
source devel/setup.bash
```
Now you are ready to run the system. It is done by using roslaunch <package_name> <launchfile.launch>. In this workspace all launchfiles are located in a package called launch.
Therefore, all our launch files are run with  the command: roslaunch launch <launchfile.launch>. Heres commands to run all the differente launchfiles in the workspace.

Simulator
```bash
roslaunch launch simulator.launch
```
## Running specific nodes
In order to run a ros node you have to have a rosmaster running. There are two ways to start a ROS master.

1. Start only one single master with no nodes
```bash
roscore
```

2. Start master and multiple nodes at the same time
```bash
roslaunch <package_name> <launchfile.launch>
```
The launchfile specifies which nodes should be launched.

After the rosmaster is running simply use rosrun in order to add wanted nodes.
```bash
rosrun <package_name> <launchfile.launch>
```
Example of running the joystick controller
```bash
rosrun controller joystick
```

# Demo
## Running just the simulator
First sourcing the ros distro.
```bash
source /opt/ros/noetic/setup.bash
```
Then change directory to the workspace directory ros_ws.
```bash
cd ~/Wave-Model/ros_ws
```
Now build the workspace.
```bash
catkin_make
```
And finally launch the simulator launch file.
```bash
roslaunch launch simulator.launch
```

## Running the model scale test in MC-Lab
First ssh into the Pi onboard the vessel. Recommend using the VSCode extension.

Now change to the workspace lab2023_ws
```bash
cd ~/lab2023_ws
```
The packages onboard the vessel should be built already so no need to catkin_make, but we still need to source.
```bash
source devel/setup.bash
```
And finally launch the csad launch file
```bash
roslaunch launch csad.launch
```

Now we need to run the control system onboard our computer connected to the vessel. Lets run a pid controller, guidance system and the observer. 
Note that all nodes needs to be run in seperate terminals
```bash
rosrun observer smallGain
```
```bash
rosrun controller pid
```
```bash
rosrun guidance path_param
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
