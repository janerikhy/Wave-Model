cmake_minimum_required(VERSION 3.0.2)
project(messages)

find_package(catkin REQUIRED)

################################
##  CONFIGURE THIS FILE HERE  ##
################################

set(ROS_DEPENDS
  message_generation
  message_runtime
  nav_msgs
  sensor_msgs
  std_msgs
  rospy
  roscpp
  )

set(EXTERNAL_DEPENDS
  #OpenCV
  )

set(INCLUDE_DIRECTORIES
  #${OpenCV_LIBS}
  )


find_package(catkin REQUIRED COMPONENTS ${ROS_DEPENDS})

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg
  FILES
  Observer_state.msg
  Reference.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${ROS_DEPENDS}
  DEPENDS
    ${EXTERNAL_DEPENDS}
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${INCLUDE_DIRECTORIES}
)

#catkin_python_setup()

