cmake_minimum_required(VERSION 3.0.2)
project(controller)

################################
##  CONFIGURE THIS FILE HERE  ##
################################

set(ROS_DEPENDS
  rospy
  std_msgs
  )

set(EXTERNAL_DEPENDS
  )

set(INCLUDE_DIRECTORIES
  )


find_package(catkin REQUIRED COMPONENTS ${ROS_DEPENDS})

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