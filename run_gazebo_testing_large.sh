#!/bin/bash

# Run Gazebo in testing mode for each robot in parallel
ros2 run turtlebot3_rl gazebo --ros-args -p mode:=testing -p namespace:=robot1 &
ros2 run turtlebot3_rl gazebo --ros-args -p mode:=testing -p namespace:=robot2 &
ros2 run turtlebot3_rl gazebo --ros-args -p mode:=testing -p namespace:=robot3 &
ros2 run turtlebot3_rl gazebo --ros-args -p mode:=testing -p namespace:=robot4 &
ros2 run turtlebot3_rl gazebo --ros-args -p mode:=testing -p namespace:=robot5 &

# Wait for all background tasks to finish
wait

