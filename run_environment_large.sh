#!/bin/bash

# Run the environment for each robot in parallel
ros2 run turtlebot3_rl environment robot1 &
ros2 run turtlebot3_rl environment robot2 &
ros2 run turtlebot3_rl environment robot3 &
ros2 run turtlebot3_rl environment robot4 &
ros2 run turtlebot3_rl environment robot5 &

# Wait for all background tasks to finish
wait

