#!/bin/bash

# Run the test agent for each robot in parallel
ros2 run turtlebot3_rl test_agent -- robot1 &
ros2 run turtlebot3_rl test_agent -- robot2 &
ros2 run turtlebot3_rl test_agent -- robot3 &
ros2 run turtlebot3_rl test_agent -- robot4 &
ros2 run turtlebot3_rl test_agent -- robot5 &

# Wait for all background tasks to finish
wait

