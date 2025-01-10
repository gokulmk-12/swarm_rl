# Robot Swarm Navigation

![Small Swarm](https://github.com/user-attachments/assets/63a3b7fe-d051-4d3a-8b6a-2ade75b043b1)

![Large Swarm](https://github.com/user-attachments/assets/b218b0fa-3b75-49d2-a1ab-2133083e50ca)

![rewardplot](https://github.com/user-attachments/assets/75c26dc9-d279-4c15-bbb4-d1a2d8a9ef20)
![epstat](https://github.com/user-attachments/assets/249c4f9b-aaec-4c58-8007-2d2605d47a81)


## Software Requirements
The entire pipeline was tested in Ubuntu 22.04 with ROS2 Humble Hawksbill and Gazebo Classic. The zip file contains all the necessary code to run the pipeline.
Please note that folders **custom_msgs**, **sim_gazebo**, **turtlebot3_rl** inside the zip are ros2 packages. All others are ordinary directories.

## LLM Requirements
We have used Groq AI Inference Engine. So the LLM requires the API key from Groq to run. Please do login in Groq, get API keys and put in the below file
> tool_caller.py in llm/LLM/LLM, line 18 and 35. Put your API Key as a string in Groq(api_key=<your_API_key>)

## Gazebo Models
There is a [gazebo_models.zip](https://drive.google.com/file/d/1RDvVamywxUlJ13DGDlOhOFfkxlMmjLl0/view?usp=sharing) file. This contains the models and obstacles in gazebo. Please extract them to .gazebo/models (hidden folder) in your home directory.

## Usage Instructions
A few videos are in the Videos folder which can help one understand how to run the pipeline. However, the steps have been mentioned in detail below.

# Create ROS2 Workspace
```
mkdir -p swarm_ws/src
cd swarm_ws/src
```
Unzip the files in the **src** directly. Then, colcon build the workspace by going one step back
```
cd ..
colcon build
```
There is a requirements.txt file including all the necessary python libraries. To install do the following
```
cd src
pip3 install -r requirements.txt
```
---
## Some Important Files to Note
### 1) params.yaml
This file contains the robot positions, goal positions, RL algorithm hyperparameters, and some import variables for a 10x10 environment
### 2) spawn_4_bots.launch.py (in launch folder)
This is the main launch file that spawns 4 robots, 3 dynamic obstacles, and 4 targets in a 10x10 environment. The associated positions of the robot are 
read from **params.yaml**
### 3) params_large.yaml
This file contains the robot positions, goal positions, RL algorithm hyperparameters, and some import variables for a 15x15 environment
### 4) spawn_5_bots.launch.py (in launch folder)
This is the main launch file that spawns 5 robots, 3 dynamic obstacles, and 5 targets in a 15x15 cluttered environment. The associated positions of the robot are 
read from **params_large.yaml**
### 5) datafinal.json (in llm folder)
This file holds the robot coordinates, target coordinates, status of the robot, and the assigned target to a robot in the 10x10 environment. This file is extensively
used by the PyGame Visualizer, LLM Chatbot and RL Agent to update status of the robot.
### 6) datafinallarge.json (in llm folder)
This file holds the robot coordinates, target coordinates, status of the robot, and the assigned target to a robot in the 15x15 environment.

---
# Test 1: Entire Workflow with 4 Robots and 10x10 Environment with LLM, PyGame
**Terminal 1:** LLM Main Script (Make sure you point to src)
```
cd swarm_ws/src
python3 python3 llm/LLM/LLM/app.py 
```
**Terminal 2:** Streamlit Interface (Make sure you point to src)
```
streamlit run llm/LLM/LLM/chat_ui.py 
```
Use the streamlit UI to assign robot the targets. An example prompt is given below
```
assign robot_1 to service_1,  robot_2 to charging_2,  robot_3 to vending_machine,  robot_4 to charging_1
```
**Terminal 3:** Spawn 4 bots launch file (Make sure you point to swarm_ws)
```
cd ..
source install/setup.bash
ros2 launch sim_gazebo spawn_4_bots.launch.py 
```
**Terminal 4:** Obstacle Detection Script (Uses laser scan to visualize object locations in RViZ) (Make sure you point to swarm_ws)
```
cd ..
source install/setup.bash
ros2 run turtlebot3_rl det_obs
```
**Terminal 5:** PyGame Operator Visualizer (Can be used to assign robot to targets by mouse click, checkout in Videos) (Make sure you point to src)
```
cd ..
source install/setup.bash
cd src
ros2 run turtlebot3_rl global_database_node
```
**Terminal 6:** ROS2-RL Environment Bash (the workspace has to be sourced), the terminal will be empty after run, dont worry unless any issues
```
source install/setup.bash
cd src
./run_environment.sh
```
**Terminal 7:** ROS2-Gazebo Environment Bash (the workspace has to be sourced)
```
source install/setup.bash
cd src
./run_gazebo_testing.sh
```
Make sure all 4 robots have been assigned coordinates. Also have a look at Terminal 6 if all the 4 robot coordinates have been recieved. This is crucial, some cases some robot coordinates would have not recieved.
This can solved by just rerunning the Terminal 6 and 7 in order. This issue is due to ROS2 service-client delay.

**Terminal 8:** ROS2-Agent Bash (the workspace has to be sourced)
```
source install/setup.bash
cd src
./run_test_agent.sh
```
The run_test_agent will load the pretrained weights from models folder. Once that succeeds, one can visulaize the robot swarm move in the Gazebo Engine.

---
# Test 2: Entire Workflow with 5 Robots and 15x15 Environment (no LLM, PyGame)
**Important**: As explained earlier, the user is supposed to uncomment a few lines in few scripts to make the code work for the larger environment
> env.py file in turtlebot3_rl package, line 18

> gazebo.py file in turtlebot3_rl package, line 21

> agent.py file in turtlebot3_rl package, line 27

> td3.py file in turtlebot3_rl package, line 13

> util.py file in turtlebot3_rl package, line 10

**Terminal 1:** Spawn 5 bots launch file (Make sure you point to swarm_ws)
```
cd ..
source install/setup.bash
ros2 launch sim_gazebo spawn_5_bots.launch.py 
```
**Terminal 2:** ROS2-RL Environment Bash (the workspace has to be sourced), the terminal will be empty after run, dont worry unless any issues
```
source install/setup.bash
cd src
./run_environment_large.sh
```
**Terminal 3:** ROS2-Gazebo Environment Bash (the workspace has to be sourced)
```
source install/setup.bash
cd src
./run_gazebo_testing_large.sh
```
Make sure all 5 robots have been assigned coordinates. Also have a look at Terminal 2 if all the 4 robot coordinates have been recieved. This is crucial, some cases some robot coordinates would have not recieved.
This can solved by just rerunning the Terminal 2 and 3 in order. This issue is due to ROS2 service-client delay.

**Terminal 4:** ROS2-Agent Bash (the workspace has to be sourced)
```
source install/setup.bash
cd src
./run_test_agent_large.sh
```
The run_test_agent will load the pretrained weights from models folder. Once that succeeds, one can visulaize the robot swarm move in the Gazebo Engine.

---
# Ongoing & Future Work

- Train the Swarm with other RL algorithms like PPO, MAPPO, MADDPG.
- Turn it into an Decentralized Training and Centralized Execution Problem.
- Code up a Graph-RL scenario for training and execution.
<<<<<<< HEAD
=======

>>>>>>> 485312a (Added Videos and GIFs)
