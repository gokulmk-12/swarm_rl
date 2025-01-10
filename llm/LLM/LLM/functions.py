import numpy as np
import json
import os

status_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "LLM", "LLM", "status_retrive.txt")
with open(status_file, "w", encoding="utf-8") as file:
        file.write("")

json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinal.json")
with open(json_file, 'r') as json_file:
   data = json.load(json_file)

robot_data=data['robot']
object_data=data['target']
object_ids=[]
object_types=[]
robot_id=[]

for i in object_data:
    if i["object_id"] not in object_ids:
        object_ids.append(i["object_id"])
    if i["type"] not in object_types:
        object_types.append(i["type"])
for i in robot_data:
    if i["robot_id"] not in robot_id:
        robot_id.append(i["robot_id"])



def load_data():
    json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinal.json")
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)

    robot_data=data['robot']
    object_data=data['target']
    return robot_data,object_data

def update_data(robot_data,object_data):
    data={"robot":robot_data,'target':object_data}
    json_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "datafinal.json")
    with open(json_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)    

def assign_object_to_robot(robot_data,robot_id, object_coords):
    
    # Find the robot by ID (reference)
    robot = next((r for r in robot_data if r['robot_id'] == robot_id), None)
    if not robot:
        print("Robot not found.")
        return robot_data
    
    robot['assigned_object_coords'] = object_coords
    return robot_data
    
def reset_robot_and_object_state(robot_data, object_data, robot_id, object_coord):
    robot = next((r for r in robot_data if r['robot_id'] == robot_id), None)
    if not robot:
        print(f"Robot with ID {robot_id} not found.")
        return robot_data, object_data

    robot['idle'] = True
    robot['assigned_object_coords'] = []
    print(f"Robot {robot_id} state reset: idle=True, assigned_object_coords=[]")

    obj = next((o for o in object_data if o['coords'] == object_coord), None)
    if not obj:
        print(f"Object with ID {object_coord} not found.")
        return robot_data, object_data

    obj['is_assigned'] = False
    print(f"Object {object_coord} is_assigned status set to False.")
    return robot_data, object_data

    robot['idle'] = True
    robot['assigned_object_coords'] = []
    print(f"Robot {robot_id} state reset: idle=True, assigned_object_coords=[]")
    return robot_data



def evaluate_objects_and_robots(object_data, robot_data, find=False):
    nearest_robot = None
    nearest_object_coord = None
    min_distance = float('inf')

    # Filter robots that are idle (idle=True)
    
    idle_robots=robot_data
    # If find is False, filter out objects with "is_assigned": True
    if not find:
        idle_robots = [robot for robot in robot_data if robot.get("idle", False)]
        print("hhd")
        object_data = [obj for obj in object_data if not obj.get("is_assigned", False)]
        if not object_data:
            return "All objects of this type are already assigned."

    # Iterate over idle robots and objects to find the minimum distance
    for robot in idle_robots:
        robot_coord = robot["position"]
        for obj in object_data:
            obj_coord = obj["coords"]
            distance = np.linalg.norm(np.array(robot_coord) - np.array(obj_coord))
            
            # Update if this robot-object pair has the smallest distance
            if distance < min_distance:
                min_distance = distance
                nearest_robot = robot["robot_id"]
                nearest_object_coord = obj_coord

    # Return the robot and object coordinates with the least distance
    return {"robot_id": nearest_robot, "object_position": nearest_object_coord, "distance": min_distance}

def find_nearest(object_type):
    robot_data,object_data=load_data()
    # Filter objects based on type
    filtered_objects = [obj for obj in object_data if obj["type"] == object_type]
    
    # Get the nearest object and robot pair using the updated function
    result = evaluate_objects_and_robots(filtered_objects, robot_data, find=True)
    
    # Custom message for success or failure
    if isinstance(result, str):
        return result
    message = f"closest  '{object_type}' from robots is at position {result['object_position']}."
    return message



def assign_general(object_type):
    robot_data,object_data=load_data()
    '''Finds the nearest object of a given type and then assigns a robot to it'''
    # Filter objects based on type
    filtered_objects = [obj for obj in object_data if obj["type"] == object_type]
    
    # Get the nearest object and robot pair
    result = evaluate_objects_and_robots(filtered_objects, robot_data)
    if isinstance(result, str):
        return result

    closest_object_coord = result["object_position"]
    closest_robot_id = result["robot_id"]

    # Update the robot and object statuses
    closest_robot = next(robot for robot in robot_data if robot["robot_id"] == closest_robot_id)
    closest_robot["idle"] = False
    closest_robot["status"]="unknown"
    closest_object = next(obj for obj in object_data if obj["coords"] == closest_object_coord)
    closest_object["is_assigned"] = True

    # Send data to RL
    robot_data = assign_object_to_robot(robot_data, closest_robot_id, closest_object["coords"])  
    update_data(robot_data,object_data)

    # Custom message
    message = f"Robot '{closest_robot_id}' has been assigned to the nearest object of type '{object_type}' at position {closest_object_coord}."
    return message




def assign_specific(object_id=None, robot_id=None, object_type=None):
    '''Assigns a robot to a specific object based on given parameters'''
    robot_data,object_data=load_data()

    # Case 1: If object_id is given but not robot_id
    if object_id and not robot_id:
        object_to_be_assigned = None
        for obj in object_data:
            if obj["object_id"] == object_id:
                if obj["is_assigned"]:
                    return f"Object '{object_id}' is already assigned."
                else:
                    object_to_be_assigned = obj
                    break
        if not object_to_be_assigned:
            return f"Object '{object_id}' not found."

        # Find the closest robot (idle robot) to this object
        filtered_objects = [object_to_be_assigned]
        result = evaluate_objects_and_robots(filtered_objects, robot_data)
        closest_robot_id = result["robot_id"]

        if closest_robot_id:
            # Update the robot and object statuses
            closest_robot = next(robot for robot in robot_data if robot["robot_id"] == closest_robot_id)
            closest_robot["idle"] = False
            closest_robot["status"]="unknown"
            object_to_be_assigned["is_assigned"] = True

            # Send data to RL
            robot_data = assign_object_to_robot(robot_data, closest_robot_id, object_to_be_assigned["coords"])  
            update_data(robot_data,object_data)
            # Custom message
            return f"Robot '{closest_robot_id}' has been assigned to object '{object_id}' at position {object_to_be_assigned['coords']}."

    # Case 2: If robot_id is given but not object_id
    elif robot_id and not object_id:
        closest_object = None
        min_distance = float('inf')

        # Find the robot and check if it is idle
        robot = next((robot for robot in robot_data if robot["robot_id"] == robot_id), None)
        
        if robot is None:
            return f"Robot with ID '{robot_id}' not found."

        if not robot["idle"]:
            return f"Robot '{robot_id}' is not idle and cannot be assigned to an object."

        # If object_type is provided, filter objects by type
        filtered_objects = [obj for obj in object_data if not obj["is_assigned"]]
        if object_type:
            filtered_objects = [obj for obj in filtered_objects if obj["type"] == object_type]

        # Find the closest object of the given type to the specific robot
        for obj in filtered_objects:
            distance = np.linalg.norm(np.array(obj["coords"]) - np.array(robot["position"]))
            if distance < min_distance:
                min_distance = distance
                closest_object = obj

        if not closest_object:
            return f"No available object of type '{object_type}' found for robot '{robot_id}'."

        # Update the robot and object statuses
        robot["idle"] = False
        robot["status"]="unknown"
        closest_object["is_assigned"] = True

        # Send data to RL
        robot_data = assign_object_to_robot(robot_data, robot_id, closest_object["coords"])  
        update_data(robot_data,object_data)

        # Custom message
        return f"Robot '{robot_id}' has been assigned to the nearest object of type '{object_type}' at position {closest_object['coords']}."

    # Case 3: If both object_id and robot_id are given
    elif object_id and robot_id:
        object_to_be_assigned = None
        for obj in object_data:
            if obj["object_id"] == object_id:
                if obj["is_assigned"]:
                    return f"Object '{object_id}' is already assigned."
                else:
                    object_to_be_assigned = obj
                    break

        if not object_to_be_assigned:
            return f"Object '{object_id}' not found."

        # Check if the given robot is idle
        robot = next((robot for robot in robot_data if robot["robot_id"] == robot_id), None)
        if not robot or not robot.get("idle", False):
            return f"Robot '{robot_id}' is either not found or not idle."

        # Assign the robot to the object
        robot["idle"] = False
        robot["status"]="unknown"
        object_to_be_assigned["is_assigned"] = True

        # Send data to RL
        robot_data = assign_object_to_robot(robot_data, robot_id, object_to_be_assigned["coords"])
        update_data(robot_data,object_data)

        # Custom message
        return f"Robot '{robot_id}' has been assigned to object '{object_id}' at position {object_to_be_assigned['coords']}."

    else:
        return "Invalid input: At least one of object_id or robot_id must be provided."



def handle_robot_task_complete(robot_id,object_coord):
    robot_data,object_data=load_data()
    robot_data,object_data=reset_robot_and_object_state(robot_data,object_data,robot_id,object_coord)
    update_data(robot_data,object_data)
    status_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "LLM", "LLM", "status_retrive.txt")
    with open(status_file, "w", encoding="utf-8") as file:
        file.write(robot_id)
    
