import os
import numpy as np
import json

dir_name = os.path.dirname(os.path.realpath(__name__))
with open(os.path.join(dir_name,"llm","datafinal.json")) as json_file:
    robot_data = json.load(json_file)["robot"]
with open(os.path.join(dir_name,"llm","datafinal.json")) as json_file:
    object_data = json.load(json_file)["target"]

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

def burn(robot_data, object_data):
    final_data = {"robot":robot_data, "target":object_data}
    with open(os.path.join(dir_name,"llm","datafinal.json"),'w') as json_file:
        json.dump(final_data, json_file, indent=4)


def evaluate_objects_and_robots(object_data, robot_data, find=False):
    nearest_robot = None
    nearest_object_coord = None
    min_distance = float('inf')

    # Filter robots that are idle (idle=True)
    
    
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


def send_to_RL(data):
    '''Send data to RL'''
    print("Sending to RL:", data)


def find_nearest(object_type):
    update_coord()
    # Filter objects based on type
    filtered_objects = [obj for obj in object_data if obj["type"] == object_type]
    
    # Get the nearest object and robot pair using the updated function
    result = evaluate_objects_and_robots(filtered_objects, robot_data, find=True)
    
    # Custom message for success or failure
    if isinstance(result, str):
        return result
    message = f"Robot '{result['robot_id']}' is closest to the object of type '{object_type}' at position {result['object_position']}."
    return message


def assign_general(object_type):
    update_coord()
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
    closest_object = next(obj for obj in object_data if obj["coords"] == closest_object_coord)
    closest_object["is_assigned"] = True

    # Send data to RL
    data_to_send = {closest_robot_id: closest_object["coords"]}
    send_to_RL(data_to_send)

    # Custom message
    message = f"Robot '{closest_robot_id}' has been assigned to the nearest object of type '{object_type}' at position {closest_object_coord}."
    burn()
    return message


def update_coord():
    with open(os.path.join(dir_name,"llm","datafinal.json"), 'r') as json_file:
        updated_robot_data = json.load(json_file)["robot"]
    updated_positions = {robot['robot_id']: robot['position'] for robot in updated_robot_data}    
    for robot in robot_data:
        if robot['robot_id'] in updated_positions:
            robot['position'] = updated_positions[robot['robot_id']]


def assign_specific(object_id=None, robot_id=None, object_type=None):
    '''Assigns a robot to a specific object based on given parameters'''
    update_coord()

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
            object_to_be_assigned["is_assigned"] = True

            # Send data to RL
            data_to_send = {closest_robot_id: object_to_be_assigned["coords"]}
            send_to_RL(data_to_send)

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
        closest_object["is_assigned"] = True

        # Send data to RL
        data_to_send = {robot_id: closest_object["coords"]}
        send_to_RL(data_to_send)
        burn()

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
        object_to_be_assigned["is_assigned"] = True

        # Send data to RL
        data_to_send = {robot_id: object_to_be_assigned["coords"]}
        for i in robot_data:
            if i["robot_id"]==robot_id:
                i["assigned_object_coords"]=object_to_be_assigned['coords']
        send_to_RL(data_to_send)

        

        # Custom message
        burn(robot_data, object_data)

        return f"Robot '{robot_id}' has been assigned to object '{object_id}' at position {object_to_be_assigned['coords']}."

    else:
        return "Invalid input: At least one of object_id or robot_id must be provided."

assign_specific(object_id="Target_1",robot_id="robot_1")
assign_specific(object_id="Target_2",robot_id="robot_2")
