import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import List
from groq import Groq
import instructor
from functions import find_nearest, assign_general, assign_specific,load_data,object_ids,object_types,robot_id

# Mapping tool names to functions


# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq()

# Define model structure for Groq response
class Argument(BaseModel):
    argument_name: str
    argument_value: str

class Tool(BaseModel):
    tool_name: str
    arguments: List[Argument]

class ResponseModel(BaseModel):
    tool_calls: List[Tool]

class ChatModel(BaseModel):
    response: str
# Initialize instructor with Groq
client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)

def run_conversation(user_prompt, tools):
    messages = [
        {
            "role": "system",
            "content": f"""
            **Instructions**:
            - Analyze the given steps, which include tool names and arguments.
            - Use these steps to generate the sequence of tool calls, ensuring correct ordering and dependency management.
            - for each single task CALL A SINGLE FUNCTION ONLY UNLESS UTILL THERE IS MULTIPLE ASSIGNMENT IN THE USER QUERY
            - You can call only one tool at a time.
            -**IF THE QUERY incomplete to call a function (e.g. "assign".. here it dosent sepicify what to assign who to assign )RETURN EMPTY TOOL_CALL**
            - You have access to the following tools: {tools}.
            - for arguments use Object types:{object_types} , Object ids:{object_ids} and Robot id:{robot_id}. Choose the closest enough matching argument to the one received in the query.
            - For example, if exsisting oject type is "Object_1" .. and user says anything close to it .. the tool call should put argument value as "Object_1" only.
            - If you receive a 
            """
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    # Make the Groq API call
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        response_model=ResponseModel,
        messages=messages,
        temperature=0.9,
        max_tokens=1000,
    )

    return response.tool_calls


def general_conversation(query):
    robot_data,object_data=load_data()
    
    messages = [
        {
            "role": "system",
            "content": f"""
            you are a freindly asistant who engages in genral conversatition with the user ,You tell that you are an asistant who will help in assigning and finding objects to robot .or if they want to know status of any robots 
            
            you have access to robot and object status for refrence
            robot_data:{robot_data}
            object_data:{object_data}
            
            """
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        response_model=ChatModel,
        messages=messages,
        temperature=0.9,
        max_tokens=1000,
    )

    return response.response

    
    
def output(query, tools):
    tool_calls = run_conversation(query, tools)
    output = []

    for i in tool_calls:
        output.append(i.json())

    return output

# Define tool descriptions and arguments
# Types_obj = ("charging","vending", "servicebot")
# obj_id = ("Object_1", "Object_2", "Object_3", "Object_4")
# robot_id = ("robot_1", "robot_2", "robot_3", "robot_4")

tools=[
     {
    "tool_name": "find_nearest",
    "tool_description": "finds the nearest object of a given type",
    "args": [
    {
    "arg_name": "object_type",
    "arg_type": "string",
    "is_array": False,
    "is_required": True,
    "one_of": object_types
    }
    ]
},
{
    "tool_name": "assign_general",
    "tool_description": "assigns a robot nearest to a given object type",
    "args": [
    {
    "arg_name": "object_type",
    "arg_type": "string",
    "is_array": False,
    "is_required": True,
    "one_of": object_types
    }
    ]
},
{
    "tool_name": "assign_specific",
    "tool_description": """The assign_specific function assigns a robot to a specific object based on three possible modes: 
    (1) When only an 'object_id' is provided, it finds the closest idle robot to the object.
    (2) When only a 'robot_id' is provided, the function identifies the closest available object of the same 'object_type' as the robot, assigns it to the robot, and updates their statuses accordingly. 
    (3) When both an 'object_id' and 'robot_id' are given, it directly assigns the specified robot to the specified object, ensuring that neither is already assigned, and then updates both their statuses before sending the data to the RL system. This flexibility allows efficient handling of various assignment scenarios.""",
    "args": [
    {
    "arg_name": "object_id",
    "arg_type": "string",
    "is_array": False,
    "is_required": False,
    "one_of": object_ids
    },
    {
    "arg_name": "robot_id",
    "arg_type": "string",
    "is_array": False,
    "is_required": False,
    "one_of": robot_id
    },
    {
    "arg_name": "object_type",
    "arg_type": "string",
    "is_array": False,
    "is_required": False,
    "one_of": object_types
    }
    
    ]
},
{
    "tool_name": "general_conversation",
    "tool_description": "this tool is used to handel general greeting such as 'hi' and 'hello' and status  of robots,to know which object is robot assigned too, other general info",
    "args": [
    {
    "arg_name": "query",
    "arg_type": "string",
    "is_array": False,
    "is_required": True,
    }
    ]
},
{
    "tool_name": "invalid_query",
    "tool_description": "called when the user query dosent makes any sense",
    "args": [
    {
    "arg_name": "object_type",
    "arg_type": "string",
    "is_array": False,
    "is_required": True,
    "one_of": object_types
    }
    ]
}]


tool_map = {
    "find_nearest": find_nearest,
    "assign_general": assign_general,
    "assign_specific": assign_specific,
    "general_conversation":general_conversation
}

def send_query(user_prompt):
    response=[]
    L = output(user_prompt, tools)
    print(L)
    if L==[]:
        return(["invalid instruction"])
    args = {}
    for i in L:
        i = json.loads(i)
        if i["tool_name"]=="invalid_query":
            return(["invalid instruction"])
        else:
            func = tool_map[i["tool_name"]]
            for arg in i["arguments"]:
                args[arg["argument_name"]] = arg["argument_value"]
            response.append(func(**args))
    return response

    