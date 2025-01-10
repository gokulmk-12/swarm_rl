from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import uvicorn 
import logging
from tool_caller import  send_query
from functions import handle_robot_task_complete
import os

# for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
#     logging.getLogger(logger_name).handlers = [logging.NullHandler()]

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

status_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "LLM", "LLM", "status_retrive.txt")


app= FastAPI()


class Query(BaseModel):
    query:str

class Update(BaseModel):
    robot_id:str
    object_coord:list



@app.post('/query')
def create_blog(req:Query):
    res=send_query(req.query)
    # return req
    return res


@app.get('/reset')
def update_status():
    with open(status_file, "w", encoding="utf-8") as file:
        file.write("")
    return "success"

@app.post('/set')
def update_status(req:Update):

    status=req.robot_id
    handle_robot_task_complete(req.robot_id, req.object_coord)
    return "success"

    
uvicorn.run(app,host="127.0.0.1",port=9000)