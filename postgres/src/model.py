from typing import List
from pydantic import BaseModel


class Task(BaseModel):
    id: int
    task: str


class Tasks(BaseModel):
    tasks: List[Task]
