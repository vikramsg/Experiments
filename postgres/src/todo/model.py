from typing import List

from pydantic import BaseModel


class Task(BaseModel):
    id: int
    task: str
