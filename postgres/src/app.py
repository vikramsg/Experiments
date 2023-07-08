from fastapi import Depends, FastAPI
from src.db import get_db
from sqlalchemy import text
from src.model import Task, Tasks

app = FastAPI()


@app.post("/tasks")
async def create_task(task: str, db=Depends(get_db)) -> None:
    query = text("INSERT INTO tasks (task) VALUES (:task)")
    db.execute(query, {"task": task})


@app.get("/tasks", response_model=Tasks)
async def get_tasks(db=Depends(get_db)) -> None:
    query = text("SELECT task FROM tasks")
    fetched_tasks = db.execute(query).all()

    tasks: Tasks = [Task(id=task[0], task=task[1]) for task in fetched_tasks]

    return tasks
