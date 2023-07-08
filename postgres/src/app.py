from fastapi import Depends, FastAPI, Response
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.db import get_db
from src.model import Task, Tasks

app = FastAPI()


@app.post("/tasks")
async def create_task(task: str, db=Depends(get_db)) -> Response:
    try:
        query = text("INSERT INTO tasks (task) VALUES (:task)")
        db.execute(query, {"task": task})
        db.commit()
    except IntegrityError as e:
        if "psycopg2.errors.UniqueViolation" in e.args[0]:
            return Response(status_code=400, content="Task already exists")
        else:
            return Response(status_code=400, content=f"Unknown error. {e.args[0]}")

    return Response(status_code=201, content="Task created successfully")


@app.get("/tasks", response_model=Tasks)
async def get_tasks(db=Depends(get_db)) -> Tasks:
    query = text("SELECT id, task FROM tasks")
    fetched_tasks = db.execute(query).all()

    tasks = [Task(id=task[0], task=task[1]) for task in fetched_tasks]

    return Tasks(tasks=tasks)
