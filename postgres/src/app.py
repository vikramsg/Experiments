import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.db import get_db
from src.model import Task, Tasks

logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI()


@app.post("/task", status_code=status.HTTP_201_CREATED)
async def create_task(task: str, db=Depends(get_db)) -> Task:
    try:
        query = text("INSERT INTO tasks (task) VALUES (:task) RETURNING id")
        result = db.execute(query, {"task": task})
        db.commit()
        task_id = result.fetchone()[0]
    except IntegrityError as e:
        logger.warning(f"Error in creating task: {e.args[0]}")
        if "psycopg2.errors.UniqueViolation" in e.args[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Task already exists"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unknown error occurred",
            )

    return Task(task=task, id=task_id)


@app.get("/tasks")
async def get_tasks(db=Depends(get_db)) -> List[Task]:
    logger.info("Fetching all tasks.")
    query = text("SELECT id, task FROM tasks")
    fetched_tasks = db.execute(query).all()

    tasks = [Task(id=task[0], task=task[1]) for task in fetched_tasks]

    return tasks


@app.put("/tasks/{id}")
async def update_tasks(id: int, new_task: str, db=Depends(get_db)) -> Task:
    logging.info(f"Updating task with id: {id}")
    query = text("UPDATE tasks SET task = :task WHERE id= :id")
    db.execute(query, {"task": new_task, "id": id})
    db.commit()

    return Task(task=new_task, id=id)


@app.delete("/tasks/{id}")
async def delete_tasks(id: int, db=Depends(get_db)) -> Task:
    query = text("SELECT id, task FROM tasks WHERE id= :id")
    query_result = db.execute(query, {"id": id}).fetchone()
    if query_result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such Id")

    logging.info(f"Deleting task with id: {id}")
    query = text("DELETE FROM tasks WHERE id= :id RETURNING id, task")
    result = db.execute(query, {"id": id})
    db.commit()

    id, task = result.fetchone()

    return Task(id=id, task=task)
