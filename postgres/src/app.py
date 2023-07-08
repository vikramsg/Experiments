from fastapi import Depends, FastAPI
from src.db import get_db
from sqlalchemy import text

app = FastAPI()


@app.post("/tasks")
async def create_task(task: str, db=Depends(get_db)) -> None:
    query = text("INSERT INTO tasks (task) VALUES (:task)")
    db.execute(query, {"task": task})


@app.get("/tasks")
async def get_tasks(db=Depends(get_db)) -> None:
    query = text("SELECT task FROM tasks")
    tasks = db.execute(query).fetchall()
