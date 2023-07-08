from fastapi import Depends, FastAPI
from src.db import get_db

app = FastAPI()


@app.post("/tasks")
async def create_task(task: str, db=Depends(get_db)) -> None:
    db.execute("INSERT INTO tasks (task) VALUES (:task)", {"task": task})
