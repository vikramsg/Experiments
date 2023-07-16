import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.db import get_db
from src.todo.routes import todo_router
from src.middleware import CustomExceptionMiddleware

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# app = FastAPI()
def init_routers(app: FastAPI) -> None:
    app.include_router(todo_router)


def init_custom_exception(app: FastAPI) -> None:
    app.add_middleware(CustomExceptionMiddleware)


def create_app() -> FastAPI:
    _app = FastAPI(
        title="Vikram's todo app",
        description="Vikram's awesome to do app",
        version="0.1.0",
    )

    @_app.on_event("startup")
    def startup_event() -> None:
        get_db()

    # Keep this on top of all the other middlewares
    init_custom_exception(app=_app)

    init_routers(app=_app)

    return _app
