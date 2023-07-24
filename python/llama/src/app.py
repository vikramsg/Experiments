from fastapi import FastAPI

from src.llama.routes import llama_router


def init_routers(app: FastAPI) -> None:
    app.include_router(llama_router)


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLAMA 2",
        description="Get response from LLAMA 2",
        version="0.1.0",
    )

    init_routers(app=app)

    return app
