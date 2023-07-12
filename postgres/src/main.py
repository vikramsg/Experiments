from src.app import create_app

app = create_app()


# poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src --no-access-log
