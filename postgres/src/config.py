import os


def postgres_url() -> str:
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")

    return f"postgresql://{user}:{password}@db/{db}"
