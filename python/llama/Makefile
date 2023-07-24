.PHONY: install test lint check 

help:
	@echo "install - install dependencies with poetry"
	@echo "lint - run linter"
	@echo "check - run static checks"
	@echo "test - run tests"

lint:
	./linter.sh
	
install:
	poetry install --no-root
	poetry shell

check:
	./static_checks.sh	

test:
	pytest


up:
	@echo "Loading environment variables from .env file"
	@dotenv run docker compose up --build 

down:
	docker-compose down