.PHONY: help install install-dev run test lint format docker-build docker-run \
       data-generate train evaluate pipeline clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install all dependencies (production + dev)
	pip install -r requirements-dev.txt

run: ## Start the FastAPI server (development)
	python -m uvicorn src.forecast.api.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run tests with coverage
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	black --check src tests scripts
	isort --check-only src tests scripts
	pylint src

format: ## Auto-format code
	black src tests scripts
	isort src tests scripts

docker-build: ## Build Docker image
	docker build -t finance-forecasting:latest .

docker-run: ## Run with docker-compose
	docker-compose up --build

docker-down: ## Stop docker-compose services
	docker-compose down -v

data-generate: ## Generate synthetic financial data
	python -m scripts.generate_data

train: ## Train all forecasting models
	python -m scripts.train_models

pipeline: ## Run end-to-end pipeline
	python -m scripts.run_pipeline

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -rf data/raw/* data/processed/* data/models/*
