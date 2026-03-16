.PHONY: help install lint test run-api run-demo clean

help:
	@echo "Bay-Max development commands:"
	@echo "  make install   - Install dependencies in dev mode"
	@echo "  make lint      - Run ruff linter"
	@echo "  make test      - Run pytest test suite"
	@echo "  make run-api   - Start FastAPI server on port 8000"
	@echo "  make run-demo  - Start Gradio demo on port 7860"
	@echo "  make clean     - Remove build artifacts and caches"

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ apps/ tests/

test:
	pytest tests/ -v

run-api:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

run-demo:
	python apps/demo/gradio_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache dist build
