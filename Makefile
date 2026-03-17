.PHONY: help install install-vector install-dialogue install-all lint test run-api run-demo smoke-test real-model-test clean

help:
	@echo "Bay-Max development commands:"
	@echo "  make install           - Install dev dependencies only (no vision/vector/dialogue)"
	@echo "  make install-vector    - Install dev + vector (LanceDB + embeddings)"
	@echo "  make install-dialogue  - Install dev + dialogue (Transformers)"
	@echo "  make install-all       - Install dev + vision + vector + dialogue"
	@echo "  make lint              - Run ruff linter"
	@echo "  make test              - Run pytest test suite"
	@echo "  make run-api           - Start FastAPI server on port 8000"
	@echo "  make run-demo          - Start Gradio demo on port 7860"
	@echo "  make smoke-test        - Run local dialogue smoke-test script"
	@echo "  make real-model-test   - Run real-model smoke-test (requires LLM backend in .env)"
	@echo "  make clean             - Remove build artifacts and caches"

install:
	pip install -e ".[dev]"

install-vector:
	pip install -e ".[dev,vector]"

install-dialogue:
	pip install -e ".[dev,dialogue]"

install-all:
	pip install -e ".[all]"

lint:
	ruff check src/ apps/ tests/ scripts/

test:
	pytest tests/ -v

run-api:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

run-demo:
	python apps/demo/gradio_app.py

smoke-test:
	python scripts/dialogue_smoke.py

real-model-test:
	python scripts/dialogue_smoke.py --real-model

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache dist build
