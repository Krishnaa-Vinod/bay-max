.PHONY: help install install-vector install-dialogue install-tts install-speech install-all lint test run-api run-demo smoke-test real-model-test live-webcam live-replay live-webcam-tts live-webcam-speech verify-007 verify-008 clean

help:
	@echo "Bay-Max development commands:"
	@echo "  make install           - Install dev dependencies only (no vision/vector/dialogue)"
	@echo "  make install-vector    - Install dev + vector (LanceDB + embeddings)"
	@echo "  make install-dialogue  - Install dev + dialogue (Transformers)"
	@echo "  make install-tts       - Install dev + TTS (Kokoro + sounddevice)"
	@echo "  make install-speech    - Install dev + Speech input (faster-whisper + sounddevice)"
	@echo "  make install-all       - Install everything"
	@echo "  make lint              - Run ruff linter"
	@echo "  make test              - Run pytest test suite"
	@echo "  make run-api           - Start FastAPI server on port 8000"
	@echo "  make run-demo          - Start Gradio demo on port 7860"
	@echo "  make smoke-test        - Run local dialogue smoke-test script"
	@echo "  make real-model-test   - Run real-model smoke-test (requires LLM backend in .env)"
	@echo "  make live-webcam       - Start live companion in webcam mode"
	@echo "  make live-webcam-tts   - Start live companion with TTS enabled"
	@echo "  make live-webcam-speech - Start live companion with TTS + speech input"
	@echo "  make live-replay       - Start live companion in replay mode (set REPLAY_PATH)"
	@echo "  make verify-007        - Run iteration 007 local verification"
	@echo "  make verify-008        - Run iteration 008 tests"
	@echo "  make clean             - Remove build artifacts and caches"

install:
	pip install -e ".[dev]"

install-vector:
	pip install -e ".[dev,vector]"

install-dialogue:
	pip install -e ".[dev,dialogue]"

install-tts:
	pip install -e ".[dev,tts]"

install-speech:
	pip install -e ".[dev,speech]"

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

live-webcam:
	python scripts/live_companion.py

live-webcam-tts:
	python scripts/live_companion.py --tts-backend kokoro

live-webcam-speech:
	python scripts/live_companion.py --tts-backend kokoro --stt-backend faster_whisper

live-replay:
	python scripts/live_companion.py --replay $(REPLAY_PATH) --no-overlay

verify-007:
	python scripts/verify_007.py --tts-only

verify-008:
	pytest tests/test_iteration_008.py -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache dist build
