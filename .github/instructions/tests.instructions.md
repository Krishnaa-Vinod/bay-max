# Testing Instructions

## Test Organization
- All tests live in `tests/`.
- Test files are named `test_*.py`.
- Use pytest as the test runner.
- Use `pytest-asyncio` for async test functions.

## Test Categories
- **Health tests** (`test_health.py`): Verify API endpoints respond correctly.
- **Schema tests** (`test_memory_models.py`): Verify Pydantic models validate correctly.
- **State tests** (`test_project_state_schema.py`): Verify project state JSON schema.

## Running Tests
```bash
make test                    # Run all tests
pytest tests/test_health.py  # Run specific test file
pytest -v                    # Verbose output
```

## Conventions
- Test functions should be descriptive: `test_create_user_returns_201`.
- Use fixtures for shared test setup.
- Use `httpx.AsyncClient` with `ASGITransport` for testing FastAPI.
- Do not depend on external services or API keys in tests.
- Keep test fixtures lightweight and committed to the repo.
