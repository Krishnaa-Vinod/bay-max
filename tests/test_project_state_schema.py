"""Tests for project_state.json schema validity."""

import json
import os

PROJECT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "project_state.json"
)


def test_project_state_file_exists():
    assert os.path.isfile(PROJECT_STATE_PATH), (
        f"project_state.json not found at {PROJECT_STATE_PATH}"
    )


def test_project_state_is_valid_json():
    with open(PROJECT_STATE_PATH) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_project_state_has_required_fields():
    with open(PROJECT_STATE_PATH) as f:
        data = json.load(f)

    required_fields = [
        "iteration",
        "branch_name",
        "status",
        "last_updated",
        "modules_implemented",
        "modules_stubbed",
        "tests_passing",
        "api_endpoints",
        "known_issues",
        "next_tasks",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"


def test_project_state_iteration_is_string():
    with open(PROJECT_STATE_PATH) as f:
        data = json.load(f)
    assert isinstance(data["iteration"], str)


def test_project_state_modules_are_lists():
    with open(PROJECT_STATE_PATH) as f:
        data = json.load(f)
    assert isinstance(data["modules_implemented"], list)
    assert isinstance(data["modules_stubbed"], list)
    assert len(data["modules_implemented"]) > 0


def test_project_state_api_endpoints_are_listed():
    with open(PROJECT_STATE_PATH) as f:
        data = json.load(f)
    assert isinstance(data["api_endpoints"], list)
    assert len(data["api_endpoints"]) >= 8
