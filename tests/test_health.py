"""Tests for the /healthz endpoint."""

import pytest
from apps.api.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_healthz_returns_ok(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_create_user(client):
    response = await client.post(
        "/v1/users",
        json={"display_name": "Test User"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["display_name"] == "Test User"
    assert "id" in data


@pytest.mark.asyncio
async def test_get_user_not_found(client):
    response = await client.get("/v1/users/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_session(client):
    response = await client.post("/v1/sessions", json={})
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "active"
    assert "id" in data
