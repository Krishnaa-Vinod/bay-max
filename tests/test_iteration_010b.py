"""Tests for Iteration 010b: Companion UI hotfixes for local readiness."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from uuid import uuid4

from baymax.live.schemas import LiveRuntimeStatus, SessionLifecycleState


class TestLiveSnapshotNonPlaceholderValues:
    """Verify that live snapshot contains real/null values, not hardcoded placeholders."""

    def test_live_runtime_status_has_perception_fields(self):
        """LiveRuntimeStatus should have perception tracking fields."""
        status = LiveRuntimeStatus()

        # These fields should exist and be nullable (None by default, not hardcoded)
        assert hasattr(status, 'recognition_confidence')
        assert hasattr(status, 'posture')
        assert hasattr(status, 'engagement')
        assert hasattr(status, 'engagement_score')

        # Default values should be None, not hardcoded placeholders
        assert status.recognition_confidence is None
        assert status.posture is None
        assert status.engagement is None
        assert status.engagement_score is None

    def test_live_runtime_status_has_memory_refs_field(self):
        """LiveRuntimeStatus should have last_memory_refs field."""
        status = LiveRuntimeStatus()

        assert hasattr(status, 'last_memory_refs')
        assert status.last_memory_refs == []

    def test_build_snapshot_message_uses_real_values(self):
        """build_snapshot_message should use status values, not placeholders."""
        # Import here to avoid issues with module loading order
        from apps.api.live_ws import build_snapshot_message

        status = LiveRuntimeStatus(
            live_mode_active=True,
            session_status=SessionLifecycleState.ACTIVE,
            current_user_id=uuid4(),
            current_user_display_name="TestUser",
            recognition_confidence=0.85,
            posture="upright",
            engagement="high",
            engagement_score=0.9,
            last_memory_refs=["Memory 1", "Memory 2"],
        )

        # Build snapshot without live_runtime set (uses status values only)
        with patch('apps.api.live_ws._live_runtime', None):
            snapshot = build_snapshot_message(status)

        # Verify perception values come from status, not hardcoded
        assert snapshot['perception']['recognition_confidence'] == 0.85
        assert snapshot['perception']['posture'] == "upright"
        # engagement is derived from engagement_score or engagement label
        assert snapshot['perception']['engagement'] == 0.9

        # Verify memory hits come from status.last_memory_refs
        assert len(snapshot['memory_hits']) == 2
        assert snapshot['memory_hits'][0]['text'] == "Memory 1"
        assert snapshot['memory_hits'][1]['text'] == "Memory 2"

    def test_build_snapshot_message_returns_null_when_unavailable(self):
        """build_snapshot_message should return null (None) when values are unavailable."""
        from apps.api.live_ws import build_snapshot_message

        status = LiveRuntimeStatus(
            live_mode_active=True,
            session_status=SessionLifecycleState.IDLE,
        )

        with patch('apps.api.live_ws._live_runtime', None):
            snapshot = build_snapshot_message(status)

        # These should be null/None, not hardcoded defaults
        assert snapshot['perception']['recognition_confidence'] is None
        assert snapshot['perception']['posture'] is None
        assert snapshot['perception']['engagement'] is None

        # Memory hits should be empty list when none available
        assert snapshot['memory_hits'] == []


class TestMemoryHitsUIPath:
    """Verify retrieved memory hits flow through to the UI data path."""

    def test_memory_refs_stored_in_status_from_response(self):
        """Memory refs from response should be stored in status.last_memory_refs."""
        status = LiveRuntimeStatus()

        # Simulate what happens when a response is generated
        mock_memory_refs = ["User likes coffee", "User works remotely", "Had a meeting yesterday"]
        refs = mock_memory_refs[:10] if mock_memory_refs else []
        status.last_memory_refs = refs

        assert len(status.last_memory_refs) == 3
        assert "User likes coffee" in status.last_memory_refs

    def test_memory_hits_included_in_snapshot(self):
        """Memory hits should appear in WebSocket snapshot."""
        from apps.api.live_ws import build_snapshot_message

        status = LiveRuntimeStatus(
            live_mode_active=True,
            session_status=SessionLifecycleState.ACTIVE,
            last_memory_refs=["Fact about user", "Previous conversation topic"],
        )

        with patch('apps.api.live_ws._live_runtime', None):
            snapshot = build_snapshot_message(status)

        # Memory hits should be present and formatted correctly
        assert 'memory_hits' in snapshot
        assert len(snapshot['memory_hits']) == 2

        # Each hit should have text and score fields
        for hit in snapshot['memory_hits']:
            assert 'text' in hit
            assert 'score' in hit


class TestPackageJsonTestScript:
    """Verify package.json has the test scripts configured."""

    def test_package_json_has_test_script(self):
        """package.json should have a test script that runs vitest."""
        import json
        from pathlib import Path

        package_json_path = Path(__file__).parent.parent / "apps" / "ui" / "package.json"

        with open(package_json_path) as f:
            pkg = json.load(f)

        assert 'scripts' in pkg
        assert 'test' in pkg['scripts']
        assert 'vitest' in pkg['scripts']['test']

        # test:watch should also exist
        assert 'test:watch' in pkg['scripts']
        assert 'vitest' in pkg['scripts']['test:watch']


class TestFrontendTypesNullable:
    """Verify frontend TypeScript types handle nullable values."""

    def test_live_ts_perception_types_allow_null(self):
        """PerceptionData in live.ts should allow null values for runtime fields."""
        from pathlib import Path

        types_path = Path(__file__).parent.parent / "apps" / "ui" / "src" / "types" / "live.ts"

        with open(types_path) as f:
            content = f.read()

        # These fields should allow null
        assert 'recognition_confidence: number | null' in content
        assert 'posture: string | null' in content
        assert 'engagement: number | null' in content

    def test_memory_hit_type_allows_null_score(self):
        """MemoryHit in live.ts should allow null score."""
        from pathlib import Path

        types_path = Path(__file__).parent.parent / "apps" / "ui" / "src" / "types" / "live.ts"

        with open(types_path) as f:
            content = f.read()

        assert 'score: number | null' in content
