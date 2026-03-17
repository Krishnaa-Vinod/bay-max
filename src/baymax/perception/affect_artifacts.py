"""Affect artifact logging and timeline tracking for Bay-Max."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from baymax.schemas.perception import AffectTimelinePoint, EmotionResult, SmoothedAffectState

logger = logging.getLogger(__name__)


class AffectArtifactLogger:
    """Logs affect analysis timeline and strategy changes for verification artifacts."""

    def __init__(self, artifact_dir: str = "./artifacts/affect"):
        """Initialize affect artifact logger.

        Args:
            artifact_dir: Directory to save affect artifacts
        """
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Timeline tracking
        self._timeline: list[AffectTimelinePoint] = []
        self._strategy_changes: list[dict[str, Any]] = []

        # Session metadata
        self._session_id: UUID | None = None
        self._session_start: datetime | None = None

        logger.debug("Initialized AffectArtifactLogger with artifact_dir: %s", self.artifact_dir)

    def start_session(self, session_id: UUID) -> None:
        """Start tracking a new session.

        Args:
            session_id: Session UUID
        """
        self._session_id = session_id
        self._session_start = datetime.utcnow()
        self._timeline.clear()
        self._strategy_changes.clear()

        logger.info("Started affect artifact logging for session %s", session_id)

    def log_emotion_result(
        self,
        emotion_result: EmotionResult,
        smoothed_state: SmoothedAffectState | None = None,
    ) -> None:
        """Log raw emotion result and smoothed state to timeline.

        Args:
            emotion_result: Raw emotion analysis result
            smoothed_state: Optional smoothed affect state
        """
        if not self._session_id:
            logger.warning("No active session for affect logging")
            return

        timeline_point = AffectTimelinePoint(
            timestamp=emotion_result.timestamp,
            valence=smoothed_state.valence if smoothed_state else emotion_result.valence,
            arousal=smoothed_state.arousal if smoothed_state else emotion_result.arousal,
            confidence=smoothed_state.confidence if smoothed_state else emotion_result.confidence,
            raw_valence=emotion_result.valence,
            raw_arousal=emotion_result.arousal,
            backend=emotion_result.backend,
        )

        self._timeline.append(timeline_point)

        logger.debug(
            "Logged affect timeline point: valence=%.3f, arousal=%.3f, confidence=%.3f",
            timeline_point.valence, timeline_point.arousal, timeline_point.confidence
        )

    def log_strategy_change(
        self,
        old_strategy: str,
        new_strategy: str,
        reason: str,
        affect_context: dict[str, Any] | None = None,
    ) -> None:
        """Log a strategy change triggered by affect analysis.

        Args:
            old_strategy: Previous response strategy
            new_strategy: New response strategy
            reason: Reason for the change
            affect_context: Optional affect context that triggered the change
        """
        if not self._session_id:
            logger.warning("No active session for strategy change logging")
            return

        change_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "old_strategy": old_strategy,
            "new_strategy": new_strategy,
            "reason": reason,
            "affect_context": affect_context or {},
            "session_id": str(self._session_id),
        }

        self._strategy_changes.append(change_record)

        # Also add to timeline with strategy hint
        if self._timeline:
            self._timeline[-1].strategy_hint = f"{old_strategy} -> {new_strategy}: {reason}"

        logger.info(
            "Logged strategy change: %s -> %s (reason: %s)",
            old_strategy, new_strategy, reason
        )

    def save_session_artifacts(self) -> list[str]:
        """Save all artifacts for the current session.

        Returns:
            List of generated artifact file paths
        """
        if not self._session_id or not self._session_start:
            logger.warning("No active session to save artifacts for")
            return []

        generated_files = []
        session_prefix = f"session_{self._session_id}_{self._session_start.strftime('%Y%m%d_%H%M%S')}"

        try:
            # Save timeline as JSON
            timeline_json_path = self.artifact_dir / f"{session_prefix}_timeline.json"
            timeline_data = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "valence": point.valence,
                    "arousal": point.arousal,
                    "confidence": point.confidence,
                    "raw_valence": point.raw_valence,
                    "raw_arousal": point.raw_arousal,
                    "backend": point.backend,
                    "strategy_hint": point.strategy_hint,
                }
                for point in self._timeline
            ]
            with open(timeline_json_path, "w", encoding="utf-8") as f:
                json.dump(timeline_data, f, indent=2)
            generated_files.append(str(timeline_json_path))

            # Save timeline as CSV for analysis
            timeline_csv_path = self.artifact_dir / f"{session_prefix}_timeline.csv"
            with open(timeline_csv_path, "w", newline="", encoding="utf-8") as f:
                if self._timeline:
                    writer = csv.DictWriter(f, fieldnames=[
                        "timestamp", "valence", "arousal", "confidence",
                        "raw_valence", "raw_arousal", "backend", "strategy_hint"
                    ])
                    writer.writeheader()
                    for point in self._timeline:
                        writer.writerow({
                            "timestamp": point.timestamp.isoformat(),
                            "valence": point.valence,
                            "arousal": point.arousal,
                            "confidence": point.confidence,
                            "raw_valence": point.raw_valence,
                            "raw_arousal": point.raw_arousal,
                            "backend": point.backend,
                            "strategy_hint": point.strategy_hint or "",
                        })
            generated_files.append(str(timeline_csv_path))

            # Save strategy changes log
            if self._strategy_changes:
                strategy_log_path = self.artifact_dir / f"{session_prefix}_strategy_changes.json"
                with open(strategy_log_path, "w", encoding="utf-8") as f:
                    json.dump(self._strategy_changes, f, indent=2)
                generated_files.append(str(strategy_log_path))

            # Save summary metadata
            summary_path = self.artifact_dir / f"{session_prefix}_summary.json"
            summary_data = {
                "session_id": str(self._session_id),
                "session_start": self._session_start.isoformat(),
                "session_end": datetime.utcnow().isoformat(),
                "timeline_points": len(self._timeline),
                "strategy_changes": len(self._strategy_changes),
                "artifacts_generated": generated_files,
                "duration_minutes": (
                    (datetime.utcnow() - self._session_start).total_seconds() / 60.0
                    if self._session_start else 0.0
                ),
            }

            # Add basic statistics
            if self._timeline:
                valences = [p.valence for p in self._timeline if p.confidence > 0.5]
                arousals = [p.arousal for p in self._timeline if p.confidence > 0.5]

                if valences and arousals:
                    summary_data["statistics"] = {
                        "mean_valence": sum(valences) / len(valences),
                        "mean_arousal": sum(arousals) / len(arousals),
                        "high_confidence_points": len(valences),
                        "total_points": len(self._timeline),
                    }

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            generated_files.append(str(summary_path))

            logger.info(
                "Saved affect artifacts for session %s: %d files",
                self._session_id, len(generated_files)
            )

        except Exception as e:
            logger.error("Error saving affect artifacts: %s", e)

        return generated_files

    def get_timeline_summary(self) -> dict[str, Any]:
        """Get a summary of the current timeline for debugging.

        Returns:
            Dictionary with timeline summary statistics
        """
        if not self._timeline:
            return {"timeline_points": 0, "status": "empty"}

        high_conf_points = [p for p in self._timeline if p.confidence > 0.5]
        valences = [p.valence for p in high_conf_points]
        arousals = [p.arousal for p in high_conf_points]

        summary = {
            "timeline_points": len(self._timeline),
            "high_confidence_points": len(high_conf_points),
            "strategy_changes": len(self._strategy_changes),
            "session_duration_minutes": (
                (datetime.utcnow() - self._session_start).total_seconds() / 60.0
                if self._session_start else 0.0
            ),
        }

        if valences and arousals:
            summary.update({
                "mean_valence": sum(valences) / len(valences),
                "mean_arousal": sum(arousals) / len(arousals),
                "valence_range": (min(valences), max(valences)),
                "arousal_range": (min(arousals), max(arousals)),
            })

        return summary

    def create_verification_manifest(self, test_name: str = "unnamed") -> str:
        """Create a verification manifest for the current session.

        Args:
            test_name: Name of the test or verification scenario

        Returns:
            Path to generated manifest file
        """
        if not self._session_id:
            logger.warning("No active session for manifest creation")
            return ""

        manifest_path = self.artifact_dir / f"verification_manifest_{test_name}.json"

        manifest = {
            "test_name": test_name,
            "session_id": str(self._session_id),
            "generated_at": datetime.utcnow().isoformat(),
            "timeline_summary": self.get_timeline_summary(),
            "verification_points": [],
        }

        # Add verification checkpoints based on timeline
        for i, point in enumerate(self._timeline):
            if point.confidence > 0.6:  # Only include confident points
                manifest["verification_points"].append({
                    "index": i,
                    "timestamp": point.timestamp.isoformat(),
                    "valence": point.valence,
                    "arousal": point.arousal,
                    "confidence": point.confidence,
                    "backend": point.backend,
                    "expected_behavior": self._generate_expected_behavior(point),
                })

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info("Created verification manifest: %s", manifest_path)
            return str(manifest_path)
        except Exception as e:
            logger.error("Error creating verification manifest: %s", e)
            return ""

    def _generate_expected_behavior(self, point: AffectTimelinePoint) -> str:
        """Generate expected behavior description for verification.

        Args:
            point: Timeline point to analyze

        Returns:
            Description of expected system behavior
        """
        if point.valence <= -0.3 and point.arousal <= 0.4:
            return "System should use gentler, more validating tone"
        elif point.valence <= -0.3 and point.arousal > 0.55:
            return "System should focus on calming, empathetic presence"
        elif point.valence >= 0.4 and point.arousal > 0.55:
            return "System can be naturally more engaging and upbeat"
        elif point.valence >= 0.4 and point.arousal <= 0.4:
            return "System should maintain warm, peaceful tone"
        else:
            return "System should use default supportive approach"