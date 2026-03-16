#!/usr/bin/env python3
"""Local test script for Bay-Max frame analysis with pose and engagement.

Usage:
    # Analyze a single image
    python scripts/local_test.py --image path/to/image.jpg

    # Analyze a folder of images (frame replay)
    python scripts/local_test.py --folder path/to/frames/

    # Create a user and enroll a face first
    python scripts/local_test.py --enroll path/to/face.jpg --name "Test User"

    # Full workflow: enroll then analyze
    python scripts/local_test.py --enroll face.jpg --name "Alice" --image test.jpg

Annotated outputs are saved to BAYMAX_ARTIFACT_DIR (default: ./artifacts).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from baymax.config.settings import get_settings
from baymax.orchestrator.service import Orchestrator
from baymax.perception.annotations import draw_annotations, save_annotated_artifact
from baymax.schemas.user import UserProfileCreate


def load_image(path: str) -> np.ndarray:
    """Load an image as RGB numpy array."""
    pil_image = Image.open(path).convert("RGB")
    return np.array(pil_image)


async def run_enroll(orch: Orchestrator, image_path: str, name: str) -> dict:
    """Enroll a face from an image."""
    user = await orch.create_user(UserProfileCreate(display_name=name))
    frame = load_image(image_path)
    enrollment = await orch.enroll_face(user.id, frame)
    return {
        "user_id": str(user.id),
        "display_name": name,
        "status": enrollment.status,
        "message": enrollment.message,
    }


async def run_analyze(
    orch: Orchestrator,
    image_path: str,
    session_id: UUID,
    artifact_dir: str,
) -> dict:
    """Analyze a single frame and save annotated output."""
    frame = load_image(image_path)
    result = await orch.analyze_frame(session_id, frame)

    # Draw annotations
    annotated = draw_annotations(
        frame,
        recognized_faces=result.recognized_faces,
        unknown_faces=result.unknown_faces,
        pose_result=result.pose_result,
        engagement=result.engagement,
    )

    # Save artifact
    artifact_ref = save_annotated_artifact(
        annotated,
        artifact_dir=artifact_dir,
        session_id=str(session_id),
    )

    output = {
        "image": image_path,
        "faces_detected": result.faces_detected,
        "recognized": [
            {"name": r.user_display_name, "confidence": round(r.match_confidence, 3)}
            for r in result.recognized_faces
        ],
        "unknown_count": len(result.unknown_faces),
        "pose_present": result.pose_result.pose_present if result.pose_result else False,
        "engagement": result.engagement.model_dump(mode="json", exclude={"timestamp"})
        if result.engagement
        else None,
        "observations_written": result.observations_written,
        "latency_ms": result.latency_ms,
        "artifact_saved": artifact_ref.path if artifact_ref else None,
    }
    return output


async def main():
    parser = argparse.ArgumentParser(description="Bay-Max local test tool")
    parser.add_argument("--image", help="Path to a single image to analyze")
    parser.add_argument("--folder", help="Path to a folder of images to analyze in sequence")
    parser.add_argument("--enroll", help="Path to an enrollment face image")
    parser.add_argument("--name", default="Test User", help="Display name for enrollment")
    args = parser.parse_args()

    if not args.image and not args.folder and not args.enroll:
        parser.print_help()
        sys.exit(1)

    settings = get_settings()
    artifact_dir = settings.artifact_dir

    orch = Orchestrator()
    await orch.initialize()

    try:
        # Enrollment
        if args.enroll:
            print(f"Enrolling face from: {args.enroll}")
            result = await run_enroll(orch, args.enroll, args.name)
            print(json.dumps(result, indent=2))
            print()

        # Create session for analysis
        if args.image or args.folder:
            session = await orch.create_session()
            print(f"Session created: {session.id}")
            print(f"Artifacts directory: {artifact_dir}")
            print()

        # Single image analysis
        if args.image:
            print(f"Analyzing: {args.image}")
            result = await run_analyze(orch, args.image, session.id, artifact_dir)
            print(json.dumps(result, indent=2))

        # Folder replay
        if args.folder:
            folder = Path(args.folder)
            image_files = sorted(
                p
                for p in folder.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            print(f"Found {len(image_files)} images in {folder}")
            print()
            for img_path in image_files:
                print(f"Analyzing: {img_path.name}")
                result = await run_analyze(orch, str(img_path), session.id, artifact_dir)
                print(json.dumps(result, indent=2))
                print()

    finally:
        await orch.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
