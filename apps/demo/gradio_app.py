"""Bay-Max Gradio demo application."""

import asyncio
import json
from uuid import UUID

import gradio as gr
import numpy as np

from baymax.orchestrator.service import Orchestrator
from baymax.perception.annotations import draw_annotations
from baymax.schemas.user import UserProfileCreate

orch = Orchestrator()
_initialized = False


async def ensure_init():
    global _initialized
    if not _initialized:
        await orch.initialize()
        _initialized = True


def run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def create_user_and_session(display_name: str) -> str:
    """Create a user and session, return status."""

    async def _inner():
        await ensure_init()
        user = await orch.create_user(UserProfileCreate(display_name=display_name))
        session = await orch.create_session(user_id=user.id)
        return json.dumps({
            "user_id": str(user.id),
            "user_name": user.display_name,
            "session_id": str(session.id),
        }, indent=2)

    return run_async(_inner())


def enroll_face_from_image(user_id: str, image: np.ndarray | None) -> str:
    """Enroll a face from an uploaded image."""

    async def _inner():
        await ensure_init()
        if image is None:
            return json.dumps({"error": "No image provided"}, indent=2)
        if not user_id.strip():
            return json.dumps({"error": "User ID is required"}, indent=2)
        try:
            uid = UUID(user_id.strip())
        except ValueError:
            return json.dumps({"error": "Invalid User ID format"}, indent=2)

        enrollment = await orch.enroll_face(uid, image)
        return json.dumps({
            "enrollment_id": str(enrollment.id),
            "user_id": str(enrollment.user_id),
            "status": enrollment.status,
            "confidence": enrollment.confidence,
            "face_count_detected": enrollment.face_count_detected,
            "message": enrollment.message,
        }, indent=2)

    return run_async(_inner())


def analyze_frame_image(
    session_id: str, image: np.ndarray | None,
) -> tuple[str, np.ndarray | None]:
    """Analyze an uploaded frame for face detection, recognition, pose, and engagement."""

    async def _inner():
        await ensure_init()
        if image is None:
            return json.dumps({"error": "No image provided"}, indent=2), None
        if not session_id.strip():
            return json.dumps({"error": "Session ID is required"}, indent=2), None
        try:
            sid = UUID(session_id.strip())
        except ValueError:
            return json.dumps({"error": "Invalid Session ID format"}, indent=2), None

        result = await orch.analyze_frame(sid, image)

        # Draw annotated overlay
        annotated = draw_annotations(
            image,
            recognized_faces=result.recognized_faces,
            unknown_faces=result.unknown_faces,
            pose_result=result.pose_result,
            engagement=result.engagement,
        )

        engagement_data = None
        if result.engagement:
            engagement_data = {
                "level": str(result.engagement.level),
                "score": round(result.engagement.score, 3),
                "posture": str(result.engagement.posture),
                "lean": str(result.engagement.lean),
                "motion": str(result.engagement.motion),
                "pose_visible": result.engagement.pose_visible,
            }

        output = json.dumps({
            "session_id": str(result.session_id),
            "faces_detected": result.faces_detected,
            "recognized_faces": [
                {
                    "user_id": str(r.user_id),
                    "name": r.user_display_name,
                    "match_confidence": round(r.match_confidence, 3),
                }
                for r in result.recognized_faces
            ],
            "unknown_faces": len(result.unknown_faces),
            "pose_detected": result.pose_result.pose_present if result.pose_result else False,
            "engagement": engagement_data,
            "observations_written": result.observations_written,
            "latency_ms": result.latency_ms,
            "frame_summary": result.state.get("last_frame_summary", ""),
        }, indent=2)

        return output, annotated

    return run_async(_inner())


def get_response(session_id: str, user_id: str, context: str) -> str:
    """Generate a supportive response."""

    async def _inner():
        await ensure_init()
        uid = UUID(user_id) if user_id else None
        response = await orch.respond(
            session_id=UUID(session_id),
            user_id=uid,
            context=context,
        )
        return json.dumps({
            "strategy": response.strategy.value,
            "message": response.message,
        }, indent=2)

    return run_async(_inner())


def query_user_memories(user_id: str) -> str:
    """Query memories for a user."""

    async def _inner():
        await ensure_init()
        result = await orch.query_memories(user_id=UUID(user_id))
        return json.dumps({
            "total_count": result.total_count,
            "episodic": [m.content for m in result.episodic_memories],
            "semantic": [f.content for f in result.semantic_facts],
        }, indent=2)

    return run_async(_inner())


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(title="Bay-Max Demo") as demo:
        gr.Markdown("# Bay-Max Companion Demo")
        gr.Markdown(
            "A memory-first empathetic companion agent with face recognition,"
            " pose estimation, and engagement tracking."
        )

        with gr.Tab("Setup"):
            name_input = gr.Textbox(label="Display Name", placeholder="Enter your name")
            setup_btn = gr.Button("Create User & Session")
            setup_output = gr.JSON(label="Setup Result")
            setup_btn.click(create_user_and_session, inputs=[name_input], outputs=[setup_output])

        with gr.Tab("Enroll Face"):
            gr.Markdown("Upload a photo containing exactly **one face** to enroll.")
            enroll_user_input = gr.Textbox(
                label="User ID",
                placeholder="Paste user ID from Setup tab",
            )
            enroll_image = gr.Image(label="Face Photo", type="numpy")
            enroll_btn = gr.Button("Enroll Face")
            enroll_output = gr.JSON(label="Enrollment Result")
            enroll_btn.click(
                enroll_face_from_image,
                inputs=[enroll_user_input, enroll_image],
                outputs=[enroll_output],
            )

        with gr.Tab("Frame Analysis"):
            gr.Markdown(
                "Upload a frame to detect/recognize faces, estimate pose,"
                " and compute engagement."
            )
            frame_session_input = gr.Textbox(
                label="Session ID",
                placeholder="Paste session ID from Setup tab",
            )
            frame_image = gr.Image(label="Frame", type="numpy")
            frame_btn = gr.Button("Analyze Frame")
            with gr.Row():
                frame_output = gr.JSON(label="Analysis Result")
                frame_annotated = gr.Image(label="Annotated Frame")
            frame_btn.click(
                analyze_frame_image,
                inputs=[frame_session_input, frame_image],
                outputs=[frame_output, frame_annotated],
            )

        with gr.Tab("Interact"):
            session_input = gr.Textbox(
                label="Session ID",
                placeholder="Paste session ID from Setup",
            )
            user_input = gr.Textbox(label="User ID", placeholder="Paste user ID from Setup")
            context_input = gr.Textbox(
                label="Context",
                placeholder="Optional context for the interaction",
            )
            respond_btn = gr.Button("Get Response")
            response_output = gr.JSON(label="Response")
            respond_btn.click(
                get_response,
                inputs=[session_input, user_input, context_input],
                outputs=[response_output],
            )

        with gr.Tab("Memory"):
            mem_user_input = gr.Textbox(label="User ID", placeholder="Enter user ID")
            mem_btn = gr.Button("Query Memories")
            mem_output = gr.JSON(label="Memories")
            mem_btn.click(query_user_memories, inputs=[mem_user_input], outputs=[mem_output])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
