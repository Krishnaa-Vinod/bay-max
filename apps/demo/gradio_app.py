"""Bay-Max Gradio demo application."""

import asyncio
import json
from uuid import UUID

import gradio as gr
import numpy as np

from baymax.core.enums import CorrectionAction, TurnRole
from baymax.orchestrator.service import Orchestrator
from baymax.perception.annotations import draw_annotations
from baymax.schemas.memory import MemoryCorrectionRequest
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
    """Generate a supportive response with memory-aware output."""

    async def _inner():
        await ensure_init()
        uid = UUID(user_id) if user_id else None
        response = await orch.respond(
            session_id=UUID(session_id),
            user_id=uid,
            context=context,
        )
        out = {
            "strategy": response.strategy.value,
            "message": response.message,
            "backend": response.backend,
            "model_name": response.model_name,
            "fallback_used": response.fallback_used,
            "safety_flags": response.safety_flags,
            "memory_refs": response.memory_refs,
            "state_summary": response.state_summary,
        }
        return json.dumps(out, indent=2)

    return run_async(_inner())


def get_dialogue_backends() -> str:
    """Return dialogue backend configuration."""

    async def _inner():
        await ensure_init()
        info = orch.get_dialogue_backends()
        return json.dumps({
            "available_backends": info.available_backends,
            "active_backend": info.active_backend,
            "active_model": info.active_model,
            "fallback_enabled": info.fallback_enabled,
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


# --- Iteration 004: new Gradio handlers ---


def add_chat_turn(session_id: str, user_id: str, role: str, text: str) -> str:
    """Add a typed conversation turn."""

    async def _inner():
        await ensure_init()
        if not session_id.strip() or not text.strip():
            return json.dumps({"error": "Session ID and text are required"}, indent=2)
        try:
            sid = UUID(session_id.strip())
        except ValueError:
            return json.dumps({"error": "Invalid Session ID"}, indent=2)
        uid = UUID(user_id.strip()) if user_id.strip() else None
        turn_role = TurnRole.USER if role == "user" else TurnRole.SYSTEM
        turn = await orch.add_turn(
            session_id=sid, role=turn_role, text=text.strip(), user_id=uid
        )
        return json.dumps({
            "turn_id": str(turn.id),
            "role": turn.role.value,
            "text": turn.text,
            "timestamp": turn.timestamp.isoformat(),
        }, indent=2)

    return run_async(_inner())


def get_session_turns(session_id: str) -> str:
    """Get all turns for a session."""

    async def _inner():
        await ensure_init()
        sid = UUID(session_id.strip())
        turns = await orch.get_turns(sid)
        return json.dumps([
            {"role": t.role.value, "text": t.text, "time": t.timestamp.isoformat()}
            for t in turns
        ], indent=2)

    return run_async(_inner())


def consolidate_session_handler(session_id: str, user_id: str) -> str:
    """Consolidate a session into memories."""

    async def _inner():
        await ensure_init()
        if not session_id.strip() or not user_id.strip():
            return json.dumps({"error": "Session ID and User ID are required"}, indent=2)
        result = await orch.consolidate(
            session_id=UUID(session_id.strip()),
            user_id=UUID(user_id.strip()),
        )
        out = {
            "session_id": str(result.session_id),
            "episodic_created": result.episodic_memories_created,
            "semantic_created": result.semantic_facts_created,
        }
        if result.summary:
            out["summary"] = result.summary.summary_text
        if result.errors:
            out["errors"] = result.errors
        return json.dumps(out, indent=2)

    return run_async(_inner())


def get_memory_summary_handler(user_id: str) -> str:
    """Get memory summary for a user."""

    async def _inner():
        await ensure_init()
        summary = await orch.get_memory_summary(UUID(user_id.strip()))
        return json.dumps({
            "episodic_count": summary.episodic_count,
            "semantic_count": summary.semantic_count,
            "session_summaries": [s.summary_text for s in summary.session_summaries],
            "recent_episodic": [m.content for m in summary.recent_episodic],
            "confirmed_facts": [
                {"id": str(f.id), "content": f.content, "confidence": f.confidence}
                for f in summary.confirmed_facts
            ],
        }, indent=2)

    return run_async(_inner())


def correct_memory_handler(fact_id: str, action: str, updated_content: str) -> str:
    """Correct a semantic fact."""

    async def _inner():
        await ensure_init()
        act = CorrectionAction(action)
        req = MemoryCorrectionRequest(
            fact_id=UUID(fact_id.strip()),
            action=act,
            updated_content=updated_content.strip() if updated_content.strip() else None,
        )
        try:
            result = await orch.correct_memory(req)
            return json.dumps({
                "fact_id": str(result.fact_id),
                "action": result.action.value,
                "previous_content": result.previous_content,
                "new_content": result.new_content,
                "new_status": result.new_status.value,
            }, indent=2)
        except ValueError as e:
            return json.dumps({"error": str(e)}, indent=2)

    return run_async(_inner())


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(title="Bay-Max Demo") as demo:
        gr.Markdown("# Bay-Max Companion Demo")
        gr.Markdown(
            "A memory-first empathetic companion agent with face recognition,"
            " pose estimation, engagement tracking, and memory recall."
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

        with gr.Tab("Chat"):
            gr.Markdown("Type conversation turns to simulate dialogue (before audio exists).")
            chat_session_input = gr.Textbox(
                label="Session ID", placeholder="Paste session ID"
            )
            chat_user_input = gr.Textbox(label="User ID", placeholder="Paste user ID")
            chat_role = gr.Radio(
                choices=["user", "system"], value="user", label="Role"
            )
            chat_text = gr.Textbox(label="Message", placeholder="Type a message...")
            chat_send_btn = gr.Button("Send Turn")
            chat_result = gr.JSON(label="Turn Result")
            chat_send_btn.click(
                add_chat_turn,
                inputs=[chat_session_input, chat_user_input, chat_role, chat_text],
                outputs=[chat_result],
            )
            chat_history_btn = gr.Button("View Turn History")
            chat_history = gr.JSON(label="Turn History")
            chat_history_btn.click(
                get_session_turns,
                inputs=[chat_session_input],
                outputs=[chat_history],
            )

        with gr.Tab("Interact"):
            gr.Markdown("Generate a memory-aware supportive response.")
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
            response_output = gr.JSON(
                label="Response (includes backend, memory_refs, safety_flags)"
            )
            respond_btn.click(
                get_response,
                inputs=[session_input, user_input, context_input],
                outputs=[response_output],
            )

        with gr.Tab("Backends"):
            gr.Markdown("### Dialogue Backend Status")
            gr.Markdown(
                "Shows the active dialogue backend, configured model, "
                "and whether rule-based fallback is enabled."
            )
            backends_btn = gr.Button("Refresh Backend Status")
            backends_output = gr.JSON(label="Backend Configuration")
            backends_btn.click(
                get_dialogue_backends,
                inputs=[],
                outputs=[backends_output],
            )

        with gr.Tab("Consolidate"):
            gr.Markdown("Consolidate a session into episodic memories and semantic facts.")
            cons_session_input = gr.Textbox(
                label="Session ID", placeholder="Paste session ID"
            )
            cons_user_input = gr.Textbox(label="User ID", placeholder="Paste user ID")
            cons_btn = gr.Button("Consolidate Session")
            cons_output = gr.JSON(label="Consolidation Result")
            cons_btn.click(
                consolidate_session_handler,
                inputs=[cons_session_input, cons_user_input],
                outputs=[cons_output],
            )

        with gr.Tab("Memory"):
            gr.Markdown("Query and inspect memories for a user.")
            mem_user_input = gr.Textbox(label="User ID", placeholder="Enter user ID")
            with gr.Row():
                mem_btn = gr.Button("Query Memories")
                mem_summary_btn = gr.Button("Memory Summary")
            mem_output = gr.JSON(label="Memories")
            mem_btn.click(query_user_memories, inputs=[mem_user_input], outputs=[mem_output])
            mem_summary_btn.click(
                get_memory_summary_handler,
                inputs=[mem_user_input],
                outputs=[mem_output],
            )
            gr.Markdown("### Correct a Semantic Fact")
            corr_fact_id = gr.Textbox(label="Fact ID", placeholder="Paste fact UUID")
            corr_action = gr.Radio(
                choices=["confirm", "reject", "update"],
                value="confirm",
                label="Action",
            )
            corr_content = gr.Textbox(
                label="Updated Content (for update only)",
                placeholder="New fact text...",
            )
            corr_btn = gr.Button("Apply Correction")
            corr_output = gr.JSON(label="Correction Result")
            corr_btn.click(
                correct_memory_handler,
                inputs=[corr_fact_id, corr_action, corr_content],
                outputs=[corr_output],
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
