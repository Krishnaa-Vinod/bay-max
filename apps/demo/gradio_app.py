"""Bay-Max Gradio demo application."""

import asyncio
import json
from uuid import UUID

import gradio as gr

from baymax.orchestrator.service import Orchestrator
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
        gr.Markdown("A memory-first empathetic companion agent.")

        with gr.Tab("Setup"):
            name_input = gr.Textbox(label="Display Name", placeholder="Enter your name")
            setup_btn = gr.Button("Create User & Session")
            setup_output = gr.JSON(label="Setup Result")
            setup_btn.click(create_user_and_session, inputs=[name_input], outputs=[setup_output])

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
