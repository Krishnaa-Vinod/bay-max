# Bay-Max Project Brief

## Vision
Bay-Max is a memory-first empathetic companion agent. It recognizes enrolled users from vision input, tracks interaction sessions, stores meaningful observations and memories, and produces supportive personalized responses.

## Current Scope (Iteration 001)
The MVP focuses on establishing the foundational architecture:
- Modular monolith codebase with clear module boundaries
- Core data schemas for users, sessions, perception, memory, and responses
- SQLite-backed metadata storage
- Rule-based dialogue provider (no external API keys required)
- Stub interfaces for vision/perception components
- FastAPI backend with health, user, session, memory, and response endpoints
- Gradio demo shell for interactive testing

## Non-Goals for Iteration 001
- Robotics integration
- Clinical diagnosis or medical advice
- Wearable device integration
- Doctor report ingestion
- Audio runtime features
- Fine-tuning or training large models
- Production deployment

## Supervisor
Krishnaa Vinod is the project supervisor and final approver.

## Engineering Principles
1. Build a modular monolith, not microservices.
2. Prefer clear interfaces over premature optimization.
3. Make repo memory explicit and persistent in files.
4. Never claim a feature is complete unless code, tests, and file evidence exist.
5. Keep large artifacts, model caches, and temporary files out of git.
6. Use environment variables for all data/model/cache locations.
7. Default to safe, non-clinical language in docs and code comments.
