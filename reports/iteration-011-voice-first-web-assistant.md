# Iteration 011 - Voice-First Web-Connected Assistant

## Goal

Deliver a smoother voice-first Bay-Max runtime with explicit mode transparency, interruption handling, recognition continuity, and real web-tool-assisted responses.

## Objective

Shift Bay-Max from a stitched perception-first demo feel to a conversation-first loop with truthful diagnostics, while preserving local fallback behavior.

## Baseline Problems

- Mic toggle/restart could leave speech tasks in partial states.
- Recognition labels and continuity were too brittle on transient weak frames.
- No real web tool broker path for current-info questions.
- Text vs speech telemetry and modality metadata diverged.
- UI did not clearly expose voice mode/fallback/tool availability reasons.

## Changes Implemented

- Added explicit voice mode configuration and status contract (`realtime_voice`, `local_chained_voice`, `text_only`).
- Hardened runtime speech lifecycle with idempotent start/stop + task recreation (`LiveRuntime.start_microphone`, `stop_microphone`).
- Added explicit speech loop state transitions and interrupt endpoint (`POST /v1/live/voice/interrupt`).
- Added sticky identity continuity windows in live runtime for short recognition degradation windows.
- Added web tool modules and broker path:
  - `search_web(query)`
  - `fetch_url(url)`
  - `summarize_sources(items)`
- Integrated tool usage and source refs into orchestrator responses and live telemetry.
- Unified text/speech turn metadata with explicit modality + source tags.
- Added realtime session provisioning endpoint (`POST /v1/realtime/session`) with clean fallback reasoning.
- Updated UI contracts/components to show mode, fallback reason, web tool availability, and latency diagnostics.

## Why These Changes Were Necessary

- Voice smoothness and interruption required lifecycle-safe mic/task management.
- Recognition should personalize, not gate, conversation continuity.
- Web-needed answers require explicit tooling and source references.
- Trustworthy UX requires runtime-truthful telemetry (mode, fallback, tool usage).

## Why the Old Design Felt Rough

- Runtime state was fragmented between direct service calls and lifecycle loops.
- Recognition state labels did not represent sticky/low-confidence continuity clearly.
- Web access was implicit/non-structured.
- Frontend diagnostics lacked a single authoritative mode/tool/fallback contract.

## New Architecture Summary

- Authoritative runtime status now carries mode/tool/latency/capability state.
- Dialogue response path can invoke web tools when heuristics indicate live knowledge needs.
- Text and speech turns feed shared orchestrator response path with modality metadata.
- UI consumes unified snapshot/bootstrap contract with explicit mode transparency.

## Architecture Before vs After

- Before: mostly local chained pipeline behavior with limited explicit mode/fallback/tool contracts.
- After: explicit mode abstraction + runtime-safe speech lifecycle + tool-broker-enabled response path + parity telemetry.

## Files Changed

- `.gitignore`
- `.env.example`
- `apps/api/main.py`
- `apps/api/live_http.py`
- `apps/api/live_ws.py`
- `apps/api/realtime_session.py`
- `src/baymax/config/settings.py`
- `src/baymax/live/schemas.py`
- `src/baymax/live/runtime.py`
- `src/baymax/orchestrator/service.py`
- `src/baymax/schemas/response.py`
- `src/baymax/tools/__init__.py`
- `src/baymax/tools/web_search.py`
- `src/baymax/tools/web_fetch.py`
- `apps/ui/src/types/live.ts`
- `apps/ui/src/App.tsx`
- `apps/ui/src/components/MemoryPanel.tsx`
- `tests/test_iteration_011.py`
- `tests/test_iteration_010b.py`
- `README.md`
- `docs/LOCAL_TESTING.md`
- `docs/project_state.json`

## Verification Steps

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_iteration_011.py tests/test_iteration_010b.py -q`
2. Start backend (`make run-local-backend-full`) and UI (`make run-ui`).
3. Confirm UI banner diagnostics show active mode and fallback reason.
4. Use `/v1/live/mic/toggle` start/stop repeatedly.
5. Use `/v1/live/voice/interrupt` during speech playback.
6. Ask a web-needed question and verify response `Sources:` + tool usage in telemetry.

## How to Run the New Voice Flow Locally

1. `make run-local-backend-full`
2. `make run-ui`
3. Open `http://localhost:3000`
4. Start voice session (mic toggle), speak or type, then interrupt via UI or `POST /v1/live/voice/interrupt`.

## How the Web Tool Path Works

1. Orchestrator checks query heuristics for live/web-needed intent.
2. Calls `search_web` with configured provider.
3. Optionally calls `fetch_url` on top result.
4. Calls `summarize_sources` and injects summary into context.
5. Response returns structured `tool_usage` and `source_refs`; message appends top source references.

## Observed Results

- Targeted tests pass for iteration 011 + 010b compatibility.
- Mic lifecycle now routes through runtime-safe idempotent APIs.
- Telemetry includes explicit mode/tool/fallback state.
- Text and speech paths now carry consistent modality/source metadata.

## Latency Observations and Bottlenecks

- `end_of_speech_to_first_audio_ms` and `interrupt_to_audio_stop_ms` are now represented in runtime status and snapshot contracts.
- Hardware-level latency values were not claimed in this headless environment.
- Main bottlenecks remain STT decode time, local model generation, and TTS synthesis/playback.

## Known Tradeoffs

- Realtime voice endpoint currently provisions mode truthfully but does not yet complete browser WebRTC media wiring in this patch.
- Web search provider uses DuckDuckGo HTML parsing, which may require robustness tuning over time.

## Known Gaps

- Full manual A/V latency measurement on real microphone/speaker hardware still pending.
- Browser realtime path handshake + media negotiation is not fully implemented yet.

## What Still Remains for the Next Iteration

- End-to-end browser realtime voice session (WebRTC) behind current provisioning endpoint.
- Better web source ranking and retry/failover strategies.
- Full manual latency benchmark capture on physical hardware.

## Next Iteration Recommendations

- Prioritize full WebRTC realtime client wiring.
- Add integration tests for mode transitions and interrupt latency bounds.
- Expand tool-provider abstraction to support additional search/fetch backends.
