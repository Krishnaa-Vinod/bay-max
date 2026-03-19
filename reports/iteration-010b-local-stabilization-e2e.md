# Iteration 010b Local Stabilization E2E Report

Date: 2026-03-19
Branch: fix/local-stabilization-e2e

## Scope

Stabilization-only pass for local usability:
- single-process local backend path
- text input error visibility and session auto-create behavior
- websocket truthfulness on reconnect/health state
- speech disabled reason plumbing
- memory panel refresh/empty-state clarity
- recognition/session debug clarity in UI

## Commands Executed

1. UI websocket test

```bash
cd apps/ui
npm test -- --run tests/ws.test.ts
```

Result:
- `6 passed`

2. Iteration 010b tests (with plugin autoload disabled due host ROS pytest plugins)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/NewVolume1/baymax-work/.conda/envs/baymax-ssd/bin/python -m pytest -q tests/test_iteration_010b.py
```

Result:
- `9 passed, 1 warning`
- warning: unknown `asyncio_mode` config option (non-blocking for this run)

3. UI production build

```bash
cd apps/ui
npm run build
```

Result:
- TypeScript + Vite build passed

4. Single-process local backend smoke (replay mode)

```bash
/mnt/NewVolume1/baymax-work/.conda/envs/baymax-ssd/bin/python scripts/run_local_backend.py \
  --replay artifacts/live_run_007 \
  --max-run-sec 120 \
  --no-overlay \
  --no-speech-input \
  --port 8010
```

5. Runtime API smoke while launcher is active

```bash
curl -sS http://127.0.0.1:8010/v1/live/ui-state
curl -sS -X POST http://127.0.0.1:8010/v1/live/text-input \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from smoke"}'
```

Observed response excerpts:

- `GET /v1/live/ui-state` included:
  - `"speech_disabled_reason":"Speech backend disabled (BAYMAX_ENABLE_SPEECH_INPUT=false)"`
  - `"recognition_state":"unknown_user"`
  - `"session_binding":"anonymous"`

- `POST /v1/live/text-input` returned:
  - `"success":true`
  - non-empty `session_id`
  - assistant `response_text`

## Key Findings

- Previous local path (`scripts/live_companion.py` with threaded API) reproduced a real cross-thread sqlite failure during text input.
- New launcher (`scripts/run_local_backend.py`) resolved that failure in smoke testing by running API + runtime in one process/thread.
- Speech remained intentionally disabled for this smoke (`--no-speech-input`), and exact disabled reason was exposed correctly.

## Manual UI Verification Still Required

The following need manual confirmation in an interactive browser session:
- face-state transitions in left/center columns across no-face/unknown/below-threshold/recognized
- memory panel visual refresh timing after each interaction
- websocket reconnect feed cleanup and single recovered event
- mic listening toggle behavior when speech input is enabled with available local mic dependencies

## Notes

No screenshots were captured in this run.
