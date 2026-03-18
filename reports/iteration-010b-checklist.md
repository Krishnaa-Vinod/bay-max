# Iteration 010B verification checklist

## Ship/no-ship gates

Fail the iteration if any of these are missing:
- no pushed branch
- no `reports/iteration-010b.md`
- no `reports/iteration-010b.json`
- no working `/ws/live` endpoint
- no one-screen three-column UI
- no proof that the UI used real backend runtime state
- docs still out of sync with code

## Preflight

- [ ] 009.1 hotfix changes are present in this branch or explicitly imported
- [ ] README install/run instructions match reality
- [ ] Makefile/package scripts are present for UI install and run

## Backend verification

- [ ] `/ws/live` accepts a connection and sends structured JSON
- [ ] snapshot messages include session, perception, pipeline state, last response, memory hits, cooldown, and last event
- [ ] event messages include timestamp, stage, status, and detail
- [ ] `/v1/live/frame/latest` returns a real annotated frame or honest unavailable state
- [ ] `/v1/live/ui-state` returns bootstrap state
- [ ] `/v1/live/text-input` accepts a typed turn
- [ ] `/v1/live/mic/toggle` works or is honestly marked partial/not available

## Frontend verification

- [ ] app runs locally in a browser
- [ ] left column shows annotated frame, identity, posture, engagement, affect plot, and session info
- [ ] center column shows Bay-Max face, activity feed, last response, and input controls
- [ ] right column shows retrieved memories, semantic facts, and memory stats
- [ ] WebSocket reconnect behavior is handled visibly
- [ ] activity feed keeps the latest 100 events and color-codes them correctly
- [ ] Bay-Max face state changes are visible for idle/listening/thinking/speaking/cooldown

## Manual browser checks

- [ ] known-user greeting flow visible in UI
- [ ] memory retrieval appears before grounded response
- [ ] cooldown visibly suppresses extra output
- [ ] safety-gate event appears after a diagnosis-style prompt
- [ ] session pause/end appears when user leaves frame
- [ ] session resume appears when user returns
- [ ] affect plot moves when affect backend is available, or UI honestly shows unavailable/unknown
- [ ] last response text matches spoken or generated response

## Required artifacts

- [ ] `artifacts/verification_010b/manifest.json`
- [ ] `artifacts/verification_010b/summary.md`
- [ ] `artifacts/verification_010b/ws_sample.jsonl`
- [ ] `artifacts/verification_010b/event_feed_sample.json`
- [ ] `artifacts/verification_010b/ui_bootstrap_state.json`
- [ ] `artifacts/verification_010b/frontend_build.log`
- [ ] `artifacts/verification_010b/backend_test.log`
- [ ] `artifacts/verification_010b/screenshots/overview.png`
- [ ] `artifacts/verification_010b/screenshots/activity_feed.png`
- [ ] `artifacts/verification_010b/screenshots/memory_panel.png`

## Report honesty checks

- [ ] automated backend tests are separated from frontend tests
- [ ] manual browser tests are separated from Sol/headless tests
- [ ] blocked items are listed explicitly
- [ ] screenshots are real, not mockups
- [ ] commit SHA in report is real and matches pushed branch

## Reviewer scorecard

1. Backend telemetry quality
2. UI usefulness for debugging
3. Memory visibility
4. Activity feed clarity
5. Runtime-state-to-avatar fidelity
6. Local run workflow quality
7. Documentation parity
8. Artifact quality
9. Honesty of reporting
10. Overall readiness for next iteration

Scoring:
- 9-10 = excellent, proceed
- 7-8 = acceptable, fix a few issues
- 5-6 = partial, needs follow-up
- 0-4 = not ready
