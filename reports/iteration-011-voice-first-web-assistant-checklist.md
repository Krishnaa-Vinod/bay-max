# Iteration 011 Verification Checklist

## Setup
- [ ] Install dependencies: `pip install -e ".[all]"`
- [ ] Copy env: `cp .env.example .env`
- [ ] Start backend: `make run-local-backend-full`
- [ ] Start UI: `make run-ui`

## Primary voice mode smoke test
- [ ] UI shows active mode (`realtime_voice` / `local_chained_voice` / `text_only`)
- [ ] Voice fallback reason is visible when realtime is unavailable
- [ ] Start mic from UI without requiring recognition first
- [ ] Spoken reply is generated and played

## Local fallback smoke test
- [ ] Disable realtime preconditions (for example, unset `OPENAI_API_KEY`)
- [ ] Confirm mode switches to `local_chained_voice`
- [ ] Confirm fallback reason appears in UI
- [ ] Conversation still works with local STT/dialogue/TTS path

## Recognition continuity test
- [ ] Known user stays attached across short weak/no-match frames
- [ ] Recognition states display truthfully:
  - [ ] `no_face`
  - [ ] `face_seen_unknown`
  - [ ] `known_low_confidence`
  - [ ] `known_attached`
- [ ] Conversation continues during recognition degradation

## Web-tool test
- [ ] Ask a live/current-info query (for example, latest price/news)
- [ ] Confirm tool usage includes `search_web`
- [ ] Confirm response includes source references
- [ ] Confirm telemetry carries `last_tools_used` and `last_web_sources`

## Interrupt/barge-in test
- [ ] Trigger assistant speech
- [ ] Interrupt with user speech or `POST /v1/live/voice/interrupt`
- [ ] Confirm state transitions through `interrupted` back to `listening`
- [ ] Confirm no dead mic/task state after interrupt

## Failure-mode test
- [ ] Repeated mic start/stop/start does not leave half-started runtime
- [ ] Camera/frame open failure cleans speech tasks cleanly
- [ ] Web tool failure returns graceful response without crash
- [ ] UI surfaces explicit reason for disabled speech/web capabilities
