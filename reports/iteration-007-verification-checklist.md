# Iteration 007 Verification Checklist

Use this after Copilot finishes Iteration 007.

## Pass/fail gates

Fail the iteration if any of these are missing:
- branch not pushed
- reports/iteration-007.md missing
- reports/iteration-007.json missing
- artifacts/verification_007/manifest.json missing
- artifacts/verification_007/summary.md missing
- no real TTS backend implemented
- no proof for webcam/TTS behavior and no honest not_run explanation
- docs drift from code

## What must be verified

### A. Core runtime still works
- face recognition did not regress
- live session auto-start still works
- pause/end still works
- resume/new-session branching still works
- proactive cooldown still works
- safety behavior still works

### B. TTS layer works
- at least one local TTS backend initializes
- text response can be turned into a WAV file
- generated WAV path is logged
- playback works locally OR is honestly marked not_run
- TTS failure does not crash the live runtime

### C. Inspection/debugability works
- GET /v1/tts/backends returns active backend/voice info
- GET /v1/live/status includes speech fields
- last_spoken_text is visible after a response
- queue depth is visible
- overlay or debug HUD is visually checked on a real display OR marked not_run with reason

### D. Artifact proof exists
Look for these under `artifacts/verification_007/`:
- manifest.json
- summary.md
- events.jsonl or equivalent event log
- responses.jsonl or equivalent response log
- speech.jsonl or equivalent speech log
- generated WAV files or refs
- selected snapshots if webcam/display path was exercised

## Manual test cases

### VT701 Real webcam startup
Expected:
- webcam opens
- frames processed
- session artifacts begin writing

### VT702 Known-user greeting with speech
Expected:
- recognized user greeted by name or known identity
- spoken output happens or WAV is generated
- if relevant memory exists, greeting reflects it naturally

### VT703 Unknown-user handling
Expected:
- no fake personalization
- generic safe wording
- spoken output, if any, stays generic

### VT704 Cooldown suppression with speech
Expected:
- no rapid repeated speech spam
- suppression reason logged

### VT705 Leave and resume session
Expected:
- leave causes pause/end after threshold
- return within resume window resumes appropriately
- return does not trigger annoying duplicate speech bursts

### VT706 Quiet companionship spoken prompt
Expected:
- one gentle check-in after quiet interval
- event and spoken output logged

### VT707 TTS backend failure fallback
Expected:
- broken backend does not crash runtime
- text path still works
- failure is logged clearly

### VT708 Artifact proof completeness
Expected:
- manifest and summary accurately reflect what ran
- artifact files referenced actually exist

### VT709 Overlay/debug status check
Expected:
- visible identity/state/speech debug info on display machine
- if not visually checked, must be marked not_run honestly

### VT710 Safety in spoken mode
Expected:
- diagnosis-style request remains safely refused or redirected
- spoken output does not make unsafe claims

## Scoring rubric

### 10-point review
1. Live runtime regression-free
2. Real TTS backend implemented
3. WAV generation proven
4. Playback proven or honestly marked not_run
5. Known-user speech case proven
6. Cooldown + anti-spam proven
7. Artifact proof complete
8. Status/debug inspection good
9. Docs updated and accurate
10. Report honest and consistent

### Score interpretation
- 9-10: ready to move to Iteration 008
- 7-8: acceptable, fix a few items first
- 5-6: partial, redo iteration
- 0-4: failed

## What to send back for review
- branch link
- latest commit SHA
- reports/iteration-007.md
- reports/iteration-007.json
- artifacts/verification_007/manifest.json
- artifacts/verification_007/summary.md
- note whether playback was truly heard on speakers or only WAV generation was verified
