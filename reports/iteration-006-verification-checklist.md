# Iteration 006 Verification Checklist

Use this after Copilot reports completion.

## Pass/fail gates

Fail Iteration 006 if any of these are missing:
- branch not pushed
- `reports/iteration-006.md` missing
- `reports/iteration-006.json` missing
- `artifacts/verification_006/manifest.json` missing
- `artifacts/verification_006/summary.md` missing
- no proof of automatic session lifecycle
- no proof of cooldown suppression
- no proof of at least one memory-grounded proactive response
- docs not updated to match the live runtime

## What to manually verify

### VT601 — Live runner startup
Expected:
- runner starts in webcam or replay mode
- initial status/log files are created
- no crash on startup

Proof:
- runner console output
- artifact manifest entry
- initial status JSON

### VT602 — Known-user arrival greeting
Expected:
- returning known user appears
- one warm greeting happens
- greeting uses memory when relevant

Proof:
- response log entry
- memory_refs not empty when appropriate
- annotated frame or event snapshot

### VT603 — Unknown-user handling
Expected:
- unknown user is not greeted with invented personal details
- system stays generic or quiet

Proof:
- response/event log
- notes in summary

### VT604 — Session auto-start
Expected:
- no manual session API call needed
- session starts after configured presence threshold

Proof:
- session timeline JSON
- event log shows start trigger

### VT605 — Session pause/end
Expected:
- leaving camera view long enough pauses or ends session
- transition recorded in artifacts

Proof:
- session timeline JSON
- event log

### VT606 — Session resume/new session
Expected:
- returning inside resume window resumes prior session
- returning after longer gap starts a new one

Proof:
- session timeline JSON with IDs
- summary notes

### VT607 — Cooldown enforcement
Expected:
- repeated identical events do not trigger repeated text every few seconds
- suppression reason is logged

Proof:
- responses.jsonl
- events.jsonl with suppressed entries

### VT608 — Quiet companionship prompt
Expected:
- after a quiet interval, one gentle companionship prompt can appear
- it should not repeat immediately

Proof:
- response log timestamps
- cooldown state or suppression proof

### VT609 — Engagement/posture event
Expected:
- stable state change triggers event or is suppressed with a reason
- no jitter spam

Proof:
- event log
- overlay snapshot or replay artifact

### VT610 — Overlay correctness
Expected:
- overlay or debug output shows identity, posture, engagement, session state, last event, and last response

Proof:
- screenshot or annotated frame
- summary notes

### VT611 — Live status endpoint
Expected:
- returns current runtime state when live mode is active

Proof:
- JSON response artifact

### VT612 — Memory-grounded proactive response
Expected:
- a proactive response references something remembered from earlier interaction/session
- `memory_refs` or equivalent proof is present

Proof:
- response JSON
- event log
- summary notes

### VT613 — No-memory graceful behavior
Expected:
- fresh user or weak-memory case does not fabricate facts

Proof:
- response JSON

### VT614 — Safety in live mode
Expected:
- diagnosis-style typed request still triggers safe redirect behavior

Proof:
- response JSON with `safety_flags`

### VT615 — Artifact proof completeness
Expected:
- manifest, summary, events, responses, and session timeline all exist and are readable

Proof:
- artifact directory tree

### VT616 — Replay-mode parity
Expected:
- replay mode uses the same core runtime logic and produces comparable logs

Proof:
- replay artifact set
- summary notes

## Scorecard

### 10-point review
1. Live runner works at all
2. Session lifecycle works automatically
3. Cooldowns prevent chatter
4. Known-user greeting is memory-grounded
5. Unknown-user handling is safe and non-fabricated
6. Overlay/debug state is understandable
7. Live artifacts are complete and auditable
8. Automated tests cover event/cooldown/session logic
9. Docs are updated and accurate
10. Report is honest about webcam vs replay verification

## Good result
- 9–10: proceed to Iteration 007
- 7–8: acceptable, patch a few things first
- 5–6: partial, redo or hotfix before moving on
- 0–4: failed live runtime milestone
