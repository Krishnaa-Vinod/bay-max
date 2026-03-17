# Iteration 008 Verification Checklist

Use this after Copilot finishes Iteration 008.

## Hard pass gates

Fail the iteration if any of these are missing:
- branch not pushed
- no real local STT backend implemented
- no proof artifacts under `artifacts/verification_008/`
- no honest distinction between automated and manual hardware tests
- docs not updated
- speech path does not enter session/memory flow
- no evidence of speaking lock or echo suppression

## Core verification cases

### VT801 — Microphone startup and status
Expected:
- speech input starts locally
- `/v1/audio/status` shows listening state
- status shows active STT and TTS backends

Proof:
- status JSON artifact
- command used
- environment info

### VT802 — VAD segmentation
Expected:
- one spoken phrase creates one segment and one transcript
- silence/noise alone does not create repeated transcripts

Proof:
- saved utterance WAV (if enabled)
- transcript JSON
- VAD decision log

### VT803 — Spoken turn enters memory pipeline
Expected:
- spoken user text is stored as a session turn with `source='speech'`
- session or memory summary reflects it

Proof:
- session turn artifact
- memory summary artifact

### VT804 — Speech-based grounded recall
Expected:
- after Session A stores at least one fact or preference, Session B voice query such as "What do you remember about me?" produces a response that mentions a real prior detail
- `memory_refs` are present

Proof:
- transcript of recall request
- response JSON
- memory refs in artifact
- summary note indicating whether prior detail was truly verbalized

### VT805 — Echo suppression
Expected:
- Bay-Max does not transcribe its own TTS output into a new user turn
- echo or speaking-lock decision is logged

Proof:
- echo decision artifact or suppression log
- no follow-on self-triggered user turn

### VT806 — Speaking lock and cooldown
Expected:
- input during Bay-Max speech is blocked or ignored
- input immediately after speech is blocked until cooldown expires

Proof:
- timing log
- audio/status snapshots
- notes showing accepted vs rejected utterances

### VT807 — No-memory graceful behavior
Expected:
- fresh user gets no fabricated memory references

Proof:
- response artifact with empty or absent memory refs

### VT808 — Safety in spoken mode
Expected:
- spoken diagnosis-style request is refused or redirected safely
- no unsupported medical certainty in spoken or text output

Proof:
- transcript artifact
- response artifact
- safety flags

### VT809 — Baymax-inspired tone
Expected:
- responses are calm, concise, literal, gentle, nonjudgmental
- not chatty, not slangy, not overexcited

Proof:
- at least 2 transcript/response pairs
- short reviewer note scoring tone quality

### VT810 — Artifact completeness
Expected:
- `artifacts/verification_008/manifest.json`
- `artifacts/verification_008/summary.md`
- transcripts and timing logs
- key audio files or explicit notes when audio files were intentionally not saved

Proof:
- artifact tree listing

## Recommended scoring rubric

### 10-point score
1. Local install actually works
2. Microphone capture works
3. VAD segmentation is sane
4. STT backend works on real speech
5. Spoken turns enter memory flow
6. Speech recall truly mentions prior memory
7. Echo suppression works
8. Safety works in spoken mode
9. Artifacts are complete and auditable
10. Docs match reality

### Interpretation
- 9-10: strong pass, ready for next perception/emotion iteration
- 7-8: usable, but fix issues before expanding scope
- 5-6: partial, needs another stabilization pass
- 0-4: not ready to build on

## What to send back for review

Send these after Copilot finishes:
- branch link
- `reports/iteration-008.md`
- `reports/iteration-008.json`
- `artifacts/verification_008/manifest.json`
- `artifacts/verification_008/summary.md`
- one note saying whether audio playback was actually heard by a human or only WAV generation was verified
