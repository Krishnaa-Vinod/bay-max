# Iteration 009 verification checklist

## Hard gates

Fail the iteration if any of these are false:
- branch pushed
- reports/iteration-009.md exists
- reports/iteration-009.json exists
- core docs updated and not obviously stale
- at least one affect backend path exists or a blocker is honestly documented with a runnable fallback
- automated tests ran and results are included
- verification artifacts exist under artifacts/verification_009/

## What this iteration must prove

This iteration is **not** proving clinical emotion recognition. It is proving that Bay-Max can:
- extract a usable facial-affect signal
- smooth it over time
- bias response strategy gently and safely
- avoid overclaiming
- write only stable, non-clinical session summaries

## Automated verification

### Backend and packaging
- VT901: install path for affect backend is documented and works
- VT902: backend selection works (`null`, `pyfeat`, optional fallback)
- VT903: if Py-Feat is the active backend, it imports and initializes successfully, or the blocker is honestly reported

### Schemas and state
- VT904: `EmotionResult` / affect schemas validate correctly
- VT905: `InteractionState` includes affect fields
- VT906: live status includes affect-enabled and affect summary fields

### Smoothing and gating
- VT907: EMA or smoothing behaves as expected
- VT908: stability duration increases only while affect state is consistent
- VT909: low-confidence affect leaves planner unchanged
- VT910: low-confidence affect is omitted from prompt context

### Planner and prompting
- VT911: low-valence + low-arousal case softens strategy appropriately
- VT912: high-valence + higher-arousal case can bias toward lighter engagement appropriately
- VT913: affect does not override safety logic
- VT914: prompt builder includes a short internal affect note only when confidence is high enough

### Memory and safety
- VT915: no per-frame affect spam enters long-term memory
- VT916: stable affect summary can be created at session level
- VT917: diagnostic / clinical words are blocked or sanitized in consolidation output
- VT918: memory correction/deletion compatibility is preserved

### APIs and docs
- VT919: `GET /v1/emotion/backends` works
- VT920: README / ARCHITECTURE / ROADMAP / MODEL_STACK / EMOTION_ARCHITECTURE / COMPANION_PERSONA match the code
- VT921: `project_state.json` is valid and current

## Replay or local manual checks

These can be done on a laptop/workstation. Replay is preferred because it is reproducible.

### Required replay/manual cases
- VT930: positive-expression replay or live sample
  - expected: valence trends positive, confidence reasonable, tone slightly warmer or more engaging
- VT931: neutral-expression replay or live sample
  - expected: valence near neutral, planner mostly unchanged
- VT932: low-positive-affect / worried / low-energy replay or live sample
  - expected: tone softens, less cheerleading, more validation
- VT933: occluded / turned-away / poor-light sample
  - expected: low confidence or unknown, no hard guess
- VT934: stable-state memory write
  - expected: after configured stability duration, a session-level affect summary may be written
- VT935: unstable/noisy affect
  - expected: no long-term memory write
- VT936: response-tone comparison
  - expected: same base prompt under two different affect states yields noticeably different but still Baymax-like tone

## Artifact proof expected

The branch should generate a structure like this:

```text
artifacts/verification_009/
  manifest.json
  summary.md
  automated/
  replay/
  live/
  affect/
  logs/
```

Expected artifacts include:
- `manifest.json`
- `summary.md`
- machine/environment info
- automated test results
- replay or live verification results
- affect timeline JSON or CSV
- strategy-change log
- selected annotated frame artifacts if implemented
- notes about which backend actually ran

## Human review questions

Use these after the artifacts are generated:
- Does Bay-Max become a little gentler when affect looks lower, without sounding pitying or clinical?
- Does Bay-Max become a little more upbeat when affect looks positive, without becoming hyped or cartoonish?
- When the face is ambiguous, does the system stay uncertain rather than invent a feeling?
- Are the stored summaries phrased as observations rather than diagnoses?
- Does the persona still sound calm, literal, and help-oriented?

## Pass standard

Approve Iteration 009 if:
- automated checks pass
- replay/manual checks show at least one positive and one lower-affect tone adaptation case
- uncertainty handling is honest
- memory summaries remain non-clinical
- docs match the implementation

Do not approve if:
- the system starts making diagnostic or mental-health claims
- single noisy frames create long-term memory writes
- affect is always forced even when confidence is low
- docs/report claim real backend execution that did not happen
