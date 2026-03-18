# Iteration 009.1 Hotfix Verification Checklist

Use this after Copilot finishes the hotfix.

## Pass/fail gates

Fail the hotfix if any of these are missing:
- live runtime still does not call the affect analyzer
- runtime status still shows default affect values only
- no replay/local proof artifacts exist
- `.env.example` still lacks affect environment variables
- `reports/iteration-009.json` still contains `pending_final_commit`
- tests or lint were claimed but not run

## What to inspect first

1. `src/baymax/live/runtime.py`
   - look for affect analyzer initialization
   - look for affect analysis inside the analysis-frame path
   - look for smoother usage
   - look for status updates of valence/arousal/confidence/backend/stable duration

2. `src/baymax/live/schemas.py`
   - confirm status fields match what runtime now populates

3. `src/baymax/perception/emotion.py`
   - confirm backend factory still works
   - confirm MediaPipe path remains the actual primary backend unless explicitly changed

4. `src/baymax/planner/supportive_planner.py`
   - confirm affect bias logic still only softly adjusts strategy
   - confirm GREET/RECALL and safety-critical flows are not overridden

5. `.env.example`
   - verify affect env vars exist and are documented

6. `reports/iteration-009.json`
   - verify real `latest_commit_sha`
   - verify no stale placeholders remain

7. `pyproject.toml`
   - verify version metadata is intentionally updated and consistent with reports/docs

## Artifact proof required

Expected root:
- `artifacts/verification_0091/manifest.json`
- `artifacts/verification_0091/summary.md`
- `artifacts/verification_0091/sweep_results.json`
- `artifacts/verification_0091/affect/replay_status_samples.json`
- `artifacts/verification_0091/affect/runtime_events.json`
- `artifacts/verification_0091/affect/annotated_frames/`

The artifacts should prove:
- which sample inputs were used
- whether a face was found
- valence/arousal/confidence values
- whether affect was stable or suppressed
- whether strategy changed due to affect
- whether memory writes were created or suppressed
- backend used
- environment limitations

## Functional checks

### VC901 - Runtime wiring present
Pass only if `runtime.py` clearly initializes and uses the affect analyzer and smoother.

### VC902 - Status propagation works
Pass only if runtime status now populates:
- `affect_enabled`
- `valence`
- `arousal`
- `affect_confidence`
- `affect_backend`
- `affect_stable_duration_sec`

### VC903 - Overlay/debug visibility
Pass only if local/replay outputs show affect values in the overlay or saved debug context.

### VC904 - Neutral sample behavior
Pass only if neutral/reduced-expression input yields either near-neutral affect or low-confidence/unknown output.

### VC905 - Positive sample behavior
Pass only if positive/smiling input yields a measurably different affect state from neutral, or the report honestly marks it ambiguous.

### VC906 - No-face behavior
Pass only if a no-face/poor-face sample yields unknown or unavailable affect without false confidence.

### VC907 - Planner bias behavior
Pass only if at least one check shows affect can influence strategy/tone when confidence is sufficient, and no change happens when confidence is low.

### VC908 - Memory gating behavior
Pass only if affect memory is written only after stability/confidence thresholds, or the report honestly shows that no sample met thresholds.

### VC909 - Clinical safety preserved
Pass only if affect summaries remain non-clinical and do not use blocked language like diagnosis terms.

### VC910 - API truthfulness
Pass only if `/v1/audio/status` and `/v1/emotion/backends` match actual runtime/backend behavior.

### VC911 - Report parity
Pass only if:
- `latest_commit_sha` is real
- docs match code
- version metadata is internally consistent
- remaining blockers are honest

## Commands that should appear in the report

At minimum:

```bash
python -m pytest tests/ -q
python -m ruff check src/ apps/ tests/ scripts/
```

And at least one replay/local affect verification command, for example:

```bash
python scripts/affect_replay_smoke.py --input <path>
```

or an equivalent documented command.

## Approval rubric

- **Approve fully**: runtime wired, artifacts prove it, docs/reports cleaned, tests pass
- **Approve with caveats**: runtime wired but replay/local proof weak or partial
- **Do not approve**: runtime still not wired, or proof/docs are still inconsistent
