# Iteration 012 - Answer-First Speech Focus

## Goal

Make Bay-Max answer spoken and typed user turns directly, stay on-topic, avoid random proactive chatter, and be truthful when running in limited fallback mode.

## Root Causes Found

1. Direct asks were mapped to `SUGGEST` instead of an answer-first strategy.
2. Prompting forced supportive/memory style even for direct Q&A.
3. Memory references could be injected even when irrelevant.
4. `auto_local` model preference did not prioritize strongest local models.
5. Speech accepted any non-empty transcript as valid.
6. Proactive scheduler default triggered arrival/state/quiet chatter.

## What Changed

1. Added deterministic intent routing:
- `src/baymax/dialogue/intent_router.py`
- New `TurnIntent` enum categories in `src/baymax/core/enums.py`

2. Added explicit strategies for answer-first and clarification:
- `ResponseStrategy.ANSWER`
- `ResponseStrategy.WEB_ANSWER`
- `ResponseStrategy.CLARIFY`
- `ResponseStrategy.PROACTIVE_CHECK_IN`

3. Refactored planner to prioritize user intent:
- `src/baymax/planner/supportive_planner.py`
- Direct question/task/follow-up -> `ANSWER`
- Emotional share -> `EMPATHIZE`
- Recall -> `RECALL`
- Unclear -> `CLARIFY`

4. Split prompt behavior into answer-first vs companion families:
- `src/baymax/dialogue/prompt_builder.py`
- Answer-first system instructions for direct Q&A
- Companion/empathy instructions for emotional/supportive turns

5. Added relevance-gated memory usage:
- `select_relevant_memories(...)` in `src/baymax/dialogue/prompt_builder.py`
- New setting: `BAYMAX_DIALOGUE_MEMORY_RELEVANCE_THRESHOLD`

6. Made backend/fallback truthfulness explicit:
- `src/baymax/orchestrator/service.py`
- `src/baymax/schemas/response.py`
- Added `limited_mode` and `limited_mode_reason`
- Added `spoken_text` and `display_text` parity fields
- Rule-based now provides explicit limited-mode wording

7. Improved `auto_local` preference ordering:
- `src/baymax/dialogue/factory.py`
- Preference: Qwen `7b -> 3b -> 1.5b -> 0.5b`

8. Added speech transcript quality gate:
- `src/baymax/live/runtime.py`
- Heuristics: short text, too few tokens, filler-only, garbled repeats, low confidence, possible echo overlap
- Low-quality result triggers clarification instead of literal answer

9. Disabled random proactive speech by default:
- `src/baymax/config/settings.py`
- New default `BAYMAX_PROACTIVE_MODE=affect_only`
- `src/baymax/live/proactive_scheduler.py` now gates by mode
- Added affect-only event `AFFECT_DISTRESS_PERSISTENT`
- `src/baymax/live/runtime.py` emits affect distress event only after stable negative evidence + cooldown

10. Updated telemetry/UI contracts:
- `src/baymax/live/schemas.py`
- `apps/api/live_ws.py`
- `apps/api/live_http.py`
- `apps/ui/src/types/live.ts`

## Backend Selection and Fallback Behavior After Fix

- `auto_local` now chooses strongest local Qwen first (7b > 3b > 1.5b > 0.5b).
- If backend resolves to `rule_based`, runtime now reports limited mode truthfully.
- If fallback occurs from LLM failure, response metadata includes `limited_mode=true` and reason.
- Spoken output is natural (`spoken_text`) while UI can include source list (`display_text`).

## Transcript Quality Gate Rules

Speech turns are rejected for clarification when quality score drops below acceptance threshold.

Signals used:
1. Transcript too short by chars.
2. Too few tokens.
3. Filler-only utterance.
4. Garbled repeated character pattern.
5. Provider confidence below threshold.
6. Possible overlap with last spoken assistant text.

Clarification response:
- "I did not catch that clearly. Could you repeat it?"

## Proactive Policy Before vs After

Before:
- Arrival/recognition gain/posture/engagement/quiet companionship could proactively trigger speech.

After:
- Default mode is `affect_only`.
- Those non-affect triggers are suppressed by default.
- Only persistent distress affect can trigger a gentle check-in, with silence gating + long cooldown.

## Files Changed

- `.env.example`
- `README.md`
- `docs/LOCAL_TESTING.md`
- `apps/api/live_http.py`
- `apps/api/live_ws.py`
- `apps/ui/src/types/live.ts`
- `src/baymax/audio/schemas.py`
- `src/baymax/audio/transcriber.py`
- `src/baymax/config/settings.py`
- `src/baymax/core/enums.py`
- `src/baymax/dialogue/factory.py`
- `src/baymax/dialogue/intent_router.py`
- `src/baymax/dialogue/prompt_builder.py`
- `src/baymax/dialogue/rule_based.py`
- `src/baymax/live/proactive_scheduler.py`
- `src/baymax/live/runtime.py`
- `src/baymax/live/schemas.py`
- `src/baymax/orchestrator/service.py`
- `src/baymax/planner/interfaces.py`
- `src/baymax/planner/supportive_planner.py`
- `src/baymax/schemas/response.py`
- `tests/test_iteration_005.py`
- `tests/test_iteration_006.py`
- `tests/test_iteration_012_answer_first.py`
- `tests/test_speech_intent_routing.py`
- `tests/test_proactive_policy_affect_only.py`
- `tests/test_backend_fallback_truthfulness.py`
- `tests/test_memory_relevance_gating.py`
- `tests/test_low_confidence_transcript_clarification.py`

## Design Decisions and Tradeoffs

1. Deterministic intent router before generation:
- Pro: predictable and easy to test.
- Tradeoff: lexical heuristic approach can misclassify edge phrasing.

2. Strict answer-first prompt family:
- Pro: reduces supportive drift for direct Q&A.
- Tradeoff: less expressive warmth in factual turns by default.

3. Relevance-gated memory injection:
- Pro: removes unrelated memory intrusions.
- Tradeoff: borderline relevant memories may be omitted if threshold is too high.

4. Aggressive proactive suppression default:
- Pro: removes random chatter.
- Tradeoff: fewer spontaneous companion moments unless mode is explicitly changed.

5. Heuristic transcript quality gate:
- Pro: prevents confident hallucinated replies on bad ASR.
- Tradeoff: can ask for repeats on some very short but valid utterances.

## Test Results

Command:
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/NewVolume1/baymax-work/.conda/envs/baymax-ssd/bin/python -m pytest tests/test_iteration_012_answer_first.py tests/test_speech_intent_routing.py tests/test_proactive_policy_affect_only.py tests/test_backend_fallback_truthfulness.py tests/test_memory_relevance_gating.py tests/test_low_confidence_transcript_clarification.py tests/test_iteration_005.py tests/test_iteration_006.py tests/test_iteration_010b.py tests/test_iteration_011.py -q`

Result:
- `142 passed, 1 warning`

## Manual Validation Checklist

See companion checklist file:
- `reports/iteration-012-answer-first-speech-focus-checklist.md`

Manual runtime verification performed in this patch:
- Code-level + test-level verification completed.
- Full interactive mic/camera real-time behavioral validation still needs supervisor-run manual pass.

## Remaining Known Issues

1. Intent router is heuristic and may need semantic upgrade for ambiguous slang.
2. Transcript quality gate currently uses simple rules, not full ASR confidence calibration.
3. Rule-based fallback remains intentionally limited for broad factual QA.

## Recommended Next Iteration

1. Add semantic/reranker intent resolution for follow-ups and ambiguous short turns.
2. Add calibrated ASR quality model + per-device threshold tuning.
3. Add structured UI badges for `limited_mode` and `proactive_mode` in main diagnostics panel.
4. Expand manual real-time evaluation harness with scripted speech scenarios and scoring.
