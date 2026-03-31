# Iteration 012 Verification Checklist

## Setup

- [ ] `pip install -e ".[all]"`
- [ ] `cp .env.example .env`
- [ ] Confirm `BAYMAX_PROACTIVE_MODE=affect_only`
- [ ] Start backend: `make run-local-backend-full`
- [ ] Start UI: `make run-ui`

## Direct Question Behavior

- [ ] Ask: `What is the capital of France?`
- [ ] Expected: answer-first (`Paris` or concise equivalent), no self-care drift
- [ ] Ask: `How do I center a div in CSS?`
- [ ] Expected: concise useful CSS answer first

## Follow-Up Grounding

- [ ] Ask a direct question and receive answer
- [ ] Say: `why?`
- [ ] Expected: coherent follow-up grounded in previous turn

## Emotion Path

- [ ] Say: `I feel sad today.`
- [ ] Expected: empathetic response (not generic factual mode)

## Proactive Suppression (Default)

- [ ] Trigger arrival / recognition changes
- [ ] Trigger posture / engagement changes
- [ ] Wait through quiet-companionship window
- [ ] Expected: no proactive speech from these triggers in default mode

## Affect-Only Check-In

- [ ] Hold stable negative affect above confidence threshold and duration
- [ ] Ensure no recent user/assistant speech before check-in window
- [ ] Expected: at most one gentle check-in
- [ ] Expected: no repeated probing after cooldown gate

## Transcript Quality Gate

- [ ] Speak clipped or filler-only utterance
- [ ] Expected: `I did not catch that clearly. Could you repeat it?`
- [ ] Speak clear utterance
- [ ] Expected: normal answer-first response path

## Limited Backend Truthfulness

- [ ] Force `BAYMAX_DIALOGUE_BACKEND=rule_based` or disable LLM backends
- [ ] Expected: runtime/UI indicate fallback or limited mode truthfully
- [ ] Expected: no pretending to full broad factual competency

## Regression Sweep

- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /mnt/NewVolume1/baymax-work/.conda/envs/baymax-ssd/bin/python -m pytest tests/test_iteration_010b.py tests/test_iteration_011.py -q`
- [ ] Expected: pass (or only intentional behavior updates)
