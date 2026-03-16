# Safety Guardrails

Bay-Max is designed as a **supportive companion**, not a clinical tool. This document describes the safety constraints that prevent the system from making unsupported medical claims or diagnoses.

## Core Principle

Bay-Max must never:
- Diagnose a disease, condition, or illness from face, pose, or behaviour
- Assert clinical certainty beyond what is directly observed
- Recommend medication, treatment, or dosage
- Replace or simulate a qualified healthcare professional

When the system detects a user request that crosses these lines, it responds with a warm redirect rather than refusing bluntly.

## Safety Layers

### Layer 1: User Input Check (`check_safety`)

Before generating any response, the orchestrator passes the user's `context` string to `check_safety()`. This function matches against two pattern categories:

**Diagnosis request patterns:**
- `"diagnose me"`, `"do i have"`, `"what disease"`, `"what condition"`, `"am i sick"`, `"detect disease"`, `"diagnose from my face"`, `"what medication"`, `"prescribe"`, `"medical advice"`, etc.

**Medical certainty patterns:**
- `"are you sure i am"`, `"confirm i have"`, `"100% certain"`, etc.

When matched, the system returns a `SafetyDecision` with `is_safe=False` and a pre-composed redirect:

> "I'm here to support you, but I'm not able to make medical diagnoses or interpret health symptoms. If you're concerned about your health, please speak with a qualified healthcare professional. I'm happy to listen and offer companionship."

The response is returned with `safety_flags` populated and the redirect as the message.

### Layer 2: LLM Output Check (`check_output_safety`)

When using a generative LLM backend (Ollama, Transformers), the system validates the generated text before returning it. This catches cases where the LLM might violate safety rules despite prompt instructions.

**Dangerous output patterns:**
- `"you have"` (asserting a condition)
- `"you are diagnosed"`, `"i diagnose"`, `"my diagnosis is"`, `"your condition is"`
- `"you should take"`, `"i recommend medication"`, `"take this drug"`

When a dangerous output pattern is detected, the LLM output is replaced with the same safe redirect message.

### Layer 3: Prompt-Level Rules

The system prompt explicitly lists the safety rules for the LLM:
```
- You are a supportive companion, not a clinician.
- You must never diagnose a disease, condition, or illness.
- You must never recommend medication, treatment, or dosage.
- You must never assert medical certainty beyond what you observe.
- If the user asks for medical advice or diagnosis, gently redirect them
  to a qualified healthcare professional.
- Never fabricate memories or facts you do not have.
```

## Configuration

| Variable | Default | Effect |
|----------|---------|--------|
| `BAYMAX_ENABLE_SAFE_HEALTH_MODE` | `true` | Enables input + output safety checks |

Setting `BAYMAX_ENABLE_SAFE_HEALTH_MODE=false` disables both the input check and LLM output validation. This is not recommended for production use.

## Testing Safety

```bash
# Run automated safety tests
pytest tests/test_iteration_005.py::TestSafetyGating -v

# Manual via API:
curl -s -X POST http://localhost:8000/v1/respond \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<sid>","user_id":"<uid>","context":"Can you diagnose me from my face?"}' \
  | python -m json.tool
# Expected: safety_flags is non-empty, message contains redirect
```

## Limitations

- Pattern-based matching may miss paraphrases or novel phrasings of unsafe requests
- LLM output check uses substring matching — not semantic understanding
- These guardrails are a first-pass safeguard, not a complete safety system
- For production clinical applications, additional domain-expert review would be required

## Scope Clarification

Bay-Max is explicitly scoped to:
- Recognising and greeting enrolled users
- Providing emotional support and encouragement
- Remembering and personalising interactions based on user history
- Suggesting simple self-care actions (breaks, relaxation) when appropriate

Bay-Max is explicitly **not** scoped to:
- Diagnosing physical or mental health conditions
- Interpreting medical symptoms
- Providing clinical recommendations
- Replacing professional psychological or medical care
