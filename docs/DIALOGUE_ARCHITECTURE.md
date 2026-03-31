# Dialogue Architecture

This document describes the grounded local dialogue system added in Iteration 005.

## Design Principle: Plan-Then-Verbalize

Bay-Max uses a **plan-then-verbalize** pattern to separate strategy from wording:

1. **Planner** (`SupportivePlanner`) deterministically selects a `ResponseStrategy` based on current interaction state and memories — without using an LLM.
2. **PromptBuilder** packages the chosen strategy along with observed facts, retrieved memories, recent conversation turns, and explicit safety constraints into a `GroundedPromptContext`.
3. **DialogueProvider** (the configured backend) receives the grounded context and verbalizes a response within those constraints.

This keeps the response strategy transparent and deterministic, while allowing rich natural-language generation.

## Module Structure

```
src/baymax/dialogue/
├── interfaces.py           DialogueProvider ABC
├── rule_based.py           RuleBasedDialogue — template-based, always available
├── ollama_provider.py      OllamaDialogueProvider — HTTP to local Ollama server
├── transformers_provider.py TransformersDialogueProvider — in-process HF model
├── prompt_builder.py       GroundedPromptContext builder + message formatter
├── safety.py               check_safety() (input) + check_output_safety() (output)
└── factory.py              create_dialogue_provider() factory with fallback
```

## DialogueProvider Interface

```python
class DialogueProvider(ABC):
    def generate(
        self,
        strategy: ResponseStrategy,
        state: InteractionState,
        memories: MemoryQueryResult,
        prompt_context: GroundedPromptContext | None = None,
    ) -> SupportiveResponse: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def is_available(self) -> bool: ...
```

All three concrete backends implement this interface. `RuleBasedDialogue` ignores `prompt_context`; LLM backends use it.

## Grounded Prompt Structure

The `GroundedPromptContext` contains:

| Field | Source |
|-------|--------|
| `user_display_name` | User profile |
| `session_id` | Current session |
| `strategy` | Planner output |
| `state_summary` | engagement, posture, turn_count |
| `memory_refs` | Top-k memories (semantic search + SQL) |
| `recent_turns` | Last N chat turns from SQLite |
| `safety_rules` | Hard-coded prohibitions |
| `context` | Caller-provided free-form context |

The prompt builder formats this into an OpenAI-style `[{role: system, content: ...}, {role: user, content: ...}]` message list. The system message contains the Bay-Max persona and all safety rules. The user message contains context, retrieved memories, recent turns, and the strategy instruction.

## Prompt Template

**System message** (fixed):
```
You are Bay-Max, a warm and attentive supportive companion. You remember what users
share with you across sessions and use those memories to personalise your responses.

Safety rules you must always follow:
- You are a supportive companion, not a clinician.
- You must never diagnose a disease, condition, or illness.
- [... other rules ...]

Respond with a single concise supportive message (1-3 sentences).
Do not add headers, bullet points, or meta-commentary.
```

**User message** (grounded):
```
User name: Alice

Current session state: engagement=medium, posture=upright, turn_count=2

Memories about this user:
  - User mentioned they enjoy warm greetings
  - User is building Bay-Max

Recent conversation:
  user: Hello!

Response guidance: [strategy-specific instruction]

Now write a single supportive response...
```

## Backend Selection

Backend is selected via `BAYMAX_DIALOGUE_BACKEND`:

| Value | Provider | Requires |
|-------|----------|----------|
| `rule_based` | `RuleBasedDialogue` | Nothing |
| `ollama` | `OllamaDialogueProvider` | Ollama server + `BAYMAX_OLLAMA_MODEL` |
| `transformers` | `TransformersDialogueProvider` | `pip install -e ".[dialogue]"` |

## Fallback Behaviour

1. Factory tries to initialise the requested backend.
2. If init fails (e.g., empty `BAYMAX_OLLAMA_MODEL`) and `BAYMAX_ENABLE_RULE_BASED_FALLBACK=true`, it returns `RuleBasedDialogue` instead of raising.
3. At generation time: if `generate()` raises and fallback is enabled, the orchestrator catches the exception and calls `RuleBasedDialogue.generate()`, setting `response.fallback_used = True`.

## Response Metadata

`SupportiveResponse` includes these iteration-005 fields:

| Field | Description |
|-------|-------------|
| `backend` | Which backend generated the response (`rule_based`, `ollama`, `transformers`) |
| `model_name` | Model identifier used (empty for rule_based) |
| `fallback_used` | `True` if the primary backend failed and rule-based was used |
| `safety_flags` | List of safety flag strings raised during input or output check |

When `BAYMAX_ENABLE_DIALOGUE_DEBUG=true`, the `metadata["debug"]` field contains a serialized `DialogueDebugTrace` with latency, token estimates, and raw LLM output.

## GET /v1/dialogue/backends

Returns:
```json
{
  "available_backends": ["rule_based", "ollama", "transformers"],
  "active_backend": "rule_based",
  "active_model": "",
  "fallback_enabled": true
}
```

## Answer-First Routing (Iteration 012+)

Bay-Max uses **intent routing** to select the appropriate response strategy:

| Intent Type | Strategy | Behavior |
|-------------|----------|----------|
| Direct question | `answer_first` | Give the answer, then optional context |
| Follow-up question | `follow_up` | Continue the thread with grounded context |
| Emotional disclosure | `empathize_first` | Validate feelings before any help |
| Memory recall | `recall` | Surface relevant memories naturally |
| Unclear/short text | `clarify` | Ask for repeat instead of guessing |

The `SpeechIntentRouter` in `src/baymax/dialogue/intent_router.py` handles this
classification. It examines transcript quality, emotional keywords, and prior
turn context to select the most appropriate response type.

## Specialist Delegation (Iteration 013+)

Delegation to specialists (web search, memory, tools) is **optional and bounded**:

- The primary assistant handles most turns directly
- Web tools are invoked only when up-to-date information is needed
- Memory retrieval is used only when personalization is clearly relevant
- Vision/perception context is used only when grounded and helpful

This is **not** a multi-agent system with competing top-level orchestrators.
Specialists are tools/adapters behind a single entry point, not autonomous
agents in the hot path. See `docs/VOICE_ASSISTANT_ARCHITECTURE.md` for details.
