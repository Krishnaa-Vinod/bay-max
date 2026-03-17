# Verification Sweep Checklist

## What Copilot must prove
- API health works
- user create/get works
- face enrollment works
- same-user recognition works
- unknown/no-match case works
- pose/body-state fields appear
- session state evolves
- typed turns persist
- session consolidation works
- memory summary works
- memory correction works
- semantic retrieval works
- /respond works in rule_based mode
- /respond works in real local-model mode when available
- grounded recall actually mentions remembered content
- no-memory response does not fabricate
- safety refusal works
- backend fallback works
- dialogue backends endpoint works
- LanceDB real round-trip is proven or honestly marked not_run
- demo/manual flow is proven or honestly marked not_run

## Artifact locations
- artifacts/verification_005/manifest.json
- artifacts/verification_005/summary.md
- artifacts/verification_005/api/
- artifacts/verification_005/perception/
- artifacts/verification_005/memory/
- artifacts/verification_005/dialogue/
- artifacts/verification_005/safety/
- artifacts/verification_005/logs/

## Must-have proof for grounded recall
- rendered prompt/messages saved
- memory_refs saved
- response text saved
- chosen strategy saved
- recall intent saved
- mentions_session_a=true

## Output you should send back for review
- branch name
- latest commit sha
- artifacts/verification_005/manifest.json
- artifacts/verification_005/summary.md
- updated reports/iteration-005.md
- updated reports/iteration-005.json
