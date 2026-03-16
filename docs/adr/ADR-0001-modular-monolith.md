# ADR-0001: Modular Monolith Architecture

## Status
Accepted

## Context
Bay-Max needs a software architecture that supports iterative development of multiple subsystems (vision, memory, dialogue) while keeping the system simple enough for a small team to develop and a single process to run.

## Decision
We adopt a modular monolith architecture. All modules live in a single Python package (`src/baymax/`) with clear interface boundaries. Each module exposes its API through an `interfaces.py` file. Modules depend on interfaces, not concrete implementations.

## Consequences
- **Positive**: Simple deployment (single process), easy to refactor, clear module boundaries, no network overhead between modules.
- **Positive**: Interfaces allow swapping implementations (e.g., stub -> real face detection) without changing dependent code.
- **Negative**: All modules share the same process; a crash in one affects all.
- **Negative**: Cannot scale modules independently (not needed at MVP stage).
- **Mitigation**: If scaling becomes necessary, modules can be extracted into services later because they already communicate through interfaces.
