# Documentation Instructions

## Documentation Structure
- `docs/PROJECT_BRIEF.md` - High-level project overview and goals
- `docs/ARCHITECTURE.md` - System architecture and module descriptions
- `docs/ROADMAP.md` - Feature roadmap by iteration
- `docs/BACKLOG.md` - Detailed backlog of work items
- `docs/PROJECT_MEMORY.md` - Persistent memory for agent continuity
- `docs/CHANGELOG.md` - Change log by iteration
- `docs/project_state.json` - Machine-readable project state
- `docs/adr/` - Architecture Decision Records

## Conventions
- Use safe, non-clinical language throughout.
- Keep docs in sync with actual code and test status.
- Never claim a feature works unless there is code and test evidence.
- ADRs follow the format: Title, Status, Context, Decision, Consequences.
- Update `project_state.json` at the end of each iteration.

## Reporting
- Iteration reports go in `reports/`.
- Each iteration produces both `.md` and `.json` report files.
- Reports must include actual test results and honest status.
