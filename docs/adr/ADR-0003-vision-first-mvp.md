# ADR-0003: Vision-First MVP

## Status
Accepted

## Context
Bay-Max aims to be a multi-modal companion. We need to choose which modality to prioritize for the MVP while keeping the architecture open to future modalities.

## Decision
- Prioritize vision (webcam/video) as the primary input modality for the MVP.
- Implement interface stubs for all perception components in iteration-001.
- Use rule-based dialogue for iteration-001 to avoid dependency on external API keys or large models.
- Define schemas that support future audio, text, and sensor inputs.

## Consequences
- **Positive**: Vision provides rich non-verbal context (engagement, emotion, identity).
- **Positive**: Rule-based dialogue makes the system fully self-contained in iteration-001.
- **Positive**: Interface stubs establish the contract that real implementations will fulfill.
- **Negative**: The system cannot actually see or recognize users until perception is implemented.
- **Negative**: Rule-based responses are limited compared to LLM-generated ones.
- **Mitigation**: The modular architecture allows adding real perception and LLM dialogue in iteration-002 and iteration-004 respectively.
