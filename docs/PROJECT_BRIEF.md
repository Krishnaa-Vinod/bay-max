# Bay-Max Project Brief

## Vision
Bay-Max is a friendly voice-first assistant with memory and empathetic understanding. It listens to users, responds naturally in voice, remembers meaningful interactions, and recognizes enrolled users to personalize the experience.

## Current Scope (Iteration 013)

Bay-Max operates as a **friendly voice assistant first**:

- **Voice-first interaction**: User speaks → Bay-Max listens → processes → responds via TTS
- **Single assistant runtime**: One primary assistant owns the interaction loop
- **Activation modes**: Push-to-talk, continuous VAD, wake phrase gate
- **Follow-up conversations**: Natural multi-turn dialogue without re-activation
- **Barge-in support**: Interrupt while assistant is speaking
- **Answer-first behavior**: Direct answers for questions, empathy-first for emotional shares

The richer perception and memory stack **enhances** responses but doesn't dominate:
- Memory retrieval adds personalization when relevant
- User recognition personalizes greetings and context
- Affect detection enables empathetic responses (when stable)

### Architecture Position
- **Primary pattern**: Single assistant runtime with explicit state machine
- **Future pattern**: Optional specialist delegation behind one triage boundary
- **Principle**: Keep the realtime loop simple; put complexity behind tools/adapters

## Non-Goals for Iteration 013
- Full proactive emotional check-ins (deferred)
- Mandatory internal multi-agent architecture
- Production wake-word quality (current is ASR-gated provisional)
- Robotics integration
- Clinical diagnosis or medical advice
- Wearable device integration

## Supervisor
Krishnaa Vinod is the project supervisor and final approver.

## Engineering Principles
1. Build a modular monolith, not microservices.
2. Prefer clear interfaces over premature optimization.
3. Make repo memory explicit and persistent in files.
4. Never claim a feature is complete unless code, tests, and file evidence exist.
5. Keep large artifacts, model caches, and temporary files out of git.
6. Use environment variables for all data/model/cache locations.
7. Default to safe, non-clinical language in docs and code comments.
8. **The assistant should be useful and responsive first, proactive second.**
