# AGENTS.md — Holo Companion

## Multi Brain (MANDATORY)

- Read `.multibrain/session.md` before starting work.
- Use `.multibrain/session.md` as the master index only.
- Open only the `.multibrain/indexes/*.md` bucket files that match the current task.
- Open `.multibrain/context/*.md` only when the selected bucket points to deeper context that matters.
- After meaningful work, update the relevant named bucket and refresh the master index if needed.

## Mission

Build a production-minded, low-latency AI voice companion that can later drive a VTuber-style avatar and a hologram display.

The first product milestone is **voice-only**: the user should be able to talk naturally in Indonesian (including Indonesian-English code switching), hear a natural expressive response quickly, interrupt the assistant while it is speaking, and continue a 10–15 minute conversation without the interaction feeling like a slow chatbot.

Do **not** optimize for a flashy demo. Optimize for a foundation that can grow into:

1. real-time voice companion,
2. personality + emotion,
3. long-term memory,
4. avatar/animation,
5. hologram presentation.

## Canonical project docs

Before non-trivial implementation work, read:

- `ARCHITECTURE.md` — system boundaries, interfaces, runtime design, portability rules.
- `ROADMAP.md` — current phase, quality gates, and order of implementation.

Do not create additional planning/docs files unless they are genuinely necessary. Prefer updating these three files.

## Core engineering rules

### 1. Build our own orchestration core; reuse mature models/libraries

Do not rewrite speech recognition, VAD, or TTS research from scratch.

Use mature components behind our own interfaces so providers can be replaced later.

Reference projects such as Open-LLM-VTuber and AIRI for ideas, but do not tightly couple the product to their internal architecture or fork them wholesale unless there is a clear technical reason.

### 2. Hardware must be portable

The project lives on an SSD that may move between PCs.

Known development machines may include:

- AMD Ryzen 5 5600 + Radeon RX 9060 XT 16 GB + Ubuntu.
- Another PC with an NVIDIA RTX 5050-class GPU around 8 GB VRAM.

Therefore:

- Never assume CUDA.
- Never assume ROCm.
- Never hardcode an absolute project path.
- CPU operation must remain a valid baseline for core audio/VAD/orchestration.
- GPU acceleration is an optional provider/runtime capability.
- Prefer `device: auto | cpu | cuda | rocm` style configuration.
- `.venv` is machine-local/disposable and must never be committed.
- The reproducible source of truth is `pyproject.toml` + `uv.lock`.
- If moving the SSD changes paths or machines, recreate/sync the environment instead of trying to preserve a broken virtualenv.

### 3. Python/runtime conventions

- Target Python 3.11 unless a dependency forces a documented change.
- Use `uv` for project/package management.
- Prefer `uv add`, `uv remove`, `uv sync`, and `uv run`.
- Do not use `sudo pip`.
- Do not modify the Ubuntu system Python.
- Keep secrets in `.env`; commit only `.env.example`.
- Keep runtime configuration declarative and validated.

### 4. Architecture before convenience

Do not create a giant `main.py`.

Keep stable interfaces around:

- audio capture/output,
- VAD,
- STT,
- LLM,
- TTS,
- response/emotion planning,
- turn management,
- runtime orchestration.

Provider-specific code must stay behind provider interfaces.

A future avatar must be able to consume structured assistant state without changing the LLM/TTS core.

### 5. Real-time behavior is a first-class requirement

The system must be designed for streaming and cancellation from the start.

Required capabilities:

- continuous/hands-free listening,
- speech-start and speech-end detection,
- streaming or low-latency STT where useful,
- streaming LLM tokens,
- incremental TTS synthesis/playback,
- barge-in/interruption,
- cancellation propagation,
- queue backpressure,
- latency measurement.

Never implement the final conversation flow as:

`record all -> transcribe all -> generate all text -> synthesize all audio -> play`

when a streaming implementation is feasible.

### 6. Indonesian quality matters

Primary conversational language is Indonesian, often mixed with English technical terms.

Benchmark real speech containing:

- casual Indonesian,
- slang,
- English code switching,
- programming terminology,
- numbers,
- quiet speech,
- fast speech,
- fan/background noise.

Do not choose STT/TTS solely from leaderboard claims. Benchmark on the user's real audio and listening preference.

### 7. TTS voice identity

The long-term voice should be original and character-specific: cute/kawaii, expressive, warm, natural, and adult-sounding rather than childish.

Do not depend on cloning a famous VTuber/voice actor identity.

TTS must be swappable. Candidate providers/models may change over time.

The response layer should eventually produce structured style information such as:

```json
{
  "text": "Hehe, akhirnya datang juga.",
  "emotion": "happy",
  "energy": 0.55,
  "speaking_style": "soft_playful"
}
```

Later, the same state can drive facial expression and animation.

### 8. Memory is intentionally deferred

Do not build long-term memory until the voice loop is stable and enjoyable.

Short conversation context is allowed. Persistent semantic memory, relationship state, schedules, and similar systems belong to later phases.

### 9. Security and privacy

- Never commit API keys, tokens, reference voice files with unclear rights, or personal recordings.
- Avoid sending microphone audio to external services unless that provider is explicitly selected/configured.
- Clearly separate local vs cloud providers.
- Do not silently upload audio.
- Prefer minimal data retention.

## Token- and time-efficient agent behavior

This project is being developed with AI coding agents, so token efficiency matters.

### Agent should do

Use tools/code autonomously for work where the agent adds substantial value:

- inspect/search the repository,
- design interfaces,
- implement code,
- modify multiple related files,
- write focused tests,
- debug non-obvious failures,
- inspect dependency/API behavior,
- run short deterministic verification commands,
- review diffs,
- reason about architecture.

### Ask the user to do simple/manual work when cheaper

If a task is simple, hardware-dependent, subjective, privileged, or would waste many agent tokens, stop and give the user a short exact instruction.

Good user-action examples:

- run one or two shell commands and paste output,
- install an OS package requiring `sudo`,
- choose/confirm microphone or speaker,
- create an API key/account,
- approve OS microphone permission,
- download a very large model,
- listen to A/B TTS samples and score them,
- move/copy hardware,
- reboot,
- test from another PC,
- report subjective audio quality.

Format such handoffs as:

```text
USER ACTION
1. <exact action>
2. <exact command if needed>
3. Send back: <only the information needed>
```

Do not give ten optional branches when one next action is enough.

### Do not hand off trivial repository work unnecessarily

If the agent can safely perform a small code edit, file creation, formatting, grep, or test in a few tool calls, do it instead of asking the user to type it manually.

The goal is **token efficiency without turning the user into the coding agent**.

## Execution style

For every meaningful task:

1. Read the relevant current code/docs.
2. Identify the smallest vertical slice that advances the active roadmap phase.
3. Implement only that slice.
4. Run focused verification.
5. Report:
   - what changed,
   - what was verified,
   - measured latency/quality when relevant,
   - the single next step.
6. Update `ROADMAP.md` only when a milestone/quality gate actually changes.

Avoid speculative large refactors.

## Definition of done

A feature is not done because code exists.

It is done when:

- the intended path runs,
- important failure paths are handled,
- configuration is documented,
- targeted tests/checks pass,
- no provider-specific assumption leaks into generic core code,
- latency-impacting changes have measurements when practical.

## Current priority

Until `ROADMAP.md` says otherwise, prioritize the voice core in this order:

`audio -> VAD -> STT -> LLM streaming -> TTS -> interruption -> latency tuning`

Avatar, Live2D/VRM, hologram rendering, long-term memory, and relationship simulation are out of scope for the current phase.
