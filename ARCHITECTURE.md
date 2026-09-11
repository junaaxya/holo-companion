# ARCHITECTURE.md — Holo Companion

## 1. Product goal

Holo Companion is a modular real-time AI companion.

The product should eventually combine:

- natural two-way speech,
- consistent character/personality,
- expressive voice,
- memory,
- VTuber-style visual embodiment,
- animation/expression control,
- hologram-style physical display.

The present architecture is deliberately centered on the **voice runtime**, because a convincing voice interaction must work before visual embodiment is added.

## 2. Design principles

1. **Streaming first** — latency is part of the product.
2. **Provider-independent core** — STT, LLM, TTS, and hardware accelerators are replaceable.
3. **Portable across AMD/NVIDIA/CPU machines**.
4. **Structured assistant state** — text, emotion, style, and later gestures share one response model.
5. **Measure before optimizing**.
6. **Local/cloud boundaries are explicit**.
7. **Do not add memory/avatar complexity before the voice loop passes its quality gate**.

## 3. High-level runtime

```text
Microphone
    |
    v
Audio Capture
    |
    +--------------------+
    |                    |
    v                    |
VAD / Turn Detection     |
    |                    |
    v                    |
STT                      |
    |                    |
    v                    |
Turn Manager             |
    |                    |
    +--> Persona/Context |
    |                    |
    v                    |
LLM (streaming)          |
    |                    |
    v                    |
Response Planner         |
    |                    |
    +--> text            |
    +--> emotion         |
    +--> energy          |
    +--> speaking_style  |
    |                    |
    v                    |
TTS (incremental)        |
    |                    |
    v                    |
Audio Playback           |
    |                    |
    +------ barge-in -----+
```

Future:

```text
Response Planner
    |
    +--> TTS -> Speaker
    |
    +--> Character State -> Avatar Renderer -> Hologram Display
```

## 4. Recommended source layout

Use a `src` layout.

```text
holo-companion/
├── AGENTS.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
├── config/
│   └── default.yaml
├── src/
│   └── holo_companion/
│       ├── __init__.py
│       ├── config/
│       │   ├── models.py
│       │   └── loader.py
│       ├── audio/
│       │   ├── capture.py
│       │   ├── playback.py
│       │   └── types.py
│       ├── vad/
│       │   ├── base.py
│       │   └── silero.py
│       ├── stt/
│       │   ├── base.py
│       │   └── faster_whisper.py
│       ├── llm/
│       │   ├── base.py
│       │   └── openai_compatible.py
│       ├── tts/
│       │   ├── base.py
│       │   └── providers/
│       ├── dialogue/
│       │   ├── persona.py
│       │   ├── response.py
│       │   └── turn_manager.py
│       ├── runtime/
│       │   ├── events.py
│       │   ├── orchestrator.py
│       │   ├── cancellation.py
│       │   └── metrics.py
│       └── cli.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

Do not create all modules immediately. Grow this structure phase-by-phase.

## 5. Internal event model

The runtime should communicate through typed events instead of direct provider-to-provider coupling.

Candidate events:

```text
AudioFrame
SpeechStarted
SpeechEnded
TranscriptPartial
TranscriptFinal
AssistantTextChunk
AssistantResponsePlan
TTSAudioChunk
PlaybackStarted
PlaybackFinished
CancelTurn
RuntimeError
```

This makes interruption, metrics, debugging, recording/replay tests, and future avatar integration much easier.

## 6. Concurrency model

Use AnyIO for orchestration so cancellation, task groups, and test coordination stay explicit.

Conceptual tasks:

```text
audio_capture_task
vad_task
stt_task
dialogue_task
tts_task
playback_task
```

Connect stages with bounded `asyncio.Queue` objects.

Why bounded queues:

- prevent unlimited memory growth,
- expose backpressure,
- make latency problems measurable,
- prevent stale audio/text from being played after interruption.

Provider code that is blocking should be wrapped appropriately rather than blocking the main event loop.

The first Phase 6 slice owns one active turn at a time in a provider-neutral state machine:

```text
IDLE
LISTENING
TRANSCRIBING
THINKING
SPEAKING
INTERRUPTING
ERROR
STOPPED
```

Transitions are explicit and invalid transitions raise typed errors. A new speech-start event invalidates the active turn/generation before cancellation, stops playback, and only then allows replacement work. STT remains synchronous at the provider boundary and is run through a serialized non-abandoning worker thread adapter, so two STT calls cannot overlap and teardown waits for the blocking call to return. This has a known latency ceiling: shutdown or interruption cannot reclaim a CPU-bound STT call until the provider returns.

Playback is currently a tiny async sink protocol only: `write_and_start`, `stop`, and `aclose`. The first playback timestamp is taken after the sink acknowledges write/start admission. It is a proxy for queued playback start, not proof that a physical DAC emitted audio.

## 7. Turn lifecycle

Each conversational turn gets a unique `turn_id`.

Example:

```text
speech starts
 -> create/user turn context
 -> speech ends
 -> STT final
 -> LLM generation
 -> TTS chunks
 -> playback
 -> turn completed
```

On barge-in:

```text
user speech detected
 -> increment/cancel active assistant turn
 -> stop playback immediately
 -> cancel pending TTS
 -> cancel LLM stream if supported
 -> discard stale queued chunks for old turn_id
 -> start new user turn
```

Every queued/generated artifact should be associated with its `turn_id` so old audio cannot leak into a newer conversation.

## 8. Provider interfaces

Use small stable protocols/abstract interfaces.

Illustrative shape only:

```python
class STTProvider(Protocol):
    async def transcribe(self, audio, *, language_hint=None) -> Transcript:
        ...

class LLMProvider(Protocol):
    async def stream(self, messages, *, cancel_token) -> AsyncIterator[str]:
        ...

class TTSProvider(Protocol):
    async def stream(self, request, *, cancel_token) -> AsyncIterator[AudioChunk]:
        ...
```

Provider-specific configuration must not leak into turn-manager logic.

## 9. Initial provider strategy

### VAD

Initial candidate: Silero VAD.

Run on CPU by default. VAD is not a good reason to reserve GPU resources.

### STT

Initial candidate: Faster-Whisper.

First benchmark CPU on the Ryzen 5 5600 before investing time in AMD-specific GPU compilation.

Suggested model comparison:

- `large-v3-turbo`
- `large-v3`
- a smaller model if latency requires it.

Select by measured Indonesian/code-switch accuracy + latency, not model size.

### LLM

Use an OpenAI-compatible streaming provider interface first.

The application must not depend on a particular commercial model name.

A provider/model change should be configuration, not architecture.

### TTS

Do not permanently select a TTS engine before listening tests.

The interface should support:

- cloud streaming TTS,
- local TTS,
- reference/custom voice,
- style/emotion metadata,
- cancellation.

Candidate families may include Fish Audio, Qwen TTS, GPT-SoVITS, or newer alternatives. Treat the list as replaceable.

## 10. Response model

Do not pass raw LLM strings directly into every downstream subsystem forever.

The target internal object is similar to:

```python
AssistantResponsePlan(
    text="Hehe, akhirnya datang juga.",
    emotion="happy",
    energy=0.55,
    speaking_style="soft_playful",
)
```

Later fields may include:

```text
facial_expression
gesture
eye_direction
animation
```

The response planner must remain simple in early phases. Do not add an elaborate emotion simulation before basic voice quality is proven.

## 11. Audio conventions

Choose one canonical internal PCM representation and convert at boundaries.

Recommended starting point for STT/VAD pipeline:

- mono,
- 16 kHz where the model expects it,
- float32 or int16 internally, chosen consistently,
- small frames suitable for VAD.

Preserve the physical input device's native sample rate at capture only as needed; resample deliberately rather than relying on hidden conversions.

TTS/playback may use a different sample rate. Do not force TTS output to 16 kHz if the model provides higher-quality audio.

## 12. Latency telemetry

Instrument timestamps from the beginning.

At minimum record:

```text
speech_end_detected
stt_start
stt_complete
llm_request_start
first_llm_token
tts_request_start
first_tts_audio
first_playback_audio
```

Derived metrics:

```text
end_of_speech_detection_ms
stt_finalize_ms
llm_ttft_ms
tts_ttfa_ms
perceived_response_latency_ms
total_turn_ms
```

The runtime clock is injectable and monotonic. Cancelled/error turn metrics are preserved for inspection, with derived speech-end-to-transcript/token/audio/playback values left nullable when a stage never completed.

Optimization decisions should be based on these numbers.

Initial aspirational perceived-response target:

- roughly 1–2 seconds after the user finishes speaking.

This is a project target, not a guaranteed provider benchmark.

## 13. Configuration model

Target configuration shape:

```yaml
runtime:
  device: auto
  log_level: INFO

audio:
  input_device: auto
  output_device: auto

vad:
  provider: silero

stt:
  provider: faster_whisper
  model: large-v3-turbo
  device: cpu

llm:
  provider: openai_compatible
  model: ${LLM_MODEL}

tts:
  provider: ${TTS_PROVIDER}

conversation:
  allow_barge_in: true
```

Environment variables hold secrets; YAML holds non-secret behavior.

## 14. Portability across the SSD / two PCs

Never commit:

```text
.venv/
__pycache__/
model caches/
machine-specific audio device IDs
secret .env files
```

When moving to a different PC:

```text
uv sync
```

should recreate/synchronize the environment.

GPU-specific runtime setup is host-specific.

Do not encode a Radeon-only or NVIDIA-only install into the generic application startup path.

If local GPU inference becomes complex, prefer one of:

1. optional dependency groups,
2. provider-specific setup documentation,
3. separate local inference service/container,

instead of polluting the core package with mutually incompatible GPU requirements.

## 15. Testing strategy

### Unit tests

Use for:

- turn state transitions,
- cancellation,
- config parsing,
- response chunking,
- queue behavior,
- provider adapters with mocked transports.

### Integration tests

Use recorded fixtures for:

- VAD -> speech segments,
- STT provider invocation,
- streaming LLM -> chunker,
- TTS stream -> playback buffer.

### Manual quality tests

The user performs subjective tests for:

- microphone quality,
- Indonesian STT accuracy,
- TTS naturalness,
- kawaii/character fit,
- interruption feel,
- 10–15 minute conversation comfort.

Do not try to replace subjective listening with a fake automated score.

## 16. Explicitly deferred systems

Until the voice-core quality gate passes, do not build:

- Live2D/VRM rendering,
- hologram output,
- complex long-term memory,
- relationship level mechanics,
- jealousy/attachment simulations,
- multi-agent roleplay,
- home automation,
- vision,
- wake-word training.

These can be revisited after the core feels natural.
