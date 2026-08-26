# ROADMAP.md — Holo Companion

## Guiding milestone

The first major milestone is complete only when the companion can hold a comfortable **10–15 minute voice conversation** without an avatar.

Required experience:

- natural hands-free input,
- strong Indonesian recognition with English code switching,
- reasonably fast response,
- expressive/non-robotic voice,
- interruption/barge-in,
- stable multi-turn conversation.

Do not proceed to avatar/hologram merely because a demo works once.

---

## Phase 0 — Foundation

Status: **COMPLETE**

### Goal

Create a portable Python 3.11 project that can move with the SSD between AMD and NVIDIA PCs.

### Already established

- Ubuntu development environment.
- Git available.
- FFmpeg available.
- `uv` installed.
- Python 3.11 installed through `uv`.
- `.venv` created.
- Project may run on:
  - Ryzen 5 5600 + RX 9060 XT 16 GB.
  - NVIDIA PC with RTX 5050-class ~8 GB VRAM.

### Remaining

- [x] Initialize/verify `pyproject.toml`.
- [x] Add `.gitignore`.
- [x] Add `.env.example`.
- [x] Establish `src/` package only when first implementation starts. (Deferred until Phase 1 implementation.)
- [x] Verify `uv sync` / `uv run`.
- [x] Do not install GPU-specific stacks yet.

### Quality gate

A clean clone/copy of the project can recreate the Python environment from project metadata without relying on system Python packages. Verified with `uv sync --locked` and `uv run --locked` on Python 3.11.16.

---

## Phase 1 — Audio I/O

Status: **COMPLETE**

### Goal

Reliably capture and play audio before adding AI.

### Tasks

- [x] Add minimal audio dependencies.
- [x] Enumerate input/output devices.
- [x] Select/test actual microphone.
- [x] Record a short WAV.
- [x] Replay/inspect it.
- [x] Confirm sample rate/channel conversion is explicit.
- [x] Check clipping, low volume, fan noise, and echo.

### User actions preferred

The user should:

- choose the physical mic/speaker,
- listen to recorded WAV quality,
- report noise/clipping,
- grant microphone permission if necessary.

### Quality gate

A repeatable recording path produces clean speech that sounds correct to the user. Verified with the user's analog headset microphone at 40% PipeWire source gain: a 10-second 16 kHz mono PCM_16 WAV was clear without audible distortion. Diagnostics reported 68 clipping samples out of 160,000 frames (0.043%) and RMS 0.096.

---

## Phase 2 — VAD / turn detection

Status: **COMPLETE**

### Goal

Hands-free detection of speech start/end.

### Initial candidate

Silero VAD.

### Tasks

- [x] Add provider interface.
- [x] Feed microphone frames continuously.
- [x] Emit `SpeechStarted` / `SpeechEnded`.
- [x] Tune minimum speech/silence parameters.
- [x] Test normal speech, pauses, fan noise, and false starts.
- [x] Measure end-of-speech detection latency.

### Quality gate

The user can talk normally without a push-to-talk button, and ordinary mid-sentence pauses do not constantly cut the utterance. Verified on the user's analog headset with threshold 0.6, minimum speech 500 ms, and minimum silence 700 ms: two intended turns were detected, four short noise/transient candidates were rejected, the mid-sentence pause stayed within one turn, no queue overrun occurred, and measured end-of-speech latency was 704 ms.

---

## Phase 3 — STT

Status: **NEXT**

### Goal

Accurate Indonesian + English-code-switch transcription with acceptable latency.

### Initial candidate

Faster-Whisper.

### Benchmark candidates

- `large-v3-turbo`
- `large-v3`
- smaller fallback if necessary.

### Hardware strategy

Benchmark CPU first on Ryzen 5 5600.

Do not spend time compiling an AMD/HIP path until CPU measurements show a real need.

NVIDIA CUDA acceleration may be tested later on the second PC.

### Test set

Create roughly 20–30 real recordings including:

- casual Indonesian,
- slang,
- English technical words,
- GitHub/repository/programming vocabulary,
- numbers,
- fast speech,
- quiet speech,
- mild room/fan noise.

### Measure

- transcript correctness,
- meaning-changing errors,
- STT finalize latency,
- CPU/RAM use.

### Quality gate

Conversational transcription is reliably understandable and does not frequently corrupt key technical terms.

Do not use the LLM to hide a bad STT system.

---

## Phase 4 — Streaming LLM

Status: **BLOCKED BY PHASE 3**

### Goal

Produce conversational responses quickly without coupling to one model vendor.

### Tasks

- [ ] Define `LLMProvider`.
- [ ] Implement one OpenAI-compatible streaming adapter.
- [ ] Add minimal persona.
- [ ] Stream tokens/chunks.
- [ ] Measure time-to-first-token.
- [ ] Keep persona concise.
- [ ] Avoid long-term memory.

### Initial persona direction

Adult, warm, playful, curious, cute, slightly teasing, supportive.

Conversational Indonesian, occasional natural English, not customer-service-like.

### Quality gate

Text-only multi-turn responses feel conversational and begin streaming quickly.

---

## Phase 5 — TTS bake-in

Status: **BLOCKED BY PHASE 4**

### Goal

Produce a natural, expressive, character-appropriate voice.

### Rule

Do not permanently choose a TTS engine from internet reputation alone.

### Provider interface must support

- streaming/incremental synthesis when available,
- cancellation,
- style/emotion hints when available,
- cloud and local implementations,
- reference/custom voice later.

### Evaluation set

Use the exact same Indonesian/code-switch sentences across providers.

Include:

- neutral,
- happy,
- shy,
- concerned,
- excited,
- whisper/soft,
- technical sentences,
- numbers,
- English package/repository names.

### Suggested scoring

| Criterion | Weight |
|---|---:|
| Indonesian pronunciation | 25% |
| Naturalness | 20% |
| Character/kawaii fit | 20% |
| Expressiveness | 15% |
| Latency | 10% |
| Cost/hardware | 5% |
| License/deployment fit | 5% |

### User action preferred

The user performs blind/subjective listening and scores samples.

### Quality gate

Chosen provider/voice is pleasant enough that the user prefers listening to it for a longer conversation rather than switching back to text.

---

## Phase 6 — Real-time orchestration + interruption

Status: **BLOCKED BY PHASE 5**

### Goal

Make the interaction feel alive instead of turn-based.

### Tasks

- [ ] Implement turn IDs.
- [ ] Bounded async queues.
- [ ] Incremental text-to-TTS chunking.
- [ ] Playback queue.
- [ ] User barge-in.
- [ ] Immediate playback stop.
- [ ] Cancel stale LLM/TTS work.
- [ ] Drop stale chunks by turn ID.
- [ ] Handle provider timeout/failure.
- [ ] Add runtime latency metrics.

### Quality gate

While the companion is speaking, the user can interrupt naturally and the companion stops quickly, listens, and responds to the new utterance without old audio leaking afterward.

---

## Phase 7 — Response planner / emotion

Status: **BLOCKED BY PHASE 6**

### Goal

Create a minimal structured state shared by voice now and avatar later.

### Target object

```text
text
emotion
energy
speaking_style
```

Do not create a complex artificial psychology engine yet.

### Quality gate

Emotion/style choices improve TTS delivery consistently instead of producing random exaggerated behavior.

---

## Phase 8 — Voice-core acceptance test

Status: **BLOCKED BY PHASE 7**

### Acceptance session

Run a continuous 10–15 minute conversation.

Test:

- casual conversation,
- technical question,
- code-switching,
- short answer,
- longer explanation,
- interruption,
- corrections such as "eh bukan itu",
- quiet speech,
- brief silence,
- user speaking while audio is playing.

Record metrics and user impressions.

### Pass criteria

- [ ] Conversation remains stable.
- [ ] No frequent STT meaning failures.
- [ ] No annoying push-to-talk requirement.
- [ ] Interruption works.
- [ ] TTS remains pleasant.
- [ ] Perceived response latency is acceptable.
- [ ] No stale audio after cancellation.
- [ ] User says the voice-only system already feels worth continuing.

---

# Deferred roadmap

Only after Phase 8 passes:

## Phase 9 — Persistent memory

- episodic facts,
- preference memory,
- retrieval,
- user-controlled deletion,
- memory boundaries/privacy.

## Phase 10 — Character/avatar

- VRM or Live2D decision,
- lip sync,
- blink/idle motion,
- facial expression,
- emotion -> animation mapping.

## Phase 11 — Hologram presentation

- render scene optimized for black background/reflection,
- display geometry,
- Pepper's Ghost prototype,
- latency between speech/emotion/avatar,
- physical enclosure.

## Phase 12 — Product hardening

- packaging,
- startup service,
- crash recovery,
- offline/cloud modes,
- profiling,
- configuration UI,
- licensing review.

---

# Rule for the next action

Always work on the **first unchecked item in the active phase**, unless a discovered blocker forces a smaller prerequisite.

Do not jump ahead because a later feature is more exciting.
