from holo_companion.runtime.metrics import TurnTimestamps


def with_tts_request(timestamps: TurnTimestamps, now: float) -> TurnTimestamps:
    return TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=timestamps.llm_invocation_start,
        llm_request_start=timestamps.llm_request_start,
        first_llm_raw_delta=timestamps.first_llm_raw_delta,
        first_llm_token=timestamps.first_llm_token,
        tts_request_start=timestamps.tts_request_start or now,
        first_tts_audio=timestamps.first_tts_audio,
        first_playback_audio=timestamps.first_playback_audio,
    )


def with_first_llm_token(timestamps: TurnTimestamps, now: float) -> TurnTimestamps:
    return TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=timestamps.llm_invocation_start,
        llm_request_start=timestamps.llm_request_start,
        first_llm_raw_delta=timestamps.first_llm_raw_delta,
        first_llm_token=timestamps.first_llm_token or now,
        tts_request_start=timestamps.tts_request_start,
        first_tts_audio=timestamps.first_tts_audio,
        first_playback_audio=timestamps.first_playback_audio,
    )


def with_first_llm_raw_delta(timestamps: TurnTimestamps, now: float) -> TurnTimestamps:
    return TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=timestamps.llm_invocation_start,
        llm_request_start=timestamps.llm_request_start,
        first_llm_raw_delta=timestamps.first_llm_raw_delta or now,
        first_llm_token=timestamps.first_llm_token,
        tts_request_start=timestamps.tts_request_start,
        first_tts_audio=timestamps.first_tts_audio,
        first_playback_audio=timestamps.first_playback_audio,
    )


def with_first_tts_audio(timestamps: TurnTimestamps, now: float) -> TurnTimestamps:
    return TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=timestamps.llm_invocation_start,
        llm_request_start=timestamps.llm_request_start,
        first_llm_raw_delta=timestamps.first_llm_raw_delta,
        first_llm_token=timestamps.first_llm_token,
        tts_request_start=timestamps.tts_request_start,
        first_tts_audio=timestamps.first_tts_audio or now,
        first_playback_audio=timestamps.first_playback_audio,
    )


def with_first_playback(timestamps: TurnTimestamps, now: float) -> TurnTimestamps:
    return TurnTimestamps(
        speech_end_detected=timestamps.speech_end_detected,
        stt_start=timestamps.stt_start,
        stt_invocation_start=timestamps.stt_invocation_start,
        stt_complete=timestamps.stt_complete,
        llm_invocation_start=timestamps.llm_invocation_start,
        llm_request_start=timestamps.llm_request_start,
        first_llm_raw_delta=timestamps.first_llm_raw_delta,
        first_llm_token=timestamps.first_llm_token,
        tts_request_start=timestamps.tts_request_start,
        first_tts_audio=timestamps.first_tts_audio,
        first_playback_audio=timestamps.first_playback_audio or now,
    )
