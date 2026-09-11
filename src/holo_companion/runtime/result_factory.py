import numpy as np

from holo_companion.runtime.metrics import TurnTimestamps
from holo_companion.runtime.ownership import OwnedTurn
from holo_companion.runtime.turn import TextTurnDiagnostic, TurnError, TurnResult
from holo_companion.stt.base import Transcript, Utterance


def turn_result(
    owned: OwnedTurn,
    transcript: Transcript | None,
    utterance: Utterance,
    assistant_text: str,
    timestamps: TurnTimestamps,
    error: TurnError | None,
    text_diagnostic: TextTurnDiagnostic | None = None,
) -> TurnResult:
    return TurnResult(
        owned.turn_id,
        owned.generation_id,
        None if transcript is None else transcript.text,
        assistant_text,
        timestamps,
        error,
        utterance.duration_seconds,
        None if transcript is None else transcript.processing_seconds,
        utterance.samples.size,
        float(np.abs(utterance.samples).max()),
        float(np.sqrt(np.mean(np.square(utterance.samples)))),
        utterance.start_frame_index,
        utterance.end_frame_index,
        text_diagnostic,
    )
