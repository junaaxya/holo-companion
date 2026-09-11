from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final


@unique
class RuntimeState(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


AllowedTransitions = dict[RuntimeState, frozenset[RuntimeState]]

ALLOWED_TRANSITIONS: Final[AllowedTransitions] = {
    RuntimeState.IDLE: frozenset((RuntimeState.LISTENING, RuntimeState.STOPPED)),
    RuntimeState.LISTENING: frozenset((RuntimeState.TRANSCRIBING, RuntimeState.INTERRUPTING, RuntimeState.ERROR, RuntimeState.STOPPED)),
    RuntimeState.TRANSCRIBING: frozenset((RuntimeState.THINKING, RuntimeState.INTERRUPTING, RuntimeState.ERROR, RuntimeState.STOPPED)),
    RuntimeState.THINKING: frozenset((RuntimeState.SPEAKING, RuntimeState.INTERRUPTING, RuntimeState.ERROR, RuntimeState.STOPPED)),
    RuntimeState.SPEAKING: frozenset((RuntimeState.IDLE, RuntimeState.INTERRUPTING, RuntimeState.ERROR, RuntimeState.STOPPED)),
    RuntimeState.INTERRUPTING: frozenset((RuntimeState.LISTENING, RuntimeState.IDLE, RuntimeState.ERROR, RuntimeState.STOPPED)),
    RuntimeState.ERROR: frozenset((RuntimeState.IDLE, RuntimeState.STOPPED)),
    RuntimeState.STOPPED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class InvalidStateTransition(Exception):
    from_state: RuntimeState
    to_state: RuntimeState

    def __str__(self) -> str:
        return f"invalid runtime transition: {self.from_state.value} to {self.to_state.value}"


def transition(from_state: RuntimeState, to_state: RuntimeState) -> RuntimeState:
    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        raise InvalidStateTransition(from_state, to_state)
    return to_state
