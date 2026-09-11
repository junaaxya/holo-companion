from dataclasses import dataclass

import anyio

from holo_companion.llm.base import CancellationHandle


@dataclass(slots=True)
class OwnedTurn:
    turn_id: int
    generation_id: int
    cancellation: CancellationHandle
    scope: anyio.CancelScope | None = None
    interruption: object | None = None
    stt_started: bool = False


class StaleOutputError(Exception):
    pass


class TurnCancelledError(Exception):
    pass


def ensure_current(active: OwnedTurn | None, owned: OwnedTurn) -> None:
    if owned.cancellation.is_cancelled():
        raise TurnCancelledError
    if not is_current(active, owned.turn_id, owned.generation_id):
        raise StaleOutputError


def is_current(active: OwnedTurn | None, turn_id: int, generation_id: int) -> bool:
    return active is not None and active.turn_id == turn_id and active.generation_id == generation_id
