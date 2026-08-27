from dataclasses import asdict
from typing import TypeAlias

from holo_companion.llm.base import ChatMessage, GenerationResult, TextChunk

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def message_payload(message: ChatMessage) -> JsonObject:
    return {"role": message.role, "content": message.content}


def chunk_payload(chunk: TextChunk) -> JsonObject:
    return {"type": "chunk", "index": chunk.index, "text": chunk.text, "elapsed_seconds": chunk.elapsed_seconds}


def result_payload(result: GenerationResult) -> JsonObject:
    return {"type": "result", "status": result.status.value, "text": result.text, "metrics": asdict(result.metrics)}
