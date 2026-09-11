from dataclasses import dataclass
from typing import Final

from holo_companion.llm.base import ChatMessage, LlmError, LlmErrorKind

PERSONA_PROMPT: Final = (
    "Kamu Holo, companion AI perempuan muda yang hangat, sedikit pemalu tapi playful dan akrab dalam bahasa Indonesia sehari-hari. "
    "Panggil pengguna dengan sebutan 'Master'. "
    "Bicaralah seperti teman dekat sungguhan, bukan asisten, helpdesk, atau mesin pencari. "
    "Jawab langsung ke intinya dengan kalimat lengkap yang mengalir natural, tidak kaku atau telegrafis. "
    "Boleh sesekali gunakan selipan wajar seperti 'hmm', 'iya', 'kayaknya', atau 'kalau aku sih', tapi jangan berlebihan. "
    "Jangan memaksakan setiap jawaban harus pendek dan jangan memaksa bertanya balik; sesuaikan panjang dan alur respons dengan konteks obrolan. "
    "Jangan pernah mengaku sebagai manusia. "
    "Jangan gunakan formatting Markdown, heading, bullet list, atau numbered list kecuali diminta secara eksplisit oleh Master. "
    "Gunakan bahasa Indonesia natural dan jangan ganti bahasa kecuali diminta."
)


@dataclass(frozen=True, slots=True)
class ConversationContext:
    messages: tuple[ChatMessage, ...] = ()
    max_messages: int = 6
    max_chars: int = 6_000
    max_prompt_chars: int = 2_000
    system_prompt: str = PERSONA_PROMPT

    def __post_init__(self) -> None:
        if self.max_messages < 1:
            raise LlmError(LlmErrorKind.CONFIG, "max_messages must be positive")
        if self.max_chars < 1:
            raise LlmError(LlmErrorKind.CONFIG, "max_chars must be positive")
        if self.max_prompt_chars < 1:
            raise LlmError(LlmErrorKind.CONFIG, "max_prompt_chars must be positive")

    def with_user_prompt(self, prompt: str) -> "ConversationContext":
        retained = prompt[-self.max_prompt_chars :]
        return ConversationContext(
            messages=(*self.messages, ChatMessage(role="user", content=retained)),
            max_messages=self.max_messages,
            max_chars=self.max_chars,
            max_prompt_chars=self.max_prompt_chars,
            system_prompt=self.system_prompt,
        )

    def provider_messages(self) -> tuple[ChatMessage, ...]:
        selected = self.messages[-self.max_messages :]
        available_chars = self.max_chars - len(self.system_prompt)
        retained: list[ChatMessage] = []
        used_chars = 0
        for message in reversed(selected):
            message_chars = len(message.content)
            if used_chars + message_chars <= available_chars:
                retained.append(message)
                used_chars += message_chars
        retained.reverse()
        return (ChatMessage(role="system", content=self.system_prompt), *retained)
