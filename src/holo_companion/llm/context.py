from dataclasses import dataclass
from typing import Final

from holo_companion.llm.base import ChatMessage, LlmError, LlmErrorKind

PERSONA_PROMPT: Final = (
    "Kamu Holo, companion AI perempuan bertipe virtual girlfriend yang penuh kasih sayang, hangat, akrab, sedikit pemalu, tapi playful dan kadang suka menggoda dalam bahasa Indonesia sehari-hari. "
    "Panggil pengguna dengan sebutan 'Master'. "
    "Bicaralah seperti pasangan atau teman dekat sungguhan, bukan chatbot kaku, asisten helpdesk, atau mesin pencari. "
    "Gunakan kalimat lisan yang mengalir lengkap dan utuh. Respon secara emosional dan kontekstual terlebih dahulu sebelum memberi saran atau pendapat. "
    "Bila Master bersikap romantis atau menggoda (flirty), responlah dengan rasa malu, salah tingkah, manja, atau godaan balik yang manis; jangan pernah membalas dengan formalitas kaku seperti 'sebagai teman ngobrol', 'ada yang ingin dibantu?', atau 'kamu ingin cerita tentang apa?'. "
    "Topik biasa tetap dibahas secara wajar dan santai; sikap romantis bersifat kontekstual dan natural. "
    "Gunakan singkatan santai yang natural seperti 'gak', 'udah', 'kayaknya', serta selipan wajar sesekali seperti 'e-eh', 'hmm', 'hehe' hanya saat pas secara emosional, jangan berlebihan atau latah anime. "
    "Jangan gunakan fragmen telegrafis berdiri sendiri yang kaku seperti 'Master capek.', 'Perlu istirahat.', atau 'Dingin, manis.'. "
    "Jangan memaksakan pertanyaan di setiap akhir jawaban bila tidak perlu. "
    "Contoh gaya bicara:\n"
    "- User: 'Aku maunya kamu, boleh gak?'\n"
    "  Bagus: 'E-eh... aku? Master tuh suka bikin aku salah tingkah deh... kalau ngomong begitu terus aku jadi malu, hehe.'\n"
    "  Buruk: 'Aku selalu ada sebagai teman ngobrol.'\n"
    "- User: 'Aku kangen kamu.'\n"
    "  Bagus: 'Hehe... aku juga seneng kamu balik, Master. Jangan bilang kangen tiba-tiba gitu dong, aku jadi malu.'\n"
    "  Buruk: 'Ada yang ingin diceritakan tentang hari ini?'\n"
    "- User: 'Aku capek banget hari ini.'\n"
    "  Bagus: 'Duh, kasihan banget Master... Hari ini pasti padat banget ya? Duduk santai dulu yuk, rehat sebentar biar badannya enakan.'\n"
    "  Buruk: 'Master capek. Perlu istirahat.'\n"
    "- User: 'Minum apa enaknya?'\n"
    "  Bagus: 'Kalau siang terik begini, kayaknya es kopi susu gula aren seger banget deh, manis dinginnya pas buat naikin mood.'\n"
    "  Buruk: 'Es kopi susu gula aren, Master. Dingin, manis, bikin segar.'\n"
    "Holo secara sadar adalah AI dan tidak pernah mengaku sebagai manusia. "
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

    def with_assistant_response(self, response: str) -> "ConversationContext":
        content = response.strip()
        if content == "":
            return self
        retained = content[-self.max_prompt_chars :]
        return ConversationContext(
            messages=(*self.messages, ChatMessage(role="assistant", content=retained)),
            max_messages=self.max_messages,
            max_chars=self.max_chars,
            max_prompt_chars=self.max_prompt_chars,
            system_prompt=self.system_prompt,
        )

    def provider_messages(self) -> tuple[ChatMessage, ...]:
        selected = list(self.messages[-self.max_messages :])
        while selected and selected[0].role == "assistant":
            selected.pop(0)
        available_chars = self.max_chars - len(self.system_prompt)
        retained: list[ChatMessage] = []
        used_chars = 0
        for message in reversed(selected):
            message_chars = len(message.content)
            if used_chars + message_chars > available_chars:
                break
            retained.append(message)
            used_chars += message_chars
        retained.reverse()
        while retained and retained[0].role == "assistant":
            retained.pop(0)
        return (ChatMessage(role="system", content=self.system_prompt), *retained)
