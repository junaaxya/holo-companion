from pathlib import Path

import pytest

from holo_companion.llm.base import CancellationHandle, ChatMessage, LlmConfig, LlmError, LlmErrorKind
from holo_companion.llm.config import load_llm_config
from holo_companion.llm.context import ConversationContext


def test_llm_config_normalizes_absolute_base_url() -> None:
    # Given / When
    config = LlmConfig(provider="openai_compatible", base_url=" https://api.example.test/v1/ ", model=" model-a ", api_key=" secret ")

    # Then
    assert config.base_url == "https://api.example.test/v1"
    assert config.model == "model-a"
    assert config.api_key == "secret"


@pytest.mark.parametrize("base_url", ["", "api.example.test/v1", "ftp://example.test", "https://u:p@example.test/v1", "https://example.test/v1?q=1", "https://example.test/v1#frag"])
def test_llm_config_rejects_unsafe_base_url(base_url: str) -> None:
    # Given / When / Then
    with pytest.raises(LlmError, match="base_url"):
        LlmConfig(provider="openai_compatible", base_url=base_url, model="model", api_key="secret")


def test_llm_config_loads_key_only_from_environment(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = "https://api.example.test/v1"\nmodel = "model-a"\ntimeout_seconds = 12.5\n', encoding="utf-8")

    # When
    config = load_llm_config(config_path, {"LLM_API_KEY": "env-secret"})

    # Then
    assert config.api_key == "env-secret"
    assert config.timeout_seconds == 12.5


def test_llm_config_allows_environment_model_override(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = "https://api.example.test/v1"\nmodel = "config-model"\n', encoding="utf-8")

    # When
    config = load_llm_config(config_path, {"LLM_API_KEY": "env-secret", "LLM_MODEL": "env-model"})

    # Then
    assert config.model == "env-model"


def test_llm_config_allows_environment_endpoint_and_model_overrides(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = "https://config.example.test/v1"\nmodel = "config-model"\n', encoding="utf-8")

    # When
    config = load_llm_config(
        config_path,
        {
            "LLM_API_KEY": "env-secret",
            "LLM_BASE_URL": "http://localhost:20128/v1",
            "LLM_MODEL": "ag/gemini-3.6-flash-low",
        },
    )

    # Then
    assert config.base_url == "http://localhost:20128/v1"
    assert config.model == "ag/gemini-3.6-flash-low"
    assert config.api_key == "env-secret"


def test_llm_config_rejects_missing_environment_key(tmp_path: Path) -> None:
    # Given
    config_path = tmp_path / "default.toml"
    config_path.write_text('[llm]\nprovider = "openai_compatible"\nbase_url = "https://api.example.test/v1"\nmodel = "model-a"\n', encoding="utf-8")

    # When / Then
    with pytest.raises(LlmError, match="LLM_API_KEY"):
        load_llm_config(config_path, {})


def test_conversation_context_keeps_system_plus_newest_six_messages() -> None:
    # Given
    messages = tuple(ChatMessage(role="user", content=f"message-{index}") for index in range(8))
    context = ConversationContext(messages=messages)

    # When
    provider_messages = context.provider_messages()

    # Then
    assert provider_messages[0].role == "system"
    assert [message.content for message in provider_messages[1:]] == [f"message-{index}" for index in range(2, 8)]


def test_conversation_context_truncates_prompt_from_left() -> None:
    # Given
    prompt = "a" * 2_050

    # When
    context = ConversationContext().with_user_prompt(prompt)

    # Then
    assert len(context.messages[-1].content) == 2_000


def test_conversation_context_preserves_custom_system_prompt() -> None:
    # Given
    custom_prompt = "Custom companion persona."
    context = ConversationContext(system_prompt=custom_prompt)

    # When
    provider_messages = context.provider_messages()

    # Then
    assert provider_messages[0].role == "system"
    assert provider_messages[0].content == custom_prompt


def test_default_conversation_context_has_nonempty_system_prompt() -> None:
    # Given
    context = ConversationContext()

    # When
    provider_messages = context.provider_messages()

    # Then
    assert provider_messages[0].role == "system"
    assert len(provider_messages[0].content) > 50
    assert "Holo" in provider_messages[0].content


def test_cancellation_handle_raises_typed_error_after_cancel() -> None:
    # Given
    cancellation = CancellationHandle()
    cancellation.cancel("barge-in")

    # When / Then
    with pytest.raises(LlmError) as raised:
        cancellation.raise_if_cancelled()
    assert raised.value.kind is LlmErrorKind.CANCELLED
    assert "barge-in" in str(raised.value)
