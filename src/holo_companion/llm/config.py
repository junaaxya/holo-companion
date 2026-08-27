import os
import tomllib
from pathlib import Path
from typing import Final, Mapping

from holo_companion.llm.base import LlmConfig, LlmError, LlmErrorKind

CONFIG_PATH: Final = Path("config/default.toml")
API_KEY_ENV: Final = "LLM_API_KEY"


def load_llm_config(config_path: Path = CONFIG_PATH, env: Mapping[str, str] = os.environ) -> LlmConfig:
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LlmError(LlmErrorKind.CONFIG, f"LLM config file not found: {config_path}") from error
    llm_data = data.get("llm")
    if not isinstance(llm_data, dict):
        raise LlmError(LlmErrorKind.CONFIG, "config/default.toml must contain [llm]")
    return LlmConfig(
        provider=str(llm_data.get("provider", "")),
        base_url=str(llm_data.get("base_url", "")),
        model=str(llm_data.get("model", "")),
        timeout_seconds=float(llm_data.get("timeout_seconds", 30.0)),
        api_key=env.get(API_KEY_ENV, ""),
    )
