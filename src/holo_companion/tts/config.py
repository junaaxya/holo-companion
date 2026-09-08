import os
import tomllib
from pathlib import Path
from typing import Final, Mapping

from holo_companion.tts.base import TtsError, TtsErrorKind
from holo_companion.tts.elevenlabs import ElevenLabsConfig

CONFIG_PATH: Final = Path("config/default.toml")
API_KEY_ENV: Final = "ELEVENLABS_API_KEY"
VOICE_ID_ENV: Final = "ELEVENLABS_VOICE_ID"
MODEL_ID_ENV: Final = "ELEVENLABS_MODEL_ID"


def load_elevenlabs_tts_config(config_path: Path = CONFIG_PATH, env: Mapping[str, str] = os.environ) -> ElevenLabsConfig:
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TtsError(TtsErrorKind.CONFIG, f"TTS config file not found: {config_path}") from error
    tts_data = data.get("tts")
    if not isinstance(tts_data, dict):
        raise TtsError(TtsErrorKind.CONFIG, "config/default.toml must contain [tts.elevenlabs]")
    elevenlabs_data = tts_data.get("elevenlabs")
    if not isinstance(elevenlabs_data, dict):
        raise TtsError(TtsErrorKind.CONFIG, "config/default.toml must contain [tts.elevenlabs]")
    return ElevenLabsConfig(
        api_key=env.get(API_KEY_ENV, ""),
        voice_id=env.get(VOICE_ID_ENV, ""),
        model_id=env.get(MODEL_ID_ENV, ""),
        base_url=str(elevenlabs_data.get("base_url", "https://api.elevenlabs.io")),
        timeout_seconds=float(elevenlabs_data.get("timeout_seconds", 30.0)),
        stability=float(elevenlabs_data.get("stability", 0.40)),
    )
