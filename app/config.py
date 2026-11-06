from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    ollama_model: str
    ollama_base_url: str
    host: str
    port: int
    whisper_model: str
    piper_voice: str
    data_dir: Path


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.environ.get("DATA_DIR", str(ROOT / "data")))
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini:online"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        host=os.environ.get("APP_HOST", "127.0.0.1"),
        port=int(os.environ.get("APP_PORT", "8765")),
        whisper_model=os.environ.get("WHISPER_MODEL", "small.en"),
        piper_voice=os.environ.get("PIPER_VOICE", "en_US-lessac-medium"),
        data_dir=data_dir,
    )
