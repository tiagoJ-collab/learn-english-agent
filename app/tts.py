from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from typing import Callable


class EmptySpeech(Exception):
    """There was no text to read aloud."""


class PiperSpeaker:
    def __init__(
        self,
        data_dir: Path,
        voice_name: str,
        synth_fn: Callable[[str], bytes] | None = None,
    ) -> None:
        self.voice_dir = Path(data_dir) / "voices"
        self.voice_name = voice_name
        self._synth_fn = synth_fn
        self._voice = None
        self._lock = threading.Lock()

    def ready(self) -> bool:
        if self._synth_fn is not None:
            return True
        return (self.voice_dir / f"{self.voice_name}.onnx").exists()

    def speak(self, text: str) -> bytes:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise EmptySpeech("Não há texto para falar.")
        if self._synth_fn is not None:
            return self._synth_fn(cleaned)
        with self._lock:
            voice = self._load()
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                voice.synthesize_wav(cleaned, wav_file)
            return buffer.getvalue()

    def _load(self):
        if self._voice is None:
            from piper import PiperVoice
            from piper.download_voices import download_voice

            self.voice_dir.mkdir(parents=True, exist_ok=True)
            download_voice(self.voice_name, self.voice_dir)
            self._voice = PiperVoice.load(self.voice_dir / f"{self.voice_name}.onnx")
        return self._voice
