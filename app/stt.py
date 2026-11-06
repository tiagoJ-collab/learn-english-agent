from __future__ import annotations

import io
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

WHISPER_RATE = 16000


class EmptyAudio(Exception):
    """The recording had no speech samples."""


class InvalidAudio(Exception):
    """The upload was not a WAV file we can read."""


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            width = wav_file.getsampwidth()
            rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InvalidAudio("Arquivo de áudio inválido.") from exc
    if width != 2 or rate <= 0 or channels <= 0:
        raise InvalidAudio("Arquivo de áudio inválido.")
    if len(frames) < 2:
        raise EmptyAudio("Não ouvi nada. Tente falar de novo.")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    if channels > 1:
        usable = samples.size - (samples.size % channels)
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        raise EmptyAudio("Não ouvi nada. Tente falar de novo.")
    return samples / 32768.0, rate


def to_whisper_audio(samples: np.ndarray, rate: int) -> np.ndarray:
    if samples.size == 0:
        raise EmptyAudio("Não ouvi nada. Tente falar de novo.")
    if rate == WHISPER_RATE:
        return samples.astype(np.float32)
    target = int(samples.size * WHISPER_RATE / rate)
    if target <= 0:
        raise EmptyAudio("Não ouvi nada. Tente falar de novo.")
    source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


class WavTranscriber:
    def __init__(
        self,
        model_name: str,
        download_root: Path,
        transcribe_fn: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.download_root = Path(download_root)
        self._transcribe_fn = transcribe_fn
        self._model = None
        self._lock = threading.Lock()

    def transcribe(self, wav_bytes: bytes) -> str:
        samples, rate = decode_wav(wav_bytes)
        audio = to_whisper_audio(samples, rate)
        if self._transcribe_fn is not None:
            text = self._transcribe_fn(audio)
        else:
            text = self._run_model(audio)
        return (text or "").strip()

    def _run_model(self, audio: np.ndarray) -> str:
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self.download_root.mkdir(parents=True, exist_ok=True)
                self._model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.download_root),
                )
            segments, _info = self._model.transcribe(audio, language="en")
            return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
