import io
import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from app.stt import EmptyAudio, InvalidAudio, WavTranscriber, decode_wav, to_whisper_audio
from app.tts import EmptySpeech, PiperSpeaker


def make_wav(samples, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        if samples:
            wav_file.writeframes(array("h", samples).tobytes())
    return buffer.getvalue()


def test_empty_wav_is_rejected():
    with pytest.raises(EmptyAudio):
        decode_wav(make_wav([]))


def test_invalid_audio_is_rejected():
    with pytest.raises(InvalidAudio):
        decode_wav(b"this is not a wav")


def test_decode_roundtrip_and_resample():
    audio, rate = decode_wav(make_wav([0, 16000, -16000], rate=8000))

    assert rate == 8000
    assert audio.shape == (3,)
    assert to_whisper_audio(audio, rate).shape == (6,)


def test_transcribe_returns_what_the_double_produces(tmp_path):
    transcriber = WavTranscriber(
        "small.en",
        tmp_path / "whisper",
        transcribe_fn=lambda samples: "hello there",
    )

    assert transcriber.transcribe(make_wav([0, 1000, -1000])) == "hello there"


def test_empty_speech_is_rejected(tmp_path):
    speaker = PiperSpeaker(tmp_path, "en_US-lessac-medium", synth_fn=lambda text: b"RIFF")

    with pytest.raises(EmptySpeech):
        speaker.speak("   ")


def test_speak_returns_the_double_audio(tmp_path):
    seen = []
    speaker = PiperSpeaker(
        tmp_path,
        "en_US-lessac-medium",
        synth_fn=lambda text: seen.append(text) or b"RIFF-fake",
    )

    assert speaker.speak("  Hello there. ") == b"RIFF-fake"
    assert seen == ["Hello there."]
    assert speaker.ready() is True
    assert not (Path(tmp_path) / "voices" / "en_US-lessac-medium.onnx").exists()
