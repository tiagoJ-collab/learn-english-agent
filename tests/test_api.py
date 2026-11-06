import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.ollama import OllamaUnavailable
from app.openrouter import Article, OpenRouterError
from app.store import Store
from app.stt import WavTranscriber
from app.tts import PiperSpeaker

ARTICLE_TEXT = (
    "Rain falls on the city every afternoon, and the streets shine after the clouds pass. "
    "People open umbrellas, buses slow down, and the river rises a little before night. "
    "Warm air lifts water from the sea, and that water returns as rain over the hills."
)


class FakeArticles:
    def __init__(self):
        self.lookups = []

    async def fetch(self, topic, level):
        return Article(
            title=f"About {topic}",
            url=f"https://example.com/{topic}",
            source="Example",
            text=ARTICLE_TEXT,
        )

    async def lookup(self, question):
        self.lookups.append(question)
        return "Joinville is rainy today, about 22 C, according to a weather report."


class FakeChat:
    def __init__(self):
        self.calls = []
        self.live = False

    async def status(self):
        return {"up": True, "model_ready": True}

    async def needs_current_facts(self, question, reading):
        return self.live

    async def stream(self, reading, history, user_text, live_notes=""):
        self.calls.append(
            {
                "reading": reading,
                "history": list(history),
                "user_text": user_text,
                "live_notes": live_notes,
            }
        )
        for piece in ("The article ", "is about rain. ", "It is short."):
            yield piece


class DownChat:
    async def status(self):
        return {"up": False, "model_ready": False}

    async def needs_current_facts(self, question, reading):
        return False

    async def stream(self, reading, history, user_text, live_notes=""):
        raise OllamaUnavailable("Ollama down")
        yield ""


class SilentChat:
    async def status(self):
        return {"up": True, "model_ready": True}

    async def needs_current_facts(self, question, reading):
        return False

    async def stream(self, reading, history, user_text, live_notes=""):
        if False:
            yield ""


def _events(body):
    found = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                found.append(json.loads(line[5:]))
    return found


@pytest.fixture
def world(tmp_path):
    settings = Settings(
        openrouter_api_key="test-key",
        openrouter_model="openai/gpt-4o-mini:online",
        ollama_model="llama3.2",
        ollama_base_url="http://127.0.0.1:11434",
        host="127.0.0.1",
        port=8765,
        whisper_model="small.en",
        piper_voice="en_US-lessac-medium",
        data_dir=tmp_path,
    )
    store = Store(tmp_path / "readings.db")
    articles = FakeArticles()
    chat = FakeChat()
    transcriber = WavTranscriber("small.en", tmp_path / "whisper", transcribe_fn=lambda audio: "hello there")
    speaker = PiperSpeaker(tmp_path, "en_US-lessac-medium", synth_fn=lambda text: b"RIFF-fake")
    client = TestClient(create_app(settings, store, articles, chat, transcriber, speaker))
    return SimpleNamespace(client=client, store=store, chat=chat, settings=settings, tmp_path=tmp_path)


def test_home_page(world):
    response = world.client.get("/")

    assert response.status_code == 200
    assert "Sala de leitura" in response.text


def test_health(world):
    response = world.client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ollama_up"] is True
    assert body["model"] == "llama3.2"
    assert body["piper_ready"] is True


def test_search_saves_a_reading_and_rejects_a_bad_request(world):
    response = world.client.post("/api/readings", json={"topic": "rain", "level": "b1"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "About rain"
    assert body["level"] == "B1"
    assert body["text"] == ARTICLE_TEXT
    assert world.client.get("/api/readings").json()[0]["id"] == body["id"]
    assert world.client.post("/api/readings", json={"topic": "   ", "level": "B1"}).status_code == 400
    assert world.client.post("/api/readings", json={"topic": "rain", "level": "Z9"}).status_code == 400
    assert world.client.get("/api/readings/999").status_code == 404


def test_chat_streams_sentences_and_keeps_the_transcript(world):
    reading = world.client.post("/api/readings", json={"topic": "rain", "level": "B1"}).json()

    response = world.client.post(
        f"/api/readings/{reading['id']}/messages",
        json={"content": "What is it about?"},
    )

    assert response.status_code == 200
    events = _events(response.text)
    sentences = [event["sentence"] for event in events if "sentence" in event]
    assert sentences == ["The article is about rain.", "It is short."]
    assert events[-1] == {"done": True}
    assert world.chat.calls[0]["user_text"] == "What is it about?"
    assert world.chat.calls[0]["history"] == []
    assert world.chat.calls[0]["live_notes"] == ""
    assert {"status": "Buscando na web…"} not in events
    assert "Rain falls" in world.chat.calls[0]["reading"]["text"]
    messages = world.store.list_messages(reading["id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "The article is about rain. It is short."


def test_chat_reports_when_ollama_is_down(world):
    reading = world.store.create_reading(
        "rain",
        "B1",
        Article(title="Rain", url="https://example.com/rain", source="Example", text=ARTICLE_TEXT),
    )
    client = TestClient(
        create_app(
            world.settings,
            world.store,
            FakeArticles(),
            DownChat(),
            WavTranscriber("small.en", world.tmp_path / "whisper", transcribe_fn=lambda audio: "hi"),
            PiperSpeaker(world.tmp_path, "en_US-lessac-medium", synth_fn=lambda text: b"RIFF"),
        )
    )

    response = client.post(f"/api/readings/{reading['id']}/messages", json={"content": "Hello"})

    assert response.status_code == 200
    assert {"error": "Ollama down"} in _events(response.text)
    assert [item["role"] for item in world.store.list_messages(reading["id"])] == ["user"]


def test_transcribe_and_speak(world):
    import io
    import wave
    from array import array

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(array("h", [0, 1000, -1000]).tobytes())

    heard = world.client.post(
        "/api/transcribe",
        files={"file": ("speech.wav", buffer.getvalue(), "audio/wav")},
    )
    spoken = world.client.post("/api/speak", json={"text": "Hello."})
    empty = world.client.post("/api/speak", json={"text": "   "})
    garbage = world.client.post(
        "/api/transcribe",
        files={"file": ("speech.wav", b"nope", "audio/wav")},
    )

    assert heard.status_code == 200
    assert heard.json() == {"text": "hello there"}
    assert spoken.status_code == 200
    assert spoken.headers["content-type"].startswith("audio/wav")
    assert spoken.content == b"RIFF-fake"
    assert empty.status_code == 400
    assert garbage.status_code == 400


def test_silent_model_does_not_save_an_assistant_message(world):
    reading = world.store.create_reading(
        "rain",
        "B1",
        Article(title="Rain", url="https://example.com/rain", source="Example", text=ARTICLE_TEXT),
    )
    client = TestClient(
        create_app(
            world.settings,
            world.store,
            FakeArticles(),
            SilentChat(),
            WavTranscriber("small.en", world.tmp_path / "whisper", transcribe_fn=lambda audio: "hi"),
            PiperSpeaker(world.tmp_path, "en_US-lessac-medium", synth_fn=lambda text: b"RIFF"),
        )
    )

    response = client.post(f"/api/readings/{reading['id']}/messages", json={"content": "Hello"})

    assert any(event.get("error") for event in _events(response.text))
    assert [item["role"] for item in world.store.list_messages(reading["id"])] == ["user"]


def _reading(world):
    return world.store.create_reading(
        "rain",
        "B1",
        Article(title="Rain", url="https://example.com/rain", source="Example", text=ARTICLE_TEXT),
    )


def _client(world, articles, chat):
    return TestClient(
        create_app(
            world.settings,
            world.store,
            articles,
            chat,
            WavTranscriber("small.en", world.tmp_path / "whisper", transcribe_fn=lambda audio: "hi"),
            PiperSpeaker(world.tmp_path, "en_US-lessac-medium", synth_fn=lambda text: b"RIFF"),
        )
    )


def test_a_current_question_searches_the_web_before_the_local_reply(world):
    reading = _reading(world)
    articles = FakeArticles()
    chat = FakeChat()
    chat.live = True
    client = _client(world, articles, chat)

    response = client.post(
        f"/api/readings/{reading['id']}/messages",
        json={"content": "What's the weather in Joinville?"},
    )

    events = _events(response.text)
    assert {"status": "Buscando na web…"} in events
    assert articles.lookups == ["What's the weather in Joinville?"]
    assert "22 C" in chat.calls[0]["live_notes"]
    assert events[-1] == {"done": True}


def test_a_failed_web_search_still_lets_the_local_model_answer(world):
    reading = _reading(world)
    articles = FakeArticles()

    async def broken(question):
        raise OpenRouterError("Não consegui buscar o texto no OpenRouter.")

    articles.lookup = broken
    chat = FakeChat()
    chat.live = True
    client = _client(world, articles, chat)

    response = client.post(
        f"/api/readings/{reading['id']}/messages",
        json={"content": "Como está o tempo no Brasil hoje?"},
    )

    assert response.status_code == 200
    assert "search failed" in chat.calls[0]["live_notes"]
    assert _events(response.text)[-1] == {"done": True}
