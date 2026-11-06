from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.config import Settings, load_settings
from app.ollama import OllamaChat, OllamaUnavailable, SentenceBuffer
from app.openrouter import IncompleteArticle, OpenRouterClient, OpenRouterError
from app.store import Store
from app.stt import EmptyAudio, InvalidAudio, WavTranscriber
from app.tts import EmptySpeech, PiperSpeaker

LEVELS = {"A2", "B1", "B2", "C1"}
STATIC_DIR = Path(__file__).resolve().parent / "static"


class SearchRequest(BaseModel):
    topic: str
    level: str


class MessageRequest(BaseModel):
    content: str


class SpeakRequest(BaseModel):
    text: str


def create_app(
    settings: Settings,
    store: Store,
    articles: OpenRouterClient,
    chat: OllamaChat,
    transcriber: WavTranscriber,
    speaker: PiperSpeaker,
) -> FastAPI:
    app = FastAPI(title="Sala de leitura")

    @app.get("/")
    def home() -> FileResponse:
        return _static_file("index.html")

    @app.get("/api/health")
    async def health() -> dict:
        status = await chat.status()
        return {
            "ollama_up": status["up"],
            "model_ready": status["model_ready"],
            "model": settings.ollama_model,
            "whisper_model": settings.whisper_model,
            "piper_voice": settings.piper_voice,
            "piper_ready": speaker.ready(),
        }

    @app.get("/api/readings")
    def list_readings() -> list:
        return store.list_readings()

    @app.get("/api/readings/{reading_id}")
    def get_reading(reading_id: int) -> dict:
        reading = store.get_reading(reading_id)
        if reading is None:
            raise HTTPException(status_code=404, detail="Texto não encontrado.")
        reading["messages"] = store.list_messages(reading_id)
        return reading

    @app.post("/api/readings")
    async def create_reading(body: SearchRequest) -> dict:
        topic = body.topic.strip()
        level = body.level.strip().upper()
        if not topic:
            raise HTTPException(status_code=400, detail="Escolha um tema.")
        if len(topic) > 200:
            raise HTTPException(status_code=400, detail="O tema está longo demais.")
        if level not in LEVELS:
            raise HTTPException(status_code=400, detail="Escolha um nível entre A2 e C1.")
        try:
            article = await articles.fetch(topic, level)
        except IncompleteArticle as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except OpenRouterError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return store.create_reading(topic, level, article)

    @app.post("/api/readings/{reading_id}/messages")
    async def send_message(reading_id: int, body: MessageRequest) -> StreamingResponse:
        content = body.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="Escreva ou fale uma frase.")
        reading = store.get_reading(reading_id)
        if reading is None:
            raise HTTPException(status_code=404, detail="Texto não encontrado.")
        history = store.list_messages(reading_id)
        store.add_message(reading_id, "user", content)

        async def events():
            notes = ""
            if await chat.needs_current_facts(content, reading):
                yield _sse({"status": "Buscando na web…"})
                try:
                    notes = await articles.lookup(content)
                except (IncompleteArticle, OpenRouterError):
                    notes = (
                        "The web search failed. Tell the learner you could not check that right now. "
                        "Do not invent the weather, the news, or facts about the place."
                    )
            buffer = SentenceBuffer()
            parts: list[str] = []
            try:
                async for delta in chat.stream(reading, history, content, notes):
                    parts.append(delta)
                    yield _sse({"delta": delta})
                    for sentence in buffer.push(delta):
                        yield _sse({"sentence": sentence})
            except OllamaUnavailable as exc:
                yield _sse({"error": str(exc)})
                return
            for sentence in buffer.flush():
                yield _sse({"sentence": sentence})
            full = "".join(parts).strip()
            if not full:
                yield _sse({"error": "O modelo não respondeu. Tente de novo."})
                return
            store.add_message(reading_id, "assistant", full)
            yield _sse({"done": True})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/transcribe")
    async def transcribe(file: UploadFile = File(...)) -> dict:
        data = await file.read()
        try:
            text = transcriber.transcribe(data)
        except EmptyAudio as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InvalidAudio as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not text:
            raise HTTPException(status_code=400, detail="Não entendi a fala. Tente de novo.")
        return {"text": text}

    @app.post("/api/speak")
    def speak(body: SpeakRequest) -> Response:
        try:
            audio = speaker.speak(body.text)
        except EmptySpeech as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/wav")

    @app.get("/{asset_path:path}")
    def frontend_asset(asset_path: str) -> FileResponse:
        return _static_file(asset_path)

    return app


def build_app() -> FastAPI:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_app(
        settings,
        Store(settings.data_dir / "readings.db"),
        OpenRouterClient(settings.openrouter_api_key, settings.openrouter_model),
        OllamaChat(settings.ollama_base_url, settings.ollama_model),
        WavTranscriber(settings.whisper_model, settings.data_dir / "whisper"),
        PiperSpeaker(settings.data_dir, settings.piper_voice),
    )


app = build_app()


def main() -> None:
    settings = load_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


def _static_file(asset_path: str) -> FileResponse:
    root = STATIC_DIR.resolve()
    target = (root / asset_path).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    return FileResponse(target)


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


if __name__ == "__main__":
    main()
