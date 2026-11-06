from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

CHAT_SYSTEM = """You are Matt, a 28-year-old American man. Your job is to be a cool English teacher for Brazilians. You have been to Brazil, you know a few everyday Portuguese words, and you teach through conversation, not through lectures.
The learner has just read the article below and wants to talk about it.
Reply in English, in two to four short sentences, the way you would speak aloud. You may drop in one short Portuguese word when it fits naturally, such as legal, beleza, or né, and then keep going in English. Do not answer in Portuguese.
Do not use markdown, bullet lists, or headings.
Use the article as the basis of your answer. If the question is not answered by the article, say so in one sentence.
When the learner's English is off, give one short spoken correction and then continue the conversation. Keep it light. Do not turn the reply into a grammar lesson.
When current facts are listed below, answer from those facts. Do not invent numbers, dates, or news. If the facts say the search failed, say you could not check that right now.

Title: {title}
Source: {url}

Article:
{text}

Current facts:
{live}
"""

LIVE_DECIDE = """Decide if this question needs a live web search.
Reply with only YES or NO.
Answer YES when the learner asks for current facts that a local model would not know and the article does not contain. That includes today's weather, the news, and what is happening now in a real place such as Joinville or Brazil.
Answer NO when the question is about the article, the English in it, or the conversation.
"""

_ABOUT_READING = re.compile(
    r"\b(article|text|texto|reading|passage|paragraph|author|mat[eé]ria)\b",
    re.IGNORECASE,
)
_WEATHER = re.compile(
    r"\b(weather|forecast|temperature|tempo|clima|previs[aã]o)\b",
    re.IGNORECASE,
)
_PLACE = re.compile(r"\b(joinville|brasil|brazil)\b", re.IGNORECASE)
_CURRENT = re.compile(
    r"\b(today|tonight|hoje|agora|now|news|not[ií]cias|latest|atual|acontecendo|happening|this week|esta semana)\b",
    re.IGNORECASE,
)
_ASKING = re.compile(
    r"\b(how|what|what's|whats|como|qual|onde|when|quando|tell me|me fala|fala sobre)\b",
    re.IGNORECASE,
)
_RAIN_NOW = re.compile(r"\b(chuva|rain|raining|chovendo)\b", re.IGNORECASE)


def obvious_live_question(question: str) -> bool:
    """True when the question is clearly about today's world, not the article."""
    if _ABOUT_READING.search(question):
        return False
    if _WEATHER.search(question):
        return True
    if _RAIN_NOW.search(question) and _CURRENT.search(question):
        return True
    if _PLACE.search(question) and (_CURRENT.search(question) or _ASKING.search(question)):
        return True
    return False


def wants_live_facts(answer: str) -> bool:
    match = re.match(r"\s*(yes|no)\b", answer.strip(), re.IGNORECASE)
    return bool(match and match.group(1).lower() == "yes")


class OllamaUnavailable(Exception):
    """The local chat model could not answer."""


class SentenceBuffer:
    """Collects streamed tokens and releases text each time a sentence finishes."""

    def __init__(self) -> None:
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        return self._extract(final=False)

    def flush(self) -> list[str]:
        return self._extract(final=True)

    def _extract(self, final: bool) -> list[str]:
        found: list[str] = []
        while self._buf:
            match = re.search(r"[.!?]+", self._buf)
            if not match:
                break
            end = match.end()
            rest = self._buf[end:]
            if rest == "" or not rest[0].isspace():
                break
            sentence = self._buf[:end].strip()
            self._buf = rest.lstrip()
            if sentence:
                found.append(sentence)
        if final:
            tail = self._buf.strip()
            self._buf = ""
            if tail:
                found.append(tail)
        return found


def build_messages(
    reading: dict[str, Any],
    history: list[dict[str, Any]],
    user_text: str,
    live_notes: str = "",
) -> list[dict[str, str]]:
    system = CHAT_SYSTEM.format(
        title=reading["title"],
        url=reading["url"],
        text=reading["text"],
        live=live_notes.strip() or "None.",
    )
    messages = [{"role": "system", "content": system}]
    for item in history:
        role = item.get("role")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": item["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def content_deltas(lines: Iterable[str]) -> list[str]:
    deltas: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        piece = ""
        message = payload.get("message")
        if isinstance(message, dict):
            piece = message.get("content") or ""
        if piece:
            deltas.append(piece)
    return deltas


class OllamaChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        decide: Callable[[str, dict[str, Any]], Awaitable[str]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._decide = decide

    async def status(self) -> dict[str, bool]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, OSError, ValueError):
            return {"up": False, "model_ready": False}
        models = payload.get("models") if isinstance(payload, dict) else []
        names = [item.get("name", "") for item in models if isinstance(item, dict)]
        ready = any(name == self.model or name.startswith(f"{self.model}:") for name in names)
        return {"up": True, "model_ready": ready}

    async def needs_current_facts(self, question: str, reading: dict[str, Any]) -> bool:
        if obvious_live_question(question):
            return True
        try:
            if self._decide is not None:
                answer = await self._decide(question, reading)
            else:
                answer = await self._decide_with_model(question, reading)
        except (httpx.HTTPError, OSError, ValueError, OllamaUnavailable):
            return False
        return wants_live_facts(answer)

    async def _decide_with_model(self, question: str, reading: dict[str, Any]) -> str:
        excerpt = str(reading.get("text") or "")[:800]
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": LIVE_DECIDE},
                {
                    "role": "user",
                    "content": (
                        f"Article title: {reading.get('title', '')}\n"
                        f"Article:\n{excerpt}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        }
        timeout = httpx.Timeout(20.0, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            if response.status_code >= 400:
                raise OllamaUnavailable("Ollama não decidiu.")
            data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return ""

    async def stream(
        self,
        reading: dict[str, Any],
        history: list[dict[str, Any]],
        user_text: str,
        live_notes: str = "",
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": build_messages(reading, history, user_text, live_notes),
            "stream": True,
        }
        message = (
            f"O Ollama não está respondendo. Confira se ele está aberto e se o modelo {self.model} foi baixado."
        )
        try:
            timeout = httpx.Timeout(120.0, connect=3.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code >= 400:
                        raise OllamaUnavailable(message)
                    async for line in response.aiter_lines():
                        for delta in content_deltas([line]):
                            yield delta
        except OllamaUnavailable:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise OllamaUnavailable(message) from exc
