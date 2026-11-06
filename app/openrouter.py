from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MIN_ARTICLE_CHARS = 200

ARTICLE_SYSTEM = """You find one real English article on the web for a language learner.
Return only a JSON object with the keys title, url, source, and text.
The text must be the article body, faithful to the source, between 400 and 700 words.
Prefer openly available pages such as Wikipedia or other public educational pages.
Match the requested reading level.
Do not wrap the JSON in markdown.
"""


class OpenRouterError(Exception):
    """The text search could not be completed."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class IncompleteArticle(OpenRouterError):
    """The model answered, but not with a usable article."""


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    source: str
    text: str


LIVE_SYSTEM = """You look up current facts on the web.
Return a short factual brief in English, four to eight sentences.
Include the place, the date when you can, and the source names.
If you cannot find a current source, say that plainly.
Do not use markdown.
"""


def build_payload(topic: str, level: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0.2,
        "plugins": [{"id": "web"}],
        "messages": [
            {"role": "system", "content": ARTICLE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\nLevel: {level}\n"
                    "Find one article and return the JSON."
                ),
            },
        ],
    }


def build_lookup_payload(question: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0.2,
        "plugins": [{"id": "web"}],
        "messages": [
            {"role": "system", "content": LIVE_SYSTEM},
            {"role": "user", "content": question},
        ],
    }


def parse_article(payload: dict[str, Any]) -> Article:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise IncompleteArticle("A busca não devolveu um texto completo.") from exc
    data = _load_json_object(_message_text(content))
    title = str(data.get("title") or "").strip()
    url = str(data.get("url") or "").strip()
    text = str(data.get("text") or "").strip()
    source = str(data.get("source") or "").strip()
    parsed = urlparse(url)
    if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IncompleteArticle("A busca não devolveu um texto completo.")
    if len(text) < MIN_ARTICLE_CHARS:
        raise IncompleteArticle("A busca não devolveu um texto completo.")
    if not source:
        source = parsed.netloc
    return Article(title=title, url=url, source=source, text=text)


def parse_live_brief(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise IncompleteArticle("A busca atual não devolveu uma resposta.") from exc
    text = " ".join(_message_text(content).split())
    if len(text) < 20:
        raise IncompleteArticle("A busca atual não devolveu uma resposta.")
    return text[:1500]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        post: Callable[[str, dict[str, str], dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._post = post or _default_post

    async def fetch(self, topic: str, level: str) -> Article:
        payload = await self._request(build_payload(topic, level, self.model))
        return parse_article(payload)

    async def lookup(self, question: str) -> str:
        payload = await self._request(build_lookup_payload(question, self.model))
        return parse_live_brief(payload)

    async def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise OpenRouterError("Falta a chave do OpenRouter no arquivo .env.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8765",
            "X-Title": "English reading agent",
        }
        last_error: OpenRouterError | None = None
        for attempt in range(2):
            try:
                return await self._post(OPENROUTER_URL, headers, body)
            except OpenRouterError as exc:
                last_error = exc
            except httpx.HTTPError as exc:
                last_error = OpenRouterError(
                    "Não consegui buscar o texto no OpenRouter.",
                    retryable=True,
                )
                last_error.__cause__ = exc
            if not last_error.retryable or attempt == 1:
                raise last_error
        raise last_error or OpenRouterError("Não consegui buscar o texto no OpenRouter.")


async def _default_post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code == 401:
        raise OpenRouterError("A chave do OpenRouter foi recusada.")
    if response.status_code >= 400:
        retryable = response.status_code == 429 or response.status_code >= 500
        raise OpenRouterError(_provider_message(response), retryable=retryable)
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise IncompleteArticle("A busca não devolveu um texto completo.") from exc


def _provider_message(response: httpx.Response) -> str:
    fallback = "Não consegui buscar o texto no OpenRouter."
    try:
        data = response.json()
    except json.JSONDecodeError:
        return fallback
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message.strip():
        return fallback
    clean = " ".join(message.split())[:180]
    return f"{fallback} {clean}"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    raise IncompleteArticle("A busca não devolveu um texto completo.")


def _load_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IncompleteArticle("A busca não devolveu um texto completo.") from exc
    if not isinstance(data, dict):
        raise IncompleteArticle("A busca não devolveu um texto completo.")
    return data
