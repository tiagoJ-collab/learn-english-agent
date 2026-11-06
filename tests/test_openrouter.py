import asyncio
import json

import httpx
import pytest

from app.openrouter import (
    IncompleteArticle,
    OpenRouterClient,
    OpenRouterError,
    build_lookup_payload,
    build_payload,
    parse_article,
    parse_live_brief,
)

ARTICLE_TEXT = (
    "Rain falls on the city every afternoon, and the streets shine after the clouds pass. "
    "People open umbrellas, buses slow down, and the river rises a little before night. "
    "Warm air lifts water from the sea, and that water returns as rain over the hills."
)


def _payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_payload_asks_for_web_search():
    payload = build_payload("rain", "B1", "openai/gpt-4o-mini:online")

    assert payload["model"] == "openai/gpt-4o-mini:online"
    assert payload["plugins"] == [{"id": "web"}]
    assert "rain" in payload["messages"][-1]["content"]
    assert "B1" in payload["messages"][-1]["content"]


def test_parse_fenced_article():
    content = "```json\n" + json.dumps(
        {
            "title": "Rain",
            "url": "https://en.wikipedia.org/wiki/Rain",
            "source": "Wikipedia",
            "text": ARTICLE_TEXT,
        }
    ) + "\n```"
    article = parse_article(_payload(content))

    assert article.title == "Rain"
    assert article.url == "https://en.wikipedia.org/wiki/Rain"
    assert article.source == "Wikipedia"
    assert "umbrellas" in article.text


def test_parse_uses_the_site_name_when_source_is_missing():
    content = json.dumps(
        {"title": "Rain", "url": "https://en.wikipedia.org/wiki/Rain", "text": ARTICLE_TEXT}
    )
    article = parse_article(_payload(content))

    assert article.source == "en.wikipedia.org"


def test_parse_rejects_an_incomplete_article():
    with pytest.raises(IncompleteArticle):
        parse_article(_payload(json.dumps({"title": "Rain"})))


def test_parse_rejects_a_text_that_is_too_short():
    content = json.dumps(
        {"title": "Rain", "url": "https://example.com/rain", "text": "It rains."}
    )
    with pytest.raises(IncompleteArticle):
        parse_article(_payload(content))


def test_parse_reads_a_list_of_content_parts():
    article = parse_article(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "title": "Rain",
                                        "url": "https://example.com/rain",
                                        "source": "Example",
                                        "text": ARTICLE_TEXT,
                                    }
                                ),
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert article.title == "Rain"


def test_fetch_sends_the_key_and_returns_the_article():
    seen = {}

    async def post(url, headers, body):
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = body
        return _payload(
            json.dumps(
                {
                    "title": "Rain",
                    "url": "https://example.com/rain",
                    "source": "Example",
                    "text": ARTICLE_TEXT,
                }
            )
        )

    client = OpenRouterClient("secret", "openai/gpt-4o-mini:online", post=post)
    article = asyncio.run(client.fetch("rain", "B1"))

    assert article.title == "Rain"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["body"]["plugins"][0]["id"] == "web"
    assert seen["url"].endswith("/chat/completions")


def test_fetch_turns_a_connection_problem_into_a_clear_error():
    calls = {"n": 0}

    async def post(url, headers, body):
        calls["n"] += 1
        raise httpx.ConnectError("offline")

    client = OpenRouterClient("secret", "openai/gpt-4o-mini:online", post=post)

    with pytest.raises(OpenRouterError, match="OpenRouter"):
        asyncio.run(client.fetch("rain", "B1"))
    assert calls["n"] == 2


def test_fetch_retries_a_temporary_failure_once():
    calls = {"n": 0}

    async def post(url, headers, body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OpenRouterError("Não consegui buscar o texto no OpenRouter.", retryable=True)
        return _payload(
            json.dumps(
                {
                    "title": "Rain",
                    "url": "https://example.com/rain",
                    "source": "Example",
                    "text": ARTICLE_TEXT,
                }
            )
        )

    client = OpenRouterClient("secret", "openai/gpt-4o-mini:online", post=post)
    article = asyncio.run(client.fetch("rain", "B1"))

    assert calls["n"] == 2
    assert article.title == "Rain"


def test_fetch_does_not_retry_a_rejected_key():
    calls = {"n": 0}

    async def post(url, headers, body):
        calls["n"] += 1
        raise OpenRouterError("A chave do OpenRouter foi recusada.")

    client = OpenRouterClient("secret", "openai/gpt-4o-mini:online", post=post)

    with pytest.raises(OpenRouterError, match="recusada"):
        asyncio.run(client.fetch("rain", "B1"))
    assert calls["n"] == 1


def test_lookup_asks_the_web_for_the_question():
    payload = build_lookup_payload("What's the weather in Joinville?", "openai/gpt-4o-mini:online")

    assert payload["plugins"] == [{"id": "web"}]
    assert payload["messages"][-1]["content"] == "What's the weather in Joinville?"


def test_lookup_returns_the_brief():
    seen = {}

    async def post(url, headers, body):
        seen["body"] = body
        return _payload("Joinville is rainy today, about 22 C, according to a weather report.")

    client = OpenRouterClient("secret", "openai/gpt-4o-mini:online", post=post)
    brief = asyncio.run(client.lookup("What's the weather in Joinville?"))

    assert "22 C" in brief
    assert seen["body"]["plugins"][0]["id"] == "web"
    with pytest.raises(IncompleteArticle):
        parse_live_brief(_payload("too short"))


def test_fetch_requires_an_api_key():
    client = OpenRouterClient("", "openai/gpt-4o-mini:online", post=None)

    with pytest.raises(OpenRouterError, match=".env"):
        asyncio.run(client.fetch("rain", "B1"))
