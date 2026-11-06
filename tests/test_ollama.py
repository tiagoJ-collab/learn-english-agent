import json

import asyncio

from app.ollama import (
    OllamaChat,
    SentenceBuffer,
    build_messages,
    content_deltas,
    obvious_live_question,
    wants_live_facts,
)


def test_streamed_tokens_become_sentences():
    buffer = SentenceBuffer()

    assert buffer.push("Hel") == []
    assert buffer.push("lo. ") == ["Hello."]
    assert buffer.push("Next") == []
    assert buffer.flush() == ["Next"]


def test_a_finished_sentence_without_a_trailing_space_is_kept_until_flush():
    buffer = SentenceBuffer()

    assert buffer.push("Really? Yes. ") == ["Really?", "Yes."]
    assert buffer.push("OK") == []
    assert buffer.flush() == ["OK"]


def test_messages_keep_the_article_in_the_prompt():
    reading = {
        "title": "Rain",
        "url": "https://example.com/rain",
        "text": "Clouds make rain.",
    }
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]

    messages = build_messages(reading, history, "Why does it rain?")

    assert "Clouds make rain." in messages[0]["content"]
    assert "two to four short sentences" in messages[0]["content"]
    assert "cool English teacher for Brazilians" in messages[0]["content"]
    assert "Do not answer in Portuguese." in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello"}
    assert messages[-1] == {"role": "user", "content": "Why does it rain?"}
    assert "Current facts:\nNone." in messages[0]["content"]


def test_live_notes_are_added_when_the_web_search_ran():
    reading = {"title": "Rain", "url": "https://example.com/rain", "text": "Clouds make rain."}

    messages = build_messages(reading, [], "Weather?", "Joinville is 22 C today.")

    assert "Joinville is 22 C today." in messages[0]["content"]


def test_obvious_current_questions_skip_the_local_model():
    assert obvious_live_question("What's the weather in Joinville?")
    assert obvious_live_question("Como está o tempo no Brasil hoje?")
    assert obvious_live_question("What's happening in Brazil today?")
    assert not obvious_live_question("Why did the city declare an emergency?")
    assert not obvious_live_question("What does the article say about Brazil?")
    assert wants_live_facts("YES")
    assert wants_live_facts("Yes, it is news.")
    assert not wants_live_facts("NO, the article answers it.")


def test_the_model_can_mark_a_question_as_current():
    seen = {}

    async def decide(question, reading):
        seen["question"] = question
        return "NO"

    chat = OllamaChat("http://127.0.0.1:9", "llama3.2", decide=decide)

    assert asyncio.run(chat.needs_current_facts("Who won last night?", {"title": "Rain", "text": "Clouds."})) is False
    assert seen["question"] == "Who won last night?"
    assert asyncio.run(chat.needs_current_facts("What's the weather in Joinville?", {"title": "Rain", "text": "Clouds."}))


def test_content_deltas_skip_blank_and_done_lines():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        "",
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"done": True}),
        "not-json",
    ]

    assert content_deltas(lines) == ["Hel", "lo"]
