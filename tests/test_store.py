from app.openrouter import Article
from app.store import Store

TEXT = (
    "Rain falls on the city every afternoon, and the streets shine after the clouds pass. "
    "People open umbrellas, buses slow down, and the river rises a little before night."
)


def _article(title: str) -> Article:
    return Article(title=title, url=f"https://example.com/{title}", source="Example", text=TEXT)


def test_readings_are_listed_newest_first_and_messages_stay_in_order(tmp_path):
    store = Store(tmp_path / "readings.db")
    first = store.create_reading("rain", "B1", _article("Rain"))
    second = store.create_reading("wind", "A2", _article("Wind"))

    listed = store.list_readings()

    assert [item["title"] for item in listed] == ["Wind", "Rain"]
    assert "text" not in listed[0]
    assert store.get_reading(first["id"])["text"] == TEXT

    store.add_message(second["id"], "user", "What is the text about?")
    store.add_message(second["id"], "assistant", "It is about wind.")

    messages = store.list_messages(second["id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "It is about wind."
    assert store.get_reading(999) is None
