from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.openrouter import Article

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    level TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def create_reading(self, topic: str, level: str, article: Article) -> dict[str, Any]:
        now = _now()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO readings (topic, level, title, url, source, body, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (topic, level, article.title, article.url, article.source, article.text, now),
            )
            conn.commit()
            reading_id = int(cursor.lastrowid)
        reading = self.get_reading(reading_id)
        if reading is None:
            raise RuntimeError("A leitura não foi gravada.")
        return reading

    def list_readings(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, topic, level, title, url, source, created_at
                FROM readings
                ORDER BY id DESC
                """
            ).fetchall()
        return [_reading(row, with_text=False) for row in rows]

    def get_reading(self, reading_id: int) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM readings WHERE id = ?", (reading_id,)).fetchone()
        if row is None:
            return None
        return _reading(row, with_text=True)

    def add_message(self, reading_id: int, role: str, content: str) -> dict[str, Any]:
        now = _now()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (reading_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (reading_id, role, content, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _message(row)

    def list_messages(self, reading_id: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE reading_id = ?
                ORDER BY id ASC
                """,
                (reading_id,),
            ).fetchall()
        return [_message(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reading(row: sqlite3.Row, with_text: bool) -> dict[str, Any]:
    data = {
        "id": row["id"],
        "topic": row["topic"],
        "level": row["level"],
        "title": row["title"],
        "url": row["url"],
        "source": row["source"],
        "created_at": row["created_at"],
    }
    if with_text:
        data["text"] = row["body"]
    return data


def _message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "reading_id": row["reading_id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
