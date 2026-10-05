"""Local SQLite storage for settings, uploaded documents, and chat history."""

import json
import sqlite3
from pathlib import Path

from rag import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    filename       TEXT NOT NULL,
    openai_file_id TEXT,
    status         TEXT NOT NULL,          -- processing | ready | failed
    error          TEXT,
    ocr            INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role         TEXT NOT NULL,            -- user | assistant
    text         TEXT NOT NULL,
    model        TEXT,
    sources_json TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class DB:
    def __init__(self, path: Path = config.DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    # --- settings ---
    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # --- documents ---
    def add_document(self, filename: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO documents (filename, status) VALUES (?, 'processing')", (filename,)
            )
        return cur.lastrowid

    def update_document(self, doc_id: int, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.conn:
            self.conn.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))

    def list_documents(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC, id DESC").fetchall()

    def ready_filenames(self) -> dict[str, str]:
        """OpenAI file id -> original filename, for documents that can be searched."""
        rows = self.conn.execute(
            "SELECT openai_file_id, filename FROM documents WHERE status = 'ready'"
        ).fetchall()
        return {r["openai_file_id"]: r["filename"] for r in rows}

    def delete_document(self, doc_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # --- chats ---
    def create_chat(self, title: str) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO chats (title) VALUES (?)", (title,))
        return cur.lastrowid

    def list_chats(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chats ORDER BY id DESC").fetchall()

    def delete_chat(self, chat_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def add_message(self, chat_id: int, role: str, text: str,
                    model: str | None = None, sources: list[dict] | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (chat_id, role, text, model, sources_json) VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, text, model, json.dumps(sources or [], ensure_ascii=False)),
            )

    def list_messages(self, chat_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()
        return [
            {"role": r["role"], "text": r["text"], "model": r["model"],
             "sources": json.loads(r["sources_json"])}
            for r in rows
        ]
