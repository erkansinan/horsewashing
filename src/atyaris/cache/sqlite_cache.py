"""SQLite tabanli basit TTL (time-to-live) onbellek.

TJK'ya asiri istek gonderilmesini onlemek icin (bkz. checklist.md, madde 1 ve 5)
kullanilir. ``diskcache`` gibi harici bir bagimliliga ihtiyac duymadan, standart
kutuphanedeki ``sqlite3`` modulu ile calisir.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SqliteTTLCache:
    """Anahtar-deger ciftlerini son gecerlilik zamaniyla birlikte saklayan cache."""

    def __init__(self, path: str | Path, default_ttl_seconds: int = 600) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl_seconds
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        row = self._conn.execute(
            "SELECT value, expires_at FROM cache_entries WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at < time.time():
            self.delete(key)
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = time.time() + ttl
        self._conn.execute(
            "INSERT OR REPLACE INTO cache_entries (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), expires_at),
        )
        self._conn.commit()

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
