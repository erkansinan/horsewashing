"""SQLite TTL cache icin testler."""
from __future__ import annotations

import time

from atyaris.cache.sqlite_cache import SqliteTTLCache


def test_set_and_get_roundtrip(tmp_path) -> None:
    cache = SqliteTTLCache(tmp_path / "test.sqlite3", default_ttl_seconds=60)
    cache.set("key", {"a": 1})
    assert cache.get("key") == {"a": 1}


def test_expired_entry_returns_none(tmp_path) -> None:
    cache = SqliteTTLCache(tmp_path / "test.sqlite3", default_ttl_seconds=60)
    cache.set("key", "value", ttl_seconds=0)
    time.sleep(0.01)
    assert cache.get("key") is None


def test_missing_key_returns_none(tmp_path) -> None:
    cache = SqliteTTLCache(tmp_path / "test.sqlite3")
    assert cache.get("missing") is None


def test_delete_removes_entry(tmp_path) -> None:
    cache = SqliteTTLCache(tmp_path / "test.sqlite3")
    cache.set("key", "value")
    cache.delete("key")
    assert cache.get("key") is None
