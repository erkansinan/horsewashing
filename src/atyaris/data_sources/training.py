"""Independent TJK source factory for retrospective data collection."""
from __future__ import annotations

from atyaris.cache.sqlite_cache import SqliteTTLCache
from atyaris.config import Settings
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource


def build_training_data_source(settings: Settings) -> TJKHtmlDataSource:
    return TJKHtmlDataSource(
        base_url=settings.tjk_base_url,
        user_agent=settings.user_agent,
        request_timeout=settings.request_timeout_seconds,
        min_request_interval=settings.min_request_interval_seconds,
        cache=SqliteTTLCache(settings.cache_path, settings.cache_ttl_seconds),
        cache_ttl_seconds=settings.cache_ttl_seconds,
        horse_history_path=settings.phase1_raw_history_jsonl_path,
        max_retries=2,
    )
