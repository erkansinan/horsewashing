"""Low-latency TJK source factory for the live prediction path."""
from __future__ import annotations

from atyaris.cache.sqlite_cache import SqliteTTLCache
from atyaris.config import Settings
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource


def build_prediction_data_source(settings: Settings) -> TJKHtmlDataSource:
    return TJKHtmlDataSource(
        base_url=settings.tjk_base_url,
        user_agent=settings.user_agent,
        request_timeout=settings.prediction_data_request_timeout_seconds,
        min_request_interval=settings.prediction_data_min_request_interval_seconds,
        cache=SqliteTTLCache(settings.cache_path, settings.cache_ttl_seconds),
        cache_ttl_seconds=settings.cache_ttl_seconds,
        horse_history_path=settings.phase1_raw_history_jsonl_path,
        max_retries=settings.prediction_data_request_max_retries,
        program_request_timeout=settings.prediction_program_request_timeout_seconds,
        program_max_retries=settings.prediction_program_request_max_retries,
    )
