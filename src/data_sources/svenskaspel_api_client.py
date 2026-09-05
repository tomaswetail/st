"""HTTP client and JSON parser for Svenska Spel Stryktipset draw API."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.objects.schema.data_classes.svenska_spel_config import SvenskaSpelConfig

logger = logging.getLogger(__name__)

DRAW_NOT_FOUND = "Resource Not Found"


def _participant_name(participants: list[dict[str, Any]], role: str) -> str:
    for p in participants:
        if p.get("type") == role:
            return str(p["name"])
    raise ValueError(f"Missing participant with type={role!r}")


def draw_is_open(draw: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Return True if draw registration is still open."""
    now = now or datetime.now(timezone.utc)
    state = draw.get("drawState")
    close_raw = draw.get("regCloseTime")
    if not close_raw:
        return state != "Finalized"
    close = datetime.fromisoformat(close_raw)
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    now_aware = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return state != "Finalized" and close > now_aware


class DrawNotFoundError(Exception):
    """Raised when a draw number does not exist."""


class SvenskaSpelClient:
    """Fetch Stryktipset draws from Svenska Spel API."""

    def __init__(self, config: SvenskaSpelConfig | None = None) -> None:
        config = config or SvenskaSpelConfig.from_env()
        self.svenskaspel_base_url = config.svenskaspel_base_url
        self.svenskaspel_access_key = config.svenskaspel_access_key or None
        self.enable_cache = config.enable_cache and config.cache_ttl_open_seconds > 0
        self.cache_dir = config.cache_dir
        self.cache_ttl_open_seconds = config.cache_ttl_open_seconds
        self.cache_ttl_finalized_seconds = config.cache_ttl_finalized_seconds
        self.cache_ttl_not_found_seconds = config.cache_ttl_not_found_seconds
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _draw_url(self, draw_number: int) -> str:
        base = self.svenskaspel_base_url.rstrip("/")
        if self.svenskaspel_access_key:
            return (
                f"{base}/external/1/draw/stryktipset/draws/{draw_number}"
                f"?accesskey={self.svenskaspel_access_key}"
            )
        return f"{base}/draw/1/stryktipset/draws/{draw_number}"

    def _cache_key(self, draw_number: int) -> str:
        return hashlib.sha256(self._draw_url(draw_number).encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, draw_number: int) -> dict[str, Any] | None:
        if not self.enable_cache:
            return None
        path = self._cache_path(self._cache_key(draw_number))
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        cached_at = envelope.get("_cached_at")
        ttl_seconds = envelope.get("_ttl_seconds")
        if cached_at is None or ttl_seconds is None:
            return None
        if time.time() - float(cached_at) > float(ttl_seconds):
            return None
        return envelope

    def _write_cache(self, draw_number: int, envelope: dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        path = self._cache_path(self._cache_key(draw_number))
        try:
            path.write_text(
                json.dumps(envelope, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to write Svenska Spel cache %s: %s", path, exc)

    def _cache_ttl_for_payload(self, payload: dict[str, Any]) -> int:
        draw = payload.get("draw") or {}
        if draw_is_open(draw):
            return self.cache_ttl_open_seconds
        return self.cache_ttl_finalized_seconds

    def fetch_draw_raw(
        self,
        draw_number: int,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Fetch raw draw JSON; raises DrawNotFoundError on 404."""
        if use_cache and self.enable_cache:
            envelope = self._read_cache(draw_number)
            if envelope is not None:
                if envelope.get("_not_found"):
                    raise DrawNotFoundError(f"Draw {draw_number} not found")
                data = envelope.get("data")
                if data is not None:
                    return data

        url = self._draw_url(draw_number)
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 404:
            self._write_cache(
                draw_number,
                {
                    "_cached_at": time.time(),
                    "_ttl_seconds": self.cache_ttl_not_found_seconds,
                    "_not_found": True,
                },
            )
            raise DrawNotFoundError(f"Draw {draw_number} not found")
        resp.raise_for_status()
        data = resp.json()
        if data.get("draw") is None:
            err = data.get("error") or {}
            if err.get("code") == 404 or err.get("message") == DRAW_NOT_FOUND:
                self._write_cache(
                    draw_number,
                    {
                        "_cached_at": time.time(),
                        "_ttl_seconds": self.cache_ttl_not_found_seconds,
                        "_not_found": True,
                    },
                )
                raise DrawNotFoundError(f"Draw {draw_number} not found")
            raise ValueError(
                f"Unexpected API response for draw {draw_number}: {err or data}"
            )

        self._write_cache(
            draw_number,
            {
                "_cached_at": time.time(),
                "_ttl_seconds": self._cache_ttl_for_payload(data),
                "_not_found": False,
                "data": data,
            },
        )
        return data

    def fetch_draw(
        self, draw_number: int
    ):
        """Fetch and parse a single draw."""
        payload = self.fetch_draw_raw(draw_number)
        return payload

    def find_highest_draw_number(self, seed: int, *, max_probe: int = 30) -> int:
        """Scan forward from seed until 404; return last existing draw number."""
        n = seed
        last_found = seed
        for _ in range(max_probe):
            try:
                self.fetch_draw_raw(n)
                last_found = n
                n += 1
            except DrawNotFoundError:
                break
        return last_found

    def find_upcoming_draw(
        self,
        seed: int,
        *,
        now: datetime | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """
        Find the next open Stryktipset draw.

        If none is open, returns the latest published draw instead.

        Returns (draw_number, raw_payload).
        """
        now = now or datetime.now(timezone.utc)
        highest = self.find_highest_draw_number(seed)

        for n in range(highest, seed - 1, -1):
            try:
                payload = self.fetch_draw_raw(n)
            except DrawNotFoundError:
                continue
            draw = payload["draw"]
            if draw_is_open(draw, now=now):
                return n, payload

        for n in range(highest + 1, highest + 6):
            try:
                payload = self.fetch_draw_raw(n)
            except DrawNotFoundError:
                continue
            draw = payload["draw"]
            if draw_is_open(draw, now=now):
                return n, payload

        try:
            latest_payload = self.fetch_draw_raw(highest)
            latest_draw = latest_payload["draw"]
            close = latest_draw.get("regCloseTime", "unknown")
            state = latest_draw.get("drawState", "unknown")
            logger.warning(
                "No open Stryktipset draw found; using latest draw %s "
                "(state=%s, closed at %s)",
                highest,
                state,
                close,
            )
            return highest, latest_payload
        except DrawNotFoundError as exc:
            raise ValueError(
                f"No open Stryktipset draw found starting from seed {seed}."
            ) from exc
