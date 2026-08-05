"""HTTP client and JSON parser for Svenska Spel Stryktipset draw API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from objects.schema.data_classes.svenska_spel_config import SvenskaSpelConfig

logger = logging.getLogger(__name__)

DRAW_NOT_FOUND = "Resource Not Found"


def parse_swedish_decimal(value: str | float | int | None) -> float | None:
    """Parse Svenska Spel decimal strings like '3,10' or '1,00'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


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

    def _draw_url(self, draw_number: int) -> str:
        base = self.svenskaspel_base_url.rstrip("/")
        if self.svenskaspel_access_key:
            return (
                f"{base}/external/1/draw/stryktipset/draws/{draw_number}"
                f"?accesskey={self.svenskaspel_access_key}"
            )
        return f"{base}/draw/1/stryktipset/draws/{draw_number}"

    def fetch_draw_raw(self, draw_number: int) -> dict[str, Any]:
        """Fetch raw draw JSON; raises DrawNotFoundError on 404."""

        url = self._draw_url(draw_number)
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        if resp.status_code == 404:
            raise DrawNotFoundError(f"Draw {draw_number} not found")
        resp.raise_for_status()
        data = resp.json()
        if data.get("draw") is None:
            err = data.get("error") or {}
            if err.get("code") == 404 or err.get("message") == DRAW_NOT_FOUND:
                raise DrawNotFoundError(f"Draw {draw_number} not found")
            raise ValueError(f"Unexpected API response for draw {draw_number}: {err or data}")

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
            latest_payload = self.fetch_draw_raw(highes)
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
