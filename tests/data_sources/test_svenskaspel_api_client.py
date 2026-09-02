from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from data_sources.svenskaspel_api_client import (
    DrawNotFoundError,
    SvenskaSpelClient,
)
from objects.schema.data_classes.data_sources import DISK_CACHE_TTL_ONE_YEAR
from objects.schema.data_classes.svenska_spel_config import SvenskaSpelConfig


def _config(tmp_path: Path, **overrides: Any) -> SvenskaSpelConfig:
    defaults = dict(
        enable_cache=True,
        cache_dir=tmp_path,
        cache_ttl_open_seconds=900,
        cache_ttl_finalized_seconds=DISK_CACHE_TTL_ONE_YEAR,
        cache_ttl_not_found_seconds=3600,
    )
    defaults.update(overrides)
    return SvenskaSpelConfig(**defaults)


def _open_draw_payload(draw_number: int = 4750) -> dict[str, Any]:
    return {
        "draw": {
            "drawNumber": draw_number,
            "drawState": "Open",
            "regCloseTime": "2099-12-31T18:00:00+00:00",
            "productId": 1,
            "productName": "Stryktipset",
            "drawEvents": [],
        }
    }


def _finalized_draw_payload(draw_number: int = 4740) -> dict[str, Any]:
    return {
        "draw": {
            "drawNumber": draw_number,
            "drawState": "Finalized",
            "regCloseTime": "2020-01-01T18:00:00+00:00",
            "productId": 1,
            "productName": "Stryktipset",
            "drawEvents": [],
        }
    }


def _http_response(*, status_code: int = 200, payload: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload or {})
    return response


def _cached_envelope(
    *,
    data: dict[str, Any] | None = None,
    not_found: bool = False,
    cached_at: float | None = None,
    ttl_seconds: int = 900,
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "_cached_at": cached_at if cached_at is not None else time.time(),
        "_ttl_seconds": ttl_seconds,
        "_not_found": not_found,
    }
    if data is not None:
        envelope["data"] = data
    return envelope


def test_cache_hit_skips_http(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path))
    payload = _open_draw_payload()
    response = _http_response(payload=payload)

    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response) as get_mock:
        first = client.fetch_draw_raw(4750)
        second = client.fetch_draw_raw(4750)

    assert first == payload
    assert second == payload
    assert get_mock.call_count == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1


def test_expired_open_draw_cache_refetches(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path))
    payload = _open_draw_payload()
    envelope = _cached_envelope(
        data=payload,
        cached_at=time.time() - 901,
        ttl_seconds=900,
    )
    cache_path = client._cache_path(client._cache_key(4750))
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    response = _http_response(payload=payload)
    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response) as get_mock:
        result = client.fetch_draw_raw(4750)

    assert result == payload
    assert get_mock.call_count == 1


def test_finalized_draw_uses_long_ttl(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path))
    payload = _finalized_draw_payload()
    response = _http_response(payload=payload)

    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response):
        client.fetch_draw_raw(4740)

    envelope = json.loads(
        client._cache_path(client._cache_key(4740)).read_text(encoding="utf-8")
    )
    assert envelope["_ttl_seconds"] == DISK_CACHE_TTL_ONE_YEAR


def test_not_found_cached_without_second_http(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path))
    response = _http_response(status_code=404)

    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response) as get_mock:
        with pytest.raises(DrawNotFoundError):
            client.fetch_draw_raw(9999)
        with pytest.raises(DrawNotFoundError):
            client.fetch_draw_raw(9999)

    assert get_mock.call_count == 1
    envelope = json.loads(
        client._cache_path(client._cache_key(9999)).read_text(encoding="utf-8")
    )
    assert envelope["_not_found"] is True
    assert envelope["_ttl_seconds"] == 3600


def test_use_cache_false_bypasses_cache(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path))
    payload = _open_draw_payload()
    envelope = _cached_envelope(data=payload)
    cache_path = client._cache_path(client._cache_key(4750))
    cache_path.write_text(json.dumps(envelope), encoding="utf-8")

    response = _http_response(payload=payload)
    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response) as get_mock:
        result = client.fetch_draw_raw(4750, use_cache=False)

    assert result == payload
    assert get_mock.call_count == 1


def test_enable_cache_false_always_hits_http(tmp_path: Path) -> None:
    client = SvenskaSpelClient(_config(tmp_path, enable_cache=False))
    payload = _open_draw_payload()
    response = _http_response(payload=payload)

    with patch("data_sources.svenskaspel_api_client.httpx.get", return_value=response) as get_mock:
        client.fetch_draw_raw(4750)
        client.fetch_draw_raw(4750)

    assert get_mock.call_count == 2
    assert list(tmp_path.glob("*.json")) == []
