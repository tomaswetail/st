"""Reusable throttled HTTP client for football data providers."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from objects.schema.data_classes.data_sources import DISK_CACHE_TTL_ONE_YEAR

logger = logging.getLogger(__name__)


class FootballDataHttpError(Exception):
    """Raised when an HTTP request fails after retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        """Store status code and whether the failure is retryable."""
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class NotFoundError(FootballDataHttpError):
    """HTTP 404 from a football data provider."""
    def __init__(self, message: str, *, status_code: int = 404) -> None:
        """Create a non-retryable 404 error."""
        super().__init__(message, status_code=status_code, retryable=False)


class ThrottledHttpClient:
    """Sync httpx client with delay, retries, optional disk cache, and logging."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_sec: float = 20.0,
        max_retries: int = 3,
        request_delay_ms: int = 500,
        cache_ttl_seconds: int = DISK_CACHE_TTL_ONE_YEAR,
        cache_dir: Path | None = None,
        user_agent: str = "st-football-data/1.0",
        enable_cache: bool = True,
    ) -> None:
        """Configure base URL, retries, delay, and optional disk cache."""
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.request_delay_ms = request_delay_ms
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache and cache_dir is not None and cache_ttl_seconds > 0
        self.user_agent = user_agent
        self._last_request_at: float | None = None
        self._client = httpx.Client(
            timeout=timeout_sec,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            follow_redirects=True,
        )
        if self.enable_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> ThrottledHttpClient:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and close the client."""
        self.close()

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """GET JSON from path with optional cache."""
        return self._request_json("GET", path, params=params, use_cache=use_cache)

    def post_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """POST JSON to path with optional cache."""
        return self._request_json(
            "POST",
            path,
            params=params,
            json_body=json_body,
            use_cache=use_cache,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Perform a throttled JSON request with retries and cache."""
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        cache_key = self._cache_key(method, url, params, json_body)
        if use_cache and self.enable_cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s %s", method, url)
                return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                logger.info(
                    "HTTP %s %s params=%s attempt=%s",
                    method,
                    url,
                    params,
                    attempt + 1,
                )
                if method.upper() == "POST":
                    response = self._client.post(url, params=params, json=json_body)
                else:
                    response = self._client.get(url, params=params)
                status = response.status_code
                if status == 404:
                    raise NotFoundError(f"Not found: {url}", status_code=404)
                if status == 429 or status >= 500:
                    wait = self._backoff_seconds(attempt)
                    logger.warning(
                        "Retryable HTTP %s for %s; sleeping %.1fs",
                        status,
                        url,
                        wait,
                    )
                    last_error = FootballDataHttpError(
                        f"HTTP {status} for {url}: {response.text[:300]}",
                        status_code=status,
                        retryable=True,
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
                        continue
                    raise last_error
                if status >= 400:
                    raise FootballDataHttpError(
                        f"HTTP {status} for {url}: {response.text[:300]}",
                        status_code=status,
                        retryable=False,
                    )
                data = response.json()
                if use_cache and self.enable_cache:
                    self._write_cache(cache_key, data)
                return data
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                wait = self._backoff_seconds(attempt)
                logger.warning(
                    "Transport error for %s: %s; sleeping %.1fs",
                    url,
                    exc,
                    wait,
                )
                last_error = FootballDataHttpError(
                    str(exc),
                    status_code=None,
                    retryable=True,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                    continue
                raise last_error from exc

        assert last_error is not None
        raise last_error

    def _throttle(self) -> None:
        """Sleep to enforce the configured request delay."""
        if self.request_delay_ms <= 0:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed_ms = (now - self._last_request_at) * 1000
            remaining = self.request_delay_ms - elapsed_ms
            if remaining > 0:
                time.sleep(remaining / 1000)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _backoff_seconds(attempt: int) -> float:
        """Exponential backoff delay for the given attempt."""
        return min(60.0, (2**attempt) * 0.5)

    def _cache_key(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        json_body: dict[str, Any] | None = None,
    ) -> str:
        """Hash method, URL, params, and body into a cache key."""
        raw = json.dumps(
            {
                "method": method.upper(),
                "url": url,
                "params": params or {},
                "json": json_body or {},
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Filesystem path for a cache key."""
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Any | None:
        """Return cached JSON if present and within TTL."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        """Persist JSON response under the cache key."""
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write cache %s: %s", path, exc)
