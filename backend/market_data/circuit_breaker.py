"""Provider-scoped market-data circuit breaker and compatibility facade."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class MarketTransportError(RuntimeError):
    """Raised when upstream market data provider experiences network, 5xx, or 429 errors."""


class MarketDataUnavailable(ValueError):
    """Raised when market data exists but is unusable (insufficient history,
    missing columns, non-finite or invalid values). Maps to HTTP 422."""


class UnknownTickerError(MarketDataUnavailable):
    """The provider returned no data at all for a syntactically valid symbol.

    Distinct from :class:`MarketDataUnavailable` so routes can answer 404
    (unknown symbol) instead of a generic client/temporary error.
    Maps to HTTP 404."""


class MarketCircuitBreaker:
    """
    Two-tier circuit breaker:
    1. Provider-wide transport breaker: trips on >= 3 consecutive network/5xx/429 failures.
       Cooldown: 30s. Single-probe half-open state.
    2. Per-ticker negative cache: LRU (max 64, TTL 60s) for missing/invalid ticker responses.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        negative_cache_ttl: float = 60.0,
        max_negative_cache: int = 64,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.negative_cache_ttl = negative_cache_ttl
        self.max_negative_cache = max_negative_cache
        self._lock = threading.Lock()
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._last_error: str | None = None
        self._last_checked_epoch: float | None = None
        self._half_open_in_flight = False
        self._negative_cache: OrderedDict[str, tuple[float, str, bool]] = OrderedDict()

    def check_can_execute(self, ticker: str) -> None:
        with self._lock:
            now = time.time()
            # Check negative cache for ticker
            if ticker in self._negative_cache:
                cached_time, cached_reason, is_unknown = self._negative_cache[ticker]
                if now - cached_time < self.negative_cache_ttl:
                    error_type = UnknownTickerError if is_unknown else MarketDataUnavailable
                    raise error_type(
                        f"Ticker {ticker} is temporarily cached as unavailable: {cached_reason}"
                    )
                del self._negative_cache[ticker]

            # Check provider circuit
            if self._state == "open":
                if now - self._opened_at >= self.cooldown_seconds:
                    self._state = "half_open"
                    self._half_open_in_flight = True
                else:
                    raise MarketTransportError(
                        "Upstream market data circuit is open due to recent failures."
                    )
            elif self._state == "half_open":
                if self._half_open_in_flight:
                    raise MarketTransportError(
                        "Upstream market data circuit probe is currently in flight."
                    )
                self._half_open_in_flight = True

    def record_success(self, ticker: str) -> None:
        with self._lock:
            now = time.time()
            self._consecutive_failures = 0
            self._state = "closed"
            self._opened_at = 0.0
            self._last_error = None
            self._last_checked_epoch = now
            self._half_open_in_flight = False
            self._negative_cache.pop(ticker, None)

    def record_failure(self, ticker: str, exception: Exception) -> None:
        with self._lock:
            now = time.time()
            self._half_open_in_flight = False
            self._last_error = str(exception)
            self._last_checked_epoch = now
            err_msg = str(exception).lower()
            is_transport = (
                isinstance(
                    exception, (MarketTransportError, TimeoutError, ConnectionError, OSError)
                )
                or (
                    hasattr(exception, "status_code")
                    and exception.status_code in (429, 500, 502, 503, 504)
                )
                or "rate limit" in err_msg
                or "timeout" in err_msg
                or "connection" in err_msg
                or "unavailable" in err_msg
            )

            if is_transport:
                self._consecutive_failures += 1
                if (
                    self._consecutive_failures >= self.failure_threshold
                    or self._state == "half_open"
                ):
                    self._state = "open"
                    self._opened_at = now
            elif isinstance(exception, (ValueError, MarketDataUnavailable)):
                if len(self._negative_cache) >= self.max_negative_cache:
                    self._negative_cache.popitem(last=False)
                self._negative_cache[ticker] = (
                    now,
                    str(exception),
                    isinstance(exception, UnknownTickerError),
                )

    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        """Evaluate readiness of the upstream market data dependency."""
        with self._lock:
            now = time.time()
            if self._state == "open" and now - self._opened_at >= self.cooldown_seconds:
                self._state = "half_open"

            is_ready = self._state != "open"
            status_dict: dict[str, Any] = {
                "status": "available" if is_ready else "unavailable",
                "circuit": self._state,
                "consecutive_failures": self._consecutive_failures,
                "last_error": self._last_error,
                "checked_at_epoch": self._last_checked_epoch,
                "cooldown_remaining_seconds": (
                    max(0.0, round(self.cooldown_seconds - (now - self._opened_at), 2))
                    if self._state == "open"
                    else 0.0
                ),
                "negative_cache_size": len(self._negative_cache),
            }
            return is_ready, status_dict

    def get_status(self) -> dict[str, Any]:
        return self.is_ready()[1]


class ProviderAwareCircuitBreakerFacade:
    """Compatibility facade delegating legacy breaker calls to provider-scoped breakers."""

    def __init__(self, service_getter: Callable[[], Any]) -> None:
        self._service_getter = service_getter

    def _target_breakers(self, ticker: str) -> list[MarketCircuitBreaker]:
        service = self._service_getter()
        if ticker == "__global__":
            names = service.us_provider_names()
        elif ticker.strip().upper().endswith(".L"):
            names = service.uk_provider_names()
        else:
            names = service.us_provider_names()
        return [service.breaker_for_provider(name) for name in names]

    def check_can_execute(self, ticker: str) -> None:
        breakers = self._target_breakers(ticker)
        last_err: Exception | None = None
        for breaker in breakers:
            try:
                breaker.check_can_execute(ticker)
                return
            except Exception as err:
                last_err = err
        if last_err is not None:
            raise last_err

    def record_success(self, ticker: str) -> None:
        for breaker in self._target_breakers(ticker):
            breaker.record_success(ticker)

    def record_failure(self, ticker: str, exception: Exception) -> None:
        for breaker in self._target_breakers(ticker):
            breaker.record_failure(ticker, exception)

    def is_ready(self) -> tuple[bool, dict[str, Any]]:
        service = self._service_getter()
        return service.us_circuit_readiness()

    def get_status(self) -> dict[str, Any]:
        return self.is_ready()[1]
