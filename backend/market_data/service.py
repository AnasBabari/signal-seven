"""Market-data provider chain, cache policy, provider-scoped breakers, and readiness state."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calendars import (
    SessionContext,
    completed_session_context,
    latest_completed_trading_session,
    resolve_calendar,
)
from market_data.alpaca import AlpacaProvider
from market_data.base import (
    MarketDataProvider,
    MarketDataProviderError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
)
from market_data.cache import MarketDataCache
from market_data.circuit_breaker import (
    MarketCircuitBreaker,
    MarketDataUnavailable,
    MarketTransportError,
    UnknownTickerError,
)
from market_data.normalization import daily_frame_fingerprint, normalize_daily_bars_with_audit
from market_data.yahoo import YahooProvider


class MarketDataService:
    """Resolve bars from fresh cache or a bounded provider chain."""

    def __init__(
        self,
        providers: Iterable[MarketDataProvider],
        *,
        cache: MarketDataCache,
        uk_providers: Iterable[MarketDataProvider] | None = None,
    ) -> None:
        self.providers = tuple(providers)
        self.uk_providers = tuple(uk_providers) if uk_providers is not None else (YahooProvider(),)
        self.cache = cache
        self._lock = threading.RLock()
        self._last_success_session: str | None = None
        self._last_success_at: str | None = None
        self._last_provider: str | None = None
        self._last_error: str | None = None
        self._uk_last_success_session: str | None = None
        self._provider_breakers: dict[str, MarketCircuitBreaker] = {}
        self._provider_state: dict[str, dict[str, str | None]] = {}

    def breaker_for_provider(self, provider_name: str) -> MarketCircuitBreaker:
        key = provider_name.strip().lower() or "default"
        with self._lock:
            breaker = self._provider_breakers.get(key)
            if breaker is None:
                breaker = MarketCircuitBreaker()
                self._provider_breakers[key] = breaker
            return breaker

    def us_provider_names(self) -> list[str]:
        configured = [p.name for p in self.providers if p.configured]
        if configured:
            return configured
        all_us = [p.name for p in self.providers]
        return all_us or ["default"]

    def uk_provider_names(self) -> list[str]:
        configured = [p.name for p in self.uk_providers if p.configured]
        if configured:
            return configured
        all_uk = [p.name for p in self.uk_providers]
        return all_uk or ["yahoo"]

    def _exchanges_for_provider(self, provider_name: str) -> list[str]:
        exchanges: list[str] = []
        if any(p.name == provider_name for p in self.providers):
            exchanges.append("US")
        if any(p.name == provider_name for p in self.uk_providers):
            exchanges.append("LSE")
        return exchanges or ["US"]

    def us_circuit_readiness(self) -> tuple[bool, dict[str, Any]]:
        """Evaluate circuit-breaker readiness strictly across US providers."""
        us_names = self.us_provider_names()
        all_names = list(dict.fromkeys(us_names + self.uk_provider_names()))
        us_statuses: list[tuple[bool, dict[str, Any]]] = []
        providers_report: dict[str, dict[str, Any]] = {}

        with self._lock:
            for name in all_names:
                breaker = self.breaker_for_provider(name)
                b_ready, b_status = breaker.is_ready()
                state = self._provider_state.get(name, {})
                providers_report[name] = {
                    **b_status,
                    "exchanges": self._exchanges_for_provider(name),
                    "last_success_session": state.get("last_success_session"),
                    "last_success_at": state.get("last_success_at"),
                    "last_error": state.get("last_error") or b_status.get("last_error"),
                }
                if name in us_names:
                    us_statuses.append((b_ready, b_status))

        ready = any(flag for flag, _ in us_statuses)
        primary_status = (
            next((status for flag, status in us_statuses if flag), us_statuses[0][1])
            if us_statuses
            else {
                "status": "available",
                "circuit": "closed",
                "consecutive_failures": 0,
                "last_error": None,
                "checked_at_epoch": None,
                "cooldown_remaining_seconds": 0.0,
                "negative_cache_size": 0,
            }
        )
        return ready, {
            **primary_status,
            "status": "available" if ready else "unavailable",
            "providers": providers_report,
        }

    def fetch_daily_bars(
        self,
        symbol: str,
        *,
        years: int,
        allow_post_close_grace: bool = False,
    ) -> MarketDataResult:
        cal_name, _ = resolve_calendar(symbol)
        if cal_name not in {"NYSE", "LSE"}:
            raise MarketDataSymbolNotFound("This exchange has no configured daily provider.")
        exchange_group = "LSE" if cal_name == "LSE" else "US"
        if cal_name == "LSE":
            required_session = latest_completed_trading_session(exchange="LSE").date().isoformat()
        else:
            required_session = latest_completed_trading_session().date().isoformat()

        failures: list[MarketDataProviderError] = []
        unknowns: list[MarketDataSymbolNotFound] = []
        grace_candidate: MarketDataResult | None = None
        active_providers = self.uk_providers if cal_name == "LSE" else self.providers

        for provider in active_providers:
            namespace = (
                f"{provider.name}:{getattr(provider, 'feed', 'daily')}:"
                f"{getattr(provider, 'adjustment', 'auto')}:{years}:session-v2"
            )
            cached = self.cache.load(
                provider.name, symbol, required_session=required_session, namespace=namespace
            )
            if cached is not None:
                cached.frame.attrs.update(
                    {
                        "data_provider": cached.provider,
                        "data_feed": cached.feed,
                        "price_adjustment": cached.adjustment,
                        "data_fingerprint": cached.data_fingerprint,
                        "raw_source_fingerprint": cached.raw_source_fingerprint,
                        "data_quality_audit": cached.data_quality_audit,
                        "post_close_grace_used": False,
                    }
                )
                self._record_success(cached, exchange=exchange_group)
                return cached
            if not provider.configured:
                continue

            breaker = self.breaker_for_provider(provider.name)
            try:
                breaker.check_can_execute(symbol)
            except MarketTransportError as err:
                failures.append(MarketDataServiceError(str(err)))
                continue
            except UnknownTickerError as err:
                unknowns.append(MarketDataSymbolNotFound(str(err)))
                continue
            except MarketDataUnavailable as err:
                failures.append(MarketDataProviderError(str(err)))
                continue

            try:
                if hasattr(provider, "fetch_raw_daily_bars"):
                    raw_result = provider.fetch_raw_daily_bars(symbol, years=years)
                else:
                    raw_result = provider.fetch_daily_bars(symbol, years=years)
                completed, raw_fp, audit = normalize_daily_bars_with_audit(
                    raw_result.frame,
                    provider=provider.name,
                    symbol=symbol,
                    timestamp_kind=getattr(raw_result, "timestamp_kind", "session_label"),
                    max_session=required_session,
                )
            except MarketDataSymbolNotFound as err:
                breaker.record_failure(symbol, UnknownTickerError(str(err)))
                self._record_failure(err, exchange=exchange_group, provider_name=provider.name)
                unknowns.append(err)
                continue
            except MarketDataServiceError as err:
                breaker.record_failure(symbol, MarketTransportError(str(err)))
                self._record_failure(err, exchange=exchange_group, provider_name=provider.name)
                failures.append(err)
                continue
            except MarketDataProviderError as err:
                breaker.record_failure(symbol, MarketDataUnavailable(str(err)))
                self._record_failure(err, exchange=exchange_group, provider_name=provider.name)
                failures.append(err)
                continue

            completed_as_of = completed.index[-1].date().isoformat() if not completed.empty else ""
            if completed_as_of == required_session:
                completed.attrs.update(
                    {
                        "data_provider": raw_result.provider,
                        "data_feed": raw_result.feed,
                        "price_adjustment": raw_result.adjustment,
                        "raw_source_fingerprint": raw_fp,
                        "data_quality_audit": audit,
                        "post_close_grace_used": False,
                    }
                )
                fingerprint = daily_frame_fingerprint(completed)
                completed.attrs["data_fingerprint"] = fingerprint
                result = MarketDataResult(
                    frame=completed,
                    feed=raw_result.feed,
                    adjustment=raw_result.adjustment,
                    data_fingerprint=fingerprint,
                    raw_source_fingerprint=raw_fp,
                    data_quality_audit=audit,
                    provider=raw_result.provider,
                    data_as_of=completed_as_of,
                    cache_status=raw_result.cache_status,
                    timestamp_kind="session_label",
                    post_close_grace_used=False,
                )
                self.cache.save(symbol, result, namespace=namespace)
                breaker.record_success(symbol)
                self._record_success(result, exchange=exchange_group)
                return result

            if allow_post_close_grace and grace_candidate is None:
                session_ctx: SessionContext | None = None
                try:
                    session_ctx = completed_session_context(exchange=cal_name)
                except ValueError:
                    session_ctx = None
                if (
                    session_ctx is not None
                    and session_ctx.in_post_close_grace
                    and session_ctx.required_session.date().isoformat() == required_session
                    and session_ctx.previous_session is not None
                    and completed_as_of == session_ctx.previous_session.date().isoformat()
                ):
                    completed.attrs.update(
                        {
                            "data_provider": raw_result.provider,
                            "data_feed": raw_result.feed,
                            "price_adjustment": raw_result.adjustment,
                            "raw_source_fingerprint": raw_fp,
                            "data_quality_audit": audit,
                            "post_close_grace_used": True,
                        }
                    )
                    fingerprint = daily_frame_fingerprint(completed)
                    completed.attrs["data_fingerprint"] = fingerprint
                    grace_candidate = MarketDataResult(
                        frame=completed,
                        feed=raw_result.feed,
                        adjustment=raw_result.adjustment,
                        data_fingerprint=fingerprint,
                        raw_source_fingerprint=raw_fp,
                        data_quality_audit=audit,
                        provider=raw_result.provider,
                        data_as_of=completed_as_of,
                        cache_status=raw_result.cache_status,
                        timestamp_kind="session_label",
                        post_close_grace_used=True,
                    )
                    continue

            stale_err = MarketDataServiceError(
                f"{provider.name} did not provide the latest completed session"
            )
            breaker.record_failure(symbol, MarketTransportError(str(stale_err)))
            self._record_failure(stale_err, exchange=exchange_group, provider_name=provider.name)
            failures.append(stale_err)
            continue

        if grace_candidate is not None:
            return grace_candidate

        if failures:
            error = MarketDataServiceError("All configured market-data providers are unavailable")
            error._provider_breaker_recorded = True  # type: ignore[attr-defined]
            self._record_failure(error, exchange=exchange_group)
            raise error from failures[-1]
        if unknowns:
            self._record_failure(unknowns[-1], exchange=exchange_group)
            not_found = MarketDataSymbolNotFound(f"No configured provider has data for {symbol}")
            not_found._provider_breaker_recorded = True  # type: ignore[attr-defined]
            raise not_found from unknowns[-1]
        error = MarketDataServiceError("No market-data provider is configured")
        error._provider_breaker_recorded = True  # type: ignore[attr-defined]
        self._record_failure(error, exchange=exchange_group)
        raise error

    def _record_success(self, result: MarketDataResult, *, exchange: str | None = None) -> None:
        if exchange is None:
            us_names = {p.name for p in self.providers}
            exchange = "US" if result.provider in us_names or not self.providers else "LSE"
        now_iso = datetime.now(UTC).isoformat()
        with self._lock:
            state = self._provider_state.setdefault(result.provider, {})
            state["last_success_session"] = result.data_as_of
            state["last_success_at"] = now_iso
            state["last_error"] = None
            if exchange == "LSE":
                self._uk_last_success_session = result.data_as_of
                return
            self._last_success_session = result.data_as_of
            self._last_success_at = now_iso
            self._last_provider = result.provider
            self._last_error = None

    def _record_failure(
        self,
        error: Exception,
        *,
        exchange: str = "US",
        provider_name: str | None = None,
    ) -> None:
        err_name = type(error).__name__
        with self._lock:
            if provider_name:
                state = self._provider_state.setdefault(provider_name, {})
                state["last_error"] = err_name
            if exchange == "LSE":
                return
            self._last_error = err_name
            if isinstance(error, MarketDataServiceError):
                self._last_success_session = None

    def readiness(self) -> tuple[bool, dict[str, Any]]:
        required_session = latest_completed_trading_session().date().isoformat()
        us_names = {p.name for p in self.providers if p.configured}
        fresh_entries = self.cache.fresh_entry_count(
            required_session=required_session,
            providers=us_names or None,
            exclude_lse_symbols=True,
        )
        configured = [provider.name for provider in self.providers if provider.configured]
        circuit_ready, circuit_report = self.us_circuit_readiness()
        with self._lock:
            recent_success = bool(
                self._last_success_session and self._last_success_session >= required_session
            )
            ready = bool((recent_success or fresh_entries > 0) and circuit_ready)
            return ready, {
                "status": "available" if ready else "unavailable",
                "configured_providers": configured,
                "exchange_providers": {
                    "US": configured,
                    "LSE": [p.name for p in self.uk_providers if p.configured],
                },
                "last_provider": self._last_provider,
                "last_success_at": self._last_success_at,
                "last_success_session": self._last_success_session,
                "required_session": required_session,
                "fresh_cache_entries": fresh_entries,
                "cache_persistence": "ephemeral" if self.cache.enabled else "disabled",
                "last_error": self._last_error,
                "providers": circuit_report.get("providers", {}),
            }


def build_market_data_service(settings: Any) -> MarketDataService:
    """Build the configured provider chain without exposing credentials."""
    requested = str(settings.market_data_provider).strip().lower()
    providers: list[MarketDataProvider] = []
    if requested in {"alpaca", "auto"}:
        providers.append(
            AlpacaProvider(
                key_id=settings.alpaca_api_key_id,
                secret_key=settings.alpaca_api_secret_key,
                base_url=settings.alpaca_data_base_url,
                feed=settings.alpaca_data_feed,
                adjustment=settings.alpaca_adjustment,
                timeout_seconds=settings.market_data_timeout_seconds,
            )
        )
    if requested in {"yahoo", "auto"} or settings.market_data_yahoo_fallback_enabled:
        providers.append(YahooProvider())
    cache_dir = Path(settings.market_data_cache_dir) if settings.market_data_cache_enabled else None
    return MarketDataService(
        providers,
        cache=MarketDataCache(cache_dir, enabled=settings.market_data_cache_enabled),
    )
