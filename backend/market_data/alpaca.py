"""Alpaca historical daily-bar provider."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from market_data.base import (
    MarketDataConfigurationError,
    MarketDataProviderError,
    MarketDataRateLimitError,
    MarketDataResult,
    MarketDataServiceError,
    MarketDataSymbolNotFound,
    MarketDataTimeoutError,
)
from market_data.normalization import normalize_daily_bars_with_audit


class AlpacaProvider:
    """Fetch split/dividend-adjusted US equity bars from Alpaca."""

    name = "alpaca"

    def __init__(
        self,
        *,
        key_id: str | None,
        secret_key: str | None,
        base_url: str = "https://data.alpaca.markets",
        feed: str = "iex",
        adjustment: str = "all",
        timeout_seconds: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.key_id = (key_id or "").strip()
        self.secret_key = (secret_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self.adjustment = adjustment
        self.timeout_seconds = timeout_seconds
        self._injected_client = client
        self._client: httpx.Client | None = client
        self._client_lock = threading.Lock()

    def _get_client(self) -> httpx.Client:
        if self._injected_client is not None:
            return self._injected_client
        with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.Client(
                    timeout=self.timeout_seconds,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
            return self._client

    def close(self) -> None:
        """Close the provider-owned pooled HTTP client without closing injected test clients."""
        if self._injected_client is not None:
            return
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.secret_key)

    @staticmethod
    def _response_reports_unknown_symbol(response: httpx.Response) -> bool:
        """Recognize Alpaca's symbol-level 400/422 errors without leaking text."""
        if response.status_code == 404:
            return True
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        text = " ".join(
            str(payload.get(key, ""))
            for key in ("code", "message", "error")
            if isinstance(payload, dict)
        ).lower()
        return any(
            phrase in text
            for phrase in ("symbol not found", "unknown symbol", "invalid symbol", "no data")
        )

    def fetch_raw_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult:
        """Fetch raw paginated bars from Alpaca with UTC instant timestamps preserved."""
        if not self.configured:
            raise MarketDataConfigurationError("Alpaca market-data credentials are not configured")
        end = datetime.now(UTC)
        start = end - timedelta(days=max(1, years) * 366)
        params: dict[str, Any] = {
            "timeframe": "1Day",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
            "limit": 10000,
        }
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }
        bars: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        pages = 0
        client = self._get_client()
        while True:
            pages += 1
            if pages > 10:
                raise MarketDataProviderError("Alpaca pagination exceeded the safety limit")
            try:
                response = client.get(
                    f"{self.base_url}/v2/stocks/{symbol}/bars",
                    params=params,
                    headers=headers,
                )
            except httpx.TimeoutException as err:
                raise MarketDataTimeoutError("Alpaca market-data request timed out") from err
            except httpx.HTTPError as err:
                raise MarketDataServiceError("Alpaca market-data transport failed") from err
            if response.status_code == 429:
                raise MarketDataRateLimitError("Alpaca market-data rate limit exceeded")
            if response.status_code in {
                400,
                404,
                422,
            } and self._response_reports_unknown_symbol(response):
                raise MarketDataSymbolNotFound(f"Alpaca has no data for {symbol}")
            if response.status_code in {401, 403}:
                raise MarketDataConfigurationError("Alpaca market-data credentials were rejected")
            if response.status_code >= 500:
                raise MarketDataServiceError("Alpaca market-data service is unavailable")
            if response.status_code >= 400:
                raise MarketDataServiceError("Alpaca rejected the server-generated bars request")
            try:
                payload = response.json()
            except ValueError as err:
                raise MarketDataProviderError("Alpaca returned malformed JSON") from err
            page = payload.get("bars") if isinstance(payload, dict) else None
            if isinstance(payload, dict) and "bars" in payload and page in (None, [], {}):
                break
            if not isinstance(page, list):
                raise MarketDataProviderError("Alpaca returned a malformed bars payload")
            bars.extend(item for item in page if isinstance(item, dict))
            if len(bars) > 100_000:
                raise MarketDataProviderError("Alpaca bars payload exceeded the safety limit")
            token = payload.get("next_page_token")
            if not token:
                break
            token_str = str(token)
            if token_str in seen_tokens:
                raise MarketDataProviderError("Alpaca returned a repeated pagination token")
            seen_tokens.add(token_str)
            params["page_token"] = token_str
        if not bars:
            raise MarketDataSymbolNotFound(f"Alpaca has no daily bars for {symbol}")
        try:
            frame = pd.DataFrame(
                {
                    "Open": [bar["o"] for bar in bars],
                    "High": [bar["h"] for bar in bars],
                    "Low": [bar["l"] for bar in bars],
                    "Close": [bar["c"] for bar in bars],
                    "Volume": [bar["v"] for bar in bars],
                },
                index=[bar["t"] for bar in bars],
            )
        except (KeyError, TypeError, ValueError) as err:
            raise MarketDataProviderError("Alpaca returned incomplete daily bars") from err
        return MarketDataResult(
            frame=frame,
            provider=self.name,
            feed=self.feed,
            adjustment=self.adjustment,
            data_as_of="",
            timestamp_kind="instant",
        )

    def fetch_daily_bars(self, symbol: str, *, years: int) -> MarketDataResult:
        raw = self.fetch_raw_daily_bars(symbol, years=years)
        normalized, raw_fp, audit = normalize_daily_bars_with_audit(
            raw.frame,
            provider=self.name,
            symbol=symbol,
            timestamp_kind="instant",
        )
        return MarketDataResult(
            frame=normalized,
            provider=self.name,
            feed=raw.feed,
            adjustment=raw.adjustment,
            data_as_of=normalized.index[-1].date().isoformat(),
            raw_source_fingerprint=raw_fp,
            data_quality_audit=audit,
            timestamp_kind="session_label",
        )

    def fetch_intraday_bars(
        self,
        symbol: str,
        *,
        timeframe: str = "15Min",
        limit: int = 96,
    ) -> list[dict[str, str | float]]:
        """Fetch latest-session IEX intraday bars (requiring >= 5 finite bars)."""
        if not self.configured:
            raise MarketDataConfigurationError("Alpaca market-data credentials are not configured")
        end = datetime.now(UTC)
        start = end - timedelta(days=5)
        params: dict[str, Any] = {
            "timeframe": timeframe,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
            "limit": max(int(limit) * 4, 200),
        }
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }
        client = self._get_client()
        try:
            response = client.get(
                f"{self.base_url}/v2/stocks/{symbol}/bars",
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as err:
            raise MarketDataTimeoutError("Alpaca intraday request timed out") from err
        except httpx.HTTPError as err:
            raise MarketDataServiceError("Alpaca intraday transport failed") from err

        if response.status_code == 429:
            raise MarketDataRateLimitError("Alpaca intraday rate limit exceeded")
        if response.status_code in {400, 404, 422} and self._response_reports_unknown_symbol(
            response
        ):
            raise MarketDataSymbolNotFound(f"Alpaca has no intraday data for {symbol}")
        if response.status_code in {401, 403}:
            raise MarketDataConfigurationError("Alpaca market-data credentials were rejected")
        if response.status_code >= 400:
            raise MarketDataServiceError("Alpaca intraday service is unavailable")
        try:
            payload = response.json()
        except ValueError as err:
            raise MarketDataProviderError("Alpaca returned malformed intraday JSON") from err

        raw_bars = payload.get("bars") if isinstance(payload, dict) else None
        if not isinstance(raw_bars, list) or not raw_bars:
            return []

        by_session: dict[str, list[dict[str, str | float]]] = {}
        for item in raw_bars:
            if not isinstance(item, dict):
                continue
            raw_t = item.get("t")
            raw_c = item.get("c")
            if not raw_t or raw_c is None:
                continue
            try:
                close_val = float(raw_c)
                ts = pd.Timestamp(str(raw_t))
                ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
                ny_date = ts.tz_convert("America/New_York").date().isoformat()
            except (TypeError, ValueError):
                continue
            if not pd.notna(close_val) or close_val <= 0 or close_val == float("inf"):
                continue
            by_session.setdefault(ny_date, []).append(
                {"t": ts.isoformat().replace("+00:00", "Z"), "c": close_val}
            )

        if not by_session:
            return []
        latest_session = max(by_session.keys())
        session_bars = by_session[latest_session][-int(limit) :]
        if len(session_bars) < 5:
            return []
        return session_bars
