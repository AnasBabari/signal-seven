"""Institutional market news service with multi-provider fallbacks.

Tries in order:
1. Alpaca news API (if credentials configured)
2. Yahoo Finance live news (headlines, summaries, publisher metadata)
3. U.S. SEC EDGAR material corporate filings (8-K, 10-Q, 10-K)
4. Curated local news archive cache (JSONL)

Returns rich headline objects with title, source, publication time, summary,
URL, and sentiment metrics (compound, polarity breakdown, bullish/bearish badges).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from cachetools import TTLCache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_cache: TTLCache = TTLCache(maxsize=256, ttl=10 * 60)
_news_cache = _cache
_negative_cache: TTLCache = TTLCache(maxsize=256, ttl=60)
_cik_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)
_cik_negative_cache: TTLCache = TTLCache(maxsize=256, ttl=60)
_lock = threading.RLock()

_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()


class _SecRateLimiter:
    """Thread-safe token interval limiter enforcing strictly < 10 req/s to data.sec.gov."""

    def __init__(self, min_interval_seconds: float = 0.12) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last_request_monotonic: float = 0.0

    def reset(self) -> None:
        with self._lock:
            self._last_request_monotonic = 0.0

    def acquire(self, *, max_wait_seconds: float | None = None) -> bool:
        with self._lock:
            now = time.monotonic()
            earliest = self._last_request_monotonic + self.min_interval_seconds
            wait = max(0.0, earliest - now)
            if max_wait_seconds is not None and wait > max_wait_seconds:
                return False
            if wait > 0:
                time.sleep(wait)
            self._last_request_monotonic = time.monotonic()
            return True


_sec_rate_limiter = _SecRateLimiter(0.12)


def _get_http_client() -> httpx.Client:
    global _http_client
    with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.Client(
                timeout=4.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return _http_client


def close_news_client() -> None:
    """Close the shared pooled news/SEC HTTP client on application shutdown."""
    global _http_client
    with _http_client_lock:
        if _http_client is not None:
            try:
                _http_client.close()
            finally:
                _http_client = None


_sentiment = SentimentIntensityAnalyzer()
_sentiment.lexicon.update(
    {
        # Bullish / Positive market terminology
        "surge": 2.5,
        "surges": 2.5,
        "surging": 2.5,
        "surged": 2.5,
        "soar": 2.8,
        "soars": 2.8,
        "soaring": 2.8,
        "soared": 2.8,
        "rally": 2.4,
        "rallies": 2.4,
        "rallying": 2.4,
        "rallied": 2.4,
        "jump": 2.0,
        "jumps": 2.0,
        "jumping": 2.0,
        "jumped": 2.0,
        "gain": 2.0,
        "gains": 2.0,
        "gaining": 2.0,
        "gained": 2.0,
        "outperform": 2.6,
        "outperforms": 2.6,
        "outperforming": 2.6,
        "outperformed": 2.6,
        "upgrade": 2.7,
        "upgrades": 2.7,
        "upgrading": 2.7,
        "upgraded": 2.7,
        "beat": 2.2,
        "beats": 2.2,
        "beating": 2.2,
        "bullish": 2.5,
        "bull": 2.2,
        "boom": 2.4,
        "booming": 2.4,
        "rebound": 2.0,
        "rebounds": 2.0,
        "breakout": 2.3,
        "upside": 2.0,
        "growth": 1.8,
        "record": 1.6,
        "buy": 1.8,
        "overweight": 2.0,
        # Bearish / Negative market terminology
        "plunge": -2.8,
        "plunges": -2.8,
        "plunging": -2.8,
        "plunged": -2.8,
        "slump": -2.5,
        "slumps": -2.5,
        "slumping": -2.5,
        "slumped": -2.5,
        "drop": -2.2,
        "drops": -2.2,
        "dropping": -2.2,
        "dropped": -2.2,
        "fall": -2.0,
        "falls": -2.0,
        "falling": -2.0,
        "fell": -2.0,
        "tumble": -2.6,
        "tumbles": -2.6,
        "tumbling": -2.6,
        "tumbled": -2.6,
        "crash": -3.0,
        "crashes": -3.0,
        "crashing": -3.0,
        "crashed": -3.0,
        "dive": -2.4,
        "dives": -2.4,
        "diving": -2.4,
        "dived": -2.4,
        "probe": -2.0,
        "probes": -2.0,
        "probing": -2.0,
        "probed": -2.0,
        "investigation": -2.0,
        "investigates": -2.0,
        "downgrade": -2.7,
        "downgrades": -2.7,
        "downgrading": -2.7,
        "downgraded": -2.7,
        "underperform": -2.4,
        "underperforms": -2.4,
        "miss": -2.2,
        "misses": -2.2,
        "missing": -2.2,
        "missed": -2.2,
        "shortfall": -2.2,
        "selloff": -2.5,
        "sell-off": -2.5,
        "selling": -1.8,
        "bearish": -2.5,
        "bear": -2.0,
        "loss": -2.3,
        "losses": -2.3,
        "lawsuit": -2.3,
        "warning": -2.0,
        "warns": -2.0,
        "warned": -2.0,
        "downside": -2.0,
        "underweight": -2.0,
    }
)

CIK_MAPPING: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "AMD": "0000002488",
    "COST": "0000909832",
    "QCOM": "0000804328",
    "JPM": "0000019617",
    "XOM": "0002115436",
    "WMT": "0000104169",
    "JNJ": "0000200406",
    "CAT": "0000018230",
    "KO": "0000021344",
    "NEE": "0000753308",
    "DIS": "0001744489",
    "BAC": "0000070858",
    "GE": "0000040545",
    "PLD": "0001045609",
    "NMM": "0001415921",
    "WBD": "0001437107",
}

_disk_cik_lock = threading.Lock()
_disk_cik_map: dict[str, str] = {}
_disk_cik_mtime_ns: int | None = None


def resolve_ticker_cik(symbol: str) -> str | None:
    """Resolve 10-digit zero-padded SEC CIK from universe_metadata.json or built-in fallback."""
    global _disk_cik_mtime_ns
    sym = symbol.strip().upper()
    if sym.endswith(".L"):
        return None
    if sym == "WBD":
        return "0001437107"
    path = Path(__file__).resolve().parents[1] / "data" / "universe_metadata.json"
    with _disk_cik_lock:
        try:
            st = path.stat()
            if st.st_mtime_ns != _disk_cik_mtime_ns:
                raw = json.loads(path.read_text(encoding="utf-8"))
                parsed: dict[str, str] = {}
                container = (
                    raw.get("tickers") or raw.get("entries") or raw.get("catalog") or raw
                    if isinstance(raw, dict)
                    else raw
                )
                if isinstance(container, dict):
                    for k, v in container.items():
                        if isinstance(v, dict) and v.get("cik"):
                            parsed[str(k).strip().upper()] = str(v["cik"]).strip().zfill(10)
                elif isinstance(container, list):
                    for item in container:
                        if isinstance(item, dict) and item.get("ticker") and item.get("cik"):
                            parsed[str(item["ticker"]).strip().upper()] = (
                                str(item["cik"]).strip().zfill(10)
                            )
                _disk_cik_map.clear()
                _disk_cik_map.update(parsed)
                _disk_cik_mtime_ns = st.st_mtime_ns
        except Exception:
            pass
        disk_cik = _disk_cik_map.get(sym)
    if disk_cik:
        return disk_cik.zfill(10)
    fallback = CIK_MAPPING.get(sym)
    return fallback.zfill(10) if fallback else None


SEC_MATERIAL_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A", "6-K"}

SEC_ITEM_DESCRIPTIONS: dict[str, str] = {
    "1.01": "Material Agreement",
    "1.02": "Termination of Agreement",
    "2.01": "Acquisition or Disposition",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Financial Obligation",
    "3.02": "Unregistered Sales of Equity",
    "4.01": "Change in Certifying Accountant",
    "5.01": "Change in Control",
    "5.02": "Departure/Appointment of Principal Officers",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Events",
}


def _safe_url(value: object) -> str | None:
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    return candidate if parsed.scheme in ("https", "http") and parsed.netloc else None


def _format_news_item(
    *,
    identifier: str | None = None,
    title: str,
    summary: str = "",
    source: str = "Financial Wire",
    published_at: str | datetime = "",
    url: str | None = None,
    provider: str = "unknown",
) -> dict[str, Any] | None:
    clean_title = str(title or "").strip()
    if not clean_title:
        return None

    clean_summary = str(summary or "").strip()
    if clean_summary == clean_title:
        clean_summary = ""

    pub_iso = ""
    if isinstance(published_at, datetime):
        if published_at.tzinfo is None:
            pub_iso = published_at.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
        else:
            pub_iso = published_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    elif published_at:
        raw = str(published_at).strip()
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            pub_iso = dt.astimezone(UTC).isoformat().replace("+00:00", "Z")
        except ValueError:
            pub_iso = raw

    if not pub_iso:
        pub_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    sentiment_target = f"{clean_title}. {clean_summary}" if clean_summary else clean_title
    scores = _sentiment.polarity_scores(sentiment_target)
    compound = float(scores["compound"])

    if compound > 0.15:
        label = "positive"
        badge = "Bullish"
    elif compound < -0.15:
        label = "negative"
        badge = "Bearish"
    else:
        label = "neutral"
        badge = "Neutral"

    if not identifier:
        identifier = hashlib.sha256(f"{clean_title}\0{pub_iso}".encode()).hexdigest()[:16]

    return {
        "id": identifier,
        "title": clean_title[:300],
        "headline": clean_title[:300],
        "source": str(source or "Financial Wire")[:100],
        "published_at": pub_iso,
        "summary": clean_summary[:1000],
        "url": _safe_url(url),
        "sentiment": compound,
        "sentiment_label": label,
        "sentiment_badge": badge,
        "sentiment_pos": float(scores["pos"]),
        "sentiment_neg": float(scores["neg"]),
        "sentiment_neu": float(scores["neu"]),
        "provider": provider,
    }


def _fetch_from_alpaca(
    symbol: str,
    key_id: str,
    secret_key: str,
    base_url: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    end = datetime.now(UTC)
    start = end - timedelta(days=14)
    client = _get_http_client()
    response = client.get(
        f"{base_url.rstrip('/')}/v1beta1/news",
        params={
            "symbols": symbol,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "sort": "desc",
            "limit": 10,
            "include_content": "false",
        },
        headers={
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        },
        timeout=max(0.1, min(float(timeout_seconds), 4.0)),
    )
    response.raise_for_status()
    payload = response.json()
    raw_items = payload.get("news", []) if isinstance(payload, dict) else []
    items: list[dict[str, Any]] = []
    for article in raw_items:
        if not isinstance(article, dict):
            continue
        headline = str(article.get("headline") or "").strip()
        summary = str(article.get("summary") or "").strip()
        item = _format_news_item(
            identifier=str(article.get("id") or ""),
            title=headline,
            summary=summary,
            source=str(article.get("source") or "Alpaca Feed"),
            published_at=str(article.get("created_at") or article.get("updated_at") or ""),
            url=article.get("url"),
            provider="alpaca",
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_from_yahoo(symbol: str) -> list[dict[str, Any]]:
    import yfinance as yf

    ticker_obj = yf.Ticker(symbol)
    raw_news: list[dict[str, Any]] = list(ticker_obj.news or [])
    try:
        search_obj = yf.Search(symbol, news_count=15)
        if search_obj.news:
            raw_news.extend(search_obj.news)
    except Exception:
        pass

    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for article in raw_news:
        if not isinstance(article, dict):
            continue
        content = article.get("content") if isinstance(article.get("content"), dict) else {}
        title = str(
            content.get("title") or article.get("headline") or article.get("title") or ""
        ).strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        summary = str(
            content.get("summary")
            or content.get("description")
            or article.get("summary")
            or article.get("description")
            or ""
        ).strip()

        pub_raw = (
            content.get("pubDate")
            or content.get("displayTime")
            or article.get("published_at")
            or article.get("created_at")
        )
        if not pub_raw and article.get("providerPublishTime"):
            try:
                pub_raw = datetime.fromtimestamp(
                    int(article["providerPublishTime"]), tz=UTC
                ).isoformat()
            except Exception:
                pub_raw = ""

        source = str(
            (content.get("provider") or {}).get("displayName")
            or article.get("source")
            or article.get("publisher")
            or "Yahoo Finance"
        )
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("previewUrl")
            or article.get("url")
            or article.get("link")
        )
        item_id = str(article.get("id") or content.get("id") or "")

        item = _format_news_item(
            identifier=item_id,
            title=title,
            summary=summary,
            source=source,
            published_at=str(pub_raw or ""),
            url=url,
            provider="yahoo",
        )
        if item is not None:
            items.append(item)
    return items


def _format_sec_filings(
    symbol: str, cik: str, raw_records: list[dict[str, str]]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for rec in raw_records[:5]:
        form = rec["form"]
        acc_time = rec["published_at"]
        acc_num = rec["accession_number"]
        filing_date = rec["filing_date"]
        desc = rec["description"]
        item_detail = rec["item_detail"]
        filing_url = rec["url"]

        title = f"{symbol} Form {form}{item_detail}: {desc}"
        summary = (
            f"Official U.S. SEC EDGAR filing by {symbol}. Form {form}{item_detail}. "
            f"Accession number: {acc_num}."
        )
        item = _format_news_item(
            identifier=f"sec-{acc_num}" if acc_num else None,
            title=title,
            summary=summary,
            source="SEC EDGAR",
            published_at=acc_time or filing_date,
            url=filing_url,
            provider="sec_edgar",
        )
        if item is not None:
            item["kind"] = "sec_filing"
            item["form"] = form
            item["filing_date"] = filing_date
            item["cik"] = cik
            items.append(item)
    return items


def _fetch_from_sec_edgar(symbol: str, *, timeout_seconds: float = 4.0) -> list[dict[str, Any]]:
    cik = resolve_ticker_cik(symbol)
    if not cik:
        return []

    with _lock:
        cached_records = _cik_cache.get(cik)
        if cached_records is not None:
            return _format_sec_filings(symbol, cik, cached_records)
        if cik in _cik_negative_cache:
            return []

    if not _sec_rate_limiter.acquire(max_wait_seconds=max(0.0, float(timeout_seconds))):
        return []

    try:
        from config import settings

        user_agent = str(
            getattr(
                settings, "sec_edgar_user_agent", "SignalSeven/2.0 (support@signalseven.example)"
            )
        ).strip()
    except Exception:
        user_agent = "SignalSeven/2.0 (support@signalseven.example)"

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    client = _get_http_client()
    try:
        response = client.get(
            url,
            headers=headers,
            timeout=max(0.1, min(float(timeout_seconds), 4.0)),
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        with _lock:
            _cik_negative_cache[cik] = True
        raise

    recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_times = recent.get("acceptanceDateTime", [])
    descriptions = recent.get("primaryDocDescription", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    items_list = recent.get("items", [])

    raw_records: list[dict[str, str]] = []
    cik_int = str(int(cik))
    for i in range(min(len(forms), 50)):
        form = str(forms[i]).strip()
        if form not in SEC_MATERIAL_FORMS:
            continue
        acc_time = str(acceptance_times[i] if i < len(acceptance_times) else "").strip()
        filing_date = str(filing_dates[i] if i < len(filing_dates) else "").strip()
        if not filing_date and acc_time:
            filing_date = acc_time[:10]
        acc_num = str(accession_numbers[i] if i < len(accession_numbers) else "").strip()
        primary_doc = str(primary_docs[i] if i < len(primary_docs) else "").strip()
        desc = str(descriptions[i] if i < len(descriptions) else form).strip() or form
        raw_item_code = str(items_list[i] if i < len(items_list) else "").strip()

        item_detail = ""
        if raw_item_code:
            matched = [
                SEC_ITEM_DESCRIPTIONS[code.strip()]
                for code in raw_item_code.split(",")
                if code.strip() in SEC_ITEM_DESCRIPTIONS
            ]
            if matched:
                item_detail = f" ({', '.join(matched)})"

        if acc_num and primary_doc:
            acc_compact = acc_num.replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_compact}/{primary_doc}"
            )
        else:
            filing_url = f"https://www.sec.gov/edgar/browse/?CIK={cik}"

        raw_records.append(
            {
                "form": form,
                "published_at": acc_time or (f"{filing_date}T00:00:00Z" if filing_date else ""),
                "filing_date": filing_date,
                "accession_number": acc_num,
                "description": desc,
                "item_detail": item_detail,
                "url": filing_url,
            }
        )
        if len(raw_records) >= 5:
            break

    with _lock:
        _cik_cache[cik] = raw_records
        _cik_negative_cache.pop(cik, None)
    return _format_sec_filings(symbol, cik, raw_records)


def _fetch_from_cache(symbol: str) -> list[dict[str, Any]]:
    candidates = [
        Path.cwd() / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
        Path(__file__).resolve().parents[2] / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
        Path(__file__).resolve().parents[1] / "data" / "news" / "alpaca" / f"{symbol}.jsonl",
    ]
    target_file = None
    for cand in candidates:
        if cand.is_file():
            target_file = cand
            break
    if not target_file:
        return []

    lines = [
        line.strip()
        for line in target_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items: list[dict[str, Any]] = []
    for line in reversed(lines[-15:]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = record.get("headline") or record.get("title") or ""
        item = _format_news_item(
            identifier=record.get("id"),
            title=title,
            summary=record.get("summary") or f"Market context record for {symbol}.",
            source=record.get("source") or "Market Wire",
            published_at=record.get("published_at") or record.get("collected_at") or "",
            url=record.get("url"),
            provider=record.get("provider") or "cached_archive",
        )
        if item is not None:
            items.append(item)
    return items


def get_exchange_market_close(symbol: str, session_date: str | None = None) -> datetime | None:
    """Resolve the regular market close UTC timestamp for the ticker's exchange."""
    try:
        import pandas as pd

        from calendars import (
            calendar_schedule,
            completed_session_context,
            resolve_calendar,
        )

        cal_name, _ = resolve_calendar(symbol)
        mcal_name = "LSE" if cal_name == "LSE" else "NYSE"
        if not session_date:
            close_utc = completed_session_context(exchange=mcal_name).market_close_utc
            return close_utc.to_pydatetime().astimezone(UTC)
        schedule = calendar_schedule(mcal_name, start_date=session_date, end_date=session_date)
        if not schedule.empty and "market_close" in schedule.columns:
            close_ts = schedule.iloc[0]["market_close"]
            if isinstance(close_ts, pd.Timestamp):
                return close_ts.to_pydatetime().astimezone(UTC)
    except Exception:
        pass
    return None


def _call_sec_edgar(symbol: str, remaining_seconds: float) -> list[dict[str, Any]]:
    if "timeout_seconds" in inspect.signature(_fetch_from_sec_edgar).parameters:
        return _fetch_from_sec_edgar(symbol, timeout_seconds=remaining_seconds)
    return _fetch_from_sec_edgar(symbol)


def fetch_recent_news(
    ticker: str,
    *,
    key_id: str | None = None,
    secret_key: str | None = None,
    base_url: str = "https://data.alpaca.markets",
    timeout_seconds: float = 4.0,
    include_sec_filings: bool = False,
) -> dict[str, Any]:
    """Fetch rich financial headlines with automatic fallback and <=4.0s shared deadline.

    Tries Alpaca -> Yahoo Finance -> SEC EDGAR -> Cached JSONL archive.
    Returns rich headline objects with title, source, publication time,
    summary, URL, sentiment metrics, and distinct ``sec_filings``.
    """
    symbol = ticker.strip().upper()
    with _lock:
        cached = _cache.get(symbol)
        if cached is not None:
            return dict(cached)
        neg_cached = _negative_cache.get(symbol)
        if neg_cached is not None:
            return dict(neg_cached)

    try:
        from config import settings

        configured_deadline = float(getattr(settings, "market_news_deadline_seconds", 4.0))
    except Exception:
        configured_deadline = 4.0
    budget_seconds = min(float(timeout_seconds), configured_deadline)
    deadline = time.monotonic() + max(0.05, budget_seconds)

    def _remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    items: list[dict[str, Any]] = []
    sec_filings: list[dict[str, Any]] = []
    provider_used = "none"

    # 1. Try Alpaca if credentials configured and not a foreign exchange (e.g. UK .L)
    clean_key = (key_id or "").strip()
    clean_secret = (secret_key or "").strip()
    if clean_key and clean_secret and not symbol.endswith(".L") and _remaining() > 0.02:
        try:
            items = _fetch_from_alpaca(symbol, clean_key, clean_secret, base_url, _remaining())
            if items:
                provider_used = "alpaca"
        except Exception as exc:
            logger.info("Alpaca news fetch failed for %s (%s); falling back to Yahoo", symbol, exc)

    # 2. Try Yahoo Finance fallback
    if not items and _remaining() > 0.02:
        try:
            items = _fetch_from_yahoo(symbol)
            if items:
                provider_used = "yahoo"
        except Exception as exc:
            logger.info(
                "Yahoo Finance news fetch failed for %s (%s); falling back to SEC EDGAR",
                symbol,
                exc,
            )

    # 3. Try SEC EDGAR fallback
    if not items and _remaining() > 0.02:
        try:
            items = _call_sec_edgar(symbol, _remaining())
            if items:
                provider_used = "sec_edgar"
                sec_filings = [dict(it) for it in items[:5]]
        except Exception as exc:
            logger.info(
                "SEC EDGAR news fetch failed for %s (%s); falling back to cached archive",
                symbol,
                exc,
            )

    # 4. Try Cached JSONL news archive fallback
    if not items:
        try:
            items = _fetch_from_cache(symbol)
            if items:
                provider_used = "cached_archive"
        except Exception as exc:
            logger.warning("Cache news load failed for %s: %s", symbol, exc)

    # Optionally enrich US tickers with SEC EDGAR filings within the shared deadline.
    if not sec_filings and not symbol.endswith(".L"):
        if include_sec_filings and _remaining() > 0.05:
            try:
                sec_filings = _call_sec_edgar(symbol, _remaining())[:5]
            except Exception as exc:
                logger.info("SEC EDGAR filing enrichment skipped for %s (%s)", symbol, exc)
        else:
            cik = resolve_ticker_cik(symbol)
            if cik:
                with _lock:
                    cached_sec = _cik_cache.get(cik)
                if cached_sec:
                    sec_filings = _format_sec_filings(symbol, cik, cached_sec)[:5]

    items.sort(key=lambda it: str(it.get("published_at") or ""), reverse=True)

    market_close_utc = get_exchange_market_close(symbol)
    for it in (*items, *sec_filings):
        pub_str = str(it.get("published_at") or "")
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).astimezone(UTC)
            if market_close_utc is not None:
                is_after = pub_dt > market_close_utc
                it["after_market_close"] = is_after
                it["session_timing"] = "after_hours" if is_after else "regular_hours"
        except Exception:
            it["after_market_close"] = False
            it["session_timing"] = "regular_hours"

    top_items = items[:10]
    top_filings = sec_filings[:5]
    has_content = bool(top_items or top_filings)
    result = {
        "status": "available" if has_content else "unavailable",
        "items": top_items,
        "articles": top_items,
        "sec_filings": top_filings,
        "role": "context_only",
        "used_by_model": False,
        "provider": provider_used,
        "exchange_close_utc": (
            market_close_utc.isoformat().replace("+00:00", "Z") if market_close_utc else None
        ),
        "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    with _lock:
        if has_content:
            _cache[symbol] = result
            _negative_cache.pop(symbol, None)
        else:
            _negative_cache[symbol] = result
    return dict(result)


def _extract_compound_score(item: dict[str, Any]) -> float | None:
    raw = item.get("sentiment")
    if isinstance(raw, dict):
        raw = raw.get("compound")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if -1.0 <= val <= 1.0 else None


def summarize_cached_news_context(ticker: str) -> dict[str, Any]:
    """Return a non-blocking aggregated news-sentiment summary from the warm cache.

    Never initiates upstream network calls so ``GET /api/v1/forecast`` preserves
    its cold latency SLO and M5-validated point/conformal predictions.
    """
    symbol = ticker.strip().upper()
    with _lock:
        cached = _cache.get(symbol)
        neg_cached = _negative_cache.get(symbol) if cached is None else None

    if cached is None:
        return {
            "status": "unavailable" if neg_cached is not None else "not_cached",
            "sentiment_label": "unavailable",
            "mean_compound": None,
            "headline_count": 0,
            "sec_filing_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "latest_published_at": None,
            "provider": "none",
            "used_in_point_model": False,
        }

    items = [it for it in (cached.get("items") or []) if isinstance(it, dict)]
    sec_filings = [it for it in (cached.get("sec_filings") or []) if isinstance(it, dict)]

    compounds: list[float] = []
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    for item in items:
        score = _extract_compound_score(item)
        if score is None:
            continue
        compounds.append(score)
        if score >= 0.08:
            bullish_count += 1
        elif score <= -0.08:
            bearish_count += 1
        else:
            neutral_count += 1

    mean_compound: float | None = None
    sentiment_label = "unavailable"
    if compounds:
        mean_compound = round(sum(compounds) / len(compounds), 4)
        if mean_compound >= 0.08:
            sentiment_label = "bullish"
        elif mean_compound <= -0.08:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

    timestamps = [
        str(entry.get("published_at") or "").strip()
        for entry in (*items, *sec_filings)
        if str(entry.get("published_at") or "").strip()
    ]
    latest_published_at = max(timestamps) if timestamps else None
    has_any = bool(items or sec_filings)

    return {
        "status": "available" if has_any else "unavailable",
        "sentiment_label": sentiment_label,
        "mean_compound": mean_compound,
        "headline_count": len(items),
        "sec_filing_count": len(sec_filings),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
        "latest_published_at": latest_published_at,
        "provider": str(cached.get("provider") or "none"),
        "used_in_point_model": False,
    }


def clear_news_cache() -> None:
    with _lock:
        _cache.clear()
        _negative_cache.clear()
        _cik_cache.clear()
        _cik_negative_cache.clear()
    _sec_rate_limiter.reset()
