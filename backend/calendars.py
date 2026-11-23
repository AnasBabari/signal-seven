"""Exchange-aware future trading-date generation and session close context."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
from cachetools import TTLCache

SUFFIX_CALENDARS = {
    ".L": "LSE",
    ".SW": "SIX",
    ".TO": "TSX",
    ".AX": "ASX",
    ".HK": "HKEX",
}

_DEFAULT_GET_CALENDAR = mcal.get_calendar
_calendar_cache_lock = threading.Lock()
_calendar_cache: dict[str, Any] = {}
_calendar_schedule_locks: dict[str, threading.Lock] = {}

_SESSION_CONTEXT_TTL_SECONDS = 60
_session_cache_lock = threading.Lock()
_session_context_cache: TTLCache[tuple[str, int], SessionContext] = TTLCache(
    maxsize=32, ttl=_SESSION_CONTEXT_TTL_SECONDS
)
_BYPASS_SESSION_CACHE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "bypass_session_cache", default=False
)


def _schedule_lock(cal_name: str) -> threading.Lock:
    with _calendar_cache_lock:
        lock = _calendar_schedule_locks.get(cal_name)
        if lock is None:
            lock = threading.Lock()
            _calendar_schedule_locks[cal_name] = lock
        return lock


def get_cached_calendar(cal_name: str) -> Any:
    """Return a cached market calendar instance unless mcal.get_calendar is patched."""
    if mcal.get_calendar is not _DEFAULT_GET_CALENDAR:
        return mcal.get_calendar(cal_name)
    with _calendar_cache_lock:
        calendar = _calendar_cache.get(cal_name)
        if calendar is None:
            calendar = mcal.get_calendar(cal_name)
            _calendar_cache[cal_name] = calendar
            _calendar_schedule_locks.setdefault(cal_name, threading.Lock())
        return calendar


def calendar_schedule(cal_name: str, *, start_date: Any, end_date: Any) -> pd.DataFrame:
    """Run ``calendar.schedule`` under a per-calendar lock using the shared instance."""
    calendar = get_cached_calendar(cal_name)
    with _schedule_lock(cal_name):
        return calendar.schedule(start_date=start_date, end_date=end_date)


@contextmanager
def bypass_session_cache() -> Iterator[None]:
    """Force fresh session-boundary evaluation (used by strict collection routes)."""
    token = _BYPASS_SESSION_CACHE.set(True)
    try:
        yield
    finally:
        _BYPASS_SESSION_CACHE.reset(token)


def clear_calendar_caches() -> None:
    """Clear calendar instance and session-context TTL caches (for tests)."""
    with _calendar_cache_lock:
        _calendar_cache.clear()
        _calendar_schedule_locks.clear()
    with _session_cache_lock:
        _session_context_cache.clear()


@dataclass(frozen=True)
class SessionContext:
    """Completed-session boundary and post-close grace window for an exchange."""

    required_session: pd.Timestamp
    previous_session: pd.Timestamp | None
    market_close_utc: pd.Timestamp
    in_post_close_grace: bool


def completed_session_context(
    now: datetime | pd.Timestamp | None = None,
    exchange: str = "NYSE",
    grace_minutes: int = 25,
    *,
    use_cache: bool = True,
) -> SessionContext:
    """Return the latest completed session, prior session, and early-close-aware grace state."""
    upper_ex = exchange.upper()
    if upper_ex in ("LSE", "XLON", ".L"):
        cal_name = "LSE"
        tz = ZoneInfo("Europe/London")
        err_msg = "Could not determine the latest completed LSE session."
    else:
        cal_name = "NYSE"
        tz = ZoneInfo("America/New_York")
        err_msg = "Could not determine the latest completed NYSE session."

    can_cache = (
        now is None
        and use_cache
        and not _BYPASS_SESSION_CACHE.get()
        and mcal.get_calendar is _DEFAULT_GET_CALENDAR
    )
    cache_key = (cal_name, int(grace_minutes))
    if can_cache:
        with _session_cache_lock:
            cached = _session_context_cache.get(cache_key)
            if cached is not None:
                return cached

    instant = pd.Timestamp(now if now is not None else datetime.now(UTC))
    instant = instant.tz_localize(UTC) if instant.tzinfo is None else instant.tz_convert(UTC)

    local_now = instant.tz_convert(tz)
    schedule = calendar_schedule(
        cal_name,
        start_date=(local_now - timedelta(days=14)).date(),
        end_date=local_now.date(),
    )
    completed = schedule.loc[schedule["market_close"] <= instant]
    if completed.empty:
        raise ValueError(err_msg)
    required_session = pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()
    previous_session = (
        pd.Timestamp(completed.index[-2]).tz_localize(None).normalize()
        if len(completed) >= 2
        else None
    )
    market_close_utc = pd.Timestamp(completed["market_close"].iloc[-1]).tz_convert(UTC)
    elapsed = instant - market_close_utc
    in_post_close_grace = timedelta(0) <= elapsed <= timedelta(minutes=grace_minutes)
    context = SessionContext(
        required_session=required_session,
        previous_session=previous_session,
        market_close_utc=market_close_utc,
        in_post_close_grace=in_post_close_grace,
    )
    if can_cache:
        with _session_cache_lock:
            _session_context_cache[cache_key] = context
    return context


def latest_completed_trading_session(
    now: datetime | pd.Timestamp | None = None,
    exchange: str = "NYSE",
    *,
    use_cache: bool = True,
) -> pd.Timestamp:
    """Return the latest session whose regular close has passed for the given exchange."""
    return completed_session_context(
        now=now, exchange=exchange, use_cache=use_cache
    ).required_session


def resolve_calendar(ticker: str) -> tuple[str, str]:
    """Return (calendar implementation, public identifier)."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")
    upper = ticker.upper()
    if upper.endswith(("-USD", "-GBP", "-EUR", "-USDT")):
        return "24/7", "CRYPTO_24_7"
    for suffix, calendar in SUFFIX_CALENDARS.items():
        if upper.endswith(suffix):
            return calendar, calendar
    if "." in upper:
        return "NYSE", "NYSE_FALLBACK"
    return "NYSE", "NYSE"


def future_trading_dates(
    ticker: str, current_date: pd.Timestamp, days: int
) -> tuple[list[str], str]:
    """Generate exactly ``days`` future dates for the instrument calendar."""
    if days < 1:
        raise ValueError("Forecast days must be positive.")

    calendar_name, identifier = resolve_calendar(ticker)
    current = pd.Timestamp(current_date).tz_localize(None).normalize()
    if calendar_name == "24/7":
        dates = pd.date_range(current + timedelta(days=1), periods=days, freq="D")
        return dates.strftime("%Y-%m-%d").tolist(), identifier

    resolved_name = calendar_name
    try:
        calendar = get_cached_calendar(calendar_name)
    except Exception:
        resolved_name = "NYSE"
        calendar = get_cached_calendar("NYSE")
        identifier = "NYSE_FALLBACK"

    # Widen the search horizon if a holiday-heavy period yields too few sessions.
    future: list[pd.Timestamp] = []
    horizon = max(days * 4 + 14, 45)
    lock = _schedule_lock(resolved_name)
    for _ in range(3):
        with lock:
            schedule = calendar.schedule(
                start_date=current + timedelta(days=1), end_date=current + timedelta(days=horizon)
            )
        future = [d for d in schedule.index if d > current][:days]
        if len(future) == days:
            break
        horizon *= 2
    if len(future) != days:
        raise ValueError(f"Could not generate {days} dates for calendar {identifier}.")
    if any(a >= b for a, b in zip(future, future[1:], strict=False)):
        raise ValueError(f"Calendar {identifier} produced non-chronological session dates.")
    future_dates = [d.strftime("%Y-%m-%d") for d in future]
    return future_dates, identifier
