"""Tests for offline 286-ticker P/E (TTM) batch refresh and /api/v1/info integration."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api
from api import app
from routes import market as market_route
from routes.common import limiter
from scripts.refresh_universe_pe import (
    SCHEMA_VERSION,
    load_universe_tickers,
    refresh_universe_fundamentals,
    validate_pe_ttm,
)

CLIENT = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_pe_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(limiter, "enabled", False)
    market_route.clear_info_cache()
    market_route.clear_fundamentals_cache()
    yield
    market_route.clear_info_cache()
    market_route.clear_fundamentals_cache()


def test_validate_pe_ttm_bounds_and_rounding() -> None:
    assert validate_pe_ttm(31.456) == 31.46
    assert validate_pe_ttm("18.2") == 18.2
    assert validate_pe_ttm(0.35) is None
    assert validate_pe_ttm(0) is None
    assert validate_pe_ttm(-12.5) is None
    assert validate_pe_ttm(1200.0) is None
    assert validate_pe_ttm(float("nan")) is None
    assert validate_pe_ttm(float("inf")) is None
    assert validate_pe_ttm(True) is None
    assert validate_pe_ttm(None) is None


def test_refresh_universe_fundamentals_atomic_write_and_error_preservation(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "universe_fundamentals.json"
    t0 = datetime(2026, 11, 20, 6, 15, tzinfo=UTC)

    class _FakeYFRun1:
        def Ticker(self, symbol: str) -> Any:
            table = {
                "AAPL": {"trailingPE": 32.149},
                "MSFT": {"trailingPE": 35.8},
                "SHEL.L": {"trailingPE": 14.25},
                "LOSS": {"trailingPE": -5.0},
            }
            return SimpleNamespace(info=table[symbol])

    payload1 = refresh_universe_fundamentals(
        ["AAPL", "MSFT", "SHEL.L", "LOSS"],
        output_path=out_path,
        workers=2,
        yf_module=_FakeYFRun1(),
        now_utc=t0,
    )
    assert payload1["schema_version"] == SCHEMA_VERSION
    assert payload1["universe_count"] == 4
    assert payload1["coverage_count"] == 3
    assert payload1["tickers"]["AAPL"]["pe_ttm"] == 32.15
    assert payload1["tickers"]["SHEL.L"]["pe_ttm"] == 14.25
    assert "LOSS" not in payload1["tickers"]

    # Second run: MSFT errors transiently (preserved), SHEL.L becomes unavailable (dropped), AAPL updates.
    t1 = datetime(2026, 11, 25, 18, 0, tzinfo=UTC)

    class _FakeTickerRun2:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def info(self) -> dict[str, Any]:
            if self._symbol == "MSFT":
                raise RuntimeError("transient yahoo 429")
            if self._symbol == "SHEL.L":
                return {}
            if self._symbol == "AAPL":
                return {"trailingPE": 33.0}
            return {"trailingPE": None}

    class _FakeYFRun2:
        def Ticker(self, symbol: str) -> Any:
            return _FakeTickerRun2(symbol)

    payload2 = refresh_universe_fundamentals(
        ["AAPL", "MSFT", "SHEL.L", "LOSS"],
        output_path=out_path,
        workers=1,
        yf_module=_FakeYFRun2(),
        now_utc=t1,
    )
    assert payload2["tickers"]["AAPL"] == {"pe_ttm": 33.0, "updated_at": "2026-11-25T18:00:00Z"}
    assert payload2["tickers"]["MSFT"] == {"pe_ttm": 35.8, "updated_at": "2026-11-20T06:15:00Z"}
    assert "SHEL.L" not in payload2["tickers"]
    assert payload2["stats"]["preserved_on_error"] == 1


def test_info_serves_offline_batch_pe_without_touching_slow_info_and_handles_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_backend = tmp_path / "backend"
    (fake_backend / "routes").mkdir(parents=True)
    (fake_backend / "data").mkdir(parents=True)
    fund_path = fake_backend / "data" / "universe_fundamentals.json"
    fund_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": "yahoo_offline_batch",
                "updated_at": "2026-11-25T18:00:00Z",
                "tickers": {
                    "MSFT": {"pe_ttm": 34.56, "updated_at": "2026-11-25T18:00:00Z"},
                    "BADPE": {"pe_ttm": -9.0, "updated_at": "2026-11-25T18:00:00Z"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(market_route, "__file__", str(fake_backend / "routes" / "market.py"))
    market_route.clear_fundamentals_cache()

    class _FastOnlyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.fast_info = {
                "market_cap": 3_100_000_000_000.0,
                "previous_close": 420.0,
                "year_high": 468.0,
                "year_low": 366.0,
                "currency": "USD",
            }

        @property
        def info(self) -> dict[str, Any]:
            raise AssertionError("slow .info must never be called when fast_info succeeds")

    class _FakeYF:
        def Ticker(self, symbol: str) -> _FastOnlyTicker:
            return _FastOnlyTicker(symbol)

    monkeypatch.setattr(api, "yf", _FakeYF())

    # MSFT gets offline P/E without touching slow .info.
    msft = CLIENT.get("/api/v1/info?ticker=MSFT")
    assert msft.status_code == 200
    body = msft.json()
    assert body["peRatio"] == pytest.approx(34.56)
    assert body["peSource"] == "offline_batch"
    assert body["peUpdatedAt"] == "2026-11-25T18:00:00Z"

    # Ticker absent from fundamentals file stays cleanly nullable.
    aapl = CLIENT.get("/api/v1/info?ticker=AAPL")
    assert aapl.status_code == 200
    aapl_body = aapl.json()
    assert aapl_body["peRatio"] is None
    assert aapl_body["peSource"] is None
    assert aapl_body["peUpdatedAt"] is None


def test_committed_universe_fundamentals_snapshot_schema_and_coverage() -> None:
    path = REPO_ROOT / "backend" / "data" / "universe_fundamentals.json"
    assert path.is_file(), "backend/data/universe_fundamentals.json must be committed"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["universe_count"] == 286
    tickers = payload["tickers"]
    assert isinstance(tickers, dict)
    assert payload["coverage_count"] == len(tickers)
    assert len(tickers) >= 200, f"Expected broad universe P/E coverage, got {len(tickers)}"

    valid_universe = set(load_universe_tickers())
    for sym, entry in tickers.items():
        assert sym in valid_universe
        pe = entry["pe_ttm"]
        assert isinstance(pe, (int, float)) and math.isfinite(pe) and 1.0 <= pe < 1000.0
        assert isinstance(entry.get("updated_at"), str) and entry["updated_at"].endswith("Z")

    for flagship in ("AAPL", "MSFT", "NVDA", "JPM", "SHEL.L"):
        assert flagship in tickers, f"Flagship ticker {flagship} missing from P/E snapshot"
