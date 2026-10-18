from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from calendars import latest_completed_trading_session
from market_data.base import MarketDataProviderError, MarketDataResult
from market_data.cache import MarketDataCache
from market_data.normalization import normalize_daily_bars
from market_data.service import MarketDataService
from services.simple_forecast import build_dataset


def bars(index):
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 102.0,
            "Low": 99.0,
            "Close": 101.0,
            "Volume": np.arange(len(index)) + 1000,
        },
        index=index,
    )


@pytest.mark.parametrize("date", ["2026-03-30", "2026-10-23", "2026-10-26"])
def test_london_midnight_is_a_session_label_during_dst(date):
    index = pd.DatetimeIndex([date], tz="Europe/London")
    result = normalize_daily_bars(bars(index), provider="yahoo", symbol="BP.L")
    assert result.index[0].date().isoformat() == date


def test_timestamp_instant_maps_to_exchange_local_session():
    index = pd.DatetimeIndex(["2026-03-31T23:30:00Z"])
    result = normalize_daily_bars(
        bars(index), provider="test", symbol="BP.L", timestamp_kind="instant"
    )
    assert result.index[0].date().isoformat() == "2026-04-01"


@pytest.mark.parametrize(
    "column,value",
    [("Low", 101.5), ("High", 100.5), ("Open", 0), ("Volume", -1), ("Close", float("nan"))],
)
def test_impossible_ohlcv_is_rejected(column, value):
    data = bars(pd.bdate_range("2026-09-01", periods=3))
    data.loc[data.index[-1], column] = value
    with pytest.raises(MarketDataProviderError):
        normalize_daily_bars(data, provider="test", symbol="BP.L")


def test_uk_partial_session_uses_explicit_provider_and_same_cached_contract(monkeypatch, tmp_path):
    now = datetime(2026, 10, 2, 10, tzinfo=UTC)
    assert latest_completed_trading_session(now, exchange="LSE").date().isoformat() == "2026-10-01"
    monkeypatch.setattr(
        "market_data.service.latest_completed_trading_session",
        lambda **kwargs: pd.Timestamp("2026-10-01"),
    )
    calls = []

    def fetch(symbol, *, years):
        calls.append(symbol)
        return MarketDataResult(
            bars(pd.bdate_range("2026-09-01", "2026-10-02")),
            "yahoo",
            "2026-10-02",
            feed="yahoo_daily",
            adjustment="all",
        )

    uk = SimpleNamespace(name="yahoo", configured=True, fetch_daily_bars=fetch)
    us = SimpleNamespace(
        name="alpaca",
        configured=True,
        fetch_daily_bars=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("US provider cannot serve UK")
        ),
    )
    service = MarketDataService([us], uk_providers=[uk], cache=MarketDataCache(tmp_path))
    first = service.fetch_daily_bars("BP.L", years=8)
    second = service.fetch_daily_bars("BP.L", years=8)
    assert first.data_as_of == second.data_as_of == "2026-10-01"
    assert first.data_fingerprint == second.data_fingerprint
    assert first.feed == second.feed == "yahoo_daily"
    assert calls == ["BP.L"]


def test_zero_range_last_bar_keeps_feature_and_declared_origin():
    data = bars(pd.bdate_range("2022-01-03", periods=700))
    data["Close"] = 100 + np.sin(np.arange(700) / 9)
    data["High"] = 103
    data["Low"] = 97
    data.loc[data.index[-1], ["Open", "High", "Low", "Close"]] = 100
    dataset = build_dataset(data)
    assert dataset.features.index[-1] == data.index[-1]
    assert dataset.features.iloc[-1]["close_location"] == 0.5


def test_training_and_serving_features_match_on_same_provider_observations():
    from research.price_forecasting.gpu_pipeline import build_price_features

    from services.simple_forecast import build_features

    data = bars(pd.bdate_range("2022-01-03", periods=700))
    data["Close"] = 100 + np.sin(np.arange(700) / 9)
    data.loc[data.index[-1], ["Open", "High", "Low", "Close"]] = 100
    training = build_price_features(data)
    serving = build_features(data)
    pd.testing.assert_frame_equal(training, serving)
