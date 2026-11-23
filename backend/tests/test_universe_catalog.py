"""Unit tests for the unified 286-ticker tri-exchange metadata catalog (Milestone 06 PR 6A)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from market_data.universe_catalog import (
    clear_universe_catalog_cache,
    get_catalog_ticker_meta,
    get_ticker_cik,
    load_universe_catalog,
    normalize_sector,
)
from scripts.generate_universe_metadata import (
    GICS_SECTORS,
    LSE_TICKERS,
    NASDAQ_TICKERS,
    NYSE_TICKERS,
    build_universe_metadata,
    render_frontend_universe_js,
)
from services.simple_forecast import TICKER_METADATA as LEGACY_30_METADATA
from services.simple_forecast import get_ticker_meta

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_universe_catalog_counts_and_exchange_classification() -> None:
    clear_universe_catalog_cache()
    catalog = load_universe_catalog()

    assert catalog["total_count"] == 286
    assert catalog["us_count"] == 195
    assert catalog["lse_count"] == 91
    assert catalog["exchange_counts"] == {"NASDAQ": 98, "NYSE": 97, "LSE": 91}

    tickers = catalog["tickers"]
    assert len(tickers) == 286

    by_exchange = {"NASDAQ": 0, "NYSE": 0, "LSE": 0}
    expected_mics = {"NASDAQ": "XNAS", "NYSE": "XNYS", "LSE": "XLON"}
    for sym, entry in tickers.items():
        ex = entry["exchange_name"]
        assert entry["exchange"] == ex
        assert entry["exchange_mic"] == expected_mics[ex]
        by_exchange[ex] += 1
        if sym.endswith(".L"):
            assert ex == "LSE"
        else:
            assert ex in {"NASDAQ", "NYSE"}

    assert by_exchange == {"NASDAQ": 98, "NYSE": 97, "LSE": 91}

    # Verify the 87 NYSE symbols not in legacy 30-ticker TICKER_METADATA are classified as NYSE (not NASDAQ)
    legacy_nyse = {
        sym for sym, info in LEGACY_30_METADATA.items() if info["exchange_name"] == "NYSE"
    }
    assert len(legacy_nyse) == 10
    newly_classified_nyse = set(NYSE_TICKERS) - legacy_nyse
    assert len(newly_classified_nyse) == 87
    for sym in newly_classified_nyse:
        meta = get_catalog_ticker_meta(sym)
        assert meta["exchange_name"] == "NYSE"
        assert meta["exchange_mic"] == "XNYS"
        forecast_meta = get_ticker_meta(sym)
        assert forecast_meta["exchange_name"] == "NYSE"
        assert forecast_meta["exchange_mic"] == "XNYS"
        assert forecast_meta["name"] == meta["name"]

    for usd_lse in ("CPG.L", "IHG.L"):
        forecast_meta = get_ticker_meta(usd_lse)
        assert forecast_meta["exchange_name"] == "LSE"
        assert forecast_meta["exchange_mic"] == "XLON"
        assert forecast_meta["currency"] == "USD"
        assert forecast_meta["currency_symbol"] == "$"


def test_all_195_us_tickers_have_10_digit_sec_ciks_and_lse_are_none() -> None:
    catalog = load_universe_catalog()
    tickers = catalog["tickers"]

    us_ciks = []
    for sym, entry in tickers.items():
        cik = get_ticker_cik(sym)
        assert cik == entry["cik"]
        if sym.endswith(".L"):
            assert cik is None
        else:
            assert isinstance(cik, str)
            assert re.fullmatch(r"\d{10}", cik), f"Invalid 10-digit CIK for {sym}: {cik!r}"
            us_ciks.append(cik)

    assert len(us_ciks) == 195
    # Verify explicit overrides and flagship CIKs
    assert get_ticker_cik("WBD") == "0001437107"
    assert get_ticker_cik("GOOGL") == "0001652044"
    assert get_ticker_cik("AAPL") == "0000320193"
    assert get_ticker_cik("MSFT") == "0000789019"
    assert get_ticker_cik("NVDA") == "0001045810"
    assert get_ticker_cik("SHEL.L") is None
    assert get_ticker_cik("UNKNOWN_SYM") is None


def test_quote_units_and_lse_usd_overrides() -> None:
    catalog = load_universe_catalog()
    tickers = catalog["tickers"]

    for sym, entry in tickers.items():
        meta = get_catalog_ticker_meta(sym)
        if sym in {"CPG.L", "IHG.L"}:
            assert entry["currency"] == "USD"
            assert entry["currency_symbol"] == "$"
            assert meta["currency"] == "USD"
            assert meta["currency_symbol"] == "$"
            assert meta["exchange_name"] == "LSE"
            assert meta["exchange_mic"] == "XLON"
        elif sym.endswith(".L"):
            assert entry["currency"] == "GBp"
            assert entry["currency_symbol"] == "p"
            assert meta["currency"] == "GBp"
            assert meta["currency_symbol"] == "p"
        else:
            assert entry["currency"] == "USD"
            assert entry["currency_symbol"] == "$"
            assert meta["currency"] == "USD"
            assert meta["currency_symbol"] == "$"


def test_names_gics_sectors_and_sector_crosswalk() -> None:
    catalog = load_universe_catalog()
    valid_sectors = set(GICS_SECTORS)
    assert set(catalog["gics_sectors"]) == valid_sectors

    for sym, entry in catalog["tickers"].items():
        assert isinstance(entry["name"], str) and len(entry["name"].strip()) >= 3
        assert entry["name"].strip().upper() != sym.upper()
        assert entry["sector"] in valid_sectors
        assert isinstance(entry["industry"], str) and len(entry["industry"].strip()) >= 3

    # Verify legacy 30 ticker company names are preserved
    for sym, legacy in LEGACY_30_METADATA.items():
        assert get_catalog_ticker_meta(sym)["name"] == legacy["name"]

    # Verify sector crosswalk normalization
    assert normalize_sector("Technology") == "Information Technology"
    assert normalize_sector("Healthcare") == "Health Care"
    assert normalize_sector("Financial Services") == "Financials"
    assert normalize_sector("Consumer Cyclical") == "Consumer Discretionary"
    assert normalize_sector("Consumer Defensive") == "Consumer Staples"
    assert normalize_sector("Basic Materials") == "Materials"
    assert normalize_sector(None) == "Unknown"
    assert normalize_sector("—") == "Unknown"


def test_parity_across_manifests_backend_json_and_frontend_universe_js() -> None:
    root_manifest = json.loads(
        (REPO_ROOT / "data" / "tri_exchange" / "manifest_broad_300.json").read_text(
            encoding="utf-8"
        )
    )
    backend_manifest = json.loads(
        (REPO_ROOT / "backend" / "data" / "tri_exchange" / "manifest_broad_300.json").read_text(
            encoding="utf-8"
        )
    )
    backend_catalog_path = REPO_ROOT / "backend" / "data" / "universe_metadata.json"
    frontend_js_path = REPO_ROOT / "frontend" / "src" / "universe.js"

    disk_json_text = backend_catalog_path.read_text(encoding="utf-8")
    disk_js_text = frontend_js_path.read_text(encoding="utf-8")
    disk_catalog = json.loads(disk_json_text)

    # Exact generator determinism check
    generated_catalog = build_universe_metadata()
    expected_json_text = json.dumps(generated_catalog, indent=2, ensure_ascii=False) + "\n"
    expected_js_text = render_frontend_universe_js(generated_catalog)
    assert disk_json_text == expected_json_text
    assert disk_js_text == expected_js_text

    # Exact 286 ticker parity
    expected_tickers = root_manifest["valid_tickers"]
    assert len(expected_tickers) == 286
    assert backend_manifest["valid_tickers"] == expected_tickers
    assert disk_catalog["valid_tickers"] == expected_tickers
    assert list(disk_catalog["tickers"].keys()) == expected_tickers
    assert list(NASDAQ_TICKERS + NYSE_TICKERS + LSE_TICKERS) == expected_tickers

    # Verify required frontend exports exist in frontend/src/universe.js
    for required_export in (
        "export const NASDAQ_TICKERS =",
        "export const NYSE_TICKERS =",
        "export const LSE_TICKERS =",
        "export const ALL_VALID_TICKERS =",
        "export const ALL_TICKERS_SET =",
        "export const SUPPORTED_TICKERS =",
        "export const SUPPORTED_TICKER_SET =",
        "export const TICKER_METADATA =",
        "export function normalizeTicker(",
        "export function isSupportedTicker(",
        "export function getTickerMetadata(",
    ):
        assert required_export in disk_js_text
