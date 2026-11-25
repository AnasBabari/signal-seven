"""Offline batch refresher for 286-ticker trailing P/E (TTM) fundamentals.

Fetches ``trailingPE`` from Yahoo Finance strictly outside the live FastAPI request path
and writes ``backend/data/universe_fundamentals.json`` atomically so ``GET /api/v1/info``
can serve fast, zero-blocking P/E (TTM) metrics without calling slow ``yfinance.Ticker.info``
on user requests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = REPO_ROOT / "backend" / "data" / "universe_metadata.json"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "backend" / "data" / "tri_exchange" / "manifest_broad_300.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "backend" / "data" / "universe_fundamentals.json"

SCHEMA_VERSION = "universe-fundamentals-v1"
MIN_VALID_PE = 1.0
MAX_VALID_PE = 1000.0


def validate_pe_ttm(value: Any) -> float | None:
    """Validate and round a trailing P/E ratio to a sane positive finite float."""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num < MIN_VALID_PE or num >= MAX_VALID_PE:
        return None
    return round(num, 2)


def load_universe_tickers(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> list[str]:
    """Load the ordered 286-ticker universe list from backend catalog or manifest."""
    if catalog_path.is_file():
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        valid = payload.get("valid_tickers")
        if isinstance(valid, list) and valid:
            return [str(t).strip().upper() for t in valid if str(t).strip()]
        tickers = payload.get("tickers")
        if isinstance(tickers, dict) and tickers:
            return [str(t).strip().upper() for t in tickers if str(t).strip()]
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = payload.get("valid_tickers")
        if isinstance(valid, list) and valid:
            return [str(t).strip().upper() for t in valid if str(t).strip()]
    raise FileNotFoundError(
        f"Could not find universe ticker list at {catalog_path} or {manifest_path}"
    )


def fetch_single_ticker_pe(
    symbol: str,
    *,
    yf_module: Any = None,
    attempts: int = 2,
    retry_delay: float = 0.75,
) -> tuple[str, float | None, str]:
    """Fetch trailing P/E (TTM) for one symbol with isolated retry handling.

    Returns ``(symbol, pe_ttm, status)`` where status is ``"ok"``, ``"unavailable"``,
    or ``"error:<ExceptionType>"``.
    """
    if yf_module is None:
        import yfinance as yf_module  # type: ignore[import-untyped]

    last_error: str | None = None
    for attempt in range(max(1, attempts)):
        try:
            ticker_obj = yf_module.Ticker(symbol)
            raw_info = ticker_obj.info
            if not isinstance(raw_info, dict):
                return symbol, None, "unavailable"
            pe = validate_pe_ttm(raw_info.get("trailingPE"))
            if pe is not None:
                return symbol, pe, "ok"
            return symbol, None, "unavailable"
        except Exception as exc:  # noqa: BLE001
            last_error = f"error:{type(exc).__name__}"
            if attempt + 1 < attempts:
                time.sleep(retry_delay * (attempt + 1))
    return symbol, None, last_error or "error:Unknown"


def refresh_universe_fundamentals(
    tickers: list[str],
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    workers: int = 6,
    preserve_existing: bool = True,
    yf_module: Any = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Fetch trailing P/E across ``tickers`` and write atomic JSON payload."""
    now = now_utc or datetime.now(UTC)
    now_iso = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    existing_tickers: dict[str, dict[str, Any]] = {}
    if preserve_existing and output_path.is_file():
        try:
            prev = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and isinstance(prev.get("tickers"), dict):
                for sym, entry in prev["tickers"].items():
                    if isinstance(entry, dict):
                        val = validate_pe_ttm(entry.get("pe_ttm"))
                        if val is not None:
                            existing_tickers[str(sym).strip().upper()] = {
                                "pe_ttm": val,
                                "updated_at": str(
                                    entry.get("updated_at") or prev.get("updated_at") or now_iso
                                ),
                            }
        except Exception:  # noqa: BLE001
            existing_tickers = {}

    results: dict[str, tuple[float | None, str]] = {}
    max_workers = max(1, min(int(workers), max(1, len(tickers))))
    if max_workers == 1:
        for sym in tickers:
            s, pe, status = fetch_single_ticker_pe(sym, yf_module=yf_module)
            results[s] = (pe, status)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fetch_single_ticker_pe, sym, yf_module=yf_module): sym
                for sym in tickers
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    s, pe, status = future.result()
                    results[s] = (pe, status)
                except Exception as exc:  # noqa: BLE001
                    results[sym] = (None, f"error:{type(exc).__name__}")

    merged: dict[str, dict[str, Any]] = dict(existing_tickers) if preserve_existing else {}
    fetched_ok = 0
    unavailable = 0
    errors = 0
    preserved_on_error = 0

    for sym in tickers:
        pe, status = results.get(sym, (None, "error:Missing"))
        if pe is not None:
            merged[sym] = {"pe_ttm": pe, "updated_at": now_iso}
            fetched_ok += 1
        elif status == "unavailable":
            # Explicitly no positive trailing P/E (e.g., negative EPS); drop stale entry.
            merged.pop(sym, None)
            unavailable += 1
        else:
            errors += 1
            if preserve_existing and sym in existing_tickers:
                merged[sym] = existing_tickers[sym]
                preserved_on_error += 1

    ordered_tickers = {sym: merged[sym] for sym in sorted(merged.keys())}
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "yahoo_offline_batch",
        "updated_at": now_iso,
        "universe_count": len(tickers),
        "coverage_count": len(ordered_tickers),
        "stats": {
            "fetched_ok": fetched_ok,
            "unavailable": unavailable,
            "errors": errors,
            "preserved_on_error": preserved_on_error,
        },
        "tickers": ordered_tickers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, output_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh offline trailing P/E (TTM) fundamentals for the 286-ticker universe."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional comma-separated subset of tickers to refresh (default: all 286).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path (default: backend/data/universe_fundamentals.json).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Maximum concurrent Yahoo info workers (default: 6).",
    )
    parser.add_argument(
        "--no-preserve-existing",
        action="store_true",
        help="Do not carry forward prior valid P/E entries when a ticker errors transiently.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tickers.strip():
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_universe_tickers()

    started = time.perf_counter()
    payload = refresh_universe_fundamentals(
        tickers,
        output_path=args.output,
        workers=args.workers,
        preserve_existing=not args.no_preserve_existing,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output": str(args.output),
                "universe_count": payload["universe_count"],
                "coverage_count": payload["coverage_count"],
                "stats": payload["stats"],
                "updated_at": payload["updated_at"],
                "elapsed_s": round(elapsed, 2),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
