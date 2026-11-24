"""Active volatility forecasting route.

This route is intentionally separate from ``volatility_v2``.  The v2 route is
the historical signed-release contract and remains fail-closed for clients
that explicitly depend on certification.  The active product route serves a
causal, reproducible statistical baseline while the offline learned-model
benchmark is being rebuilt.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
import secrets
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from calendars import bypass_session_cache
from data_pipeline import (
    MarketDataUnavailable,
    MarketTransportError,
    UnknownTickerError,
    _download_ohlcv,
)
from features.market import MarketContextUnavailable
from routes.common import limiter, validate_ticker
from services.forecast_ledger import (
    LedgerConflictError,
    LedgerUnavailableError,
    compute_forecast_fingerprint,
    get_current_code_commit,
    get_forecast_ledger,
)
from services.live_collection import (
    LIVE_MODEL_POLICY_V1,
    LIVE_MODEL_POLICY_V2,
    LIVE_START_DATE,
    LIVE_UNIVERSE_VERSION,
    LIVE_UNIVERSE_VERSION_V2,
    resolve_live_cohort_version,
    validate_live_collection_item,
)
from services.live_volatility import (
    GPU_G3_MODEL,
    SUPPORTED_BASELINES,
    G3BatchContext,
    build_live_volatility_forecast,
    prepare_g3_batch_context,
)
from services.volatility_contract import (
    VOLATILITY_FEATURE_SET_VERSION,
    VOLATILITY_MODEL_POLICY_VERSION,
    VOLATILITY_MODEL_VERSION,
    resolve_auto_model_policy,
    resolve_volatility_policy_name,
    validate_volatility_horizon,
)
from services.volatility_snapshot import VOLATILITY_HORIZONS, build_volatility_inference_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(tags=["volatility"])

MARKET_DATA_UNAVAILABLE_BODY = {
    "error": "MARKET_DATA_UNAVAILABLE",
    "message": "Current market data is temporarily unavailable. Please try again later.",
}

FORECAST_LEDGER_UNAVAILABLE_BODY = {
    "error": "FORECAST_LEDGER_UNAVAILABLE",
    "message": "The forecast was not recorded because the forecast ledger is temporarily unavailable.",
}

COLLECTOR_TOKEN_ENV = "FORECAST_COLLECTOR_TOKEN"


@dataclass(frozen=True)
class _PreparedForecast:
    result: dict[str, Any]
    data_as_of: str
    record_kwargs: dict[str, Any]
    post_close_grace_used: bool = False


def require_collector_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Require the server-side collector bearer token without logging it."""
    configured = os.getenv(COLLECTOR_TOKEN_ENV, "").strip().strip("\"'")
    if not configured:
        raise HTTPException(status_code=503, detail="Collector authentication is unavailable.")
    scheme, separator, candidate = (authorization or "").partition(" ")
    candidate = candidate.strip().strip("\"'")
    if not authorization:
        scheme_kind = "none"
    elif separator == " " and scheme.lower() == "bearer":
        scheme_kind = "bearer"
    else:
        scheme_kind = "other"
    authenticated = (
        scheme_kind == "bearer"
        and bool(candidate)
        and secrets.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))
    )
    if not authenticated:
        logger.warning(
            "Collector auth failed: scheme=%s, has_candidate=%s",
            scheme_kind,
            bool(candidate),
        )
        raise HTTPException(
            status_code=401,
            detail="Collector authentication failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _load_snapshot(symbol: str, *, allow_post_close_grace: bool) -> Any:
    if (
        "allow_post_close_grace"
        in inspect.signature(build_volatility_inference_snapshot).parameters
    ):
        return build_volatility_inference_snapshot(
            symbol, allow_post_close_grace=allow_post_close_grace
        )
    return build_volatility_inference_snapshot(symbol)


def _prepare_forecast_from_snapshot(
    snapshot: Any,
    symbol: str,
    horizon: int,
    requested_model: str,
    *,
    g3_context: G3BatchContext | None = None,
) -> _PreparedForecast:
    if (
        g3_context is not None
        and "g3_context" in inspect.signature(build_live_volatility_forecast).parameters
    ):
        forecast_result = build_live_volatility_forecast(
            snapshot,
            horizon=horizon,
            model=requested_model,
            g3_context=g3_context,
        )
    else:
        forecast_result = build_live_volatility_forecast(
            snapshot,
            horizon=horizon,
            model=requested_model,
        )
    code_commit = get_current_code_commit()
    data_as_of = str(getattr(snapshot, "data_as_of", None) or snapshot.origin_date)
    data_provider = getattr(snapshot, "data_provider", "unknown")
    active_policy = resolve_auto_model_policy()
    evidence = forecast_result.setdefault("evidence", {})
    evidence.update(
        {
            "code_commit": code_commit,
            "model_policy_version": VOLATILITY_MODEL_POLICY_VERSION,
            "default_policy": resolve_volatility_policy_name(),
            "auto_model_policy": {
                str(policy_horizon): policy_model
                for policy_horizon, policy_model in active_policy.items()
            },
        }
    )

    future_dates = forecast_result.get("forecast", {}).get("future_dates", [])
    target_date = future_dates[-1] if future_dates else ""
    quantiles = forecast_result.get("forecast", {}).get("price_quantiles", {})
    p05 = quantiles.get("p05", [snapshot.origin_close])[-1]
    p95 = quantiles.get("p95", [snapshot.origin_close])[-1]
    pred_vol = float(forecast_result.get("forecast", {}).get("predicted_volatility", 0.0))
    recent_values = snapshot.baseline_candidates["rolling_c2c_20"]
    recent_vol = math.sqrt(252.0 * float(recent_values[0]))
    evidence["volatility_units"] = "annualized_sigma_fraction"
    evidence.update(
        {
            "data_fingerprint": getattr(snapshot, "data_fingerprint", ""),
            "data_feed": getattr(snapshot, "data_feed", "unknown"),
            "price_adjustment": getattr(snapshot, "price_adjustment", "unknown"),
            "feature_origin": snapshot.origin_date,
        }
    )
    evidence["ledger_metric_version"] = "annualized_sigma_v1"
    active_model = str(forecast_result.get("forecast", {}).get("model", requested_model))
    record_kwargs = {
        "forecast_date": snapshot.origin_date,
        "ticker": symbol,
        "horizon": horizon,
        "target_date": target_date,
        "model_name": active_model,
        "predicted_volatility": pred_vol,
        "recent_realized_volatility": recent_vol,
        "origin_price": float(snapshot.origin_close),
        "lower_scenario_price": float(p05),
        "upper_scenario_price": float(p95),
        "record_source": "live",
        "model_version": VOLATILITY_MODEL_VERSION + ":annualized_sigma_v1",
        "feature_set_version": VOLATILITY_FEATURE_SET_VERSION,
        "code_commit": code_commit,
        "data_as_of": data_as_of,
        "data_provider": (
            f"{data_provider}|feed={getattr(snapshot, 'data_feed', 'unknown')}|adjustment={getattr(snapshot, 'price_adjustment', 'unknown')}|sha256={snapshot.data_fingerprint}"
            if getattr(snapshot, "data_fingerprint", "")
            else data_provider
        ),
    }
    grace_used = bool(getattr(snapshot, "post_close_grace_used", False))
    evidence["post_close_grace_used"] = grace_used
    evidence["forecast_fingerprint"] = compute_forecast_fingerprint(**record_kwargs)
    return _PreparedForecast(
        result=forecast_result,
        data_as_of=data_as_of,
        record_kwargs=record_kwargs,
        post_close_grace_used=grace_used,
    )


def _prepare_forecast(
    symbol: str,
    horizon: int,
    requested_model: str,
    *,
    allow_post_close_grace: bool = False,
) -> _PreparedForecast:
    snapshot = _load_snapshot(symbol, allow_post_close_grace=allow_post_close_grace)
    return _prepare_forecast_from_snapshot(snapshot, symbol, horizon, requested_model)


def _execute_forecast(
    symbol: str,
    horizon: int,
    requested_model: str,
    *,
    record_live: bool,
    live_universe_version: str = LIVE_UNIVERSE_VERSION,
) -> Any:
    try:
        allow_grace = not record_live
        if "allow_post_close_grace" in inspect.signature(_prepare_forecast).parameters:
            prepared = _prepare_forecast(
                symbol, horizon, requested_model, allow_post_close_grace=allow_grace
            )
        else:
            prepared = _prepare_forecast(symbol, horizon, requested_model)
        evidence = prepared.result.setdefault("evidence", {})
        if not record_live:
            evidence["ledger_write"] = "disabled_public_preview"
            if prepared.post_close_grace_used:
                return JSONResponse(content=prepared.result, headers={"Cache-Control": "no-store"})
            return prepared.result

        if date.fromisoformat(prepared.data_as_of) < LIVE_START_DATE:
            return JSONResponse(
                status_code=409,
                content={
                    "error": "LIVE_COLLECTION_NOT_ELIGIBLE",
                    "message": "The market snapshot predates the frozen live collection start.",
                },
            )
        policy_map = (
            LIVE_MODEL_POLICY_V2
            if live_universe_version == LIVE_UNIVERSE_VERSION_V2
            else LIVE_MODEL_POLICY_V1
        )
        expected_model = policy_map[horizon]
        if prepared.record_kwargs["model_name"] != expected_model:
            logger.error("Frozen model policy mismatch for %s horizon %s", symbol, horizon)
            return JSONResponse(
                status_code=409,
                content={
                    "error": "LIVE_COLLECTION_POLICY_MISMATCH",
                    "message": "The forecast did not match the frozen live model policy.",
                },
            )
        ledger = get_forecast_ledger()
        record = ledger.record_forecast(**prepared.record_kwargs)
        evidence.update(
            {
                "ledger_write": "recorded_live",
                "live_universe_version": live_universe_version,
                "ledger_record_id": record.id,
            }
        )
        return prepared.result
    except LedgerConflictError:
        logger.error("Conflicting immutable ledger evidence for %s", symbol)
        return JSONResponse(
            status_code=409,
            content={
                "error": "FORECAST_LEDGER_CONFLICT",
                "message": "A forecast with this origin already exists with different evidence.",
            },
        )
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except UnknownTickerError as err:
        raise HTTPException(
            status_code=404, detail="No market data is available for this ticker."
        ) from err
    except (MarketTransportError, MarketContextUnavailable) as err:
        logger.warning("Market data unavailable for %s: %s", symbol, type(err).__name__)
        return JSONResponse(status_code=503, content=MARKET_DATA_UNAVAILABLE_BODY)
    except MarketDataUnavailable as err:
        raise HTTPException(
            status_code=422,
            detail="Not enough valid market history is available for this ticker.",
        ) from err
    except ValueError as err:
        logger.info("Volatility request rejected for %s: %s", symbol, err)
        raise HTTPException(
            status_code=422,
            detail="The volatility snapshot could not be formed from valid market data.",
        ) from err
    except Exception as err:
        logger.exception("Volatility forecast failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail="Volatility forecasting is temporarily unavailable. Please retry shortly.",
        ) from err


@router.get("/api/v1/volatility/forecast")
@limiter.limit("30/minute")
def volatility_forecast(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int = Query(default=5, ge=1, le=20),
    model: str = Query(default="auto"),
) -> Any:
    """Return a read-only causal volatility preview for one trading horizon.

    The endpoint never accepts client-provided features or model paths.  It
    downloads the latest validated OHLCV snapshot, computes features through
    the last observed session, and applies the selected train-free baseline.
    Public GET requests never mutate the forecast ledger.
    """

    symbol = validate_ticker(ticker)
    try:
        horizon = validate_volatility_horizon(horizon)
    except ValueError as err:
        raise HTTPException(
            status_code=400,
            detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
        ) from err
    requested_model = str(model).strip().lower()
    if requested_model not in SUPPORTED_BASELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {list(SUPPORTED_BASELINES)}.",
        )
    return _execute_forecast(symbol, horizon, requested_model, record_live=False)


def _parse_batch_horizons(raw_horizons: str) -> list[int]:
    parts = [part.strip() for part in raw_horizons.split(",") if part.strip()]
    if not parts:
        raise HTTPException(
            status_code=400,
            detail=f"Horizons must be a comma-separated subset of {list(VOLATILITY_HORIZONS)}.",
        )
    parsed: list[int] = []
    for part in parts:
        try:
            value = validate_volatility_horizon(int(part))
        except (TypeError, ValueError) as err:
            raise HTTPException(
                status_code=400,
                detail=f"Horizons must be a comma-separated subset of {list(VOLATILITY_HORIZONS)}.",
            ) from err
        if value not in parsed:
            parsed.append(value)
    return parsed


@router.get("/api/v1/volatility/forecast/batch")
@limiter.limit("30/minute")
def volatility_forecast_batch(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizons: str = Query(default="5,10,20", min_length=1, max_length=40),
    model: str = Query(default="auto"),
) -> Any:
    """Return read-only causal volatility forecasts across multiple horizons in one pass."""
    symbol = validate_ticker(ticker)
    parsed_horizons = _parse_batch_horizons(horizons)
    requested_model = str(model).strip().lower()
    if requested_model not in SUPPORTED_BASELINES:
        raise HTTPException(
            status_code=400,
            detail=f"Model must be one of {list(SUPPORTED_BASELINES)}.",
        )

    try:
        snapshot = _load_snapshot(symbol, allow_post_close_grace=True)
    except UnknownTickerError as err:
        raise HTTPException(
            status_code=404, detail="No market data is available for this ticker."
        ) from err
    except (MarketTransportError, MarketContextUnavailable) as err:
        logger.warning("Market data unavailable for %s: %s", symbol, type(err).__name__)
        return JSONResponse(status_code=503, content=MARKET_DATA_UNAVAILABLE_BODY)
    except MarketDataUnavailable as err:
        raise HTTPException(
            status_code=422,
            detail="Not enough valid market history is available for this ticker.",
        ) from err
    except ValueError as err:
        logger.info("Volatility batch request rejected for %s: %s", symbol, err)
        raise HTTPException(
            status_code=422,
            detail="The volatility snapshot could not be formed from valid market data.",
        ) from err
    except Exception as err:
        logger.exception("Volatility batch snapshot failed for %s", symbol)
        raise HTTPException(
            status_code=503,
            detail="Volatility forecasting is temporarily unavailable. Please retry shortly.",
        ) from err

    g3_context: G3BatchContext | None = None
    active_policy = resolve_auto_model_policy()
    needs_g3 = requested_model == GPU_G3_MODEL or (
        requested_model == "auto"
        and any(active_policy.get(h) == GPU_G3_MODEL for h in parsed_horizons)
    )
    if needs_g3:
        try:
            g3_context = prepare_g3_batch_context(snapshot)
        except ValueError as err:
            logger.info("Volatility batch G3 context rejected for %s: %s", symbol, err)
            raise HTTPException(
                status_code=422,
                detail="The volatility snapshot could not be formed from valid market data.",
            ) from err

    forecasts: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}
    has_internal_error = False

    for horizon in parsed_horizons:
        try:
            prepared = _prepare_forecast_from_snapshot(
                snapshot,
                symbol,
                horizon,
                requested_model,
                g3_context=g3_context,
            )
            prepared.result.setdefault("evidence", {})["ledger_write"] = "disabled_public_preview"
            forecasts[str(horizon)] = prepared.result
        except ValueError as err:
            logger.info("Volatility batch horizon %s rejected for %s: %s", horizon, symbol, err)
            errors[str(horizon)] = {
                "code": "HORIZON_UNAVAILABLE",
                "message": "The volatility forecast could not be formed for this horizon.",
            }
        except Exception:
            logger.exception("Volatility batch forecast failed for %s horizon %s", symbol, horizon)
            has_internal_error = True
            errors[str(horizon)] = {
                "code": "FORECAST_ERROR",
                "message": "Volatility forecasting is temporarily unavailable for this horizon.",
            }

    if not forecasts:
        if has_internal_error:
            raise HTTPException(
                status_code=503,
                detail="Volatility forecasting is temporarily unavailable. Please retry shortly.",
            )
        raise HTTPException(
            status_code=422,
            detail="The volatility snapshot could not be formed from valid market data.",
        )

    data_as_of = str(getattr(snapshot, "data_as_of", None) or snapshot.origin_date)
    grace_used = bool(getattr(snapshot, "post_close_grace_used", False))
    payload = {
        "ticker": symbol,
        "as_of": data_as_of,
        "requested_model": requested_model,
        "horizons": parsed_horizons,
        "forecasts": forecasts,
        "errors": errors,
        "post_close_grace_used": grace_used,
    }
    if grace_used:
        return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
    return payload


@router.post("/api/v1/volatility/collect")
@limiter.limit("30/minute")
def collect_volatility_forecast(
    request: Request,
    _authorization: Annotated[None, Depends(require_collector_auth)],
    ticker: str = Query(min_length=1, max_length=12),
    horizon: int = Query(ge=1, le=20),
    cohort: Literal["v1", "v2", "live_universe_v1", "live_v2_2026_11"] = Query(default="v1"),
) -> Any:
    """Record one authenticated item from the frozen live collection contract."""
    symbol = validate_ticker(ticker)
    try:
        cohort_version = resolve_live_cohort_version(cohort)
        if cohort_version == LIVE_UNIVERSE_VERSION:
            symbol, normalized_horizon = validate_live_collection_item(symbol, horizon)
        else:
            symbol, normalized_horizon = validate_live_collection_item(
                symbol, horizon, cohort=cohort_version
            )
    except (TypeError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    policy_map = (
        LIVE_MODEL_POLICY_V2 if cohort_version == LIVE_UNIVERSE_VERSION_V2 else LIVE_MODEL_POLICY_V1
    )
    pinned_model = policy_map[normalized_horizon]
    with bypass_session_cache():
        return _execute_forecast(
            symbol,
            normalized_horizon,
            pinned_model,
            record_live=True,
            live_universe_version=cohort_version,
        )


@router.get("/api/v1/volatility/ledger/summary")
@limiter.limit("30/minute")
def get_volatility_ledger_summary_route(
    request: Request,
    ticker: str | None = Query(default=None, min_length=1, max_length=12),
    cohort: Literal["v1", "v2", "live_universe_v1", "live_v2_2026_11"] | None = Query(default=None),
    record_source: Literal["live", "historical_replay"] | None = Query(default=None),
    horizon: int | None = Query(default=None, ge=1, le=20),
) -> Any:
    """Return indexed SQL aggregate ledger summary grouped by cohort, source, horizon, policy, and settlement."""
    symbol = validate_ticker(ticker) if ticker is not None else None
    if horizon is not None:
        try:
            horizon = validate_volatility_horizon(horizon)
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
            ) from err
    cohort_version = resolve_live_cohort_version(cohort) if cohort is not None else None
    try:
        ledger = get_forecast_ledger()
        groups = ledger.get_ledger_summary(
            ticker=symbol,
            cohort=cohort_version,
            record_source=record_source,
            horizon=horizon,
        )
        return {
            "ticker": symbol,
            "cohort": cohort_version,
            "record_source": record_source,
            "horizon": horizon,
            "total_groups": len(groups),
            "groups": groups,
        }
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while reading summary")
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Could not read forecast ledger summary")
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


@router.get("/api/v1/volatility/ledger")
@limiter.limit("30/minute")
def get_volatility_ledger_route(
    request: Request,
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
    horizon: int | None = Query(default=None, ge=1, le=20),
    record_source: Literal["live", "historical_replay"] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Retrieve historical forecast ledger and empirical accuracy track record."""
    symbol = validate_ticker(ticker)
    if horizon is not None:
        try:
            horizon = validate_volatility_horizon(horizon)
        except ValueError as err:
            raise HTTPException(
                status_code=400,
                detail=f"Horizon must be one of {list(VOLATILITY_HORIZONS)}.",
            ) from err
    try:
        ledger = get_forecast_ledger()
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while reading %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)

    try:
        if hasattr(ledger, "get_ledger_overview"):
            overview = ledger.get_ledger_overview(
                ticker=symbol, horizon=horizon, record_source=record_source, limit=limit
            )
            entries = overview["entries"]
            live_metrics = overview["live_track_record"]
            replay_metrics = overview["replay_track_record"]
        else:
            entries = ledger.get_ledger_entries(
                ticker=symbol, horizon=horizon, record_source=record_source, limit=limit
            )
            live_metrics = ledger.get_track_record_metrics(
                ticker=symbol, horizon=horizon, record_source="live"
            )
            replay_metrics = ledger.get_track_record_metrics(
                ticker=symbol, horizon=horizon, record_source="historical_replay"
            )

        return {
            "ticker": symbol,
            "horizon": horizon,
            "live_track_record": live_metrics,
            "replay_track_record": replay_metrics,
            "track_record": live_metrics,
            "entries": entries,
        }
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while reading %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Could not read forecast ledger for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


@router.get("/api/v1/volatility/export-ledger")
@limiter.limit("10/minute")
def export_live_volatility_ledger(
    request: Request,
    response: Response,
    _authorization: Annotated[None, Depends(require_collector_auth)],
) -> Any:
    """Return the complete deterministic live track for authenticated backups."""
    try:
        ledger = get_forecast_ledger()
        records = [record.to_dict() for record in ledger.export_records(record_source="live")]
        response.headers["Cache-Control"] = "no-store"
        return {
            "record_source": "live",
            "storage_backend": ledger.storage_kind,
            "entries": records,
        }
    except LedgerUnavailableError:
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Could not export the live volatility ledger")
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


@router.post("/api/v1/volatility/score-ledger")
@limiter.limit("10/minute")
def score_volatility_ledger_route(
    request: Request,
    _authorization: Annotated[None, Depends(require_collector_auth)],
    ticker: str = Query(default="AAPL", min_length=1, max_length=12),
) -> dict[str, Any]:
    """Score pending ledger forecasts against subsequent realized volatility."""
    symbol = validate_ticker(ticker)
    try:
        ledger = get_forecast_ledger()
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while scoring %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    try:
        with bypass_session_cache():
            df = _download_ohlcv(symbol)
        if df is None or df.empty:
            raise HTTPException(
                status_code=422, detail="No historical market data available for scoring."
            )
        scored_count = ledger.score_pending_forecasts(symbol, df)
        live_metrics = ledger.get_track_record_metrics(ticker=symbol, record_source="live")
        replay_metrics = ledger.get_track_record_metrics(
            ticker=symbol, record_source="historical_replay"
        )
        return {
            "ticker": symbol,
            "scored_count": scored_count,
            "live_track_record": live_metrics,
            "replay_track_record": replay_metrics,
            "track_record": live_metrics,
        }
    except HTTPException:
        raise
    except (MarketDataUnavailable, MarketTransportError, UnknownTickerError):
        return JSONResponse(status_code=503, content=MARKET_DATA_UNAVAILABLE_BODY)
    except LedgerUnavailableError:
        logger.error("Forecast ledger unavailable while scoring %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)
    except Exception:
        logger.exception("Failed to score volatility ledger for %s", symbol)
        return JSONResponse(status_code=503, content=FORECAST_LEDGER_UNAVAILABLE_BODY)


__all__ = [
    "collect_volatility_forecast",
    "export_live_volatility_ledger",
    "require_collector_auth",
    "router",
    "volatility_forecast",
    "volatility_forecast_batch",
    "get_volatility_ledger_route",
    "get_volatility_ledger_summary_route",
    "score_volatility_ledger_route",
]
