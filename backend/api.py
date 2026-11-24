"""Signal Seven forecasting API — FastAPI backend application.

Serves causal equity volatility forecasts, scenario cones, company metadata,
and immutable forecast ledger track records.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import yfinance as yf  # type: ignore[import-untyped]
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from config import APP_VERSION, settings
from data_pipeline import (
    MarketDataUnavailable,
    MarketTransportError,
    market_circuit_breaker,
)
from routes.common import limiter, validate_ticker
from routes.health import router as health_router
from routes.market import router as market_router
from routes.simple_forecast import router as simple_forecast_router
from routes.volatility import router as volatility_router

__all__ = [
    "APP_VERSION",
    "MarketDataUnavailable",
    "MarketTransportError",
    "app",
    "limiter",
    "validate_ticker",
    "yf",
]

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def _shutdown_resources() -> None:
    """Release pooled HTTP and database connections cleanly on shutdown."""
    try:
        import data_pipeline

        close_md = getattr(data_pipeline.market_data_service, "close", None)
        if callable(close_md):
            close_md()
    except Exception:
        logger.warning("Failed to close market-data HTTP clients during shutdown", exc_info=True)
    try:
        from services import forecast_ledger

        close_ledger = getattr(forecast_ledger, "close_forecast_ledger", None)
        if callable(close_ledger):
            close_ledger()
    except Exception:
        logger.warning("Failed to close forecast ledger pool during shutdown", exc_info=True)
    try:
        from services import market_news

        close_news = getattr(market_news, "close_news_client", None)
        if callable(close_news):
            close_news()
    except Exception:
        logger.warning("Failed to close market news HTTP client during shutdown", exc_info=True)


# ── Application Setup ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start background warm-up without blocking readiness.

    The learned-forecast path costs ~9s on its first request in a fresh process
    (PyTorch import plus model fitting). Warming it in a daemon thread moves
    that cost off the request path; ``/health`` stays responsive because the
    thread never blocks startup.
    """
    from services.forecast_warmup import start_forecast_warmup

    start_forecast_warmup()
    try:
        yield
    finally:
        _shutdown_resources()


app = FastAPI(
    title="Signal Seven Forecast API",
    version=APP_VERSION,
    description="Seven-session learned price estimates with chronological historical backtests.",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please wait before trying again."},
    )


# ── Compression & CORS Middleware ────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=4)

_preview_cors_regex = settings.preview_cors_origin_regex or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=_preview_cors_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "If-None-Match", "X-Prediction-Request-ID", "X-Client-Class"],
    expose_headers=["ETag"],
)


# ── Security Headers ─────────────────────────────────────────────────
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


# ── Router Registration ──────────────────────────────────────────────
app.include_router(health_router)
app.include_router(market_router)
app.include_router(simple_forecast_router)
app.include_router(volatility_router)


def _record_upstream(status: str, error: str | None = None) -> None:
    """Reset or update upstream circuit breaker state for tests/diagnostics."""
    if status == "available":
        market_circuit_breaker.record_success("__global__")
    elif status == "unavailable":
        market_circuit_breaker.record_failure(
            "__global__", MarketTransportError(error or "Unavailable")
        )
