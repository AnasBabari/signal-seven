from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app
from config import Settings, resolve_allowed_origins, settings
from routes import volatility

CLIENT = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SECRET = "expected-collector-secret-28"
SAME_LEN_WRONG_SECRET = "wrong-value-same-length-28ch"
assert len(TEST_SECRET) == len(SAME_LEN_WRONG_SECRET) == 28


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/volatility/collect?ticker=MSFT&horizon=5"),
        ("GET", "/api/v1/volatility/export-ledger"),
        ("POST", "/api/v1/volatility/score-ledger?ticker=MSFT"),
    ],
)
@pytest.mark.parametrize(
    "auth_header",
    [
        None,
        "Bearer short",
        f"Bearer {SAME_LEN_WRONG_SECRET}",
        TEST_SECRET,
    ],
)
def test_collector_routes_401_do_not_leak_token_length_or_raw_scheme(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    method: str,
    path: str,
    auth_header: str | None,
) -> None:
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", TEST_SECRET)
    headers = {"Authorization": auth_header} if auth_header is not None else {}

    with caplog.at_level(logging.WARNING, logger="routes.volatility"):
        response = CLIENT.request(method, path, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Collector authentication failed."}
    assert response.headers.get("WWW-Authenticate") == "Bearer"

    body_lower = response.text.lower()
    logs_lower = caplog.text.lower()
    for forbidden in ("length", "expected", "received", TEST_SECRET.lower()):
        assert forbidden not in body_lower
        assert forbidden not in logs_lower


def test_collector_auth_handles_non_ascii_bearer_without_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", TEST_SECRET)
    with pytest.raises(volatility.HTTPException) as exc_info:
        volatility.require_collector_auth(
            authorization="Bearer na\u00efve-t\u00f6k\u00ebn-\U0001f680"
        )
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Collector authentication failed."

    # Raw Latin-1 HTTP header bytes (decoded by ASGI/uvicorn) must also return 401, not 500.
    response = CLIENT.get(
        "/api/v1/volatility/export-ledger",
        headers=[(b"authorization", b"Bearer naive-token-\xe9-\xfc")],
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Collector authentication failed."}


def test_export_ledger_sets_no_store_cache_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from services.forecast_ledger import ForecastLedger

    ledger = ForecastLedger(tmp_path / "export.db")
    monkeypatch.setenv("FORECAST_COLLECTOR_TOKEN", "test-collector-secret")
    monkeypatch.setattr(volatility, "get_forecast_ledger", lambda: ledger)

    response = CLIENT.get(
        "/api/v1/volatility/export-ledger",
        headers={"Authorization": "Bearer test-collector-secret"},
    )

    assert response.status_code == 200
    assert response.headers.get("Cache-Control") == "no-store"


def test_ledger_rejects_invalid_record_source() -> None:
    response = CLIENT.get("/api/v1/volatility/ledger?ticker=MSFT&record_source=bogus")
    assert response.status_code == 422


def test_cors_preflight_restricts_allowed_methods() -> None:
    origin = settings.allowed_origins[0] if settings.allowed_origins else "http://localhost:5173"
    response = CLIENT.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    methods = {
        part.strip().upper()
        for part in response.headers.get("Access-Control-Allow-Methods", "").split(",")
        if part.strip()
    }
    assert {"GET", "POST", "OPTIONS"}.issubset(methods)
    assert methods.isdisjoint({"DELETE", "PUT", "PATCH"})


def test_health_includes_hardened_security_headers() -> None:
    response = CLIENT.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )


def test_resolve_allowed_origins_fails_closed_in_production() -> None:
    defaults = ["http://127.0.0.1:5173", "http://localhost:5173"]
    assert resolve_allowed_origins(
        defaults,
        "https://stock-predictor-lstm-two.vercel.app",
        is_production=True,
    ) == ["https://stock-predictor-lstm-two.vercel.app"]
    assert (
        resolve_allowed_origins(
            defaults,
            None,
            is_production=True,
            has_explicit_allowed_origins=False,
        )
        == []
    )
    assert resolve_allowed_origins(
        ["https://custom.example.com"],
        None,
        is_production=True,
        has_explicit_allowed_origins=True,
    ) == ["https://custom.example.com"]
    assert resolve_allowed_origins(
        defaults,
        "https://preview.vercel.app",
        is_production=False,
    ) == [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://preview.vercel.app",
    ]


def test_settings_accepts_legacy_volatility_release_vars_without_public_key() -> None:
    cfg = Settings(
        _env_file=None,
        volatility_release_archive_url="https://example.com/release.tar.gz",
    )
    assert cfg.volatility_public_key_path is None


def test_vercel_config_sets_immutable_assets_and_production_csp() -> None:
    vercel_cfg = json.loads((REPO_ROOT / "frontend" / "vercel.json").read_text(encoding="utf-8"))
    headers_by_source = {
        entry["source"]: {h["key"]: h["value"] for h in entry["headers"]}
        for entry in vercel_cfg["headers"]
    }
    assert (
        headers_by_source["/assets/(.*)"]["Cache-Control"] == "public, max-age=31536000, immutable"
    )
    csp = headers_by_source["/(.*)"]["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "http://localhost" not in csp
