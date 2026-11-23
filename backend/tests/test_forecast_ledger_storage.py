from __future__ import annotations

import pytest

from services.forecast_ledger import ForecastLedger, LedgerUnavailableError, _LedgerConnection


class _Cursor:
    rowcount = 0

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _RawConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, query: str, params=()):
        self.queries.append((query, tuple(params)))
        return _Cursor()

    def executemany(self, query: str, params):
        self.queries.append((query, tuple()))
        return _Cursor()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_postgres_adapter_translates_parameters_without_interpolation() -> None:
    raw = _RawConnection()
    adapter = _LedgerConnection(raw, "postgres")

    adapter.execute("SELECT * FROM forecast_ledger WHERE ticker=? AND horizon=?", ["MSFT", 5])

    assert raw.queries == [
        ("SELECT * FROM forecast_ledger WHERE ticker=%s AND horizon=%s", ("MSFT", 5))
    ]


def test_postgres_url_is_validated_before_connection_attempt() -> None:
    with pytest.raises(LedgerUnavailableError, match="PostgreSQL connection URL"):
        ForecastLedger(database_url="mysql://user:password@example/ledger")


def test_configured_postgres_never_falls_back_to_sqlite(monkeypatch) -> None:
    import psycopg

    def fail_connect(*args, **kwargs):
        raise OSError("database is offline")

    monkeypatch.setattr(psycopg, "connect", fail_connect)
    with pytest.raises(LedgerUnavailableError, match="PostgreSQL forecast ledger is unavailable"):
        ForecastLedger(database_url="postgresql://user:password@example/ledger")


def test_postgres_pool_configuration_lifecycle_and_single_connection_overview(monkeypatch) -> None:
    """Verify ConnectionPool parameters, close lifecycle, and 1-connection overview reuse."""
    import psycopg_pool

    from services import forecast_ledger as fl

    pool_init_kwargs: list[dict[str, object]] = []
    checkouts = {"count": 0, "returns": 0, "closed": False}

    class _FakePgCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

        def fetchall(self):
            return list(self._rows)

    class _FakePgConn:
        def __init__(self) -> None:
            self.queries: list[tuple[str, tuple[object, ...]]] = []

        def execute(self, query: str, params=()):
            self.queries.append((query, tuple(params)))
            q_low = query.lower()
            if "information_schema.columns" in q_low:
                return _FakePgCursor([{"column_name": c} for c in fl._POSTGRES_REQUIRED_COLUMNS])
            if "percentile_cont(0.5)" in q_low:
                return _FakePgCursor(
                    [
                        {
                            "total_forecasts": 3,
                            "pending_forecasts": 1,
                            "scored_forecasts": 2,
                            "mean_mae": 0.04,
                            "mean_sq_error": 0.0025,
                            "mean_qlike": 0.12,
                            "qlike_count": 2,
                            "direction_eligible_forecasts": 2,
                            "direction_hits": 1,
                            "median_qlike": 0.11,
                        }
                    ]
                )
            return _FakePgCursor([])

        def commit(self):
            pass

        def rollback(self):
            pass

    fake_conn = _FakePgConn()

    class _FakePoolCM:
        def __enter__(self):
            checkouts["count"] += 1
            return fake_conn

        def __exit__(self, *args):
            checkouts["returns"] += 1

    class _FakePool:
        check_connection = psycopg_pool.ConnectionPool.check_connection

        def __init__(self, **kwargs):
            pool_init_kwargs.append(kwargs)

        def connection(self, timeout=None):
            return _FakePoolCM()

        def close(self):
            checkouts["closed"] = True

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", _FakePool)
    ledger = ForecastLedger(database_url="postgresql://user:secret@db.internal/ledger")
    assert len(pool_init_kwargs) == 1
    cfg = pool_init_kwargs[0]
    assert cfg["min_size"] == 0
    assert cfg["max_size"] == 4
    assert cfg["max_idle"] == 120
    assert cfg["check"] is psycopg_pool.ConnectionPool.check_connection
    assert checkouts["count"] == 1  # init_postgres_db checkout
    assert checkouts["returns"] == 1

    # One overview request (entries + live metrics + replay metrics) checks out ONE connection.
    overview = ledger.get_ledger_overview(ticker="MSFT", horizon=5, limit=25)
    assert checkouts["count"] == 2
    assert checkouts["returns"] == 2
    assert overview["live_track_record"]["scored_forecasts"] == 2
    assert overview["live_track_record"]["mean_rmse"] == pytest.approx(0.05)
    assert overview["live_track_record"]["median_qlike"] == pytest.approx(0.11)
    assert overview["live_track_record"]["direction_accuracy_pct"] == pytest.approx(50.0)
    # Confirm PERCENTILE_CONT(0.5) was emitted on Postgres path.
    assert any("PERCENTILE_CONT(0.5)" in q for q, _ in fake_conn.queries)

    ledger.close()
    assert checkouts["closed"] is True


def test_sqlite_and_postgres_sql_metrics_match_reference_on_odd_and_even_counts(
    tmp_path,
) -> None:
    """Verify portable SQL aggregation matches Python reference for empty, odd, and even counts."""
    import math

    import numpy as np

    ledger = ForecastLedger(tmp_path / "metrics_parity.db")

    # Empty table
    empty = ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    assert empty["total_forecasts"] == 0
    assert empty["pending_forecasts"] == 0
    assert empty["scored_forecasts"] == 0
    assert empty["median_qlike"] is None
    assert empty["mean_rmse"] is None
    assert empty["direction_accuracy_pct"] is None

    # Insert 1 pending + 3 scored (odd) then + 1 more scored (even = 4), with mixed model_versions
    rows = [
        # (date, pred, recent, actual, mv, status)
        ("2026-01-02", 0.25, 0.20, None, "deployable_v5:annualized_sigma_v1", "pending"),
        ("2026-01-05", 0.30, 0.20, 0.26, "deployable_v5:annualized_sigma_v1", "scored"),  # dir hit
        ("2026-01-06", 0.15, 0.20, 0.25, "deployable_v5:annualized_sigma_v1", "scored"),  # dir miss
        ("2026-01-07", 0.22, 0.20, 0.21, "deployable_v5", "scored"),  # legacy non-eligible version
    ]

    def _insert_row(item):
        f_date, pred, recent, actual, mv, status = item
        f_err = (pred - actual) if actual is not None else None
        a_err = abs(f_err) if f_err is not None else None
        if actual is not None:
            ratio = (actual**2) / (pred**2)
            qlike = max(ratio - math.log(ratio) - 1.0, 0.0)
        else:
            qlike = None
        with ledger._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO forecast_ledger (
                    forecast_date, ticker, horizon, target_date, model_name,
                    predicted_volatility, recent_realized_volatility, origin_price,
                    lower_scenario_price, upper_scenario_price, actual_realized_volatility,
                    forecast_error, abs_error, qlike_loss, status, record_source,
                    model_version, feature_set_version, code_commit, data_as_of,
                    data_provider, forecast_fingerprint, created_at
                ) VALUES (?, 'AAPL', 5, '2026-01-15', 'rolling_mean', ?, ?, 100.0, 95.0, 105.0,
                          ?, ?, ?, ?, ?, 'live', ?, 'v5', 'abc', ?, 'alpaca', ?, '2026-01-02T00:00:00Z')
                """,
                (
                    f_date,
                    pred,
                    recent,
                    actual,
                    f_err,
                    a_err,
                    qlike,
                    status,
                    mv,
                    f_date,
                    f"fp-{f_date}",
                ),
            )
            conn.commit()

    def _python_reference(subset):
        scored = [r for r in subset if r[5] == "scored"]
        abs_e = [abs(r[1] - r[3]) for r in scored]
        sq_e = [(r[1] - r[3]) ** 2 for r in scored]
        qlikes = []
        hits = 0
        elig = 0
        for r in scored:
            ratio = (r[3] ** 2) / (r[1] ** 2)
            qlikes.append(max(ratio - math.log(ratio) - 1.0, 0.0))
            if r[4].endswith(":annualized_sigma_v1"):
                elig += 1
                if (r[1] - r[2] >= 0 and r[3] - r[2] >= 0) or (r[1] - r[2] < 0 and r[3] - r[2] < 0):
                    hits += 1
        return {
            "total_forecasts": len(subset),
            "pending_forecasts": sum(r[5] == "pending" for r in subset),
            "scored_forecasts": len(scored),
            "direction_eligible_forecasts": elig,
            "mean_mae": float(np.mean(abs_e)),
            "mean_rmse": float(np.sqrt(np.mean(sq_e))),
            "mean_qlike": float(np.mean(qlikes)),
            "median_qlike": float(np.median(qlikes)),
            "direction_accuracy_pct": float(hits / elig * 100.0) if elig else None,
        }

    for row in rows:
        _insert_row(row)

    # Odd scored count (3)
    odd_got = ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    odd_exp = _python_reference(rows)
    for key, expected in odd_exp.items():
        assert odd_got[key] == pytest.approx(expected, rel=1e-12), f"odd mismatch on {key}"

    # Add 4th scored row (even scored count = 4)
    fourth = ("2026-01-08", 0.18, 0.25, 0.16, "deployable_v5:annualized_sigma_v1", "scored")
    rows.append(fourth)
    _insert_row(fourth)
    even_got = ledger.get_track_record_metrics(ticker="AAPL", record_source="live")
    even_exp = _python_reference(rows)
    for key, expected in even_exp.items():
        assert even_got[key] == pytest.approx(expected, rel=1e-12), f"even mismatch on {key}"
