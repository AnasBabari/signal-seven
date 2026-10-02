"""Fixed small local development comparison; never claims independent skill."""

# ruff: noqa: E402
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from calendars import latest_completed_trading_session

from research.price_forecasting.baselines import fit_ridge_validation
from research.price_forecasting.gpu_pipeline import (
    PriceTrainingConfig,
    build_global_price_dataset,
    dataset_protocol,
    train_cuda_price_model,
)
from research.price_forecasting.paired_validation import (
    checkpoint_predictions,
    paired_tests,
    ridge_predictions,
    validation_table,
)


def main():
    output = ROOT / "artifacts" / "audit_price_comparison_20261002"
    if output.exists():
        raise FileExistsError("Use a fresh artifact directory; existing evidence is preserved")
    # Frozen before scores are inspected: three cached assets, Ridge alpha100,
    # one small 2-epoch LSTM, persistence and train-majority direction.
    spec = PriceTrainingConfig(
        hidden_size=16,
        layers=1,
        embed_dim=4,
        maximum_epochs=2,
        patience=1,
        batch_size=128,
        dropout=0,
        use_attention=False,
    )
    frames, sources = {}, {}
    for ticker in ("AAPL", "ARM", "BP.L"):
        path = ROOT / "data" / "tri_exchange" / "cache" / f"{ticker}.parquet"
        data = pd.read_parquet(path)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        completed = latest_completed_trading_session(
            exchange="LSE" if ticker.endswith(".L") else "NYSE"
        )
        frames[ticker] = data.loc[data.index <= completed].tail(1000)
        sources[ticker] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provider_equivalence": "unverified",
        }
    dataset = build_global_price_dataset(frames, spec)
    protocol = dataset_protocol(dataset)
    assert protocol["train_target_max"] < protocol["validation_origin_min"]
    assert protocol["validation_target_max"] < protocol["test_origin_min"]
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                **protocol,
                "sources": sources,
                "fixed_candidates": [
                    "ridge_alpha100",
                    "lstm_2epochs",
                    "persistence",
                    "train_majority_direction",
                ],
            },
            indent=2,
        )
    )
    ridge = fit_ridge_validation(dataset)
    (output / "ridge.json").write_text(json.dumps(ridge, indent=2))
    train_cuda_price_model(dataset, output / "gpu", spec, validation_only=True)
    table = validation_table(
        dataset,
        ridge_predictions(dataset, ridge),
        checkpoint_predictions(dataset, output / "gpu" / "selection_model.pt"),
    )
    table.to_parquet(output / "predictions.parquet", index=False)
    for name in ("ridge", "lstm"):
        table[f"absolute_loss_{name}"] = 100 * np.abs(
            np.exp(table[f"y_pred_{name}"]) - np.exp(table.y_true)
        )
    table["absolute_loss_persistence"] = 100 * np.abs(1 - np.exp(table.y_true))
    table.to_parquet(output / "losses.parquet", index=False)
    (output / "paired.json").write_text(json.dumps(paired_tests(table), indent=2))
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(
        json.dumps(
            {
                "status": "development_comparison_complete",
                "path": str(output),
                "test_evaluated": False,
                "protocol": protocol,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
