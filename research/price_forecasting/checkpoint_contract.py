"""Selection/deployment artifact contract; no certification claims."""

import hashlib
from pathlib import Path

import numpy as np


def load_verified_checkpoint(path, *, feature_names, historical_origin=None):
    import torch

    path = Path(path)
    expected = Path(str(path) + ".sha256").read_text().strip()
    if len(expected) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint hash differs from saved artifact")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(
        checkpoint, feature_names=feature_names, historical_origin=historical_origin
    )
    return checkpoint


def validate_checkpoint(checkpoint, *, feature_names, historical_origin=None):
    if checkpoint.get("feature_names") != list(feature_names):
        raise ValueError("Checkpoint ordered feature schema differs")
    if checkpoint.get("feature_definition_version") != "stationary-zero-range-neutral-v2":
        raise ValueError("Checkpoint feature definitions are unsupported")
    config = checkpoint.get("config", {})
    if config.get("horizon") != 7 or config.get("lookback") != 60:
        raise ValueError("Checkpoint lookback/horizon is unsupported")
    required = {"hidden_size", "layers", "dropout", "use_attention", "embed_dim"}
    if not required.issubset(config) or not checkpoint.get("ticker_names"):
        raise ValueError("Checkpoint architecture/ticker schema is incomplete")
    role = checkpoint.get("artifact_role")
    if role not in {"validation_selected_model", "all_data_deployment_refit"}:
        raise ValueError("Checkpoint role is unsupported")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("version") != "global-calendar-purged-v1":
        raise ValueError("Checkpoint global training provenance is unavailable")
    if (
        not checkpoint.get("training_target_max")
        or checkpoint.get("scaler_target_max") != checkpoint["training_target_max"]
    ):
        raise ValueError("Checkpoint training/scaler cutoffs are unavailable or inconsistent")
    if (
        not checkpoint.get("training_rows_sha256")
        or checkpoint.get("scaler_rows_sha256") != checkpoint["training_rows_sha256"]
    ):
        raise ValueError("Checkpoint training/scaler row identity differs")
    if historical_origin is not None:
        if (
            role != "validation_selected_model"
            or checkpoint["training_target_max"] >= historical_origin
        ):
            raise ValueError("Checkpoint overlaps historical evaluation")
        if protocol.get("train_target_max") != checkpoint["training_target_max"]:
            raise ValueError("Selection cutoff differs from training cutoff")
    for key, size in [
        ("feature_mean", len(feature_names)),
        ("feature_std", len(feature_names)),
        ("target_mean", 7),
        ("target_std", 7),
    ]:
        values = np.asarray(checkpoint.get("scalers", {}).get(key, []), dtype=float)
        if (
            values.shape != (size,)
            or not np.isfinite(values).all()
            or (key.endswith("std") and (values <= 0).any())
        ):
            raise ValueError("Checkpoint scalers have invalid shape/values")
    return checkpoint
