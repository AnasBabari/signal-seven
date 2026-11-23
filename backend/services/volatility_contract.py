"""Shared production contract for deployable volatility forecasts.

The empirical benchmark and the active HTTP service use the same four
session horizons.  Keeping this contract in a small dependency-free module
prevents the API, ledger, and snapshot builder from drifting independently.
"""

from __future__ import annotations

import os
from types import MappingProxyType

SUPPORTED_VOLATILITY_HORIZONS = (1, 5, 10, 20)
VOLATILITY_MAX_HORIZON = max(SUPPORTED_VOLATILITY_HORIZONS)
VOLATILITY_MODEL_POLICY_VERSION = "empirical_volatility_benchmark_v3"
VOLATILITY_MODEL_VERSION = "deployable_v5"
VOLATILITY_FEATURE_SET_VERSION = "deployable_feature_columns_v5"
DEFAULT_VOLATILITY_POLICY = "v3_baseline"

# Versioned auto routing policies. Every policy preserves H=1 -> "garch_11"
# because no g3_h1.onnx artifact exists. Default remains "v3_baseline" because
# Milestone 05 purged out-of-sample evaluation returned PROMOTE_G3_TO_AUTO=FAIL.
VOLATILITY_POLICY_MAPS = MappingProxyType(
    {
        "v3_baseline": MappingProxyType(
            {
                1: "garch_11",
                5: "rolling_mean",
                10: "rolling_mean",
                20: "rolling_mean",
            }
        ),
        "v4_g3_h5": MappingProxyType(
            {
                1: "garch_11",
                5: "gpu_g3",
                10: "rolling_mean",
                20: "rolling_mean",
            }
        ),
        "v4_g3": MappingProxyType(
            {
                1: "garch_11",
                5: "gpu_g3",
                10: "gpu_g3",
                20: "gpu_g3",
            }
        ),
    }
)

# Frozen v3 baseline policy constant kept for backward compatibility.
AUTO_MODEL_POLICY = VOLATILITY_POLICY_MAPS[DEFAULT_VOLATILITY_POLICY]


def resolve_volatility_policy_name(policy_name: str | None = None) -> str:
    """Resolve and validate the active volatility auto-routing policy name."""
    if policy_name is not None:
        candidate = str(policy_name).strip().lower()
    else:
        configured = DEFAULT_VOLATILITY_POLICY
        try:
            from config import settings

            configured = (
                str(getattr(settings, "volatility_default_policy", DEFAULT_VOLATILITY_POLICY))
                .strip()
                .lower()
            )
        except Exception:
            configured = DEFAULT_VOLATILITY_POLICY
        env_override = (os.getenv("VOLATILITY_DEFAULT_POLICY") or "").strip().lower()
        if configured != DEFAULT_VOLATILITY_POLICY:
            candidate = configured
        elif env_override:
            candidate = env_override
        else:
            candidate = configured
    if candidate not in VOLATILITY_POLICY_MAPS:
        raise ValueError(f"VOLATILITY_DEFAULT_POLICY must be one of {list(VOLATILITY_POLICY_MAPS)}")
    return candidate


def resolve_auto_model_policy(policy_name: str | None = None) -> dict[int, str]:
    """Return the validated horizon->model mapping for ``model=auto``."""
    name = resolve_volatility_policy_name(policy_name)
    mapping = dict(VOLATILITY_POLICY_MAPS[name])
    if mapping.get(1) != "garch_11":
        raise ValueError("H=1 auto policy must always remain 'garch_11'")
    return mapping


def is_supported_volatility_horizon(horizon: int) -> bool:
    """Return whether ``horizon`` is part of the production contract."""

    try:
        validate_volatility_horizon(horizon)
        return True
    except (TypeError, ValueError):
        return False


def validate_volatility_horizon(horizon: int) -> int:
    """Normalize and validate a production volatility horizon."""

    if isinstance(horizon, bool):
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}")
    try:
        normalized = int(horizon)
        if isinstance(horizon, float) and not horizon.is_integer():
            raise ValueError
    except (TypeError, ValueError) as err:
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}") from err
    if normalized not in SUPPORTED_VOLATILITY_HORIZONS:
        raise ValueError(f"horizon must be one of {list(SUPPORTED_VOLATILITY_HORIZONS)}")
    return normalized


__all__ = [
    "AUTO_MODEL_POLICY",
    "DEFAULT_VOLATILITY_POLICY",
    "SUPPORTED_VOLATILITY_HORIZONS",
    "VOLATILITY_FEATURE_SET_VERSION",
    "VOLATILITY_MAX_HORIZON",
    "VOLATILITY_MODEL_POLICY_VERSION",
    "VOLATILITY_MODEL_VERSION",
    "VOLATILITY_POLICY_MAPS",
    "is_supported_volatility_horizon",
    "resolve_auto_model_policy",
    "resolve_volatility_policy_name",
    "validate_volatility_horizon",
]
