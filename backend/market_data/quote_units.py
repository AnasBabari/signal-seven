"""Authoritative quote-unit and currency overrides for non-GBp LSE listings."""

from __future__ import annotations

LSE_QUOTE_UNIT_OVERRIDES: dict[str, dict[str, str]] = {
    "CPG.L": {
        "name": "Compass Group plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
    "IHG.L": {
        "name": "InterContinental Hotels Group plc",
        "exchange_mic": "XLON",
        "exchange_name": "LSE",
        "currency": "USD",
        "currency_symbol": "$",
    },
}


def resolve_quote_unit(symbol: str) -> tuple[str, str]:
    """Return ``(currency, currency_symbol)`` for a ticker symbol."""
    sym = symbol.strip().upper()
    override = LSE_QUOTE_UNIT_OVERRIDES.get(sym)
    if override is not None:
        return override["currency"], override["currency_symbol"]
    if sym.endswith(".L"):
        return "GBp", "p"
    return "USD", "$"
