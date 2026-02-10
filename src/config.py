"""Global configuration for GeoQuant AI."""

from __future__ import annotations

from datetime import date

TICKERS: list[str] = ["^GSPC", "^FCHI", "EURUSD=X"]
START_DATE: str = "2010-01-01"
END_DATE: str = date.today().isoformat()
