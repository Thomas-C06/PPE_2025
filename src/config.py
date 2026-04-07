"""Global configuration for GeoQuant AI."""

from __future__ import annotations

# ── Fichiers de donnees (noms relatifs a data/processed/) ─────────────────────
DATASET_FINAL_FILE:   str = "dataset_final.csv"
GEO_SCORES_FILE:      str = "geo_scores.csv"
NEWS_PROCESSED_FILE:  str = "news_sp500_news_2024_processed.csv"
NEWS_AGGREGATED_FILE: str = "news_aggregated.csv"
NEWS_SCORED_FILE:     str = "news_scored.csv"

# ── Tickers ───────────────────────────────────────────────────────────────────
TICKERS: list[str] = [
    "^GSPC",      # S&P 500
    "^FCHI",      # CAC 40
    "EURUSD=X",   # EUR/USD
    "BTC-USD",    # Bitcoin
    "GC=F",       # Gold (XAU/USD)
    "CL=F",       # WTI Crude Oil
    "^VIX",       # CBOE Volatility Index
    "^IXIC",      # NASDAQ Composite
    "^DJI",       # Dow Jones Industrial Average
]

# Ticker utilisé comme référence principale pour le dataset_final
PRIMARY_TICKER: str = "^GSPC"

# Nom lisible pour chaque ticker (utilisé dans les graphiques et les logs)
TICKER_NAMES: dict[str, str] = {
    "^GSPC":    "SP500",
    "^FCHI":    "CAC40",
    "EURUSD=X": "EURUSD",
    "BTC-USD":  "Bitcoin",
    "GC=F":     "Gold",
    "CL=F":     "WTI_Oil",
    "^VIX":     "VIX",
    "^IXIC":    "NASDAQ",
    "^DJI":     "DowJones",
}

# ── Période cible (couvre Ukraine, inflation, SVB, Hamas, crash Japon, Trump 2024)
START_DATE: str = "2022-01-01"
END_DATE: str   = "2024-12-31"
