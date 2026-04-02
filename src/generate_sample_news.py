"""
Generate a realistic sample news CSV for GeoQuant AI testing.

This script produces ``data/raw/combined_news_data.csv`` in the
``financial_news`` format (columns: date, headline, publisher) so that
``NewsLoader`` can process it immediately, without needing a Kaggle account.

Once you have downloaded the real Kaggle dataset, simply replace the CSV in
``data/raw/`` and delete this script — it is only a development stub.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Sample data pools
# ---------------------------------------------------------------------------

HEADLINES = [
    # Géopolitique / crises
    "US imposes new sanctions on Russia over Ukraine conflict",
    "North Korea fires ballistic missile toward Japan",
    "Oil prices surge as Middle East tensions escalate",
    "Brexit deal uncertainty rattles European markets",
    "China-US trade war escalates with new tariffs",
    "Federal Reserve raises interest rates amid inflation concerns",
    "IMF warns of global recession risk following energy crisis",
    "OPEC cuts oil production, sending crude prices higher",
    "European Central Bank signals aggressive rate hike path",
    "Germany enters recession as energy costs weigh on industry",
    # Marchés financiers
    "S&P 500 drops 3% as inflation data disappoints investors",
    "Tech stocks rally on strong earnings from major firms",
    "Bitcoin falls below $20,000 as crypto winter deepens",
    "Gold hits all-time high as investors seek safe haven assets",
    "Dollar strengthens against euro on hawkish Fed comments",
    "Wall Street closes higher after positive jobs report",
    "Nasdaq enters correction territory amid rate hike fears",
    "Bond yields hit 15-year high, pressuring equity valuations",
    "Bank stocks surge after stress test results released",
    "Emerging market currencies under pressure from strong dollar",
    # Entreprises / secteurs
    "Apple reports record quarterly revenue despite supply chain issues",
    "Tesla shares fall after disappointing delivery numbers",
    "Amazon announces major layoffs amid slowing growth",
    "Oil majors post record profits as energy prices soar",
    "Banking sector faces headwinds from rising credit defaults",
    "Semiconductor shortage eases as chip production ramps up",
    "Renewable energy stocks surge on new climate legislation",
    "Pharmaceutical company announces breakthrough cancer treatment",
    "Merger talks between two major airlines collapse",
    "Retail sales data signals consumer slowdown",
    # Macro / indicateurs
    "US inflation cools to lowest level in two years",
    "Unemployment rate falls to multi-decade low",
    "GDP growth beats expectations in strong Q2 report",
    "Consumer confidence index drops sharply in August survey",
    "Housing market shows signs of cooling after rate hikes",
    "Manufacturing PMI contracts for third consecutive month",
    "Trade deficit widens as imports outpace exports",
    "Central bank holds rates steady, signals pause in tightening",
    "Wage growth accelerates, fueling inflation concerns",
    "Retail inventory levels rise, suggesting demand slowdown",
]

PUBLISHERS = [
    "Reuters",
    "Bloomberg",
    "Financial Times",
    "Wall Street Journal",
    "CNBC",
    "MarketWatch",
    "Forbes",
    "The Economist",
    "AP Finance",
    "Business Insider",
]

# ---------------------------------------------------------------------------
# Build the DataFrame
# ---------------------------------------------------------------------------

N_ROWS = 2_000
date_range = pd.date_range(start="2015-01-01", end="2023-12-31", freq="D")
dates = rng.choice(date_range, size=N_ROWS, replace=True)
dates = pd.Series(dates).sort_values().reset_index(drop=True)

headline_idx = rng.integers(0, len(HEADLINES), size=N_ROWS)
publisher_idx = rng.integers(0, len(PUBLISHERS), size=N_ROWS)

df = pd.DataFrame(
    {
        "date": dates.dt.strftime("%Y-%m-%d"),
        "headline": [HEADLINES[i] for i in headline_idx],
        "publisher": [PUBLISHERS[i] for i in publisher_idx],
    }
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

base_dir = Path(__file__).resolve().parents[1]
out_path = base_dir / "data" / "raw" / "combined_news_data.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Sample news CSV generated: {out_path}")
print(f"Rows : {len(df):,}")
print(f"Date range : {df['date'].min()} to {df['date'].max()}")
print("\nPreview:")
print(df.head(5).to_string(index=False))
