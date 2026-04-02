"""
GeoQuant AI -- Hello World
Affiche le prix du S&P500 a cote d'une headline de news pour chaque date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ajouter src/ au path si execute depuis la racine
sys.path.insert(0, str(Path(__file__).resolve().parent))

from news_config import COLUMN_MAPS  # noqa: F401 (verif import)
from news_loader import NewsLoader

BASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    # ── 1. Prix S&P500 ───────────────────────────────────────────────────────
    print("Telechargement des prix S&P500...")
    prices = yf.download("^GSPC", start="2022-01-01", end="2024-12-31",
                         progress=False, auto_adjust=False)

    # Aplatir le MultiIndex si present (yfinance >= 0.2)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [col[0] for col in prices.columns]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    prices = prices[["Close"]].rename(columns={"Close": "SP500_Close"})
    print(f"  {len(prices)} jours de prix charges.")

    # ── 2. News ──────────────────────────────────────────────────────────────
    print("\nChargement des news...")
    loader = NewsLoader(
        csv_path=BASE_DIR / "data" / "raw" / "sp500_news_headlines.csv",
        dataset="sp500_news_2024",
        start_date="2022-01-01",
        end_date="2024-12-31",
        base_dir=BASE_DIR,
    )
    raw_df = loader.load_raw()
    news_df = loader.process(raw_df)

    # Garder la premiere headline par jour
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
    news_daily = (
        news_df
        .sort_values("date")
        .groupby("date")
        .first()
        [["headline", "source"]]
    )

    # ── 3. Jointure Prix + News ──────────────────────────────────────────────
    merged = prices.join(news_daily, how="inner")

    print(f"\n{'='*70}")
    print(f"  GEOQUANT AI -- Hello World : Prix + News ({len(merged)} jours)")
    print(f"{'='*70}")
    print(f"{'Date':<12}  {'S&P500':>8}  {'Source':<20}  Headline")
    print(f"{'-'*70}")
    for date, row in merged.head(15).iterrows():
        date_str = str(date)[:10]
        price = f"{row['SP500_Close']:.1f}"
        source = str(row['source'])[:18]
        headline = str(row['headline'])[:40]
        print(f"{date_str:<12}  {price:>8}  {source:<20}  {headline}")
    print(f"{'='*70}")

    # ── 4. Sauvegarde ────────────────────────────────────────────────────────
    out_path = BASE_DIR / "data" / "processed" / "hello_world_merged.csv"
    merged.to_csv(out_path)
    print(f"\nFichier joint sauvegarde dans : {out_path}")


if __name__ == "__main__":
    main()
