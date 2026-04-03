"""Entry point for GeoQuant AI news data pipeline."""

from __future__ import annotations

from pathlib import Path

from news_config import NEWS_END_DATE, NEWS_START_DATE
from kaggle_news_loader import NewsLoader


def main() -> None:
    """Orchestrate news download, processing, and persistence."""
    base_dir = Path(__file__).resolve().parents[1]

    # ── Dataset principal 2022-2024 ──────────────────────────────────────────
    # Kaggle "S&P 500 with Financial News Headlines (2008-2024)"
    # URL : https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024
    #
    # Instructions :
    #   1. Télécharger le ZIP sur Kaggle
    #   2. Extraire et placer le CSV dans data/raw/
    #   3. Vérifier que CSV_FILE correspond au nom exact du fichier extrait

    DATASET  = "sp500_news_2024"
    CSV_FILE = "sp500_news_headlines.csv"                # nom du fichier après extraction

    csv_path = base_dir / "data" / "raw" / CSV_FILE

    loader = NewsLoader(
        csv_path=csv_path,
        dataset=DATASET,
        start_date=NEWS_START_DATE,
        end_date=NEWS_END_DATE,
        base_dir=base_dir,
    )

    raw_df = loader.load_raw()
    processed_df = loader.process(raw_df)
    loader.save_data(raw_df, processed_df)

    print("\nPreview of processed news data:")
    print(processed_df[["date", "headline", "source"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
