"""
Entry point for GeoQuant AI -- Bloc 2 NLP pipeline.

Usage:
    py -3 src/run_geo_scorer.py

Prerequisites:
    1. News CSV must exist:
       py -3 src/news_main.py          (generates news_sp500_news_2024_processed.csv)
    2. (Optional) dataset_final.csv must exist to enable geo_score injection:
       py -3 src/main.py

On first run, FinBERT (~500 MB) is downloaded automatically from Hugging Face.
Subsequent runs reuse the local cache and skip already-scored headlines.
"""

from __future__ import annotations

from pathlib import Path

from geo_scorer import GeoScorer


def main() -> None:
    """Run the full Bloc 2 NLP pipeline."""
    base_dir = Path(__file__).resolve().parents[1]

    scorer = GeoScorer(
        base_dir=base_dir,
        smoothing_days=3,   # rolling mean window to reduce daily noise
        batch_size=32,      # headlines per FinBERT forward pass (reduce if OOM)
        max_length=128,     # token limit (128 is enough for headlines)
    )

    print("=" * 65)
    print("  GEOQUANT AI -- Bloc 2 NLP / FinBERT  --  Geo-Score Pipeline")
    print("=" * 65)

    daily_df = scorer.run()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  PIPELINE SUMMARY -- BLOC 2 NLP")
    print("=" * 65)
    print(f"  Days scored       : {len(daily_df)}")
    print(f"  Date range        : {daily_df['date'].min().date()} "
          f"to {daily_df['date'].max().date()}")
    print(f"  Geo-Score mean    : {daily_df['geo_score'].mean():.3f}")
    print(f"  Geo-Score min     : {daily_df['geo_score'].min():.3f}")
    print(f"  Geo-Score max     : {daily_df['geo_score'].max():.3f}")
    print(f"  Positive days     : "
          f"{(daily_df['geo_score'] > 0).sum()} "
          f"({(daily_df['geo_score'] > 0).mean():.0%})")
    print(f"  Negative days     : "
          f"{(daily_df['geo_score'] < 0).sum()} "
          f"({(daily_df['geo_score'] < 0).mean():.0%})")
    print()
    print("  Output files:")
    print("    data/processed/news_scored.csv         <- article-level FinBERT scores")
    print("    data/processed/geo_scores.csv         <- daily Geo-Score")
    print("    data/processed/sentiment_cache.json   <- FinBERT cache")
    print("    data/processed/dataset_final.csv      <- updated with geo_score")
    print()
    print("  Preview (last 10 days):")
    print(
        daily_df.tail(10)[["date", "geo_score", "nb_articles_scored"]]
        .to_string(index=False)
    )
    print("=" * 65)
    print("  Bloc 2 complete. Ready for Bloc 3 -- Strategy Engine.")
    print("=" * 65)


if __name__ == "__main__":
    main()
