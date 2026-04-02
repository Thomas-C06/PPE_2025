"""News pipeline configuration for GeoQuant AI."""

from __future__ import annotations

from datetime import date

# Date range — aligned with Bloc 1 config.py (2022-2024)
NEWS_START_DATE: str = "2022-01-01"
NEWS_END_DATE: str   = "2024-12-31"

# ── Column mappings ──────────────────────────────────────────────────────────
# Each entry maps a known dataset format to the canonical schema:
#   date | headline | source | content
#
# Key   = dataset identifier (passed as `dataset` argument to NewsLoader)
# Value = dict mapping canonical column name -> actual column name in the CSV

COLUMN_MAPS: dict[str, dict[str, str]] = {
    # ── DATASET PRINCIPAL (2022-2024) ────────────────────────────────────────
    # Kaggle "S&P 500 with Financial News Headlines (2008-2024)"
    # https://www.kaggle.com/datasets/dyutidasmahaptra/s-and-p-500-with-financial-news-headlines-20082024
    # Columns: date, headline, close
    # Coverage: 2008-2024  |  ~19 000 articles  |  ~510 KB CSV
    "sp500_news_2024": {
        "date":     "Date",
        "headline": "Title",
        "source":   None,    # no publisher column in this dataset
        "content":  None,    # no full text in this dataset
    },
    # ── DATASETS ALTERNATIFS (référence) ────────────────────────────────────
    # Kaggle "All the News 2.0"
    # https://www.kaggle.com/datasets/snapcrack/all-the-news
    # Columns: id, title, publication, author, date, year, month, url, content
    "all_the_news": {
        "date":     "date",
        "headline": "title",
        "source":   "publication",
        "content":  "content",
    },
    # Kaggle "US Financial News Articles" (2018 seulement)
    # https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles
    # Columns: date, headline, publisher
    "financial_news": {
        "date":     "date",
        "headline": "headline",
        "source":   "publisher",
        "content":  None,
    },
}

# Canonical output columns (always present in the processed DataFrame)
CANONICAL_COLUMNS: list[str] = ["date", "headline", "source", "content"]
