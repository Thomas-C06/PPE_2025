"""
Geo-Score engine for GeoQuant AI -- unified historical NLP pipeline.

This module now keeps one coherent lineage:
    processed Kaggle news -> article scores -> daily Geo-Score
    -> aligned market dataset -> strategy -> backtest -> dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DATASET_FINAL_FILE,
    GEO_SCORES_FILE,
    NEWS_PROCESSED_FILE,
    NEWS_SCORED_FILE,
)
from finbert_sentiment import FinBertSentiment
from sentiment_cache import load_cache, save_cache


@dataclass
class GeoScorer:
    """
    Compute a daily Geo-Score from the unified historical news source.

    The article-level score is:
        geo_score_article = p_positive - p_negative

    The market dataset receives:
        - geo_score_fresh  : score computed on days with fresh headlines
        - geo_score        : forward-filled daily sentiment used by strategy
        - geo_score_source : "fresh_news" or "carried_forward"
    """

    base_dir: Path
    news_file: Optional[Path] = None
    smoothing_days: int = 3
    batch_size: int = 32
    max_length: int = 128
    model_name: str = "ProsusAI/finbert"

    _clf: Optional[FinBertSentiment] = field(default=None, init=False, repr=False)

    def _news_path(self) -> Path:
        if self.news_file is not None:
            path = Path(self.news_file)
            return path if path.is_absolute() else self.base_dir / path
        return self.base_dir / "data" / "processed" / NEWS_PROCESSED_FILE

    def _dataset_path(self) -> Path:
        return self.base_dir / "data" / "processed" / DATASET_FINAL_FILE

    def _geo_scores_path(self) -> Path:
        return self.base_dir / "data" / "processed" / GEO_SCORES_FILE

    def _scored_news_path(self) -> Path:
        return self.base_dir / "data" / "processed" / NEWS_SCORED_FILE

    def _get_clf(self) -> FinBertSentiment:
        """Lazy-load FinBERT once, then reuse it through the session."""
        if self._clf is None:
            self._clf = FinBertSentiment(model_name=self.model_name)
        return self._clf

    def load_news(self) -> pd.DataFrame:
        """Load the processed historical news CSV used by the full pipeline."""
        path = self._news_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Processed news file not found: {path}\n"
                "Run `python src/news_main.py` first."
            )

        df = pd.read_csv(path, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "headline"])
        df["headline"] = df["headline"].astype(str).str.strip()
        df = df[df["headline"] != ""]
        df = df.sort_values("date").reset_index(drop=True)

        print(
            f"  [GeoScorer] Loaded {len(df):,} headlines "
            f"({df['date'].min().date()} to {df['date'].max().date()})."
        )
        return df

    def score_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Run FinBERT on every headline, reusing the on-disk cache."""
        clf = self._get_clf()
        cache = load_cache()

        headlines = news_df["headline"].tolist()
        uncached = [headline for headline in headlines if headline not in cache]
        total = len(headlines)
        cached_count = total - len(uncached)

        print(
            f"  [GeoScorer] {total:,} headlines total | "
            f"{cached_count:,} cached | {len(uncached):,} to score."
        )

        if uncached:
            print(
                f"  [GeoScorer] Running FinBERT on {len(uncached):,} headlines "
                f"(batch_size={self.batch_size}, max_length={self.max_length})..."
            )
            df_new = clf.predict_df(
                uncached,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
            for i, text in enumerate(uncached):
                cache[text] = df_new.iloc[i].to_dict()
            save_cache(cache)
            print(f"  [GeoScorer] Cache updated ({len(cache):,} entries).")

        pred_df = pd.DataFrame([cache[text] for text in headlines])
        for col in ("p_positive", "p_neutral", "p_negative"):
            if col not in pred_df.columns:
                pred_df[col] = 0.0

        scored = pd.concat(
            [news_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )
        scored["geo_score_article"] = scored["p_positive"] - scored["p_negative"]
        return scored

    def save_scored_news(self, scored_df: pd.DataFrame) -> Path:
        """Persist the article-level scored headlines for the dashboard demo."""
        output = self._scored_news_path()
        output.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(output, index=False)
        print(f"  [GeoScorer] Scored headlines saved to {output}")
        return output

    def aggregate_by_day(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate article-level scores into one fresh daily Geo-Score.

        The daily score is confidence-weighted, smoothed with a causal rolling
        average, then clipped into [-1, 1].
        """
        scored_df = scored_df.copy()
        scored_df["confidence"] = scored_df[
            ["p_positive", "p_neutral", "p_negative"]
        ].max(axis=1)

        def weighted_mean(group: pd.DataFrame) -> float:
            weights = group["confidence"]
            total_weight = float(weights.sum())
            if total_weight == 0.0:
                return float(group["geo_score_article"].mean())
            return float((group["geo_score_article"] * weights).sum() / total_weight)

        daily = (
            scored_df.groupby("date")
            .apply(
                lambda group: pd.Series(
                    {
                        "geo_score_raw": weighted_mean(group),
                        "nb_articles_scored": len(group),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )

        daily = daily.sort_values("date").reset_index(drop=True)
        daily["geo_score"] = (
            daily["geo_score_raw"]
            .rolling(window=self.smoothing_days, min_periods=1, center=False)
            .mean()
            .clip(-1.0, 1.0)
        )

        print(
            f"  [GeoScorer] Aggregated to {len(daily)} days "
            f"(smoothing={self.smoothing_days} days)."
        )
        print(
            f"  [GeoScorer] Geo-Score range: "
            f"{daily['geo_score'].min():.3f} to {daily['geo_score'].max():.3f} | "
            f"mean: {daily['geo_score'].mean():.3f}"
        )

        return daily[["date", "geo_score", "geo_score_raw", "nb_articles_scored"]]

    def save_geo_scores(self, daily_df: pd.DataFrame) -> Path:
        """Save the fresh daily Geo-Score table to disk."""
        output = self._geo_scores_path()
        output.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output, index=False)
        print(f"  [GeoScorer] Geo-scores saved to {output}")
        return output

    def inject_into_dataset(self, daily_df: pd.DataFrame) -> Optional[Path]:
        """
        Align the Geo-Score to the market dataset without silently using 0.

        Strategy/backtest use:
            geo_score = fresh score on news days, otherwise last known score
        over the exact overlap period between prices and news coverage.
        """
        dataset_path = self._dataset_path()
        if not dataset_path.exists():
            print(
                "  [GeoScorer] dataset_final.csv not found -- skipping injection.\n"
                "              Run `python src/main.py` first."
            )
            return None

        df = pd.read_csv(dataset_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        daily = daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        coverage_start = daily["date"].min()
        coverage_end = daily["date"].max()

        df = df[(df["date"] >= coverage_start) & (df["date"] <= coverage_end)].copy()

        for col in (
            "geo_score",
            "geo_score_raw",
            "geo_score_fresh",
            "geo_score_raw_fresh",
            "geo_score_source",
            "has_fresh_news",
            "nb_articles_scored",
            "price",
            "news",
        ):
            if col in df.columns:
                df = df.drop(columns=[col])

        daily = daily.rename(
            columns={
                "geo_score": "geo_score_fresh",
                "geo_score_raw": "geo_score_raw_fresh",
            }
        )

        df = df.merge(
            daily[
                [
                    "date",
                    "geo_score_fresh",
                    "geo_score_raw_fresh",
                    "nb_articles_scored",
                ]
            ],
            on="date",
            how="left",
        )

        df = df.sort_values("date").reset_index(drop=True)
        df["nb_articles_scored"] = df["nb_articles_scored"].fillna(0).astype(int)
        df["has_fresh_news"] = df["geo_score_fresh"].notna()
        df["geo_score"] = df["geo_score_fresh"].ffill()

        if df["geo_score"].isna().any():
            missing = int(df["geo_score"].isna().sum())
            raise ValueError(
                f"Cannot inject Geo-Score: {missing} rows still have no score after alignment."
            )

        df["geo_score_raw"] = df["geo_score_raw_fresh"]
        df["geo_score_source"] = np.where(
            df["has_fresh_news"], "fresh_news", "carried_forward"
        )

        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        df["price"] = df[price_col]
        if "titles" in df.columns:
            df["news"] = df["titles"].fillna("").astype(str)
        else:
            df["news"] = ""

        matched = int(df["has_fresh_news"].sum())
        print(
            f"  [GeoScorer] Injected aligned Geo-Score into dataset_final.csv: "
            f"{matched} fresh-news days across {len(df)} trading days."
        )

        df.to_csv(dataset_path, index=False)
        print(f"  [GeoScorer] dataset_final.csv updated -> {dataset_path}")
        return dataset_path

    def run(self) -> pd.DataFrame:
        """Execute the full historical Geo-Score pipeline."""
        print("[GeoScorer] Step 1 -- Loading news...")
        news_df = self.load_news()

        print("\n[GeoScorer] Step 2 -- Scoring with FinBERT...")
        scored_df = self.score_articles(news_df)

        print("\n[GeoScorer] Step 3 -- Saving article-level scores...")
        self.save_scored_news(scored_df)

        print("\n[GeoScorer] Step 4 -- Aggregating by day...")
        daily_df = self.aggregate_by_day(scored_df)

        print("\n[GeoScorer] Step 5 -- Saving geo_scores.csv...")
        self.save_geo_scores(daily_df)

        print("\n[GeoScorer] Step 6 -- Injecting into dataset_final.csv...")
        self.inject_into_dataset(daily_df)

        return daily_df
