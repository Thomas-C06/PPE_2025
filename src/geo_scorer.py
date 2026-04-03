"""
Geo-Score engine for GeoQuant AI -- Bloc 2 NLP.

This module is the missing link between:
    - the news data loaded from Kaggle (NewsLoader / Elena)
    - the FinBERT sentiment model (FinBertSentiment / Arthur)
    - the dataset_final.csv produced by Bloc 1 (DataMerger / Mathias & Thomas)

Pipeline
--------
    1. Load processed news CSV  (data/processed/news_financial_news_processed.csv)
    2. Score every headline with FinBERT  ->  p_positive, p_neutral, p_negative
    3. Compute per-article geo_score  =  p_positive - p_negative  (range: -1 to +1)
    4. Aggregate by calendar day      ->  mean geo_score + smoothing
    5. Save  data/processed/geo_scores.csv
    6. Inject the geo_score column into  data/processed/dataset_final.csv

Output column added to dataset_final.csv
-----------------------------------------
    geo_score : float in [-1, +1]
        > 0   -> positive sentiment  (calm / growth)
        = 0   -> neutral
        < 0   -> negative sentiment  (fear / panic)
        NaN   -> no news available for that trading day
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATASET_FINAL_FILE, GEO_SCORES_FILE, NEWS_PROCESSED_FILE
from finbert_sentiment import FinBertSentiment
from sentiment_cache import load_cache, save_cache


@dataclass
class GeoScorer:
    """
    Compute a daily Geo-Score from financial news headlines using FinBERT.

    Attributes:
        base_dir:        Root directory of the GeoQuant AI project.
        news_file:       Path to the processed news CSV (relative to base_dir
                         if not absolute). Defaults to
                         data/processed/news_financial_news_processed.csv.
        smoothing_days:  Window size for rolling-mean smoothing of the daily
                         score. Set to 1 to disable smoothing.
        batch_size:      Number of headlines per FinBERT forward pass.
        max_length:      Maximum token length passed to FinBERT (longer texts
                         are truncated).
        model_name:      Hugging Face model identifier.
    """

    base_dir: Path
    news_file: Optional[Path] = None
    smoothing_days: int = 3
    batch_size: int = 32
    max_length: int = 128
    model_name: str = "ProsusAI/finbert"

    # Resolved lazily
    _clf: Optional[FinBertSentiment] = field(default=None, init=False, repr=False)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _news_path(self) -> Path:
        if self.news_file is not None:
            p = Path(self.news_file)
            return p if p.is_absolute() else self.base_dir / p
        return self.base_dir / "data" / "processed" / NEWS_PROCESSED_FILE

    def _dataset_path(self) -> Path:
        return self.base_dir / "data" / "processed" / DATASET_FINAL_FILE

    def _geo_scores_path(self) -> Path:
        return self.base_dir / "data" / "processed" / GEO_SCORES_FILE

    def _get_clf(self) -> FinBertSentiment:
        """Lazy-load FinBERT (downloaded once, then cached by Hugging Face)."""
        if self._clf is None:
            self._clf = FinBertSentiment(
                model_name=self.model_name,
            )
        return self._clf

    # ── Step 1 : Load news ────────────────────────────────────────────────────

    def load_news(self) -> pd.DataFrame:
        """
        Load the processed news CSV produced by NewsLoader.

        Returns:
            DataFrame with at least columns: date (datetime64), headline (str).

        Raises:
            FileNotFoundError: If the news file does not exist.
        """
        path = self._news_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Processed news file not found: {path}\n"
                "Run  py -3 src/news_main.py  first to generate it."
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

    # ── Step 2 : Score articles with FinBERT ──────────────────────────────────

    def score_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run FinBERT on every headline, using the disk cache to skip duplicates.

        For each article, three probabilities are added:
            p_positive, p_neutral, p_negative  (sum to 1.0)

        The per-article geo_score = p_positive - p_negative  (range -1 to +1).

        Args:
            news_df: DataFrame with a 'headline' column.

        Returns:
            Input DataFrame extended with columns:
            label | p_positive | p_neutral | p_negative | geo_score_article
        """
        clf   = self._get_clf()
        cache = load_cache()

        headlines    = news_df["headline"].tolist()
        uncached     = [h for h in headlines if h not in cache]
        total        = len(headlines)
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

        # Rebuild results in original order
        rows = [cache[h] for h in headlines]
        pred_df = pd.DataFrame(rows)

        # Normalise column names (cache may store them as p_positive etc.)
        for col in ("p_positive", "p_neutral", "p_negative"):
            if col not in pred_df.columns:
                pred_df[col] = 0.0

        scored = pd.concat(
            [news_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )

        # Per-article geo_score: positive pull minus negative pull
        scored["geo_score_article"] = scored["p_positive"] - scored["p_negative"]

        return scored

    # ── Step 3 : Aggregate by day ─────────────────────────────────────────────

    def aggregate_by_day(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate article-level scores into one daily Geo-Score.

        Aggregation method: confidence-weighted mean of geo_score_article.
        Confidence = max(p_positive, p_neutral, p_negative).

        A smoothed version is also computed (rolling mean over
        ``smoothing_days`` days) to reduce day-to-day noise.

        Args:
            scored_df: DataFrame returned by :meth:`score_articles`.

        Returns:
            DataFrame with columns:
            date | geo_score | geo_score_raw | nb_articles_scored
        """
        scored_df = scored_df.copy()
        scored_df["confidence"] = scored_df[
            ["p_positive", "p_neutral", "p_negative"]
        ].max(axis=1)

        def weighted_mean(g: pd.DataFrame) -> float:
            w = g["confidence"]
            total_w = float(w.sum())
            if total_w == 0.0:
                return float(g["geo_score_article"].mean())
            return float((g["geo_score_article"] * w).sum() / total_w)

        daily = (
            scored_df.groupby("date")
            .apply(
                lambda g: pd.Series(
                    {
                        "geo_score_raw": weighted_mean(g),
                        "nb_articles_scored": len(g),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )

        daily = daily.sort_values("date").reset_index(drop=True)

        # Smoothing lookback-only : la moyenne du jour J utilise J, J-1, J-2
        # center=False garantit qu'on ne regarde que dans le passe (causal).
        daily["geo_score"] = (
            daily["geo_score_raw"]
            .rolling(window=self.smoothing_days, min_periods=1, center=False)
            .mean()
        )

        # Clamp to [-1, +1] after smoothing
        daily["geo_score"] = daily["geo_score"].clip(-1.0, 1.0)

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

    # ── Step 4 : Save geo_scores.csv ─────────────────────────────────────────

    def save_geo_scores(self, daily_df: pd.DataFrame) -> Path:
        """
        Save the daily Geo-Score table to disk.

        Output: data/processed/geo_scores.csv
        Columns: date | geo_score | geo_score_raw | nb_articles_scored

        Args:
            daily_df: DataFrame returned by :meth:`aggregate_by_day`.

        Returns:
            Path to the saved file.
        """
        out = self._geo_scores_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(out, index=False)
        print(f"  [GeoScorer] Geo-scores saved to {out}")
        return out

    # ── Step 5 : Inject into dataset_final.csv ────────────────────────────────

    def inject_into_dataset(self, daily_df: pd.DataFrame) -> Optional[Path]:
        """
        Add the ``geo_score`` column to dataset_final.csv (Bloc 1 output).

        If dataset_final.csv does not exist yet, this step is skipped
        gracefully and a warning is printed instead.

        Args:
            daily_df: DataFrame returned by :meth:`aggregate_by_day`.

        Returns:
            Path to the updated dataset_final.csv, or None if it did not exist.
        """
        dataset_path = self._dataset_path()

        if not dataset_path.exists():
            print(
                "  [GeoScorer] dataset_final.csv not found -- skipping injection.\n"
                "              Run Bloc 1 (py -3 src/main.py) first, then re-run "
                "this script to inject the geo_score."
            )
            return None

        df = pd.read_csv(dataset_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        daily = daily_df.copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

        # Drop old geo_score columns if they already exist (re-run scenario)
        for col in ("geo_score", "geo_score_raw", "nb_articles_scored"):
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.merge(
            daily[["date", "geo_score", "geo_score_raw", "nb_articles_scored"]],
            on="date",
            how="left",
        )

        matched = df["geo_score"].notna().sum()
        print(
            f"  [GeoScorer] Injected geo_score into dataset_final.csv: "
            f"{matched} / {len(df)} trading days matched."
        )

        df.to_csv(dataset_path, index=False)
        print(f"  [GeoScorer] dataset_final.csv updated -> {dataset_path}")
        return dataset_path

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full Geo-Score pipeline.

        Steps:
            1. load_news
            2. score_articles  (FinBERT + cache)
            3. aggregate_by_day
            4. save_geo_scores
            5. inject_into_dataset

        Returns:
            Daily Geo-Score DataFrame (date | geo_score | geo_score_raw |
            nb_articles_scored).
        """
        print("[GeoScorer] Step 1 -- Loading news...")
        news_df = self.load_news()

        print("\n[GeoScorer] Step 2 -- Scoring with FinBERT...")
        scored_df = self.score_articles(news_df)

        print("\n[GeoScorer] Step 3 -- Aggregating by day...")
        daily_df = self.aggregate_by_day(scored_df)

        print("\n[GeoScorer] Step 4 -- Saving geo_scores.csv...")
        self.save_geo_scores(daily_df)

        print("\n[GeoScorer] Step 5 -- Injecting into dataset_final.csv...")
        self.inject_into_dataset(daily_df)

        return daily_df
