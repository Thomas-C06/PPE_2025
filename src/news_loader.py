"""News data loader for GeoQuant AI -- Bloc 1 Data Engineering."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class NewsLoader:
    """
    Load, clean, and aggregate financial/geopolitical news data.

    Supports a local sample CSV and is designed to accommodate Kaggle
    datasets with automatic column detection.

    Attributes:
        base_dir: Base directory of the GeoQuant AI project.
        news_file: Path to the input CSV (absolute, or relative to base_dir).
                   Defaults to data/raw/sample_news.csv.
        date_col: Name of the date column in the source CSV.
        title_col: Name of the title/headline column in the source CSV.
    """

    base_dir: Path
    news_file: Optional[Path] = None
    date_col: str = "date"
    title_col: str = "title"

    # Listes des noms de colonnes candidats pour la détection automatique
    _KNOWN_DATE_COLS: list[str] = field(
        default_factory=lambda: [
            "date", "Date", "published", "publishedAt",
            "datetime", "timestamp", "pub_date", "Datetime",
        ]
    )
    _KNOWN_TITLE_COLS: list[str] = field(
        default_factory=lambda: [
            "title", "Title", "headline", "Headline",
            "text", "Text", "content", "Content", "description",
        ]
    )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_news_path(self) -> Path:
        """Return the default sample news CSV path."""
        return self.base_dir / "data" / "raw" / "sample_news.csv"

    def _resolve_path(self) -> Path:
        """Return the resolved (absolute) path to the news file."""
        if self.news_file is None:
            return self._default_news_path()
        p = Path(self.news_file)
        return p if p.is_absolute() else self.base_dir / p

    def _detect_columns(self, df: pd.DataFrame) -> tuple[str, str]:
        """
        Auto-detect date and title columns from a DataFrame.

        Args:
            df: Source DataFrame.

        Returns:
            Tuple (date_col, title_col) of detected column names.

        Raises:
            ValueError: If a required column cannot be auto-detected.
        """
        date_col: Optional[str] = None
        title_col: Optional[str] = None

        for candidate in self._KNOWN_DATE_COLS:
            if candidate in df.columns:
                date_col = candidate
                break

        for candidate in self._KNOWN_TITLE_COLS:
            if candidate in df.columns:
                title_col = candidate
                break

        if date_col is None:
            raise ValueError(
                f"Cannot auto-detect date column. "
                f"Available columns: {list(df.columns)}"
            )
        if title_col is None:
            raise ValueError(
                f"Cannot auto-detect title column. "
                f"Available columns: {list(df.columns)}"
            )

        return date_col, title_col

    # ── Public API ────────────────────────────────────────────────────────────

    def load_raw(self) -> pd.DataFrame:
        """
        Load raw news CSV from disk.

        Returns:
            Raw DataFrame as read from the CSV file.

        Raises:
            FileNotFoundError: If the news file does not exist.
        """
        path = self._resolve_path()
        if not path.exists():
            raise FileNotFoundError(
                f"News file not found: {path}\n"
                "Place a CSV at data/raw/sample_news.csv or pass news_file= to NewsLoader."
            )

        df = pd.read_csv(path, encoding="utf-8")

        # Auto-détection des colonnes si les noms configurés sont absents
        if self.date_col not in df.columns or self.title_col not in df.columns:
            detected_date, detected_title = self._detect_columns(df)
            print(
                f"  [NewsLoader] Auto-detected columns: "
                f"date='{detected_date}', title='{detected_title}'"
            )
            self.date_col = detected_date
            self.title_col = detected_title

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw news data.

        Drops rows with invalid dates, empty titles, and exact duplicates.
        Normalises the date column to midnight (no time component).

        Args:
            df: Raw news DataFrame.

        Returns:
            Cleaned DataFrame with standardised 'date' (datetime) and 'title' columns.
        """
        df = df.copy()

        # Renomme vers les noms standardisés
        rename_map: dict[str, str] = {}
        if self.date_col != "date":
            rename_map[self.date_col] = "date"
        if self.title_col != "title":
            rename_map[self.title_col] = "title"
        if rename_map:
            df = df.rename(columns=rename_map)

        # Parse dates -- les valeurs invalides deviennent NaT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["date"])
        dropped_dates = before - len(df)
        if dropped_dates:
            print(f"  [NewsLoader] Dropped {dropped_dates} rows with invalid dates.")

        # Supprime les titres vides ou manquants
        df["title"] = df["title"].astype(str).str.strip()
        df = df[~df["title"].isin(["", "nan", "NaN", "None"])]
        dropped_titles = (before - dropped_dates) - len(df)
        if dropped_titles:
            print(f"  [NewsLoader] Dropped {dropped_titles} rows with empty titles.")

        # Supprime les doublons (même date + même titre)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["date", "title"])
        dropped_dedup = before_dedup - len(df)
        if dropped_dedup:
            print(f"  [NewsLoader] Dropped {dropped_dedup} duplicate rows.")

        # Normalise à la date (sans heure)
        df["date"] = df["date"].dt.normalize()

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def aggregate_by_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news by calendar day.

        For each day, computes the number of articles and concatenates
        all headlines into a single string separated by ' | '.

        Args:
            df: Cleaned news DataFrame with 'date' and 'title' columns.

        Returns:
            DataFrame with columns: date, nb_articles, titles.
        """
        grouped = (
            df.groupby("date")
            .agg(
                nb_articles=("title", "count"),
                titles=("title", lambda x: " | ".join(x)),
            )
            .reset_index()
        )
        grouped = grouped.sort_values("date").reset_index(drop=True)
        return grouped

    def save_aggregated(
        self,
        df: pd.DataFrame,
        filename: str = "news_aggregated.csv",
    ) -> Path:
        """
        Save aggregated news to data/processed/.

        Args:
            df: Aggregated news DataFrame.
            filename: Output filename (default: news_aggregated.csv).

        Returns:
            Absolute path to the saved file.
        """
        processed_dir = self.base_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"  [NewsLoader] Saved aggregated news -> {output_path}")
        return output_path

    def run(self) -> pd.DataFrame:
        """
        Execute the full news loading pipeline.

        Steps: load_raw -> clean -> aggregate_by_day -> save_aggregated.

        Returns:
            Aggregated news DataFrame (date, nb_articles, titles).
        """
        print("[NewsLoader] Loading raw news...")
        raw = self.load_raw()
        print(f"  Loaded {len(raw)} raw articles.")

        print("[NewsLoader] Cleaning...")
        cleaned = self.clean(raw)
        print(f"  {len(cleaned)} articles after cleaning.")

        print("[NewsLoader] Aggregating by day...")
        aggregated = self.aggregate_by_day(cleaned)
        print(f"  {len(aggregated)} days with news.")

        self.save_aggregated(aggregated)
        return aggregated
