"""News data loader for GeoQuant AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from news_config import CANONICAL_COLUMNS, COLUMN_MAPS


@dataclass
class NewsLoader:
    """
    Load, normalise, and persist financial news data from CSV datasets.

    Supported datasets (pass the key as ``dataset``):
        - ``"all_the_news"``  : Kaggle "All the News 2.0"
          https://www.kaggle.com/datasets/snapcrack/all-the-news
        - ``"financial_news"``: Kaggle "US Financial News Articles"
          https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles

    The normalised output always exposes four columns::

        date (datetime64) | headline (str) | source (str) | content (str | NaN)

    Attributes:
        csv_path:   Absolute path to the downloaded Kaggle CSV file.
        dataset:    Dataset identifier key (see ``COLUMN_MAPS`` in news_config).
        start_date: Keep only articles on or after this date (YYYY-MM-DD).
        end_date:   Keep only articles on or before this date (YYYY-MM-DD).
        base_dir:   Root directory of the GeoQuant AI project.
    """

    csv_path: Path
    dataset: str
    start_date: str
    end_date: str
    base_dir: Path
    # Resolved once at load time — not part of the public interface
    _col_map: dict[str, Optional[str]] = field(init=False, repr=False)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate inputs and resolve the column mapping."""
        if self.dataset not in COLUMN_MAPS:
            supported = list(COLUMN_MAPS.keys())
            raise ValueError(
                f"Unknown dataset '{self.dataset}'. "
                f"Supported values: {supported}"
            )
        self._col_map = COLUMN_MAPS[self.dataset]

        if not Path(self.csv_path).exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}\n"
                "Download the dataset from Kaggle and place it under data/raw/."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> tuple[Path, Path]:
        """
        Create raw and processed output directories if they do not exist.

        Returns:
            A tuple (raw_dir, processed_dir).
        """
        raw_dir = self.base_dir / "data" / "raw"
        processed_dir = self.base_dir / "data" / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir, processed_dir

    def _rename_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename dataset-specific columns to the canonical schema.

        Columns not present in the mapping are dropped. The ``content``
        column is added as NaN when the dataset does not provide full text.

        Args:
            df: Raw DataFrame loaded from CSV.

        Returns:
            DataFrame with exactly the columns in ``CANONICAL_COLUMNS``.

        Raises:
            KeyError: If a required source column is missing from the CSV.
        """
        rename: dict[str, str] = {}
        for canonical, original in self._col_map.items():
            if original is None:
                continue
            if original not in df.columns:
                raise KeyError(
                    f"Expected column '{original}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            rename[original] = canonical

        df = df.rename(columns=rename)

        # Keep only canonical columns; create missing ones as NaN
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        return df[CANONICAL_COLUMNS]

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the ``date`` column to ``datetime64[ns]`` and drop unparseable rows.

        Args:
            df: DataFrame with a raw ``date`` column.

        Returns:
            DataFrame with a proper ``datetime64`` date column.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        # Remove timezone info if present so comparisons are tz-naive
        if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        n_before = len(df)
        df = df.dropna(subset=["date"])
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  [NewsLoader] Dropped {n_dropped} rows with unparseable dates.")
        return df

    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only rows whose date falls within [start_date, end_date].

        Args:
            df: DataFrame with a parsed ``date`` column.

        Returns:
            Filtered DataFrame.
        """
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        mask = (df["date"] >= start) & (df["date"] <= end)
        filtered = df.loc[mask].copy()
        print(
            f"  [NewsLoader] Date filter {self.start_date} to {self.end_date}: "
            f"{len(df)} to {len(filtered)} articles."
        )
        return filtered

    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip whitespace from ``headline`` and ``source``; normalise encoding.

        Args:
            df: DataFrame after date filtering.

        Returns:
            DataFrame with cleaned text columns.
        """
        df = df.copy()
        for col in ("headline", "source"):
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("nan", pd.NA)
            )
        df = df.dropna(subset=["headline"])
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """
        Read the CSV file as-is, without any transformation.

        Returns:
            Raw DataFrame exactly as stored in the file.

        Raises:
            RuntimeError: If the CSV cannot be read.
        """
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV '{self.csv_path}': {exc}") from exc

        print(f"  [NewsLoader] Loaded {len(df):,} rows from '{self.csv_path.name}'.")
        return df

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw DataFrame into the canonical news schema.

        Pipeline:
            1. Rename columns to canonical names.
            2. Parse dates.
            3. Filter by date range.
            4. Clean text.
            5. Reset index.

        Args:
            raw_df: DataFrame returned by :meth:`load_raw`.

        Returns:
            Processed DataFrame with columns: date | headline | source | content.
        """
        df = self._rename_to_canonical(raw_df)
        df = self._parse_dates(df)
        df = self._filter_dates(df)
        df = self._clean_text(df)
        df = df.reset_index(drop=True)
        print(f"  [NewsLoader] Processing complete. {len(df):,} clean articles.")
        return df

    def save_data(
        self,
        raw_df: pd.DataFrame,
        processed_df: pd.DataFrame,
    ) -> None:
        """
        Persist raw and processed DataFrames to CSV.

        Output files:
            - ``data/raw/news_{dataset}_raw.csv``
            - ``data/processed/news_{dataset}_processed.csv``

        Args:
            raw_df:       DataFrame returned by :meth:`load_raw`.
            processed_df: DataFrame returned by :meth:`process`.
        """
        raw_dir, processed_dir = self._ensure_dirs()

        raw_path = raw_dir / f"news_{self.dataset}_raw.csv"
        processed_path = processed_dir / f"news_{self.dataset}_processed.csv"

        raw_df.to_csv(raw_path, index=False)
        processed_df.to_csv(processed_path, index=False)

        print(f"  [NewsLoader] Raw data saved to      {raw_path}")
        print(f"  [NewsLoader] Processed data saved to {processed_path}")

    def get_processed_path(self) -> Path:
        """
        Return the expected path of the processed CSV (may not exist yet).

        Returns:
            Path to ``data/processed/news_{dataset}_processed.csv``.
        """
        return self.base_dir / "data" / "processed" / f"news_{self.dataset}_processed.csv"
