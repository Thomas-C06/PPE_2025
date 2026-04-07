"""Data merger and technical feature engineering for GeoQuant AI -- Bloc 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class DataMerger:
    """
    Merge price data with aggregated news and compute technical indicators.

    The merge is a left join on the price calendar (trading days only).
    Days without fresh headlines receive nb_articles=0 and empty text fields.

    Technical indicators added:
        - MA20, MA50, MA200  : Simple moving averages
        - RSI_14             : Relative Strength Index (14-day, Wilder EMA method)
        - Volatility_20      : Rolling 20-day annualised volatility of log-returns
        - Drawdown           : Rolling drawdown from the running maximum

    Attributes:
        base_dir: Base directory of the GeoQuant AI project.
        primary_ticker: Ticker symbol used as the price source (for logging).
    """

    base_dir: Path
    primary_ticker: str = "^GSPC"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten MultiIndex columns produced by some yfinance versions.

        Args:
            df: DataFrame potentially with MultiIndex columns.

        Returns:
            DataFrame with single-level string column names.
        """
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [
                col[0] if isinstance(col, tuple) else col
                for col in df.columns
            ]
        return df

    @staticmethod
    def _price_column(df: pd.DataFrame) -> str:
        """
        Return the price column to use for technical indicators.

        Prefers 'Adj Close' when available, falls back to 'Close'.

        Args:
            df: Price DataFrame.

        Returns:
            Column name to use as the price series.

        Raises:
            ValueError: If neither 'Adj Close' nor 'Close' is found.
        """
        for col in ("Adj Close", "Close"):
            if col in df.columns:
                return col
        raise ValueError(
            f"No price column found. Available columns: {list(df.columns)}"
        )

    # ── Core logic ────────────────────────────────────────────────────────────

    def merge(
        self,
        prices: pd.DataFrame,
        news: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join a price DataFrame with the aggregated news DataFrame on date.

        Args:
            prices: Price DataFrame with a DatetimeIndex (from PriceLoader).
            news: Aggregated news DataFrame with a 'date' column.

        Returns:
            Merged DataFrame indexed from 0, with a 'date' column of dtype
            datetime64 and columns from both inputs.
        """
        df = self._flatten_columns(prices.copy())

        # Normalise l'index en colonne 'date'
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # yfinance peut nommer l'index "Date" ou "Datetime"
            for candidate in ("Date", "Datetime", "index"):
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "date"})
                    break
        else:
            df = df.reset_index(drop=True)

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Normalise la colonne date des news
        news = news.copy()
        news["date"] = pd.to_datetime(news["date"]).dt.normalize()

        merged = df.merge(news, on="date", how="left")

        # Remplit les jours sans news
        merged["nb_articles"] = merged["nb_articles"].fillna(0).astype(int)
        merged["titles"] = merged["titles"].fillna("")
        if "news" not in merged.columns:
            merged["news"] = merged["titles"]
        else:
            merged["news"] = merged["news"].fillna("")

        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    def compute_technical_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add technical indicators to a merged DataFrame.

        Args:
            df: DataFrame containing at least a 'Close' or 'Adj Close' column.

        Returns:
            DataFrame with the following columns added:
            MA20, MA50, MA200, RSI_14, Volatility_20, Drawdown.
        """
        df = df.copy()
        price_col = self._price_column(df)
        prices = df[price_col]
        df["price"] = prices

        # ── Moving averages ───────────────────────────────────────────────────
        df["MA20"]  = prices.rolling(window=20,  min_periods=1).mean()
        df["MA50"]  = prices.rolling(window=50,  min_periods=1).mean()
        df["MA200"] = prices.rolling(window=200, min_periods=1).mean()

        # ── RSI 14 (Wilder EMA method) ────────────────────────────────────────
        delta    = prices.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # ── Volatilité glissante 20j (annualisée) ─────────────────────────────
        log_ret = np.log(prices / prices.shift(1))
        df["Volatility_20"] = (
            log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252)
        )

        # ── Drawdown depuis le plus haut ──────────────────────────────────────
        running_max    = prices.cummax()
        df["Drawdown"] = (prices - running_max) / running_max

        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "dataset_final.csv",
    ) -> Path:
        """
        Save the final merged dataset to data/processed/.

        Args:
            df: Final merged and feature-enriched DataFrame.
            filename: Output filename.

        Returns:
            Absolute path to the saved file.
        """
        processed_dir = self.base_dir / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / filename
        df.to_csv(output_path, index=False)
        print(f"  [DataMerger] Saved dataset -> {output_path}")
        return output_path

    def run(
        self,
        prices: pd.DataFrame,
        news: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the full merge pipeline.

        Steps: merge prices + news -> compute_technical_features -> save.

        Args:
            prices: Processed price DataFrame from PriceLoader.
            news: Aggregated news DataFrame from NewsLoader.

        Returns:
            Final merged dataset (also saved to disk).
        """
        print(f"[DataMerger] Merging {self.primary_ticker} prices with news...")
        merged = self.merge(prices, news)
        print(
            f"  {len(merged)} trading days | "
            f"{(merged['nb_articles'] > 0).sum()} days with news"
        )

        print("[DataMerger] Computing technical features...")
        final = self.compute_technical_features(merged)

        self.save_dataset(final)
        return final
