"""Market data loader for GeoQuant AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PriceLoader:
    """
    Download, process, and persist market price data.

    Attributes:
        tickers: List of tickers to download.
        start_date: Start date in YYYY-MM-DD.
        end_date: End date in YYYY-MM-DD.
        base_dir: Base directory of the GeoQuant_AI project.
    """

    tickers: list[str]
    start_date: str
    end_date: str
    base_dir: Path

    def _ensure_dirs(self) -> tuple[Path, Path]:
        """
        Ensure required folders exist.

        Returns:
            A tuple (raw_dir, processed_dir).
        """
        raw_dir = self.base_dir / "data" / "raw"
        processed_dir = self.base_dir / "data" / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir, processed_dir

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data from Yahoo Finance.

        Returns:
            A dictionary mapping ticker -> DataFrame.

        Raises:
            RuntimeError: If data download fails or returns empty data.
        """
        data: Dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

            if df is None or df.empty:
                raise RuntimeError(f"No data returned for {ticker}")

            df = df.copy()
            df["Ticker"] = ticker
            data[ticker] = df

        return data

    def compute_returns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute simple and log returns for each ticker.

        Args:
            data: Dictionary mapping ticker -> DataFrame.

        Returns:
            Dictionary of processed DataFrames.
        """
        processed: Dict[str, pd.DataFrame] = {}

        for ticker, df in data.items():
            df = df.copy()
            if "Adj Close" in df.columns:
                price_col = "Adj Close"
            else:
                price_col = "Close"

            df["Returns"] = df[price_col].pct_change()
            df["Log_Returns"] = np.log(df[price_col] / df[price_col].shift(1))
            processed[ticker] = df

        return processed

    def save_data(
        self,
        raw_data: Dict[str, pd.DataFrame],
        processed_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Save raw and processed data to CSV.

        Args:
            raw_data: Dictionary of raw DataFrames.
            processed_data: Dictionary of processed DataFrames.
        """
        raw_dir, processed_dir = self._ensure_dirs()

        for ticker, df in raw_data.items():
            safe_ticker = ticker.replace("/", "_")
            raw_path = raw_dir / f"{safe_ticker}_raw.csv"
            df.to_csv(raw_path, index=True)

        for ticker, df in processed_data.items():
            safe_ticker = ticker.replace("/", "_")
            processed_path = processed_dir / f"{safe_ticker}_processed.csv"
            df.to_csv(processed_path, index=True)
