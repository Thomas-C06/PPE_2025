"""Chargeur de donnees de marche pour GeoQuant AI."""

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
    Telecharge, traite et sauvegarde les donnees de prix de marche.

    Attributes:
        tickers: Liste des tickers a telecharger.
        start_date: Date de debut au format YYYY-MM-DD.
        end_date: Date de fin au format YYYY-MM-DD.
        base_dir: Repertoire de base du projet GeoQuant_AI.
    """

    tickers: list[str]
    start_date: str
    end_date: str
    base_dir: Path

    def _ensure_dirs(self) -> tuple[Path, Path]:
        """
        Verifie que les dossiers requis existent.

        Returns:
            Un tuple (raw_dir, processed_dir).
        """
        raw_dir = self.base_dir / "data" / "raw"
        processed_dir = self.base_dir / "data" / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir, processed_dir

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Telecharge les donnees de prix depuis Yahoo Finance.

        Returns:
            Un dictionnaire mappant ticker -> DataFrame.

        Raises:
            RuntimeError: Si le telechargement echoue ou retourne des donnees vides.
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
        Calcule les rendements simples et logarithmiques pour chaque ticker.

        Args:
            data: Dictionnaire mappant ticker -> DataFrame.

        Returns:
            Dictionnaire des DataFrames traites.
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

    def compute_technical_features(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Add technical indicators to each ticker's processed DataFrame.

        Indicators added:
            - MA20, MA50, MA200  : Simple moving averages
            - RSI_14             : Relative Strength Index (14-day, Wilder EMA)
            - Volatility_20      : Rolling 20-day annualised volatility
            - Drawdown           : Rolling drawdown from the running maximum

        Args:
            data: Dictionary mapping ticker -> processed DataFrame
                  (output of compute_returns).

        Returns:
            Dictionary with the same keys, DataFrames enriched with
            technical indicator columns.
        """
        enriched: Dict[str, pd.DataFrame] = {}

        for ticker, df in data.items():
            df = df.copy()

            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            prices = df[price_col]

            # Moving averages
            df["MA20"]  = prices.rolling(window=20,  min_periods=1).mean()
            df["MA50"]  = prices.rolling(window=50,  min_periods=1).mean()
            df["MA200"] = prices.rolling(window=200, min_periods=1).mean()

            # RSI 14 (Wilder EMA method)
            delta    = prices.diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
            rs       = avg_gain / avg_loss.replace(0, np.nan)
            df["RSI_14"] = 100 - (100 / (1 + rs))

            # Volatilité glissante 20j (annualisée)
            log_ret = np.log(prices / prices.shift(1))
            df["Volatility_20"] = (
                log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252)
            )

            # Drawdown depuis le plus haut
            running_max    = prices.cummax()
            df["Drawdown"] = (prices - running_max) / running_max

            enriched[ticker] = df

        return enriched

    def save_data(
        self,
        raw_data: Dict[str, pd.DataFrame],
        processed_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Sauvegarde les donnees brutes et traitees en CSV.

        Args:
            raw_data: Dictionnaire des DataFrames bruts.
            processed_data: Dictionnaire des DataFrames traites.
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
