"""Trading strategy engine for GeoQuant AI -- unified Geo-Score overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    """
    Apply the historical GeoQuant rule on the aligned market dataset.

    Logic:
        1. Use the previous day's Geo-Score to avoid look-ahead bias.
        2. Stay long only when the market trend is positive (MA50 > MA200).
        3. Reduce exposure linearly when the Geo-Score deteriorates.
    """

    base_dir: Path
    seuil_geo: float = -0.25
    sizing_range: float = 0.55

    _dataset_path: Path = field(init=False, repr=False)
    _geo_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset_path = self.base_dir / "data" / "processed" / "dataset_final.csv"
        self._geo_path = self.base_dir / "data" / "processed" / "geo_scores.csv"

    def load(self) -> pd.DataFrame:
        """
        Load the aligned historical dataset.

        If `geo_score` is already present in dataset_final.csv, it is trusted as
        the canonical strategy input. Otherwise, the method falls back to a
        direct merge with `geo_scores.csv`.
        """
        df = pd.read_csv(self._dataset_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if "geo_score" not in df.columns:
            if not self._geo_path.exists():
                raise FileNotFoundError(
                    "No Geo-Score found in dataset_final.csv and geo_scores.csv is missing."
                )
            geo = pd.read_csv(self._geo_path, parse_dates=["date"])
            df = df.merge(geo[["date", "geo_score"]], on="date", how="left")

        missing = int(df["geo_score"].isna().sum())
        if missing:
            raise ValueError(
                f"dataset_final.csv contains {missing} missing Geo-Score values. "
                "Re-run `python src/run_geo_scorer.py` to rebuild the aligned dataset."
            )

        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the strategy signal, position sizing, and human-readable decision.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

        geo_lag = df["geo_score"].shift(1)
        golden_cross = df["MA50"] > df["MA200"]
        scale = ((geo_lag - self.seuil_geo) / max(self.sizing_range, 1e-9)).clip(0.0, 1.0)

        valid_signal = geo_lag.notna() & df["MA200"].notna()
        trend_multiplier = np.where(golden_cross, 1.0, 0.35)
        position = np.where(valid_signal, trend_multiplier * scale, 0.0)

        df["geo_score_lag"] = geo_lag
        df["golden_cross"] = golden_cross
        df["signal"] = (position > 0).astype(int)
        df["position"] = position

        conditions = [
            ~valid_signal,
            geo_lag < self.seuil_geo,
            golden_cross & (position >= 0.80),
            golden_cross & (position > 0.0),
            position > 0.0,
        ]
        choices = [
            "WAIT",
            "REDUCE",
            "BUY",
            "LIGHT BUY",
            "HOLD LIGHT",
        ]
        df["decision"] = np.select(conditions, choices, default="CASH")
        df["decision_reason"] = np.select(
            [
                ~valid_signal,
                geo_lag < self.seuil_geo,
                golden_cross,
                position > 0.0,
            ],
            [
                "Geo-Score indisponible la veille",
                "Geo-Score sous le seuil de risque",
                "Tendance haussiere et sentiment acceptable",
                "Tendance faible mais sentiment encore exploitable",
            ],
            default="Tendance insuffisante ou sentiment trop faible",
        )

        return df

    def run(self) -> pd.DataFrame:
        """Load the dataset and apply the GeoQuant strategy."""
        return self.apply(self.load())
