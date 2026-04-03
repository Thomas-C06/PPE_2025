"""Trading strategy engine for GeoQuant AI -- Bloc 3.

Implements the Golden Cross + Geo-Score rule:
    SI (MA50 > MA200)  <- tendance haussiere
    ET (geo_score > seuil_geo)  <- pas de panique geopolitique
    ALORS -> Long  (signal = 1)
    SINON -> Cash  (signal = 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Strategy:
    """
    Merge geo_scores into dataset_final and generate trading signals.

    Attributes:
        base_dir:   Root directory of the GeoQuant AI project.
        seuil_geo:  Geo-Score threshold below which the position is closed.
                    Default -0.5 (cut position only on strong negative sentiment).
    """

    base_dir: Path
    seuil_geo: float = -0.5

    # Chemins derives
    _dataset_path: Path = field(init=False, repr=False)
    _geo_path:     Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset_path = self.base_dir / "data" / "processed" / "dataset_final.csv"
        self._geo_path     = self.base_dir / "data" / "processed" / "geo_scores.csv"

    # ── Chargement ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load dataset_final.csv and merge geo_scores.csv on date.

        Returns:
            DataFrame with geo_score column added.
            Days without a geo-score receive 0 (neutral).
        """
        df = pd.read_csv(self._dataset_path, parse_dates=["date"])

        if self._geo_path.exists():
            geo = pd.read_csv(self._geo_path, parse_dates=["date"])
            geo = geo[["date", "geo_score"]].rename(
                columns={"geo_score": "geo_score"}
            )
            df = df.merge(geo, on="date", how="left")
            df["geo_score"] = df["geo_score"].fillna(0.0)
        else:
            df["geo_score"] = 0.0

        return df

    # ── Signal ────────────────────────────────────────────────────────────────

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'signal' column (1 = Long, 0 = Cash) to the DataFrame.

        Rule:
            signal = 1  if  MA50 > MA200  AND  geo_score >= seuil_geo
            signal = 0  otherwise

        Args:
            df: DataFrame containing MA50, MA200, and geo_score columns.

        Returns:
            Copy of df with 'signal' column added.
        """
        df = df.copy()

        golden_cross   = df["MA50"]  > df["MA200"]
        geo_ok         = df["geo_score"] >= self.seuil_geo

        df["signal"] = ((golden_cross) & (geo_ok)).astype(int)

        # Les premieres lignes sans MA200 fiable restent en Cash
        df.loc[df["MA200"].isna(), "signal"] = 0

        return df

    # ── Pipeline complet ──────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Load data, merge geo_scores, apply signal rule.

        Returns:
            Final DataFrame with geo_score and signal columns.
        """
        df = self.load()
        df = self.apply(df)
        return df
