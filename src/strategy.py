"""Trading strategy engine for GeoQuant AI -- Bloc 3.

Implements the Golden Cross + Geo-Score rule:
    SI (MA50 > MA200)               <- tendance haussiere (Golden Cross)
    ET (geo_score[t-1] >= seuil)    <- sentiment d'hier acceptable
    ALORS -> Long  (position = 1.0)
    SINON -> Cash  (position = 0.0)

Correction look-ahead : geo_score[t-1] est utilise (connu la veille)
pour generer le signal applique le jour J.

Position sizing : quand le signal est Long, la position est modulee
lineairement selon la force du Geo-Score entre seuil_geo et seuil_geo + 0.5.
Exemples (seuil = -0.5) :
    geo_score = -0.50 -> position = 0.0  (seuil exactement atteint)
    geo_score = -0.25 -> position = 0.5
    geo_score >= 0.00 -> position = 1.0  (pleinement investi)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Strategy:
    """
    Merge geo_scores into dataset_final and generate trading signals.

    Attributes:
        base_dir:       Root directory of the GeoQuant AI project.
        seuil_geo:      Geo-Score threshold below which the position is closed.
                        Default -0.5.
        sizing_range:   Geo-Score range over which position scales from 0 to 1.
                        Default 0.5  (full position reached at seuil_geo + 0.5).
    """

    base_dir:     Path
    seuil_geo:    float = -0.5
    sizing_range: float = 0.5

    _dataset_path: Path = field(init=False, repr=False)
    _geo_path:     Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dataset_path = self.base_dir / "data" / "processed" / "dataset_final.csv"
        self._geo_path     = self.base_dir / "data" / "processed" / "geo_scores.csv"

    # ── Chargement ────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """
        Load dataset_final.csv and merge geo_scores.csv on date.

        The geo_score is already a lookback-smoothed value from geo_scorer.py.
        Days without a geo-score receive 0.0 (neutral).
        """
        df = pd.read_csv(self._dataset_path, parse_dates=["date"])

        # Supprime l'eventuelle colonne geo_score deja injectee par geo_scorer.py
        # pour eviter le conflit geo_score_x / geo_score_y lors du merge
        if "geo_score" in df.columns:
            df = df.drop(columns=["geo_score"])

        if self._geo_path.exists():
            geo = pd.read_csv(self._geo_path, parse_dates=["date"])
            df = df.merge(geo[["date", "geo_score"]], on="date", how="left")
        else:
            df["geo_score"] = 0.0

        df["geo_score"] = df["geo_score"].fillna(0.0)
        return df

    # ── Signal + position sizing ──────────────────────────────────────────────

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'signal' (int 0/1) and 'position' (float 0.0-1.0) columns.

        Look-ahead fix: geo_score is shifted by 1 day so that the score
        of day t-1 drives the signal of day t.

        Args:
            df: DataFrame with MA50, MA200, geo_score columns.

        Returns:
            Copy of df with 'signal' and 'position' columns added.
        """
        df = df.copy()

        # ── Correction look-ahead : utiliser geo_score d'hier ─────────────────
        geo_lag = df["geo_score"].shift(1).fillna(0.0)

        golden_cross = df["MA50"] > df["MA200"]
        geo_ok       = geo_lag >= self.seuil_geo

        # Signal binaire (0 ou 1)
        df["signal"] = ((golden_cross) & (geo_ok)).astype(int)

        # Les premieres lignes sans MA200 fiable restent en Cash
        df.loc[df["MA200"].isna(), "signal"] = 0

        # ── Position sizing : proportion 0.0 -> 1.0 ───────────────────────────
        # Quand signal = 0 -> position = 0
        # Quand signal = 1 -> position scale lineairement avec le geo_score
        scale = (geo_lag - self.seuil_geo) / max(self.sizing_range, 1e-9)
        scale = scale.clip(0.0, 1.0)

        df["position"] = np.where(df["signal"] == 1, scale, 0.0)

        return df

    # ── Pipeline complet ──────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Load, merge geo_scores, apply signal + position sizing."""
        df = self.load()
        df = self.apply(df)
        return df
