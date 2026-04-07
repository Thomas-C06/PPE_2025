"""Trading strategy engine for GeoQuant AI -- unified Geo-Score overlay.

Règles de trading :
    1. Correction look-ahead : geo_score[t-1] utilisé pour le signal du jour t.
    2. Long uniquement quand la tendance est haussière (MA50 > MA200).
    3. Position réduite linéairement quand le Geo-Score se dégrade.
    4. Règle d'urgence : si geo_score[t-1] < emergency_threshold → Cash immédiat.
    5. Mode TechOnly : si la dernière news date de plus de max_days_no_news jours,
       on ignore le Geo-Score et on suit le Golden Cross seul (position = 0.5).
"""

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
        4. Emergency exit if Geo-Score drops below emergency_threshold.
        5. TechOnly mode (Golden Cross alone) when news are stale.

    Attributes:
        seuil_geo:           Geo-Score threshold below which position is reduced.
        sizing_range:        Range over which position scales from 0 to 1.
        emergency_threshold: If geo_score < this → Cash regardless of trend.
                             Default -0.7 (strong negative signal = exit now).
        max_days_no_news:    After this many days without news, switch to
                             TechOnly mode (Golden Cross only, half position).
    """

    base_dir: Path
    seuil_geo: float = -0.25
    sizing_range: float = 0.55
    emergency_threshold: float = -0.7
    max_days_no_news: int = 14

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

        Modes:
        - GeoQuant : news recentes (days_since_news <= max_days_no_news)
                     -> Golden Cross ET Geo-Score, position 0.0-1.0
        - TechOnly  : news trop vieilles -> Golden Cross seul, position fixe 0.5
        - Urgence   : geo_score < emergency_threshold -> Cash immediat
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

        # ── Look-ahead correction : on utilise le score et le compteur d'hier ──
        geo_lag  = df["geo_score"].shift(1)
        days_lag = (
            df["days_since_news"].shift(1).fillna(999).astype(int)
            if "days_since_news" in df.columns
            else pd.Series(0, index=df.index)
        )

        golden_cross = df["MA50"] > df["MA200"]
        scale = ((geo_lag - self.seuil_geo) / max(self.sizing_range, 1e-9)).clip(0.0, 1.0)

        valid_signal    = geo_lag.notna() & df["MA200"].notna()
        news_fresh      = days_lag <= self.max_days_no_news
        emergency_exit  = geo_lag < self.emergency_threshold

        # ── Calcul de position selon le mode ─────────────────────────────────
        trend_multiplier = np.where(golden_cross, 1.0, 0.35)

        # Mode GeoQuant : position proportionnelle au score
        pos_geoquant = np.where(
            valid_signal & ~emergency_exit,
            trend_multiplier * scale,
            0.0,
        )
        # Mode TechOnly : Golden Cross seul, demi-position (incertitude elevee)
        pos_techonly = np.where(
            valid_signal & golden_cross & ~emergency_exit,
            0.5,
            0.0,
        )

        position = np.where(news_fresh, pos_geoquant, pos_techonly)

        df["geo_score_lag"] = geo_lag
        df["golden_cross"]  = golden_cross
        df["signal"]        = (position > 0).astype(int)
        df["position"]      = position
        df["news_mode"]     = np.where(news_fresh, "GeoQuant", "TechOnly")

        # ── Labels lisibles pour le dashboard ────────────────────────────────
        conditions = [
            ~valid_signal,
            emergency_exit,
            geo_lag < self.seuil_geo,
            golden_cross & (position >= 0.80),
            golden_cross & (position > 0.0),
            position > 0.0,
        ]
        choices = [
            "WAIT",
            "URGENCE",
            "REDUCE",
            "BUY",
            "LIGHT BUY",
            "HOLD LIGHT",
        ]
        df["decision"] = np.select(conditions, choices, default="CASH")
        df["decision_reason"] = np.select(
            [
                ~valid_signal,
                emergency_exit,
                geo_lag < self.seuil_geo,
                ~news_fresh,
                golden_cross,
                position > 0.0,
            ],
            [
                "Geo-Score indisponible la veille",
                "Geo-Score en zone de danger (sortie d'urgence)",
                "Geo-Score sous le seuil de risque",
                f"News trop anciennes (>{self.max_days_no_news}j) -- Golden Cross seul",
                "Tendance haussiere et sentiment acceptable",
                "Tendance faible mais sentiment encore exploitable",
            ],
            default="Tendance insuffisante ou sentiment trop faible",
        )

        return df

    def run(self) -> pd.DataFrame:
        """Load the dataset and apply the GeoQuant strategy."""
        return self.apply(self.load())
