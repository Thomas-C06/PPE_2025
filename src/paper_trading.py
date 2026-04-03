"""Paper Trading engine for GeoQuant AI.

Simule un portefeuille virtuel en temps reel en utilisant :
    - Les prix S&P 500 en direct via yfinance (cours du jour)
    - Les actualites du jour via Yahoo Finance RSS
    - FinBERT pour calculer le Geo-Score sur les news d'aujourd'hui
    - La meme regle de trading que le Bloc 3 (Golden Cross + Geo-Score)

Le portefeuille est persiste dans data/processed/paper_portfolio.json
entre les sessions pour permettre un suivi dans le temps.

Fonctionnement :
    1. Recupere le prix actuel du S&P 500
    2. Recupere les news d'aujourd'hui via Yahoo RSS
    3. Calcule le Geo-Score des news avec FinBERT (avec cache)
    4. Applique la regle : MA50 > MA200 ET geo_score >= seuil => Long
    5. Met a jour le portefeuille virtuel (capital, position, historique)
    6. Retourne le snapshot complet pour affichage dans le dashboard
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ── Constantes ────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = "paper_portfolio.json"
CAPITAL_INITIAL = 10_000.0   # capital de depart en euros virtuels


# ── Structures de donnees ──────────────────────────────────────────────────────

@dataclass
class LiveSnapshot:
    """Etat du marche et signal au moment du refresh."""

    timestamp:    str          # ex. "2024-03-15 14:32:00"
    ticker:       str
    prix_actuel:  float
    ma50:         float
    ma200:        float
    golden_cross: bool
    geo_score:    float
    nb_news:      int
    headlines:    List[str]    # titres des news d'aujourd'hui
    signal:       int          # 1 = Long, 0 = Cash
    position:     float        # 0.0 a 1.0
    seuil_geo:    float


@dataclass
class PortfolioState:
    """Etat du portefeuille virtuel."""

    capital_initial:  float = CAPITAL_INITIAL
    capital_cash:     float = CAPITAL_INITIAL
    capital_investi:  float = 0.0
    prix_entree:      Optional[float] = None
    date_entree:      Optional[str] = None
    nb_parts:         float = 0.0         # unites de l'indice detenues
    nb_trades:        int = 0
    trades_gagnants:  int = 0
    historique:       List[Dict[str, Any]] = field(default_factory=list)

    @property
    def valeur_totale(self) -> float:
        return self.capital_cash + self.capital_investi

    @property
    def rendement_total(self) -> float:
        return (self.valeur_totale / self.capital_initial) - 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PortfolioState":
        hist = d.pop("historique", [])
        obj = cls(**d)
        obj.historique = hist
        return obj


# ── Fonctions de marche ────────────────────────────────────────────────────────

def _prix_live(ticker: str = "^GSPC") -> Dict[str, float]:
    """
    Recupere le prix actuel et calcule MA50 / MA200 depuis yfinance.

    Returns:
        Dict avec cles : prix, ma50, ma200
    """
    hist = yf.download(
        ticker,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if hist is None or hist.empty:
        raise RuntimeError(f"Impossible de recuperer les prix pour {ticker}")

    # Aplatir MultiIndex si present (yfinance >= 0.2.40)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [col[0] for col in hist.columns]

    price_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    prices    = hist[price_col].dropna()

    prix   = float(prices.iloc[-1])
    ma50   = float(prices.rolling(50,  min_periods=1).mean().iloc[-1])
    ma200  = float(prices.rolling(200, min_periods=1).mean().iloc[-1])

    return {"prix": prix, "ma50": ma50, "ma200": ma200}


def _news_live(ticker: str = "^GSPC", count: int = 30) -> List[Dict[str, Any]]:
    """
    Recupere les actualites du jour via Yahoo Finance RSS.
    Retourne une liste de dicts {title, published, summary}.
    """
    # Mapping indice -> symbole lisible pour Yahoo RSS
    _RSS_MAP = {
        "^GSPC": "SPY",   # Yahoo RSS ne reconnait pas ^GSPC directement
        "^FCHI": "EWQ",
    }
    rss_ticker = _RSS_MAP.get(ticker, ticker)

    try:
        from yahoo_news import fetch_yahoo_news
        return fetch_yahoo_news(rss_ticker, count=count)
    except Exception:
        return []


def _geo_score_live(
    headlines: List[str],
    cache_path: Path,
) -> float:
    """
    Calcule le Geo-Score moyen sur une liste de titres avec FinBERT.
    Utilise le cache existant pour eviter les re-calculs.

    Returns:
        Geo-Score moyen dans [-1, +1], ou 0.0 si aucun titre.
    """
    if not headlines:
        return 0.0

    from finbert_sentiment import FinBertSentiment
    from sentiment_cache import load_cache, save_cache

    cache = load_cache(cache_path) if cache_path.exists() else {}

    uncached = [h for h in headlines if h not in cache]

    if uncached:
        clf     = FinBertSentiment()
        pred_df = clf.predict_df(uncached)
        for i, text in enumerate(uncached):
            cache[text] = pred_df.iloc[i].to_dict()
        save_cache(cache, cache_path)

    scores = []
    for h in headlines:
        if h in cache:
            row   = cache[h]
            score = float(row.get("p_positive", 0)) - float(row.get("p_negative", 0))
            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def _position_sizing(geo_score: float, seuil_geo: float, sizing_range: float = 0.5) -> float:
    """Meme logique que strategy.py."""
    scale = (geo_score - seuil_geo) / max(sizing_range, 1e-9)
    return float(np.clip(scale, 0.0, 1.0))


# ── Classe principale ──────────────────────────────────────────────────────────

class PaperTrader:
    """
    Moteur de paper trading en temps reel.

    Attributes:
        base_dir:   Racine du projet.
        ticker:     Ticker a surveiller (defaut: ^GSPC).
        seuil_geo:  Seuil Geo-Score (synchronise avec le slider du dashboard).
    """

    def __init__(
        self,
        base_dir: Path,
        ticker:    str   = "^GSPC",
        seuil_geo: float = -0.5,
    ) -> None:
        self.base_dir  = base_dir
        self.ticker    = ticker
        self.seuil_geo = seuil_geo

        self._portfolio_path = base_dir / "data" / "processed" / PORTFOLIO_FILE
        self._cache_path     = base_dir / "data" / "processed" / "sentiment_cache.json"

    # ── Persistance du portefeuille ───────────────────────────────────────────

    def load_portfolio(self) -> PortfolioState:
        """Charge le portefeuille depuis le fichier JSON, ou cree un nouveau."""
        if self._portfolio_path.exists():
            with open(self._portfolio_path, encoding="utf-8") as f:
                data = json.load(f)
            return PortfolioState.from_dict(data)
        return PortfolioState()

    def save_portfolio(self, portfolio: PortfolioState) -> None:
        self._portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._portfolio_path, "w", encoding="utf-8") as f:
            json.dump(portfolio.to_dict(), f, ensure_ascii=False, indent=2)

    # ── Signal live ───────────────────────────────────────────────────────────

    def get_live_snapshot(self, use_finbert: bool = True) -> LiveSnapshot:
        """
        Recupere les donnees en temps reel et calcule le signal du moment.

        Args:
            use_finbert: Si False, utilise geo_score = 0 (mode rapide sans GPU).

        Returns:
            LiveSnapshot avec le signal et toutes les donnees du marche.
        """
        # 1. Prix + MA
        market = _prix_live(self.ticker)

        # 2. News du jour
        raw_news = _news_live(self.ticker, count=30)
        headlines = [n.get("title", "") for n in raw_news if n.get("title")]

        # 3. Geo-Score
        if use_finbert and headlines:
            geo_score = _geo_score_live(headlines, self._cache_path)
        else:
            geo_score = 0.0

        # 4. Signal
        golden_cross = market["ma50"] > market["ma200"]
        geo_ok       = geo_score >= self.seuil_geo
        signal       = 1 if (golden_cross and geo_ok) else 0
        position     = _position_sizing(geo_score, self.seuil_geo) if signal == 1 else 0.0

        return LiveSnapshot(
            timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker       = self.ticker,
            prix_actuel  = market["prix"],
            ma50         = market["ma50"],
            ma200        = market["ma200"],
            golden_cross = golden_cross,
            geo_score    = geo_score,
            nb_news      = len(headlines),
            headlines    = headlines[:10],  # afficher les 10 premiers
            signal       = signal,
            position     = position,
            seuil_geo    = self.seuil_geo,
        )

    # ── Execution du signal ───────────────────────────────────────────────────

    def execute_signal(
        self,
        snapshot:  LiveSnapshot,
        portfolio: PortfolioState,
    ) -> tuple[PortfolioState, str]:
        """
        Applique le signal au portefeuille virtuel.

        Logique :
        - Si signal=1 et position=0 -> ENTREE (acheter position * capital)
        - Si signal=0 et position>0 -> SORTIE (tout vendre)
        - Sinon -> maintenir l'etat actuel

        Args:
            snapshot:  Signal live du moment.
            portfolio: Etat actuel du portefeuille.

        Returns:
            (portfolio mis a jour, message de l'action effectuee)
        """
        portfolio  = PortfolioState.from_dict(portfolio.to_dict())   # copie
        prix       = snapshot.prix_actuel
        cible_pos  = snapshot.position    # fraction cible du capital a investir
        pos_actuelle = portfolio.nb_parts > 0

        message = "Aucune action — position inchangee."

        # ── SORTIE totale si signal = 0 et position ouverte ───────────────────
        if cible_pos == 0.0 and pos_actuelle:
            valeur_pos    = portfolio.nb_parts * prix
            pnl           = valeur_pos - portfolio.capital_investi
            pnl_pct       = pnl / portfolio.capital_investi if portfolio.capital_investi > 0 else 0.0

            portfolio.capital_cash    += valeur_pos
            portfolio.capital_investi  = 0.0
            portfolio.nb_parts         = 0.0

            portfolio.nb_trades += 1
            if pnl > 0:
                portfolio.trades_gagnants += 1

            portfolio.historique.append({
                "date":    snapshot.timestamp,
                "action":  "SORTIE",
                "prix":    round(prix, 2),
                "nb_parts": 0.0,
                "valeur":  round(valeur_pos, 2),
                "pnl_eur": round(pnl, 2),
                "pnl_pct": f"{pnl_pct:+.2%}",
                "geo_score": round(snapshot.geo_score, 4),
                "signal":  0,
            })
            message = f"SORTIE — Vendu a {prix:,.2f} | P&L : {pnl:+.2f} EUR ({pnl_pct:+.2%})"

        # ── ENTREE si signal = 1 et pas de position ───────────────────────────
        elif cible_pos > 0.0 and not pos_actuelle:
            montant_a_investir = portfolio.capital_cash * cible_pos
            nb_parts           = montant_a_investir / prix

            portfolio.capital_cash    -= montant_a_investir
            portfolio.capital_investi  = montant_a_investir
            portfolio.nb_parts         = nb_parts
            portfolio.prix_entree      = prix
            portfolio.date_entree      = snapshot.timestamp

            portfolio.historique.append({
                "date":    snapshot.timestamp,
                "action":  "ENTREE",
                "prix":    round(prix, 2),
                "nb_parts": round(nb_parts, 6),
                "valeur":  round(montant_a_investir, 2),
                "pnl_eur": 0.0,
                "pnl_pct": "--",
                "geo_score": round(snapshot.geo_score, 4),
                "signal":  1,
            })
            message = (
                f"ENTREE — Achat {cible_pos:.0%} du capital a {prix:,.2f} "
                f"({montant_a_investir:,.2f} EUR | {nb_parts:.6f} parts)"
            )

        # ── Mise a jour de la valeur de marche si investi ─────────────────────
        elif pos_actuelle:
            portfolio.capital_investi = portfolio.nb_parts * prix
            message = f"Position maintenue — valeur actuelle : {portfolio.capital_investi:,.2f} EUR"

        return portfolio, message

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_portfolio(self) -> PortfolioState:
        """Remet le portefeuille a zero (capital initial)."""
        portfolio = PortfolioState()
        self.save_portfolio(portfolio)
        return portfolio
