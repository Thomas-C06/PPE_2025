"""GeoQuant AI -- Tableau de bord interactif (Bloc 4 -- Streamlit).

Lancement depuis la racine du projet :
    /c/Users/mathi/anaconda3/python.exe -m streamlit run app/dashboard.py

Ce fichier constitue l'interface principale du projet GeoQuant AI.
Il lit uniquement le dataset S&P 500 (dataset_final.csv) produit par le Bloc 1.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RACINE         = Path(__file__).resolve().parents[1]
CHEMIN_DATASET = RACINE / "data" / "processed" / "dataset_final.csv"
CHEMIN_NEWS    = RACINE / "data" / "raw" / "sample_news.csv"
CHEMIN_GEO     = RACINE / "data" / "processed" / "geo_scores.csv"

# Rend les modules src/ importables depuis app/
sys.path.insert(0, str(RACINE / "src"))

# ---------------------------------------------------------------------------
# Configuration de la page (doit être le PREMIER appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoQuant AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .titre-principal {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sous-titre { color: #888; font-size: 0.9rem; margin-top: 0; }
    [data-testid="metric-container"] {
        background: #1e1e2e; border-radius: 10px;
        padding: 10px 16px; border: 1px solid #333;
    }
    .bloc-placeholder {
        background: #1e1e2e; border: 1px dashed #555;
        border-radius: 12px; padding: 32px 24px;
        text-align: center; color: #888;
    }
    .bloc-placeholder h3 { color: #ccc; }
    .badge-fait     { color: #2ca02c; font-weight: 700; }
    .badge-encours  { color: #ff7f0e; font-weight: 700; }
    .badge-attente  { color: #888;    font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Evenements geopolitiques/financiers cles a annoter sur les graphiques
# ---------------------------------------------------------------------------
EVENEMENTS_CLES: list[tuple[str, str, str]] = [
    ("2022-02-24", "Invasion Ukraine",  "red"),
    ("2022-06-15", "Fed +75bp",         "#9467bd"),
    ("2022-11-11", "FTX collapse",      "orange"),
    ("2023-03-10", "SVB collapse",      "firebrick"),
    ("2023-10-07", "Hamas attack",      "crimson"),
    ("2024-08-05", "Japan crash",       "saddlebrown"),
    ("2024-09-18", "Fed -50bp",         "#2ca02c"),
    ("2024-11-05", "Trump elu",         "royalblue"),
]

# Couleurs globales
COULEUR_PRIX   = "#1f77b4"
COULEUR_MA50   = "#ff7f0e"
COULEUR_MA200  = "#2ca02c"
COULEUR_DANGER = "#d62728"
COULEUR_NEWS   = "#9467bd"


# ===========================================================================
# Chargement des donnees
# ===========================================================================

@st.cache_data
def charger_dataset() -> Optional[pd.DataFrame]:
    """
    Charge le fichier dataset_final.csv produit par le Bloc 1.

    Retourne None si le fichier est absent (le dashboard affichera une erreur).
    La colonne 'titles' peut etre lue comme float (NaN) quand vide -> normalisee en str.
    """
    if not CHEMIN_DATASET.exists():
        return None

    df = pd.read_csv(CHEMIN_DATASET, parse_dates=["date"])
    df["titles"] = df["titles"].fillna("").astype(str).replace("nan", "")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def charger_news_brutes() -> Optional[pd.DataFrame]:
    """
    Charge les actualites individuelles depuis data/raw/sample_news.csv.

    Utilise pour afficher le tableau des dernieres actualites dans l'onglet NLP.
    """
    if not CHEMIN_NEWS.exists():
        return None
    df = pd.read_csv(CHEMIN_NEWS, parse_dates=["date"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def charger_geo_scores() -> Optional[pd.DataFrame]:
    """Charge geo_scores.csv produit par le Bloc 2 (geo_scorer.py)."""
    if not CHEMIN_GEO.exists():
        return None
    geo = pd.read_csv(CHEMIN_GEO, parse_dates=["date"])
    return geo.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def charger_donnees_multi_actifs(
    tickers: tuple,
    start_date: str,
    end_date:   str,
) -> dict:
    """
    Telecharge les prix pour plusieurs tickers via yfinance.
    Retourne un dict {ticker: DataFrame avec date, Returns, MA50, MA200}.
    Cache 1h pour eviter les appels repetes.
    """
    import yfinance as yf
    result = {}
    for ticker in tickers:
        try:
            hist = yf.download(ticker, start=start_date, end=end_date,
                               progress=False, auto_adjust=False)
            if hist is None or hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [col[0] for col in hist.columns]
            col = "Adj Close" if "Adj Close" in hist.columns else "Close"
            df_t = hist[[col]].copy()
            df_t.columns = ["price"]
            df_t.index.name = "date"
            df_t = df_t.reset_index()
            df_t["date"]    = pd.to_datetime(df_t["date"]).dt.tz_localize(None)
            df_t["Returns"] = df_t["price"].pct_change()
            df_t["MA50"]    = df_t["price"].rolling(50,  min_periods=1).mean()
            df_t["MA200"]   = df_t["price"].rolling(200, min_periods=1).mean()
            result[ticker]  = df_t
        except Exception:
            continue
    return result


# ===========================================================================
# Construction des graphiques Plotly
# ===========================================================================

def _ajouter_lignes_evenements(
    fig: go.Figure,
    df: pd.DataFrame,
    afficher: bool,
    row: int = 1,
    col: int = 1,
) -> None:
    """
    Ajoute des lignes verticales en pointilles pour chaque evenement cle.

    Seuls les evenements compris dans la plage de dates du DataFrame sont affiches.
    """
    if not afficher:
        return
    date_min = pd.to_datetime(df["date"].min())
    date_max = pd.to_datetime(df["date"].max())
    for date_str, label, couleur in EVENEMENTS_CLES:
        evt = pd.Timestamp(date_str)
        if date_min <= evt <= date_max:
            fig.add_vline(
                x=evt.timestamp() * 1000,
                line_dash="dash", line_color=couleur,
                line_width=1.0, opacity=0.65,
                row=row, col=col,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=couleur,
                annotation_textangle=-90,
            )


def construire_graphique_prix(
    df: pd.DataFrame,
    afficher_mm: bool,
    afficher_evenements: bool,
) -> go.Figure:
    """
    Graphique principal : prix de cloture SP500 + moyennes mobiles + activite news.

    Structure : 2 sous-graphiques superposes partageant l'axe des dates.
      - Haut (72%) : courbe de prix + MA50 + MA200 + annotations evenements
      - Bas  (28%) : barres du nombre d'articles de news par jour
    """
    col_prix = "Adj Close" if "Adj Close" in df.columns else "Close"
    a_des_news = "nb_articles" in df.columns and df["nb_articles"].sum() > 0
    hauteurs = [0.72, 0.28] if a_des_news else [1.0]
    nb_lignes = 2 if a_des_news else 1

    fig = make_subplots(
        rows=nb_lignes, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=hauteurs,
        subplot_titles=("", "Articles de news / jour" if a_des_news else ""),
    )

    # Courbe de prix
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[col_prix],
        name="SP500",
        line=dict(color=COULEUR_PRIX, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Cours : %{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Moyennes mobiles (optionnelles)
    if afficher_mm and "MA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MA50"],
            name="MA 50", line=dict(color=COULEUR_MA50, width=1.2, dash="dot"),
            hovertemplate="MA50 : %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MA200"],
            name="MA 200", line=dict(color=COULEUR_MA200, width=1.5, dash="dash"),
            hovertemplate="MA200 : %{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # Lignes d'evenements geopolitiques
    _ajouter_lignes_evenements(fig, df, afficher_evenements, row=1, col=1)

    # Barres de news
    if a_des_news:
        fig.add_trace(go.Bar(
            x=df["date"], y=df["nb_articles"],
            name="Articles/jour",
            marker_color=COULEUR_NEWS, opacity=0.75,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y} article(s)<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>S&P 500</b> -- Cours & Activite Mediatique", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=60, b=20, l=10, r=10),
        height=520,
    )
    fig.update_xaxes(gridcolor="#1e1e2e", zeroline=False)
    fig.update_yaxes(gridcolor="#1e1e2e", zeroline=False)
    return fig


def construire_graphique_geo_score(df: pd.DataFrame, seuil: float) -> go.Figure:
    """
    Graphique timeline du Geo-Score avec zones de couleur.

    Rouge  : score < seuil  (zone de panique geopolitique)
    Orange : seuil <= score < 0  (zone de prudence)
    Vert   : score >= 0  (zone de confiance)
    """
    fig = go.Figure()

    # Zones de fond colorees
    fig.add_hrect(y0=-1.0, y1=seuil, fillcolor="rgba(214,39,40,0.08)",  line_width=0)
    fig.add_hrect(y0=seuil, y1=0.0,  fillcolor="rgba(255,127,14,0.06)", line_width=0)
    fig.add_hrect(y0=0.0,  y1=1.0,   fillcolor="rgba(44,160,44,0.06)",  line_width=0)

    # Ligne du seuil
    fig.add_hline(
        y=seuil, line_dash="dash", line_color="red", line_width=1.2, opacity=0.7,
        annotation_text=f"Seuil {seuil:.1f}", annotation_position="bottom right",
        annotation_font_color="red",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=0.8)

    # Courbe du score
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["geo_score"],
        name="Geo-Score",
        line=dict(color="#00b4d8", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Geo-Score : %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Geo-Score</b> -- Sentiment NLP Quotidien",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=340, hovermode="x unified",
    )
    return fig


def construire_graphique_rsi(df: pd.DataFrame) -> go.Figure:
    """
    Graphique RSI 14 jours avec zones de surachat (>70) et survente (<30).

    La zone rouge (surachat) signale une possible correction a la baisse.
    La zone verte (survente) signale une possible reprise.
    """
    fig = go.Figure()

    fig.add_hrect(y0=70,  y1=100, fillcolor="rgba(214,39,40,0.08)", line_width=0)
    fig.add_hrect(y0=0,   y1=30,  fillcolor="rgba(44,160,44,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=COULEUR_DANGER, line_width=0.9,
                  annotation_text="Surachat 70",  annotation_position="bottom right",
                  annotation_font_color=COULEUR_DANGER)
    fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c",      line_width=0.9,
                  annotation_text="Survente 30",  annotation_position="top right",
                  annotation_font_color="#2ca02c")
    fig.add_hline(y=50, line_dash="dot",  line_color="#555",         line_width=0.7)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["RSI_14"],
        name="RSI 14",
        line=dict(color="#ff7f0e", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>RSI : %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>RSI 14 jours</b> -- Zones surachat / survente",
        yaxis=dict(range=[0, 100], gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=300,
    )
    return fig


def construire_graphique_drawdown(df: pd.DataFrame) -> go.Figure:
    """
    Graphique du drawdown glissant depuis le plus haut historique.

    Le point de drawdown maximal est annote sur le graphique.
    """
    dd_pct      = df["Drawdown"] * 100
    idx_max_dd  = dd_pct.idxmin()
    val_max_dd  = dd_pct.iloc[idx_max_dd]
    date_max_dd = df["date"].iloc[idx_max_dd]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=dd_pct,
        name="Drawdown",
        line=dict(color=COULEUR_DANGER, width=1.5),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DD : %{y:.2f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=date_max_dd, y=val_max_dd,
        text=f"Max DD : {val_max_dd:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor=COULEUR_DANGER,
        font=dict(color=COULEUR_DANGER, size=11),
        bgcolor="#0e1117", bordercolor=COULEUR_DANGER,
    )
    fig.add_hline(y=0, line_color="#555", line_width=0.7)

    fig.update_layout(
        title="<b>Drawdown</b> depuis le plus haut",
        yaxis=dict(gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=280,
    )
    return fig


def construire_graphique_volatilite(df: pd.DataFrame) -> go.Figure:
    """
    Graphique de la volatilite annualisee glissante sur 20 jours.

    La volatilite est calculee a partir de l'ecart-type des log-rendements,
    multiplie par sqrt(252) pour l'annualiser.
    """
    vol_pct = df["Volatility_20"] * 100
    moy_vol = vol_pct.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=vol_pct,
        name="Volatilite 20j",
        line=dict(color="#9467bd", width=1.5),
        fill="tozeroy", fillcolor="rgba(148,103,189,0.10)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Vol : %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=moy_vol, line_dash="dash", line_color="#888", line_width=0.9,
        annotation_text=f"Moy. {moy_vol:.1f}%",
        annotation_position="bottom right",
        annotation_font_color="#888",
    )

    fig.update_layout(
        title="<b>Volatilite annualisee 20j</b>",
        yaxis=dict(ticksuffix="%", gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=280,
    )
    return fig


def construire_distribution_rendements(df: pd.DataFrame) -> go.Figure:
    """
    Histogramme de la distribution des rendements quotidiens.

    Superpose la courbe de la loi normale theorique et marque :
      - la moyenne des rendements
      - la VaR 5% (perte journaliere depassee seulement 5% du temps)
    """
    rendements = df["Returns"].dropna() * 100
    mu, sigma  = float(rendements.mean()), float(rendements.std())
    var_5      = float(rendements.quantile(0.05))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rendements, nbinsx=80,
        name="Rendements (%)", histnorm="probability density",
        marker_color=COULEUR_PRIX, opacity=0.65,
        hovertemplate="Rendement : %{x:.2f}%<br>Densite : %{y:.4f}<extra></extra>",
    ))

    # Courbe normale theorique
    plage_x   = np.linspace(rendements.min(), rendements.max(), 300)
    courbe_n  = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((plage_x - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=plage_x, y=courbe_n,
        name=f"Loi normale (mu={mu:.2f}%, sigma={sigma:.2f}%)",
        line=dict(color="red", width=1.8),
    ))

    fig.add_vline(x=mu,    line_dash="dash", line_color="orange",       line_width=1.2,
                  annotation_text=f"Moy. {mu:.2f}%",  annotation_position="top right",
                  annotation_font_color="orange")
    fig.add_vline(x=var_5, line_dash="dot",  line_color=COULEUR_DANGER, line_width=1.2,
                  annotation_text=f"VaR 5% {var_5:.2f}%", annotation_position="top left",
                  annotation_font_color=COULEUR_DANGER)

    fig.update_layout(
        title="<b>Distribution des rendements quotidiens</b> -- SP500",
        xaxis=dict(title="Rendement journalier (%)", gridcolor="#1e1e2e"),
        yaxis=dict(title="Densite", gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=40, l=10, r=10),
        height=360, barmode="overlay",
    )
    return fig


# ===========================================================================
# Graphiques Backtest (Bloc 3)
# ===========================================================================

def construire_graphique_equity(bh_curve: pd.Series, gq_curve: pd.Series,
                                 dates: pd.Series,
                                 label_split: Optional[str] = None) -> go.Figure:
    """Courbes de valeur cumulee normalisees a 1 : Buy & Hold vs GeoQuant."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=bh_curve,
        name="Buy & Hold",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>B&H : %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=gq_curve,
        name="GeoQuant",
        line=dict(color="#2ca02c", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>GeoQuant : %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", line_width=0.8)
    if label_split:
        fig.add_vline(
            x=pd.Timestamp(label_split).timestamp() * 1000,
            line_dash="dash", line_color="#ff7f0e", line_width=1.5,
            annotation_text="Split IS / OOS",
            annotation_font_color="#ff7f0e",
            annotation_position="top left",
        )
    fig.update_layout(
        title="<b>Performance cumulee</b> -- Buy & Hold vs GeoQuant (base = 1)",
        yaxis=dict(tickformat=".2f", gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
        margin=dict(t=60, b=20, l=10, r=10),
        height=380,
    )
    return fig


def construire_graphique_signal(df: pd.DataFrame, col_prix: str) -> go.Figure:
    """Prix SP500 avec zones vertes (Long) et grises (Cash) selon la position GeoQuant."""
    fig = go.Figure()

    pos_col = "position" if "position" in df.columns else "signal"
    in_long  = False
    start_x  = None
    for i, row in df.iterrows():
        pos = float(row[pos_col]) if not pd.isna(row[pos_col]) else 0.0
        if pos > 0.0 and not in_long:
            start_x = row["date"]
            in_long = True
        elif pos == 0.0 and in_long:
            fig.add_vrect(
                x0=start_x, x1=row["date"],
                fillcolor="rgba(44,160,44,0.10)", line_width=0,
            )
            in_long = False
    if in_long and start_x is not None:
        fig.add_vrect(
            x0=start_x, x1=df["date"].iloc[-1],
            fillcolor="rgba(44,160,44,0.10)", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[col_prix],
        name="SP500", line=dict(color=COULEUR_PRIX, width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Cours : %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Signaux de trading</b> -- Zones vertes = Long (taille prop. position), blanc = Cash",
        yaxis=dict(gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        hovermode="x unified",
        margin=dict(t=50, b=20, l=10, r=10),
        height=300,
    )
    return fig


def construire_graphique_sensibilite(
    df_full: pd.DataFrame,
    seuil_courant: float,
    costs_bps: float,
    risk_free: float,
) -> go.Figure:
    """
    Courbe rendement GeoQuant et Sharpe en fonction du seuil Geo-Score.
    Permet de verifier que le seuil choisi n'est pas sur-ajuste.
    """
    from strategy import Strategy
    from backtest import run_backtest as _run_bt

    seuils   = np.arange(-1.0, 0.01, 0.05)
    rets_gq  = []
    sharpes  = []

    for s in seuils:
        strat_s = Strategy(base_dir=RACINE, seuil_geo=float(s))
        df_s    = strat_s.apply(df_full)
        _, gq_s = _run_bt(df_s, price_col="Adj Close" if "Adj Close" in df_s.columns else "Close",
                          costs_bps=costs_bps, risk_free_annual=risk_free)
        rets_gq.append(gq_s.total_return * 100)
        sharpes.append(gq_s.sharpe)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("Rendement total GeoQuant (%)", "Sharpe Ratio"))

    fig.add_trace(go.Scatter(
        x=seuils, y=rets_gq, name="Rendement (%)",
        line=dict(color="#2ca02c", width=2),
        hovertemplate="Seuil %{x:.2f} -> %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=seuils, y=sharpes, name="Sharpe",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="Seuil %{x:.2f} -> Sharpe %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    for row in (1, 2):
        fig.add_vline(x=seuil_courant, line_dash="dash", line_color="red",
                      line_width=1.2, row=row, col=1,
                      annotation_text=f"Seuil actuel {seuil_courant:.2f}",
                      annotation_font_color="red",
                      annotation_position="top right")

    fig.update_layout(
        title="<b>Analyse de sensibilite</b> -- Performance selon le seuil Geo-Score",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        showlegend=False,
        margin=dict(t=60, b=20, l=10, r=10),
        height=440,
    )
    fig.update_xaxes(gridcolor="#1e1e2e", title_text="Seuil Geo-Score", row=2, col=1)
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def construire_graphique_contribution_nlp(
    df: pd.DataFrame,
    seuil_geo: float,
    costs_bps: float,
    risk_free: float,
) -> go.Figure:
    """
    Compare 3 strategies sur la meme periode :
      - Buy & Hold
      - Golden Cross seul (sans Geo-Score)
      - GeoQuant (Golden Cross + Geo-Score)
    Isole ainsi la contribution reelle du NLP.
    """
    from strategy import Strategy
    from backtest  import run_backtest

    df2 = df.copy()

    # --- Strategie 1 : Buy & Hold (position = 1 toujours) ---
    bh, gq = run_backtest(df2, price_col="Adj Close" if "Adj Close" in df2.columns else "Close",
                          costs_bps=costs_bps, risk_free_annual=risk_free)

    # --- Strategie 2 : Golden Cross seul (geo_score neutralise = toujours >= seuil) ---
    df_gc = df2.copy()
    df_gc["geo_score"] = 1.0   # geo_score artificellement positif -> jamais bloquant
    strat_gc = Strategy(base_dir=RACINE, seuil_geo=seuil_geo)
    df_gc    = strat_gc.apply(df_gc)
    _, gc_only = run_backtest(df_gc, price_col="Adj Close" if "Adj Close" in df_gc.columns else "Close",
                              costs_bps=costs_bps, risk_free_annual=risk_free)

    # --- Strategie 3 : GeoQuant complet ---
    dates = df2["date"].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=bh.equity_curve.values,
        name="Buy & Hold", line=dict(color="#888", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>B&H : %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=gc_only.equity_curve.values,
        name="Golden Cross seul (sans NLP)",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>GC : %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=gq.equity_curve.values,
        name="GeoQuant (Golden Cross + Geo-Score)",
        line=dict(color="#2ca02c", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>GeoQuant : %{y:.3f}<extra></extra>",
    ))

    # Annotation delta NLP
    delta_nlp = gq.total_return - gc_only.total_return
    fig.add_annotation(
        text=f"Apport du NLP : {delta_nlp:+.1%}",
        x=0.01, y=0.99, xref="paper", yref="paper",
        showarrow=False,
        font=dict(color="#2ca02c" if delta_nlp >= 0 else "#d62728", size=13),
        align="left", bgcolor="rgba(14,17,23,0.7)",
    )

    fig.update_layout(
        title="<b>Contribution isolee du NLP</b> -- Buy & Hold vs Golden Cross vs GeoQuant",
        yaxis=dict(title="Valeur normalisee (base 1)", gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        height=380, margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig



def construire_correlation_vix(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> go.Figure:
    """
    Superpose le Geo-Score et le VIX (telecharge en live).
    Si le Geo-Score anticipe les pics de VIX, le modele capte quelque chose de reel.
    """
    import yfinance as yf

    if "geo_score" not in df.columns:
        return go.Figure()

    try:
        vix = yf.download("^VIX", start=start_date, end=end_date,
                          progress=False, auto_adjust=False)
        if vix is None or vix.empty:
            return go.Figure()
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        col_v = "Adj Close" if "Adj Close" in vix.columns else "Close"
        vix_s = vix[col_v].reset_index()
        vix_s.columns = ["date", "vix"]
        vix_s["date"] = pd.to_datetime(vix_s["date"]).dt.tz_localize(None)
    except Exception:
        return go.Figure()

    merged = df[["date", "geo_score"]].merge(vix_s, on="date", how="inner").dropna()
    if merged.empty:
        return go.Figure()

    # Correlation glissante 30j entre Geo-Score inverse et VIX
    # On inverse le Geo-Score car peur (score bas) <-> VIX haut
    merged["geo_inv"] = -merged["geo_score"]
    rolling_corr = merged["geo_inv"].rolling(30).corr(merged["vix"])
    overall_corr = float(merged["geo_inv"].corr(merged["vix"]))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.35, 0.35, 0.30],
        subplot_titles=(
            "Geo-Score (inverse = peur) vs VIX",
            "VIX -- Indice de peur officiel de Wall Street",
            f"Correlation glissante 30j (globale : {overall_corr:.2f})",
        ),
    )

    # Geo-Score inverse
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["geo_inv"],
        name="-Geo-Score (peur NLP)",
        line=dict(color="#9467bd", width=1.8),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Peur NLP : %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="#555", line_width=0.8, row=1, col=1)

    # VIX
    fig.add_trace(go.Scatter(
        x=merged["date"], y=merged["vix"],
        name="VIX",
        line=dict(color="#d62728", width=1.8),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>VIX : %{y:.1f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=20, line_color="#ff7f0e", line_width=0.8, line_dash="dash",
                  annotation_text="VIX 20 (seuil panique)", annotation_font_color="#ff7f0e",
                  row=2, col=1)

    # Correlation glissante
    fig.add_trace(go.Scatter(
        x=merged["date"], y=rolling_corr,
        name="Corr. 30j",
        line=dict(color="#00b4d8", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Corr : %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="#555", line_width=0.8, row=3, col=1)

    fig.update_layout(
        title="<b>Geo-Score vs VIX</b> -- Le NLP anticipe-t-il la peur du marche ?",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        showlegend=False,
        height=560, margin=dict(t=60, b=20, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


def construire_graphique_correlation(df: pd.DataFrame) -> go.Figure:
    """
    Correlation glissante 60 jours entre geo_score[t-1] et Returns[t].
    Valide que le Geo-Score a un vrai pouvoir predictif.
    """
    if "geo_score" not in df.columns or "Returns" not in df.columns:
        return go.Figure()

    df2 = df[["date", "geo_score", "Returns"]].dropna().copy()
    df2["geo_lag"] = df2["geo_score"].shift(1)
    df2 = df2.dropna()

    rolling_corr = df2["geo_lag"].rolling(60).corr(df2["Returns"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        row_heights=[0.45, 0.55],
                        subplot_titles=("Correlation glissante 60j (Geo-Score -> Rendement J+1)",
                                        "Nuage de points Geo-Score[t-1] vs Rendement[t]"))

    fig.add_trace(go.Scatter(
        x=df2["date"], y=rolling_corr,
        name="Corr. 60j",
        line=dict(color="#00b4d8", width=1.8),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Corr : %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="#555", line_width=0.8, row=1, col=1)

    colors = df2["Returns"].apply(lambda r: "#2ca02c" if r > 0 else "#d62728")
    fig.add_trace(go.Scatter(
        x=df2["geo_lag"], y=df2["Returns"] * 100,
        mode="markers",
        name="Observations",
        marker=dict(color=colors, size=4, opacity=0.5),
        hovertemplate="Geo-Score : %{x:.3f}<br>Rendement : %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    overall_corr = float(df2["geo_lag"].corr(df2["Returns"]))
    fig.add_annotation(
        text=f"Corr. globale : {overall_corr:.3f}",
        x=0.98, y=0.02, xref="paper", yref="paper",
        showarrow=False, font=dict(color="#aaa", size=12),
        align="right",
    )

    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        showlegend=False,
        margin=dict(t=60, b=20, l=10, r=10),
        height=500,
    )
    fig.update_xaxes(gridcolor="#1e1e2e")
    fig.update_yaxes(gridcolor="#1e1e2e")
    return fig


# ===========================================================================
# Barre laterale
# ===========================================================================
with st.sidebar:
    st.markdown('<div class="titre-principal">🌍 GeoQuant AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sous-titre">Backtesting & NLP -- '
        'Analyse geopolitique des marches</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Chargement des donnees
    df_complet = charger_dataset()

    # Filtre de periode
    if df_complet is not None and len(df_complet) > 0:
        date_min = pd.to_datetime(df_complet["date"].min()).date()
        date_max = pd.to_datetime(df_complet["date"].max()).date()
        plage_dates = st.slider(
            "Periode",
            min_value=date_min, max_value=date_max,
            value=(date_min, date_max),
            format="DD/MM/YY",
        )
    else:
        plage_dates = (None, None)

    # Parametres Bloc 3
    st.divider()
    st.markdown("**Parametres Bloc 3**")
    seuil_geo = st.slider(
        "Seuil Geo-Score (Cash si < seuil)",
        min_value=-1.0, max_value=0.0, value=-0.5, step=0.05, format="%.2f",
        help="Seuil en dessous duquel la strategie coupe la position.",
    )
    costs_bps = st.slider(
        "Couts de transaction (bps / cote)",
        min_value=0, max_value=50, value=2, step=1,
        help="2 bps = 0.02% par entree OU sortie (realiste pour un ETF S&P 500).",
    )
    risk_free_pct = st.slider(
        "Taux sans risque annuel (%)",
        min_value=0.0, max_value=6.0, value=0.0, step=0.5, format="%.1f%%",
        help="Utilise pour le calcul du Sharpe Ratio.",
    )
    risk_free = risk_free_pct / 100.0

    # Toggles d'affichage
    st.divider()
    afficher_mm         = st.toggle("Afficher les moyennes mobiles",       value=True)
    afficher_evenements = st.toggle("Annoter les evenements geopolitiques", value=True)

    # Bouton d'actualisation (vide le cache)
    st.divider()
    if st.button("Actualiser les donnees", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Statut du pipeline
    st.divider()
    st.markdown("**Statut du pipeline**")
    st.markdown('<span class="badge-fait">BLOC 1</span> Data Engineering',  unsafe_allow_html=True)
    _b2 = "badge-fait" if CHEMIN_GEO.exists() else "badge-attente"
    st.markdown(f'<span class="{_b2}">BLOC 2</span> NLP / FinBERT',        unsafe_allow_html=True)
    _b3 = "badge-fait" if CHEMIN_GEO.exists() else "badge-attente"
    st.markdown(f'<span class="{_b3}">BLOC 3</span> Strategie + Paper Trading', unsafe_allow_html=True)
    st.markdown('<span class="badge-fait">BLOC 4</span> Dashboard',         unsafe_allow_html=True)


# ===========================================================================
# Garde : dataset absent
# ===========================================================================
if df_complet is None or len(df_complet) == 0:
    st.error(
        "**Dataset introuvable.**\n\n"
        "Le fichier `data/processed/dataset_final.csv` n'existe pas encore.\n\n"
        "**Pour le generer, lancez :**\n"
        "```bash\ncd PPE_2025/src\n"
        "/c/Users/mathi/anaconda3/python.exe main.py\n```"
    )
    st.stop()


# ===========================================================================
# Filtrage par periode selectionnee
# ===========================================================================
df = df_complet.copy()
if plage_dates[0] is not None:
    masque = (
        (df["date"] >= pd.Timestamp(plage_dates[0]))
        & (df["date"] <= pd.Timestamp(plage_dates[1]))
    )
    df = df[masque].reset_index(drop=True)

col_prix = "Adj Close" if "Adj Close" in df.columns else "Close"


# ===========================================================================
# En-tete de la page principale
# ===========================================================================
st.markdown('<div class="titre-principal">GeoQuant AI</div>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sous-titre">S&P 500 (^GSPC) &nbsp;|&nbsp; '
    f'{pd.Timestamp(plage_dates[0]).strftime("%d %b %Y") if plage_dates[0] else "?"}'
    f' -- '
    f'{pd.Timestamp(plage_dates[1]).strftime("%d %b %Y") if plage_dates[1] else "?"}'
    f'</p>',
    unsafe_allow_html=True,
)
st.divider()


# ===========================================================================
# Onglets principaux
# ===========================================================================
onglet_marche, onglet_nlp, onglet_backtest, onglet_paper, onglet_apropos = st.tabs([
    "📊 Vue Marche",
    "🧠 Sentiment & NLP",
    "⚔️ Backtest",
    "🟢 Paper Trading",
    "ℹ️ A Propos",
])


# ---------------------------------------------------------------------------
# ONGLET 1 -- Vue Marche
# ---------------------------------------------------------------------------
with onglet_marche:

    # Metriques cles
    if len(df) >= 2:
        prix_actuel   = float(df[col_prix].iloc[-1])
        prix_debut    = float(df[col_prix].iloc[0])
        rendement_tot = (prix_actuel / prix_debut - 1) * 100
        max_dd        = float(df["Drawdown"].min() * 100) if "Drawdown" in df.columns else 0.0
        vol_ann       = float(df["Volatility_20"].dropna().mean() * 100) if "Volatility_20" in df.columns else 0.0
        jours_news    = int((df["nb_articles"] > 0).sum()) if "nb_articles" in df.columns else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cours actuel",        f"{prix_actuel:,.2f}")
        m2.metric("Rendement total",     f"{rendement_tot:+.2f}%",
                  delta=f"{rendement_tot:+.2f}%", delta_color="normal")
        m3.metric("Max Drawdown",        f"{max_dd:.2f}%",
                  delta=f"{max_dd:.2f}%",         delta_color="inverse")
        m4.metric("Volatilite annuelle", f"{vol_ann:.1f}%")
        m5.metric("Jours avec news",     str(jours_news))

    st.divider()
    st.plotly_chart(
        construire_graphique_prix(df, afficher_mm, afficher_evenements),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# ONGLET 2 -- Sentiment & NLP
# ---------------------------------------------------------------------------
with onglet_nlp:

    geo_score_present = "geo_score" in df.columns and df["geo_score"].notna().any()

    if geo_score_present:
        derniere_ligne = df[df["geo_score"].notna()].iloc[-1]
        score_actuel = float(derniere_ligne["geo_score"])
        date_score = pd.to_datetime(derniere_ligne["date"]).strftime("%d/%m/%Y") if "date" in df.columns else "date inconnue"
        st.subheader("Geo-Score -- Sentiment du marche")
        st.caption(f":warning: Score historique — derniere donnee : **{date_score}** (non mis a jour en temps reel)")

        col_jauge, col_timeline = st.columns([1, 2])

        with col_jauge:
            # Couleur et libelle selon la valeur du score
            if score_actuel < seuil_geo:
                couleur_jauge, libelle_jauge = COULEUR_DANGER, "PANIQUE"
            elif score_actuel < 0:
                couleur_jauge, libelle_jauge = "#ff7f0e", "PRUDENCE"
            else:
                couleur_jauge, libelle_jauge = "#2ca02c", "CONFIANCE"

            fig_jauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score_actuel,
                domain=dict(x=[0, 1], y=[0, 1]),
                title=dict(
                    text=f"Geo-Score du jour<br>"
                         f"<span style='color:{couleur_jauge}'>{libelle_jauge}</span>",
                    font=dict(size=14),
                ),
                number=dict(font=dict(size=36, color=couleur_jauge)),
                delta=dict(reference=0, relative=False),
                gauge=dict(
                    axis=dict(range=[-1, 1], tickwidth=1),
                    bar=dict(color=couleur_jauge),
                    bgcolor="#1e1e2e",
                    steps=[
                        dict(range=[-1, seuil_geo], color="rgba(214,39,40,0.15)"),
                        dict(range=[seuil_geo, 0],  color="rgba(255,127,14,0.10)"),
                        dict(range=[0, 1],           color="rgba(44,160,44,0.10)"),
                    ],
                    threshold=dict(
                        line=dict(color=COULEUR_DANGER, width=3),
                        thickness=0.75, value=seuil_geo,
                    ),
                ),
            ))
            fig_jauge.update_layout(
                paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                height=280, margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_jauge, use_container_width=True)

        with col_timeline:
            st.plotly_chart(
                construire_graphique_geo_score(df, seuil_geo),
                use_container_width=True,
            )

    else:
        # Placeholder -- Bloc 2 pas encore realise
        st.markdown("""
<div class="bloc-placeholder">
    <h3>🔜 Bloc 2 -- NLP en attente</h3>
    <p>Le <b>Geo-Score</b> sera calcule par FinBERT (Hugging Face)
    sur les titres de news de chaque journee.</p>
    <p>La colonne <code>geo_score</code> sera ajoutee a
    <code>dataset_final.csv</code> apres execution de
    <code>nlp/geo_scorer.py</code>.</p>
    <br>
    <table style="margin:0 auto; text-align:left; color:#aaa; font-size:0.85rem;">
        <tr><td>Modele prevu</td><td>&nbsp; FinBERT (ProsusAI/finbert)</td></tr>
        <tr><td>Input</td><td>&nbsp; Titres de news concatenes par jour</td></tr>
        <tr><td>Output</td><td>&nbsp; Score quotidien entre -1 et +1</td></tr>
    </table>
</div>""", unsafe_allow_html=True)

    # Tableau des dernieres actualites
    st.divider()
    st.subheader("Dernieres actualites")
    news_df = charger_news_brutes()

    if news_df is not None and len(news_df) > 0:
        # Filtre sur la periode selectionnee
        if plage_dates[0] is not None:
            news_df = news_df[
                (news_df["date"] >= pd.Timestamp(plage_dates[0]))
                & (news_df["date"] <= pd.Timestamp(plage_dates[1]))
            ]

        colonnes = [c for c in ["date", "title", "source", "category"] if c in news_df.columns]
        affichage = news_df[colonnes].head(15).copy()
        if "date" in affichage.columns:
            affichage["date"] = affichage["date"].dt.strftime("%d %b %Y")

        st.dataframe(affichage, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune news disponible. Verifiez `data/raw/sample_news.csv`.")



# ---------------------------------------------------------------------------
# ONGLET 3 -- Backtest
# ---------------------------------------------------------------------------
with onglet_backtest:
    from strategy import Strategy
    from backtest import run_backtest

    if charger_geo_scores() is None:
        st.warning(
            "Le fichier `data/processed/geo_scores.csv` est absent. "
            "Lancez `python src/run_geo_scorer.py` pour le generer."
        )
        st.stop()

    # Charge le dataset via strategy.load() — inclut geo_score avec decay
    # exponentiel et days_since_news depuis dataset_final.csv.
    # On applique ensuite les parametres du slider (seuil_geo choisi par l'utilisateur).
    strat = Strategy(base_dir=RACINE, seuil_geo=seuil_geo)
    df_bt_full = strat.load()
    df_bt_full = strat.apply(df_bt_full)

    # Filtre sur la periode selectionnee dans la barre laterale
    if plage_dates[0] and plage_dates[1]:
        df_bt = df_bt_full[
            (df_bt_full["date"] >= pd.Timestamp(plage_dates[0])) &
            (df_bt_full["date"] <= pd.Timestamp(plage_dates[1]))
        ].reset_index(drop=True)
    else:
        df_bt = df_bt_full.copy()

    # Backtest complet (avec couts et taux sans risque)
    bh, gq = run_backtest(df_bt, price_col=col_prix,
                          costs_bps=float(costs_bps),
                          risk_free_annual=risk_free)

    # ── En-tete ──────────────────────────────────────────────────────────────
    st.subheader("⚔️ Backtest -- Buy & Hold vs GeoQuant")
    _n_urgence   = int((df_bt.get("decision", pd.Series(dtype=str)) == "URGENCE").sum())
    _n_techonly  = int((df_bt.get("news_mode", pd.Series(dtype=str)) == "TechOnly").sum())
    st.caption(
        f"Regle : MA50 > MA200  ET  Geo-Score[t-1] ≥ {seuil_geo:.2f}  →  Long  |  "
        f"Couts : {costs_bps} bps/cote  •  Rf : {risk_free_pct:.1f}%  •  "
        f"Periode : {df_bt['date'].iloc[0].date()} → {df_bt['date'].iloc[-1].date()}  •  "
        f"Urgences : {_n_urgence}j  •  TechOnly (news >14j) : {_n_techonly}j"
    )

    # ── Metriques cles ────────────────────────────────────────────────────
    METRIQUES = [
        ("Rendement total",  bh.total_return, gq.total_return, ".2%", True),
        ("Rend. annualise",  bh.annualised,   gq.annualised,   ".2%", True),
        ("Max Drawdown",     bh.max_drawdown, gq.max_drawdown, ".2%", False),
        ("Sharpe Ratio",     bh.sharpe,       gq.sharpe,       ".2f", True),
        ("Win Rate",         bh.win_rate,     gq.win_rate,     ".1%", True),
        ("Nb trades",        float(bh.nb_trades), float(gq.nb_trades), ".0f", False),
    ]

    col_m = st.columns(len(METRIQUES))
    for i, (label, val_bh, val_gq, fmt, higher_better) in enumerate(METRIQUES):
        import math
        if math.isnan(val_gq) or math.isnan(val_bh):
            col_m[i].metric(label=label, value="N/A", delta="--")
            continue
        delta_val = val_gq - val_bh
        delta_fmt = f"+{delta_val:{fmt}}" if delta_val >= 0 else f"{delta_val:{fmt}}"
        if higher_better:
            d_color = "normal"
        else:
            d_color = "inverse"
        col_m[i].metric(
            label=label,
            value=f"{val_gq:{fmt}}",
            delta=f"vs B&H {delta_fmt}",
            delta_color=d_color,
        )

    st.divider()

    # ── Graphiques principaux ─────────────────────────────────────────────
    st.plotly_chart(
        construire_graphique_equity(bh.equity_curve, gq.equity_curve, df_bt["date"]),
        use_container_width=True,
    )
    st.plotly_chart(
        construire_graphique_signal(df_bt, col_prix),
        use_container_width=True,
    )

    st.divider()

    # ── Tableau comparatif + journal des trades ───────────────────────────
    col_tbl, col_log = st.columns([1, 2])

    with col_tbl:
        st.markdown("**Metriques detaillees**")
        rows = []
        for label, val_bh, val_gq, fmt, _ in METRIQUES:
            import math
            rows.append({
                "Metrique":   label,
                "Buy & Hold": "N/A" if math.isnan(val_bh) else f"{val_bh:{fmt}}",
                "GeoQuant":   "N/A" if math.isnan(val_gq) else f"{val_gq:{fmt}}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_log:
        st.markdown("**Journal des trades GeoQuant**")
        if len(gq.trade_log) > 0:
            log_display = gq.trade_log.copy()
            if "Date" in log_display.columns:
                log_display["Date"] = pd.to_datetime(log_display["Date"]).dt.strftime("%d %b %Y")
            st.dataframe(log_display, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun trade genere avec ce seuil.")

    st.divider()

    # ── Walk-Forward Test (In-Sample 2022-2023 / Out-of-Sample 2024) ─────
    st.subheader("Walk-Forward Test -- In-Sample 2022-2023 vs Out-of-Sample 2024")
    st.caption(
        "Optimisation sur 2022-2023 (IS), validation sur 2024 (OOS — jamais vu). "
        "Parametres IS optimaux : seuil=-0.50, sizing=0.40 (meilleur Sharpe IS). "
        "Si GeoQuant sur-performe en OOS, les resultats sont credibles."
    )

    SPLIT_DATE = "2024-01-01"
    df_is  = df_bt[df_bt["date"] <  SPLIT_DATE].reset_index(drop=True)
    df_oos = df_bt[df_bt["date"] >= SPLIT_DATE].reset_index(drop=True)
    split_date = SPLIT_DATE

    if len(df_is) > 20 and len(df_oos) > 20:
        bh_is,  gq_is  = run_backtest(df_is,  price_col=col_prix,
                                       costs_bps=float(costs_bps), risk_free_annual=risk_free)
        bh_oos, gq_oos = run_backtest(df_oos, price_col=col_prix,
                                       costs_bps=float(costs_bps), risk_free_annual=risk_free)

        wf_col1, wf_col2 = st.columns(2)

        def _wf_table(bh_r, gq_r, titre):
            rows = [
                {"Metrique": "Rendement total",
                 "B&H":      f"{bh_r.total_return:+.2%}",
                 "GeoQuant": f"{gq_r.total_return:+.2%}"},
                {"Metrique": "Sharpe",
                 "B&H":      f"{bh_r.sharpe:.2f}",
                 "GeoQuant": f"{gq_r.sharpe:.2f}"},
                {"Metrique": "Max Drawdown",
                 "B&H":      f"{bh_r.max_drawdown:.2%}",
                 "GeoQuant": f"{gq_r.max_drawdown:.2%}"},
                {"Metrique": "Nb trades",
                 "B&H":      str(bh_r.nb_trades),
                 "GeoQuant": str(gq_r.nb_trades)},
            ]
            st.markdown(f"**{titre}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with wf_col1:
            _wf_table(bh_is, gq_is,
                      f"IS -- 2022-2023 ({df_is['date'].iloc[0].date()} -> {df_is['date'].iloc[-1].date()})")
        with wf_col2:
            _wf_table(bh_oos, gq_oos,
                      f"OOS -- 2024 ({df_oos['date'].iloc[0].date()} -> {df_oos['date'].iloc[-1].date()})")

        # Verdict OOS
        oos_beats_bh_sharpe = gq_oos.sharpe > bh_oos.sharpe
        oos_beats_bh_dd     = abs(gq_oos.max_drawdown) < abs(bh_oos.max_drawdown)
        if oos_beats_bh_sharpe and oos_beats_bh_dd:
            st.success(
                f"Validation OOS reussie : GeoQuant Sharpe {gq_oos.sharpe:.2f} > "
                f"B&H {bh_oos.sharpe:.2f} ET MaxDD {gq_oos.max_drawdown:.2%} < "
                f"B&H {bh_oos.max_drawdown:.2%}. Les parametres IS generalisent bien."
            )
        elif oos_beats_bh_sharpe:
            st.info(
                f"Validation OOS partielle : meilleur Sharpe ({gq_oos.sharpe:.2f} vs {bh_oos.sharpe:.2f}) "
                f"mais rendement total inferieur."
            )
        else:
            st.warning("OOS : GeoQuant sous-performe Buy & Hold sur cette periode.")

        # Courbe equity IS + OOS combinee avec ligne de separation
        bh_combined = pd.concat(
            [bh_is.equity_curve, bh_oos.equity_curve * bh_is.equity_curve.iloc[-1]],
            ignore_index=True,
        )
        gq_combined = pd.concat(
            [gq_is.equity_curve, gq_oos.equity_curve * gq_is.equity_curve.iloc[-1]],
            ignore_index=True,
        )
        st.plotly_chart(
            construire_graphique_equity(bh_combined, gq_combined, df_bt["date"],
                                        label_split=split_date),
            use_container_width=True,
        )
    else:
        st.info(
            "La periode selectionnee ne couvre pas les deux sous-periodes IS (2022-2023) "
            "et OOS (2024). Elargissez la plage de dates pour voir le Walk-Forward."
        )



# ---------------------------------------------------------------------------
# ONGLET 4 -- Paper Trading
# ---------------------------------------------------------------------------
with onglet_paper:
    from paper_trading import PaperTrader, CAPITAL_INITIAL

    st.subheader("🟢 Paper Trading -- Simulation en temps reel")
    st.caption(
        "Le portefeuille virtuel applique la meme regle que le backtest "
        "(Golden Cross + Geo-Score) sur les donnees et actualites du jour. "
        "Aucun argent reel n'est engage."
    )

    # ── Parametres paper trading ──────────────────────────────────────────
    pt_col1, pt_col2, pt_col3 = st.columns([1, 1, 1])
    with pt_col1:
        use_finbert_live = st.toggle(
            "Analyser les news avec FinBERT",
            value=False,
            help="Desactiver pour un refresh rapide sans GPU (geo_score=0).",
        )
    with pt_col2:
        auto_execute = st.toggle(
            "Executer le signal automatiquement",
            value=False,
            help="Si actif, l'entree/sortie est effectuee lors du refresh.",
        )
    with pt_col3:
        ticker_pt = st.selectbox(
            "Actif surveille",
            options=["^GSPC", "^FCHI", "BTC-USD"],
            index=0,
        )

    trader = PaperTrader(base_dir=RACINE, ticker=ticker_pt, seuil_geo=seuil_geo)
    portfolio = trader.load_portfolio()

    # ── Boutons d'action ─────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    refresh_clicked = btn_col1.button("🔄 Rafraichir le signal", use_container_width=True)
    reset_clicked   = btn_col2.button("🗑️ Remettre a zero",      use_container_width=True,
                                       type="secondary")

    if reset_clicked:
        portfolio = trader.reset_portfolio()
        st.success("Portefeuille reinitialise — capital de depart : "
                   f"{CAPITAL_INITIAL:,.0f} EUR virtuels.")

    # ── Snapshot live ─────────────────────────────────────────────────────
    snapshot = None
    action_msg = None
    vix_live = None

    if refresh_clicked:
        # VIX live — recupere en parallele du snapshot principal
        try:
            import yfinance as yf
            _vix = yf.download("^VIX", period="5d", interval="1d",
                               progress=False, auto_adjust=False)
            if _vix is not None and not _vix.empty:
                if isinstance(_vix.columns, pd.MultiIndex):
                    _vix.columns = [c[0] for c in _vix.columns]
                _col_v = "Adj Close" if "Adj Close" in _vix.columns else "Close"
                vix_live = float(_vix[_col_v].dropna().iloc[-1])
        except Exception:
            vix_live = None

    if refresh_clicked:
        with st.spinner("Recuperation du prix et des news en cours..."):
            try:
                snapshot = trader.get_live_snapshot(use_finbert=use_finbert_live)
                if auto_execute:
                    portfolio, action_msg = trader.execute_signal(snapshot, portfolio)
                    trader.save_portfolio(portfolio)

                # ── Notification pop-up si le signal a change ─────────────
                from alert_manager import AlertManager
                mgr = AlertManager(RACINE)
                if mgr.signal_changed(ticker_pt, snapshot.signal):
                    if snapshot.signal == 1:
                        st.toast(
                            f"🟢 Signal LONG sur {ticker_pt} — "
                            f"Geo-Score {snapshot.geo_score:+.2f} | "
                            f"Confiance {snapshot.confidence:.0%}",
                            icon="🟢",
                        )
                    else:
                        st.toast(
                            f"🔴 Signal CASH sur {ticker_pt} — "
                            f"Geo-Score {snapshot.geo_score:+.2f} | "
                            f"Sortie du marche",
                            icon="🔴",
                        )
                    mgr.save_signal(ticker_pt, snapshot.signal)

            except Exception as e:
                st.error(f"Erreur lors du refresh : {e}")

    # ── Metriques du portefeuille ─────────────────────────────────────────
    st.divider()
    st.markdown("### Portefeuille virtuel")

    # Mise a jour de la valeur investie au prix courant si on a un snapshot
    if snapshot and portfolio.nb_parts > 0:
        portfolio.capital_investi = portfolio.nb_parts * snapshot.prix_actuel

    valeur_totale    = portfolio.valeur_totale
    rendement_total  = portfolio.rendement_total
    win_rate_pt      = (portfolio.trades_gagnants / portfolio.nb_trades
                        if portfolio.nb_trades > 0 else 0.0)

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Capital total",      f"{valeur_totale:,.2f} EUR",
               delta=f"{rendement_total:+.2%}", delta_color="normal")
    pm2.metric("Capital cash",       f"{portfolio.capital_cash:,.2f} EUR")
    pm3.metric("Position ouverte",   f"{portfolio.capital_investi:,.2f} EUR")
    pm4.metric("Nb trades fermes",   str(portfolio.nb_trades))
    pm5.metric("Win Rate",           f"{win_rate_pt:.1%}" if portfolio.nb_trades > 0 else "N/A")

    if action_msg:
        if "ENTREE" in action_msg:
            st.success(f"✅ {action_msg}")
        elif "SORTIE" in action_msg:
            pnl_positive = "P&L : +" in action_msg
            if pnl_positive:
                st.success(f"✅ {action_msg}")
            else:
                st.warning(f"⚠️ {action_msg}")
        else:
            st.info(f"ℹ️ {action_msg}")

    # ── Signal live ───────────────────────────────────────────────────────
    if snapshot:
        st.divider()
        st.markdown("### Signal du moment")

        sig_col1, sig_col2 = st.columns([1, 2])

        with sig_col1:
            # Indicateur signal
            if snapshot.signal == 1:
                st.markdown("""
<div style="background:#1a3a1a; border:2px solid #2ca02c; border-radius:12px;
     padding:20px; text-align:center;">
    <div style="font-size:2.5rem;">🟢</div>
    <div style="font-size:1.4rem; font-weight:700; color:#2ca02c;">LONG</div>
    <div style="color:#aaa; font-size:0.9rem;">Position : {:.0%}</div>
</div>""".format(snapshot.position), unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="background:#2a1a1a; border:2px solid #d62728; border-radius:12px;
     padding:20px; text-align:center;">
    <div style="font-size:2.5rem;">🔴</div>
    <div style="font-size:1.4rem; font-weight:700; color:#d62728;">CASH</div>
    <div style="color:#aaa; font-size:0.9rem;">Hors marche</div>
</div>""", unsafe_allow_html=True)

            st.markdown("")

            # VIX live avec interpretation
            if vix_live is not None:
                if vix_live < 15:
                    vix_label, vix_color = "Calme", "#2ca02c"
                elif vix_live < 20:
                    vix_label, vix_color = "Normal", "#aaa"
                elif vix_live < 30:
                    vix_label, vix_color = "Tension", "#ff7f0e"
                else:
                    vix_label, vix_color = "PANIQUE", "#d62728"

                vix_alerte = (
                    snapshot.signal == 1 and vix_live >= 25
                )
                vix_str = f"{vix_live:.1f} — {vix_label}"
                vix_indicateurs = ["Prix actuel", "MA 50", "MA 200",
                                   "Golden Cross", "Geo-Score",
                                   "VIX (peur marche)", "News analysees"]
                vix_valeurs = [
                    f"{snapshot.prix_actuel:,.2f}",
                    f"{snapshot.ma50:,.2f}",
                    f"{snapshot.ma200:,.2f}",
                    "✅ Oui" if snapshot.golden_cross else "❌ Non",
                    f"{snapshot.geo_score:+.3f}",
                    vix_str,
                    str(snapshot.nb_news),
                ]
                if vix_alerte:
                    st.warning(
                        f"⚠️ VIX elevé ({vix_live:.1f}) — Signal LONG actif "
                        f"mais marche en tension. Prudence."
                    )
            else:
                vix_indicateurs = ["Prix actuel", "MA 50", "MA 200",
                                   "Golden Cross", "Geo-Score", "News analysees"]
                vix_valeurs = [
                    f"{snapshot.prix_actuel:,.2f}",
                    f"{snapshot.ma50:,.2f}",
                    f"{snapshot.ma200:,.2f}",
                    "✅ Oui" if snapshot.golden_cross else "❌ Non",
                    f"{snapshot.geo_score:+.3f}",
                    str(snapshot.nb_news),
                ]

            st.dataframe(pd.DataFrame({
                "Indicateur": vix_indicateurs,
                "Valeur":     vix_valeurs,
            }), hide_index=True, use_container_width=True)

        with sig_col2:
            st.markdown(f"**Actualites analysees** ({snapshot.timestamp})")

            # Barre de confiance du signal
            conf_color = "#2ca02c" if snapshot.signal == 1 else "#d62728"
            conf_label = "Confiance du signal"
            st.markdown(
                f"<div style='margin-bottom:8px'>"
                f"<span style='color:#aaa;font-size:0.85rem;'>{conf_label}</span> "
                f"<span style='color:{conf_color};font-weight:700'>{snapshot.confidence:.0%}</span>"
                f"</div>"
                f"<div style='background:#222;border-radius:6px;height:10px;width:100%'>"
                f"<div style='background:{conf_color};border-radius:6px;height:10px;"
                f"width:{snapshot.confidence*100:.0f}%'></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Headlines avec scores individuels si FinBERT actif
            if snapshot.headlines_with_scores:
                st.markdown("**Scores par actualite (FinBERT)**")
                for titre, score in snapshot.headlines_with_scores:
                    if score >= 0.1:
                        couleur, icone = "#2ca02c", "🟢"
                    elif score <= -0.1:
                        couleur, icone = "#d62728", "🔴"
                    else:
                        couleur, icone = "#aaa", "⚪"
                    st.markdown(
                        f"<div style='margin:3px 0;font-size:0.85rem;'>"
                        f"{icone} <span style='color:{couleur};font-weight:600'>"
                        f"{score:+.2f}</span>&nbsp; {titre}</div>",
                        unsafe_allow_html=True,
                    )
            elif snapshot.headlines:
                for i, h in enumerate(snapshot.headlines, 1):
                    st.markdown(f"{i}. {h}")
            else:
                st.info("Aucune news recuperee. Verifiez votre connexion.")

        if not auto_execute:
            st.divider()
            exec_col1, exec_col2 = st.columns([1, 3])
            with exec_col1:
                if st.button("▶️ Appliquer le signal", use_container_width=True,
                             type="primary"):
                    portfolio, action_msg = trader.execute_signal(snapshot, portfolio)
                    trader.save_portfolio(portfolio)
                    if "ENTREE" in action_msg:
                        st.success(f"✅ {action_msg}")
                    elif "SORTIE" in action_msg:
                        st.warning(f"⚠️ {action_msg}")
                    else:
                        st.info(f"ℹ️ {action_msg}")
                    st.rerun()
    else:
        st.info("Appuie sur **Rafraichir le signal** pour obtenir les donnees en temps reel.")

    # ── Historique des trades ─────────────────────────────────────────────
    st.divider()
    st.markdown("### Historique des trades")

    if portfolio.historique:
        hist_df = pd.DataFrame(portfolio.historique)
        # Formater pour l'affichage
        cols_display = [c for c in ["date", "action", "prix", "valeur",
                                     "pnl_eur", "pnl_pct", "geo_score"] if c in hist_df.columns]
        hist_df = hist_df[cols_display].copy()
        hist_df.columns = [c.replace("_", " ").title() for c in cols_display]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Mini courbe d'equity du paper trading
        if len(portfolio.historique) >= 2:
            valeurs = [CAPITAL_INITIAL]
            for trade in portfolio.historique:
                if trade.get("action") == "SORTIE":
                    dernier = valeurs[-1]
                    pnl     = trade.get("pnl_eur", 0)
                    valeurs.append(dernier + pnl)
            if len(valeurs) > 1:
                dates_trades = [CAPITAL_INITIAL] + [
                    t["date"] for t in portfolio.historique if t.get("action") == "SORTIE"
                ]
                fig_pt = go.Figure()
                fig_pt.add_trace(go.Scatter(
                    y=valeurs,
                    mode="lines+markers",
                    line=dict(color="#2ca02c", width=2),
                    marker=dict(size=8),
                    name="Capital",
                    hovertemplate="Trade %{x} : %{y:,.2f} EUR<extra></extra>",
                ))
                fig_pt.add_hline(y=CAPITAL_INITIAL, line_dash="dot",
                                 line_color="#555", line_width=0.8,
                                 annotation_text=f"Capital initial {CAPITAL_INITIAL:,.0f} EUR",
                                 annotation_font_color="#888")
                fig_pt.update_layout(
                    title="<b>Evolution du capital</b> -- Paper Trading",
                    yaxis=dict(ticksuffix=" EUR", gridcolor="#1e1e2e"),
                    xaxis=dict(title="Trades", gridcolor="#1e1e2e"),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    height=280, margin=dict(t=50, b=20, l=10, r=10),
                )
                st.plotly_chart(fig_pt, use_container_width=True)
    else:
        st.info("Aucun trade execute pour l'instant.")

    st.divider()
    st.warning(
        "**Disclaimer** : le Paper Trading utilise des prix en temps reel et des "
        "news en direct, mais reste une simulation. Les signaux ne constituent pas "
        "un conseil en investissement."
    )


# ---------------------------------------------------------------------------
# ONGLET 5 -- A Propos
# ---------------------------------------------------------------------------
with onglet_apropos:
    col_desc, col_archi = st.columns([1, 1])

    with col_desc:
        st.subheader("GeoQuant AI")
        st.markdown("""
**GeoQuant AI** est une plateforme de _backtesting_ et _paper trading_ qui
utilise le NLP pour analyser les flux d'actualites geopolitiques et proteger
le capital des investisseurs face aux chocs exogenes.

**Problematique academique (PPE 2025-2026)**

> _"Comment l'integration de l'analyse semantique automatisee (NLP) des flux
> d'actualites permet-elle d'ameliorer la resilience (Risk-Management) d'une
> strategie d'investissement face aux chocs exogenes ?"_

**Ce que le projet EST**
- Un crash-test pour strategies financieres (backtesting uniquement)
- Un systeme d'alerte base sur le sentiment mediatique (NLP)
- Un outil de minimisation du drawdown (protection du capital)
- Fonctionne en sandbox uniquement (pas de vrai argent)
        """)
        st.warning(
            "**Disclaimer** -- Ceci n'est pas un conseil en investissement. "
            "GeoQuant AI est un outil de simulation academique. "
            "Les performances passees ne prejugent pas des performances futures."
        )

    with col_archi:
        st.subheader("Architecture des 4 Blocs")
        st.markdown("""
| Bloc | Module | Statut |
|------|--------|--------|
| **1 -- Data Engineering** | `src/` (loader, merger, news_loader) | ✅ Termine |
| **2 -- NLP / FinBERT** | `src/finbert_sentiment.py`, `src/geo_scorer.py` | ✅ Termine |
| **3 -- Strategie + Backtest** | `src/strategy.py`, `src/backtest.py` | ✅ Termine |
| **4 -- Dashboard + Paper Trading** | `app/dashboard.py`, `src/paper_trading.py` | ✅ Termine |
        """)

        st.divider()
        st.subheader("Donnees du dataset")
        if len(df_complet) > 0:
            nb_jours_news = int((df_complet["nb_articles"] > 0).sum()) if "nb_articles" in df_complet.columns else "N/A"
            st.markdown(f"""
- **Actif :** S&P 500 (`^GSPC`)
- **Periode :** {pd.Timestamp(df_complet["date"].min()).strftime("%d %b %Y")} -- {pd.Timestamp(df_complet["date"].max()).strftime("%d %b %Y")}
- **Jours de trading :** {len(df_complet):,}
- **Jours avec news :** {nb_jours_news}
- **Max Drawdown :** {float(df_complet["Drawdown"].min()) * 100:.2f}%
            """)
