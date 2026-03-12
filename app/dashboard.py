"""GeoQuant AI -- Interactive Dashboard (Bloc 4 -- Streamlit).

Run from the project root with:
    streamlit run app/dashboard.py
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
DATASET_PATH  = ROOT / "data" / "processed" / "dataset_final.csv"
NEWS_RAW_PATH = ROOT / "data" / "raw" / "sample_news.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))

# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="GeoQuant AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Titre principal */
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #888; font-size: 0.9rem; margin-top: 0; }

    /* Cartes de métriques */
    [data-testid="metric-container"] {
        background: #1e1e2e; border-radius: 10px;
        padding: 10px 16px; border: 1px solid #333;
    }

    /* Bloc "placeholder Bloc X" */
    .bloc-placeholder {
        background: #1e1e2e; border: 1px dashed #555;
        border-radius: 12px; padding: 32px 24px;
        text-align: center; color: #888;
    }
    .bloc-placeholder h3 { color: #ccc; }

    /* Badge de statut */
    .badge-done    { color: #2ca02c; font-weight: 700; }
    .badge-wip     { color: #ff7f0e; font-weight: 700; }
    .badge-pending { color: #888;    font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
KEY_EVENTS: list[tuple[str, str, str]] = [
    ("2022-02-24", "Invasion Ukraine",  "red"),
    ("2022-06-15", "Fed +75bp",         "#9467bd"),
    ("2022-11-11", "FTX collapse",      "orange"),
    ("2023-03-10", "SVB collapse",      "firebrick"),
    ("2023-10-07", "Hamas attack",      "crimson"),
    ("2024-08-05", "Japan crash",       "saddlebrown"),
    ("2024-09-18", "Fed -50bp",         "#2ca02c"),
    ("2024-11-05", "Trump elected",     "royalblue"),
]

# Mapping symbol -> nom lisible
TICKER_NAMES: dict[str, str] = {
    "^GSPC":    "SP500",
    "^FCHI":    "CAC40",
    "EURUSD=X": "EURUSD",
    "BTC-USD":  "Bitcoin",
    "GC=F":     "Gold",
    "CL=F":     "WTI Oil",
    "^VIX":     "VIX",
    "^IXIC":    "NASDAQ",
    "^DJI":     "Dow Jones",
}
NAME_TO_TICKER: dict[str, str] = {v: k for k, v in TICKER_NAMES.items()}

CLR_PRICE  = "#1f77b4"
CLR_MA50   = "#ff7f0e"
CLR_MA200  = "#2ca02c"
CLR_DANGER = "#d62728"
CLR_NEWS   = "#9467bd"

# ── Data helpers ──────────────────────────────────────────────────────────────

def _compute_tech_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute MA, RSI, Volatility and Drawdown for a raw price DataFrame."""
    df = df.copy()
    prices = df[price_col]

    df["MA20"]  = prices.rolling(20,  min_periods=1).mean()
    df["MA50"]  = prices.rolling(50,  min_periods=1).mean()
    df["MA200"] = prices.rolling(200, min_periods=1).mean()

    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    log_ret = np.log(prices / prices.shift(1))
    df["Volatility_20"] = log_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

    running_max = prices.cummax()
    df["Drawdown"] = (prices - running_max) / running_max

    if "Returns" not in df.columns:
        df["Returns"] = prices.pct_change()

    return df


@st.cache_data
def load_dataset_final() -> Optional[pd.DataFrame]:
    """Load the primary merged dataset (Bloc 1 output)."""
    if not DATASET_PATH.exists():
        return None
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    # La colonne titles peut etre lue comme float si vide (NaN) -> normalise
    df["titles"] = df["titles"].fillna("").astype(str).replace("nan", "")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_raw_news() -> Optional[pd.DataFrame]:
    """Load individual news articles from sample_news.csv."""
    if not NEWS_RAW_PATH.exists():
        return None
    df = pd.read_csv(NEWS_RAW_PATH, parse_dates=["date"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


@st.cache_data
def list_available_tickers() -> list[str]:
    """Return display names for tickers that have a processed CSV on disk."""
    names: list[str] = []
    for symbol, name in TICKER_NAMES.items():
        safe = symbol.replace("/", "_")
        if (PROCESSED_DIR / f"{safe}_processed.csv").exists():
            names.append(name)
    # Primary ticker (dataset_final) first
    if not names and DATASET_PATH.exists():
        names = ["SP500"]
    return names


@st.cache_data
def load_ticker_csv(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load a per-ticker processed CSV and normalise it into a date-indexed DataFrame.

    Falls back to dataset_final.csv for the primary ticker so that news
    columns are preserved.
    """
    # Use the rich merged dataset for the primary ticker
    if DATASET_PATH.exists():
        master = pd.read_csv(DATASET_PATH, nrows=1)
        primary_symbol = master["Ticker"].iloc[0] if "Ticker" in master.columns else ""
        if symbol == primary_symbol:
            return load_dataset_final()

    safe = symbol.replace("/", "_")
    path = PROCESSED_DIR / f"{safe}_processed.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, index_col=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df = _compute_tech_features(df, price_col)
    return df


# ── Chart builders ────────────────────────────────────────────────────────────

def _add_event_lines(
    fig: go.Figure,
    df: pd.DataFrame,
    show_events: bool,
    row: int = 1,
    col: int = 1,
    y_ref: float = 1.0,
) -> None:
    """Add vertical dashed lines for key geopolitical events."""
    if not show_events:
        return
    date_min = pd.to_datetime(df["date"].min())
    date_max = pd.to_datetime(df["date"].max())
    for evt_date_str, label, color in KEY_EVENTS:
        evt_dt = pd.Timestamp(evt_date_str)
        if date_min <= evt_dt <= date_max:
            fig.add_vline(
                x=evt_dt.timestamp() * 1000,
                line_dash="dash", line_color=color,
                line_width=1.0, opacity=0.65,
                row=row, col=col,
                annotation_text=label,
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color=color,
                annotation_textangle=-90,
            )


def build_price_chart(
    df: pd.DataFrame,
    ticker_name: str,
    show_ma: bool,
    show_events: bool,
    geo_threshold: float,
) -> go.Figure:
    """
    Build the main price + news activity chart.

    Two vertically stacked subplots:
      1. Price line + optional MA50/MA200 + event annotations
      2. Daily news article count (bar)
    """
    has_news = "nb_articles" in df.columns and df["nb_articles"].sum() > 0
    row_heights = [0.72, 0.28] if has_news else [1.0]
    rows = 2 if has_news else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=("", "News articles / day" if has_news else ""),
    )

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # Prix
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[price_col],
        name=ticker_name,
        line=dict(color=CLR_PRICE, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Price: %{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Moyennes mobiles
    if show_ma and "MA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MA50"],
            name="MA 50", line=dict(color=CLR_MA50, width=1.2, dash="dot"),
            hovertemplate="MA50: %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["MA200"],
            name="MA 200", line=dict(color=CLR_MA200, width=1.5, dash="dash"),
            hovertemplate="MA200: %{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # Events
    _add_event_lines(fig, df, show_events, row=1, col=1)

    # News bars
    if has_news:
        fig.add_trace(go.Bar(
            x=df["date"], y=df["nb_articles"],
            name="Articles/day",
            marker_color=CLR_NEWS, opacity=0.75,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y} article(s)<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker_name}</b> -- Price & Media Activity",
            font=dict(size=16),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=60, b=20, l=10, r=10),
        height=520,
    )
    fig.update_xaxes(gridcolor="#1e1e2e", zeroline=False)
    fig.update_yaxes(gridcolor="#1e1e2e", zeroline=False)
    return fig


def build_geo_score_chart(df: pd.DataFrame, threshold: float) -> go.Figure:
    """Build the Geo-Score timeline with colored background zones."""
    fig = go.Figure()

    col = "geo_score"
    x = df["date"]
    y = df[col]

    # Zone rouge (panique)
    fig.add_hrect(y0=-1.0, y1=threshold,
                  fillcolor="rgba(214,39,40,0.08)", line_width=0)
    # Zone orange
    fig.add_hrect(y0=threshold, y1=0.0,
                  fillcolor="rgba(255,127,14,0.06)", line_width=0)
    # Zone verte
    fig.add_hrect(y0=0.0, y1=1.0,
                  fillcolor="rgba(44,160,44,0.06)", line_width=0)

    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  line_width=1.2, opacity=0.7,
                  annotation_text=f"Seuil {threshold:.1f}",
                  annotation_position="bottom right",
                  annotation_font_color="red")
    fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=0.8)

    fig.add_trace(go.Scatter(
        x=x, y=y,
        name="Geo-Score",
        line=dict(color="#00b4d8", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Geo-Score: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Geo-Score</b> -- Sentiment NLP Quotidien",
        yaxis=dict(range=[-1.05, 1.05], gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=340,
        hovermode="x unified",
    )
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Build RSI 14-day chart with overbought / oversold zones."""
    fig = go.Figure()

    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(214,39,40,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(44,160,44,0.08)",  line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=CLR_DANGER, line_width=0.9,
                  annotation_text="Surachat 70",
                  annotation_position="bottom right",
                  annotation_font_color=CLR_DANGER)
    fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c", line_width=0.9,
                  annotation_text="Survente 30",
                  annotation_position="top right",
                  annotation_font_color="#2ca02c")
    fig.add_hline(y=50, line_dash="dot", line_color="#555", line_width=0.7)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["RSI_14"],
        name="RSI 14",
        line=dict(color="#ff7f0e", width=1.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>RSI: %{y:.1f}<extra></extra>",
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


def build_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Build the rolling drawdown chart."""
    fig = go.Figure()

    dd_pct = df["Drawdown"] * 100
    max_dd_idx = dd_pct.idxmin()
    max_dd_val = dd_pct.iloc[max_dd_idx]
    max_dd_date = df["date"].iloc[max_dd_idx]

    fig.add_trace(go.Scatter(
        x=df["date"], y=dd_pct,
        name="Drawdown",
        line=dict(color=CLR_DANGER, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(214,39,40,0.15)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig.add_annotation(
        x=max_dd_date, y=max_dd_val,
        text=f"Max DD: {max_dd_val:.1f}%",
        showarrow=True, arrowhead=2, arrowcolor=CLR_DANGER,
        font=dict(color=CLR_DANGER, size=11),
        bgcolor="#0e1117", bordercolor=CLR_DANGER,
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


def build_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Build rolling 20-day annualised volatility chart."""
    fig = go.Figure()

    vol_pct = df["Volatility_20"] * 100
    avg_vol = vol_pct.mean()

    fig.add_trace(go.Scatter(
        x=df["date"], y=vol_pct,
        name="Volatilite 20j",
        line=dict(color="#9467bd", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(148,103,189,0.10)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Vol: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=avg_vol, line_dash="dash", line_color="#888", line_width=0.9,
        annotation_text=f"Moy. {avg_vol:.1f}%",
        annotation_position="bottom right",
        annotation_font_color="#888",
    )

    fig.update_layout(
        title="<b>Volatilite annualisee 20j</b> (ecart-type des log-rendements)",
        yaxis=dict(ticksuffix="%", gridcolor="#1e1e2e"),
        xaxis=dict(gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=50, b=20, l=10, r=10),
        height=280,
    )
    return fig


def build_return_distribution(df: pd.DataFrame, ticker_name: str) -> go.Figure:
    """Build histogram of daily returns with normal PDF overlay."""
    returns = df["Returns"].dropna() * 100
    mu, sigma = float(returns.mean()), float(returns.std())
    var_5 = float(returns.quantile(0.05))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=80,
        name="Rendements (%)", histnorm="probability density",
        marker_color=CLR_PRICE, opacity=0.65,
        hovertemplate="Rendement: %{x:.2f}%<br>Densite: %{y:.4f}<extra></extra>",
    ))

    # Courbe normale theorique
    x_range = np.linspace(returns.min(), returns.max(), 300)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf,
        name=f"Normale (mu={mu:.2f}%, sigma={sigma:.2f}%)",
        line=dict(color="red", width=1.8),
    ))

    fig.add_vline(x=mu,    line_dash="dash", line_color="orange", line_width=1.2,
                  annotation_text=f"Moy. {mu:.2f}%", annotation_position="top right",
                  annotation_font_color="orange")
    fig.add_vline(x=var_5, line_dash="dot",  line_color=CLR_DANGER, line_width=1.2,
                  annotation_text=f"VaR 5% {var_5:.2f}%", annotation_position="top left",
                  annotation_font_color=CLR_DANGER)

    fig.update_layout(
        title=f"<b>Distribution des rendements quotidiens</b> -- {ticker_name}",
        xaxis=dict(title="Rendement journalier (%)", gridcolor="#1e1e2e"),
        yaxis=dict(title="Densite", gridcolor="#1e1e2e"),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=40, l=10, r=10),
        height=360,
        barmode="overlay",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="main-title">🌍 GeoQuant AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Backtesting & NLP -- Analyse geopolitique des marches</p>',
                unsafe_allow_html=True)
    st.divider()

    # Choix de l'actif
    available = list_available_tickers()
    if not available:
        st.warning("Aucun actif disponible. Lancez d'abord `python src/main.py`.")
        selected_name = "SP500"
    else:
        selected_name = st.selectbox(
            "Actif",
            options=available,
            index=0,
            help="Actif principal = SP500 (dataset complet avec news). "
                 "Autres actifs : donnees de prix uniquement.",
        )

    selected_symbol = NAME_TO_TICKER.get(selected_name, "^GSPC")

    # Chargement des donnees pour l'actif selectionne
    df_raw = load_ticker_csv(selected_symbol)

    # Plage de dates
    st.divider()
    if df_raw is not None and len(df_raw) > 0:
        date_min = pd.to_datetime(df_raw["date"].min()).date()
        date_max = pd.to_datetime(df_raw["date"].max()).date()
        date_range = st.slider(
            "Periode",
            min_value=date_min,
            max_value=date_max,
            value=(date_min, date_max),
            format="DD/MM/YY",
        )
    else:
        date_range = (None, None)

    # Seuil Geo-Score
    st.divider()
    st.markdown("**Seuil Geo-Score** (Bloc 3)")
    geo_threshold = st.slider(
        "Si Geo-Score < seuil -> Cash",
        min_value=-1.0, max_value=0.0, value=-0.5, step=0.05,
        format="%.2f",
        help="Ce seuil sera utilise par le moteur de strategie (Bloc 3) "
             "pour couper les positions en cas de panique geopolitique.",
    )
    st.caption(f"Seuil actuel : **{geo_threshold:.2f}**")

    # Toggles
    st.divider()
    show_ma     = st.toggle("Afficher les moyennes mobiles", value=True)
    show_events = st.toggle("Annoter les evenements geopolitiques", value=True)

    # Bouton actualiser (vide le cache)
    st.divider()
    if st.button("Actualiser les donnees", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Statut du pipeline
    st.divider()
    st.markdown("**Statut du pipeline**")
    st.markdown('<span class="badge-done">BLOC 1</span> Data Engineering', unsafe_allow_html=True)
    st.markdown('<span class="badge-pending">BLOC 2</span> NLP / FinBERT', unsafe_allow_html=True)
    st.markdown('<span class="badge-pending">BLOC 3</span> Strategie', unsafe_allow_html=True)
    st.markdown('<span class="badge-wip">BLOC 4</span> Dashboard (en cours)', unsafe_allow_html=True)

# ── Garde: dataset manquant ───────────────────────────────────────────────────
if df_raw is None or len(df_raw) == 0:
    st.error(
        "**Dataset introuvable.**\n\n"
        "Le fichier `data/processed/dataset_final.csv` n'a pas ete genere.\n\n"
        "**Pour le creer, lancez :**\n"
        "```bash\ncd PPE_2025/src\npython main.py\n```"
    )
    st.stop()

# ── Filtrage par periode ──────────────────────────────────────────────────────
df = df_raw.copy()
if date_range[0] is not None:
    mask = (
        (df["date"] >= pd.Timestamp(date_range[0]))
        & (df["date"] <= pd.Timestamp(date_range[1]))
    )
    df = df[mask].reset_index(drop=True)

price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

# ── En-tete principal ─────────────────────────────────────────────────────────
col_title, col_period = st.columns([3, 1])
with col_title:
    st.markdown(f'<div class="main-title">GeoQuant AI</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="subtitle">{selected_name} &nbsp;|&nbsp; '
        f'{pd.Timestamp(date_range[0]).strftime("%d %b %Y") if date_range[0] else "?"}'
        f' -- '
        f'{pd.Timestamp(date_range[1]).strftime("%d %b %Y") if date_range[1] else "?"}'
        f'</p>',
        unsafe_allow_html=True,
    )

# ── Onglets ───────────────────────────────────────────────────────────────────
tab_market, tab_nlp, tab_backtest, tab_tech, tab_about = st.tabs([
    "📊 Vue Marche",
    "🧠 Sentiment & NLP",
    "⚔️ Backtest",
    "📈 Analyse Technique",
    "ℹ️ A Propos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 -- VUE MARCHE
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:

    # Metriques cles
    if len(df) >= 2:
        current_price  = float(df[price_col].iloc[-1])
        first_price    = float(df[price_col].iloc[0])
        total_return   = (current_price / first_price - 1) * 100
        max_dd         = float(df["Drawdown"].min() * 100) if "Drawdown" in df.columns else 0.0
        ann_vol        = float(df["Volatility_20"].dropna().mean() * 100) if "Volatility_20" in df.columns else 0.0
        days_with_news = int((df["nb_articles"] > 0).sum()) if "nb_articles" in df.columns else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Prix actuel",         f"{current_price:,.2f}")
        m2.metric("Rendement total",     f"{total_return:+.2f}%",
                  delta=f"{total_return:+.2f}%",
                  delta_color="normal")
        m3.metric("Max Drawdown",        f"{max_dd:.2f}%",
                  delta=f"{max_dd:.2f}%",
                  delta_color="inverse")
        m4.metric("Volatilite annuelle", f"{ann_vol:.1f}%")
        m5.metric("Jours avec news",     str(days_with_news))

    st.divider()

    # Graphique principal prix + news
    fig_price = build_price_chart(df, selected_name, show_ma, show_events, geo_threshold)
    st.plotly_chart(fig_price, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 -- SENTIMENT & NLP
# ══════════════════════════════════════════════════════════════════════════════
with tab_nlp:

    has_geo_score = "geo_score" in df.columns and df["geo_score"].notna().any()

    if has_geo_score:
        latest_score = float(df["geo_score"].dropna().iloc[-1])
        st.subheader("Geo-Score -- Sentiment du marche")

        g1, g2 = st.columns([1, 2])
        with g1:
            # Jauge Geo-Score
            color = CLR_DANGER if latest_score < geo_threshold else (
                "#ff7f0e" if latest_score < 0 else "#2ca02c"
            )
            label = "PANIQUE" if latest_score < geo_threshold else (
                "PRUDENCE" if latest_score < 0 else "CONFIANCE"
            )
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=latest_score,
                domain=dict(x=[0, 1], y=[0, 1]),
                title=dict(text=f"Geo-Score du jour<br><span style='color:{color}'>{label}</span>",
                           font=dict(size=14)),
                number=dict(font=dict(size=36, color=color)),
                delta=dict(reference=0, relative=False),
                gauge=dict(
                    axis=dict(range=[-1, 1], tickwidth=1),
                    bar=dict(color=color),
                    bgcolor="#1e1e2e",
                    steps=[
                        dict(range=[-1, geo_threshold], color="rgba(214,39,40,0.15)"),
                        dict(range=[geo_threshold, 0],  color="rgba(255,127,14,0.10)"),
                        dict(range=[0, 1],              color="rgba(44,160,44,0.10)"),
                    ],
                    threshold=dict(
                        line=dict(color=CLR_DANGER, width=3),
                        thickness=0.75,
                        value=geo_threshold,
                    ),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0e1117", font=dict(color="#fafafa"),
                height=280, margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with g2:
            fig_geo = build_geo_score_chart(df, geo_threshold)
            st.plotly_chart(fig_geo, use_container_width=True)

    else:
        st.markdown("""
<div class="bloc-placeholder">
    <h3>🔜 Bloc 2 -- NLP en attente</h3>
    <p>Le <b>Geo-Score</b> sera calcule par FinBERT (Hugging Face) sur les titres de news.</p>
    <p>La colonne <code>geo_score</code> sera ajoutee a <code>dataset_final.csv</code>
    apres execution du module <code>nlp/geo_scorer.py</code>.</p>
    <br>
    <table style="margin: 0 auto; text-align:left; color:#aaa; font-size:0.85rem;">
        <tr><td>Modele prevu</td><td>&nbsp; FinBERT (ProsusAI/finbert)</td></tr>
        <tr><td>Input</td><td>&nbsp; Titres de news concatenes par jour</td></tr>
        <tr><td>Output</td><td>&nbsp; Score quotidien entre -1 (Panique) et +1 (Confiance)</td></tr>
    </table>
</div>""", unsafe_allow_html=True)

    # Tableau des dernieres news (depuis sample_news.csv)
    st.divider()
    st.subheader("Dernieres actualites")

    news_df = load_raw_news()
    if news_df is not None and len(news_df) > 0:
        # Filtrer sur la periode selectionnee
        news_filtered = news_df.copy()
        if date_range[0] is not None:
            news_filtered = news_filtered[
                (news_filtered["date"] >= pd.Timestamp(date_range[0]))
                & (news_filtered["date"] <= pd.Timestamp(date_range[1]))
            ]

        # Afficher les 15 plus recentes
        cols_to_show = [c for c in ["date", "title", "source", "category"] if c in news_filtered.columns]
        display_news = news_filtered[cols_to_show].head(15).copy()
        if "date" in display_news.columns:
            display_news["date"] = display_news["date"].dt.strftime("%d %b %Y")

        col_config: dict = {}
        if "category" in display_news.columns:
            col_config["category"] = st.column_config.TextColumn("Categorie", width="small")

        st.dataframe(display_news, use_container_width=True, hide_index=True,
                     column_config=col_config)
    else:
        st.info("Aucune news disponible. Verifiez `data/raw/sample_news.csv`.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 -- BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:

    st.markdown("""
<div class="bloc-placeholder">
    <h3>⚔️ Bloc 3 -- Moteur de Strategie en attente</h3>
    <p>Le backtesting comparatif <b>Buy &amp; Hold vs GeoQuant</b> sera implemente ici.</p>
    <br>
    <p style="color:#666; font-size:0.8rem;">
        Regle prevue : SI (MA50 &gt; MA200) ET (Geo-Score &gt; seuil) ALORS Long SINON Cash
    </p>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Emplacements prevus pour le Bloc 3
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Graphique comparatif (a venir)**")
        fig_placeholder = go.Figure()
        fig_placeholder.add_annotation(
            text="Disponible apres Bloc 3", x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="#555"),
        )
        fig_placeholder.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0e1117", paper_bgcolor="#1e1e2e",
            height=280, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)

    with c2:
        st.markdown("**Metriques de performance (a venir)**")
        placeholder_metrics = pd.DataFrame({
            "Metrique":   ["Rendement total", "Max Drawdown", "Sharpe Ratio", "Win Rate"],
            "Buy & Hold": ["?", "?", "?", "?"],
            "GeoQuant":   ["?", "?", "?", "?"],
        })
        st.dataframe(placeholder_metrics, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Journal des trades virtuels (a venir)**")
    placeholder_trades = pd.DataFrame({
        "Date":   ["--", "--", "--"],
        "Signal": ["--", "--", "--"],
        "Prix":   ["--", "--", "--"],
        "P&L":    ["--", "--", "--"],
    })
    st.dataframe(placeholder_trades, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 -- ANALYSE TECHNIQUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_tech:

    missing = [c for c in ("RSI_14", "Drawdown", "Volatility_20", "Returns")
               if c not in df.columns]
    if missing:
        st.warning(f"Colonnes manquantes : {missing}. Relancez `python src/main.py`.")
    else:
        # RSI
        st.plotly_chart(build_rsi_chart(df), use_container_width=True)
        st.divider()

        # Drawdown + Volatilite cote a cote
        col_dd, col_vol = st.columns(2)
        with col_dd:
            st.plotly_chart(build_drawdown_chart(df), use_container_width=True)
        with col_vol:
            st.plotly_chart(build_volatility_chart(df), use_container_width=True)

        st.divider()

        # Distribution des rendements
        st.plotly_chart(build_return_distribution(df, selected_name), use_container_width=True)

        # Statistiques descriptives
        st.divider()
        st.subheader("Statistiques descriptives")
        returns_pct = df["Returns"].dropna() * 100
        stats = pd.DataFrame({
            "Metrique": [
                "Rendement moyen/jour", "Ecart-type",
                "Skewness", "Kurtosis",
                "VaR 5%", "CVaR 5%",
                "Max perte 1j", "Max gain 1j",
            ],
            "Valeur": [
                f"{returns_pct.mean():.4f}%",
                f"{returns_pct.std():.4f}%",
                f"{returns_pct.skew():.3f}",
                f"{returns_pct.kurtosis():.3f}",
                f"{returns_pct.quantile(0.05):.4f}%",
                f"{returns_pct[returns_pct <= returns_pct.quantile(0.05)].mean():.4f}%",
                f"{returns_pct.min():.4f}%",
                f"{returns_pct.max():.4f}%",
            ],
        })
        st.dataframe(stats, use_container_width=False, hide_index=True, width=420)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 -- A PROPOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:

    col_desc, col_archi = st.columns([1, 1])

    with col_desc:
        st.subheader("GeoQuant AI")
        st.markdown("""
**GeoQuant AI** est une plateforme SaaS de _backtesting_ et _paper trading_
qui utilise le NLP pour analyser les flux d'actualites geopolitiques
et proteger le capital des investisseurs face aux chocs exogenes.

**Problematique academique (PPE 2025-2026)**

> _"Comment l'integration de l'analyse semantique automatisee (NLP) des flux
> d'actualites permet-elle d'ameliorer la resilience (Risk-Management) d'une
> strategie d'investissement face aux chocs exogenes ?"_

**Ce que le projet EST**
- Un crash-test pour strategies financieres (backtesting)
- Un systeme d'alerte precoce base sur le sentiment mediatique (NLP)
- Un outil de minimisation du drawdown (protection du capital)
- Fonctionne en sandbox uniquement (paper trading, pas de vrai argent)
        """)

        # Disclaimer legal
        st.warning(
            "**Disclaimer** -- Ceci n'est pas un conseil en investissement. "
            "GeoQuant AI est un outil de simulation academique uniquement. "
            "Les performances passees ne prejugent pas des performances futures. "
            "N'investissez pas sur la base de ces simulations."
        )

    with col_archi:
        st.subheader("Architecture des 4 Blocs")
        st.markdown("""
| Bloc | Module | Statut |
|------|--------|--------|
| **1 -- Data Engineering** | `src/` (loader, merger, news_loader) | ✅ Termine |
| **2 -- NLP / FinBERT** | `nlp/` (a creer) | ⏳ A venir |
| **3 -- Strategie** | `strategy/` (a creer) | ⏳ A venir |
| **4 -- Dashboard** | `app/dashboard.py` | 🔄 En cours |
        """)

        st.divider()
        st.subheader("Stack technique")
        st.markdown("""
| Couche | Technologie |
|--------|-------------|
| Donnees de marche | `yfinance` + `pandas` |
| Features techniques | `numpy` (MA, RSI, Vol, Drawdown) |
| NLP | `FinBERT` via Hugging Face `transformers` |
| Visualisation | `plotly` + `streamlit` |
| Interface | `streamlit` |
        """)

        st.divider()
        st.subheader("Dataset (Bloc 1)")
        if len(df) > 0:
            st.markdown(f"""
- **Actif principal :** SP500 (`^GSPC`)
- **Periode :** {pd.Timestamp(df['date'].min()).strftime('%d %b %Y')} -- {pd.Timestamp(df['date'].max()).strftime('%d %b %Y')}
- **Jours de trading :** {len(df_raw):,}
- **Jours avec news :** {int((df_raw['nb_articles'] > 0).sum()) if 'nb_articles' in df_raw.columns else 'N/A'}
- **Colonnes :** {', '.join(f'`{c}`' for c in df.columns[:8])} ...
            """)
