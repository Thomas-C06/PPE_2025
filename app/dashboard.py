from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from backtest import run_backtest
from config import GEO_SCORES_FILE, NEWS_SCORED_FILE
from strategy import Strategy

try:
    from paper_trading import PaperTrader
except Exception:
    PaperTrader = None

CHEMIN_DATASET = RACINE / "data" / "processed" / "dataset_final.csv"
CHEMIN_NEWS = RACINE / "data" / "processed" / NEWS_SCORED_FILE
CHEMIN_GEO = RACINE / "data" / "processed" / GEO_SCORES_FILE

EVENEMENTS = [
    ("2022-02-24", "Ukraine", "#d96c6c"),
    ("2022-06-15", "Fed +75 pb", "#8b7cf7"),
    ("2023-03-10", "SVB", "#d48c5d"),
    ("2023-10-07", "Moyen-Orient", "#d9b35d"),
]

BLEU = "#6ccff6"
VERT = "#59c79a"
OR = "#d9b35d"
ROUGE = "#d96c6c"
FOND = "#07111f"
TEXTE = "#edf2f8"

st.set_page_config(page_title="GeoQuant AI", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.stApp{background:radial-gradient(circle at top right,rgba(108,207,246,.12),transparent 25%),linear-gradient(180deg,#07111f 0%,#091423 100%);color:#edf2f8}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{display:none}
[data-testid="stDecoration"]{display:none}
[data-testid="collapsedControl"]{top:.85rem;left:.85rem;background:rgba(12,23,39,.92);border:1px solid rgba(143,167,189,.18);border-radius:12px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1526 0%,#0b182a 100%);border-right:1px solid rgba(143,167,189,.12)}
.block-container{padding-top:.8rem;padding-bottom:2rem;max-width:1420px}
.hero,.box,.callout{border:1px solid rgba(143,167,189,.14);border-radius:18px;background:linear-gradient(180deg,rgba(16,31,50,.98) 0%,rgba(12,23,39,.98) 100%);box-shadow:0 14px 28px rgba(2,8,16,.18)}
.hero{padding:1.05rem 1.2rem 1rem 1.2rem;margin-bottom:1rem}
.kicker{font-size:.74rem;text-transform:uppercase;letter-spacing:.12em;color:#8db4d8;font-weight:700}
.title{font-size:2.15rem;line-height:1.05;font-weight:780;color:#f4f7fb;margin:.2rem 0}
.subtitle{color:#9db3c8;font-size:.96rem;margin:.2rem 0 .65rem 0;max-width:940px}
.pipe{display:flex;flex-wrap:wrap;gap:.5rem}.step{padding:.38rem .72rem;border-radius:999px;border:1px solid rgba(143,167,189,.16);background:rgba(12,27,44,.72);font-size:.84rem;font-weight:650}
.section-title{font-size:1.2rem;font-weight:760;margin-bottom:.12rem}.section-subtitle{color:#8fa7bd;margin-bottom:.9rem}
.market-wrap{padding-top:.65rem;padding-bottom:.85rem}
.box{padding:1rem}.label{color:#8fa7bd;font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;margin-bottom:.45rem}
.value{color:#f3f7fb;font-size:1.72rem;font-weight:770;line-height:1.05}.note{color:#9db3c8;font-size:.9rem;line-height:1.4}
.headline{color:#f3f7fb;font-size:1.08rem;font-weight:730;line-height:1.38;margin-bottom:.55rem}
.callout{padding:.9rem 1rem;margin-bottom:.9rem;line-height:1.45}.warn{border-color:rgba(217,179,93,.24);background:rgba(44,35,14,.42)}.good{border-color:rgba(89,199,154,.24);background:rgba(15,42,32,.42)}
.badge{display:inline-block;padding:.28rem .56rem;border-radius:999px;font-size:.78rem;font-weight:750;letter-spacing:.04em}.buy{background:rgba(89,199,154,.18);color:#9de3c7}.reduce{background:rgba(217,108,108,.18);color:#f1b0b0}.cash{background:rgba(217,179,93,.18);color:#f1d59a}.wait{background:rgba(143,167,189,.18);color:#cdd9e6}.light{background:rgba(108,207,246,.15);color:#a6e6fb}
.focus-card{min-height:220px}
.focus-news{min-height:220px}
.focus-score{border-color:rgba(108,207,246,.22)}
.focus-decision{border-color:rgba(89,199,154,.22)}
.focus-impact{border-color:rgba(217,179,93,.20)}
.minor-card{min-height:146px}
[data-testid="stExpander"]{border:1px solid rgba(143,167,189,.10);border-radius:16px;background:rgba(12,23,39,.55)}
[data-testid="metric-container"]{background:linear-gradient(180deg,rgba(16,31,50,.98) 0%,rgba(12,23,39,.98) 100%);border:1px solid rgba(143,167,189,.12);border-radius:16px;padding:12px 14px;box-shadow:0 14px 28px rgba(2,8,16,.18)}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}%}"


def fmt_signed(x: float | None, digits: int = 3) -> str:
    return "n/d" if x is None or pd.isna(x) else f"{x:+.{digits}f}"


def fmt_date(x: Any) -> str:
    return pd.Timestamp(x).strftime("%d/%m/%Y")


def badge(decision: str) -> str:
    css = {
        "BUY": "buy",
        "LIGHT BUY": "light",
        "HOLD LIGHT": "light",
        "REDUCE": "reduce",
        "CASH": "cash",
        "WAIT": "wait",
    }.get(decision, "wait")
    return f'<span class="badge {css}">{decision}</span>'


def card(label: str, value: str, note: str = "") -> str:
    return f'<div class="box"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'


def focus_card(label: str, value: str, note: str = "", extra_class: str = "") -> str:
    css = f"box focus-card {extra_class}".strip()
    return f'<div class="{css}"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'


def section(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div><div class="section-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def charger_dataset() -> Optional[pd.DataFrame]:
    if not CHEMIN_DATASET.exists():
        return None
    df = pd.read_csv(CHEMIN_DATASET, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for col in ("titles", "news"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


@st.cache_data(show_spinner=False)
def charger_news_scorees() -> Optional[pd.DataFrame]:
    if not CHEMIN_NEWS.exists():
        return None
    return pd.read_csv(CHEMIN_NEWS, parse_dates=["date"]).sort_values(["date", "geo_score_article"], ascending=[False, True]).reset_index(drop=True)


def style_plot(fig: go.Figure, title: str, height: int, ytitle: str | None = None, xtitle: str | None = None) -> go.Figure:
    fig.update_layout(
        title=f"<b>{title}</b>",
        plot_bgcolor=FOND,
        paper_bgcolor=FOND,
        font=dict(color=TEXTE),
        hovermode="x unified",
        height=height,
        margin=dict(t=72, b=24, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="left", x=0.0),
    )
    fig.update_xaxes(gridcolor="#1b2940", title=xtitle)
    fig.update_yaxes(gridcolor="#1b2940", title=ytitle)
    return fig


def ajouter_evenements(fig: go.Figure, df: pd.DataFrame) -> None:
    for date_str, label, color in EVENEMENTS:
        evt = pd.Timestamp(date_str)
        if pd.Timestamp(df["date"].min()) <= evt <= pd.Timestamp(df["date"].max()):
            fig.add_vline(x=evt.timestamp() * 1000, line_dash="dash", line_color=color, line_width=1.0, opacity=0.7, annotation_text=label, annotation_position="top left", annotation_font_color=color, annotation_textangle=-90)


def fig_marche(df: pd.DataFrame, mm: bool) -> go.Figure:
    col_prix = "Adj Close" if "Adj Close" in df.columns else "Close"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.72, 0.28], subplot_titles=("", "Actualites fraiches par seance"))
    fig.add_trace(go.Scatter(x=df["date"], y=df[col_prix], name="S&P 500", line=dict(color=BLEU, width=2.2), hovertemplate="<b>%{x|%d %b %Y}</b><br>Cours: %{y:,.2f}<extra></extra>"), row=1, col=1)
    if mm and {"MA50", "MA200"}.issubset(df.columns):
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA50"], name="MM50", line=dict(color=OR, width=1.3, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA200"], name="MM200", line=dict(color=VERT, width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df.get("nb_articles_scored", pd.Series(0, index=df.index)), name="News fraîches", marker_color="#6e8efb", opacity=0.72), row=2, col=1)
    ajouter_evenements(fig, df)
    return style_plot(fig, "Vue de marché alignée : prix, tendance et activité news", 560)


def fig_score(df: pd.DataFrame, seuil: float) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=seuil, fillcolor="rgba(217,108,108,.10)", line_width=0)
    fig.add_hrect(y0=seuil, y1=0, fillcolor="rgba(217,179,93,.08)", line_width=0)
    fig.add_hrect(y0=0, y1=1, fillcolor="rgba(89,199,154,.08)", line_width=0)
    fig.add_hline(y=seuil, line_dash="dash", line_color=ROUGE, line_width=1.2, annotation_text=f"Seuil de risque {seuil:.2f}", annotation_position="bottom right", annotation_font_color=ROUGE)
    fig.add_trace(go.Scatter(x=df["date"], y=df["geo_score"], name="Geo-Score", line=dict(color=BLEU, width=2), fill="tozeroy", fillcolor="rgba(108,207,246,.10)", hovertemplate="<b>%{x|%d %b %Y}</b><br>Geo-Score: %{y:+.3f}<extra></extra>"))
    fig.update_yaxes(range=[-1.05, 1.05])
    ajouter_evenements(fig, df)
    return style_plot(fig, "Geo-Score journalier utilisé par la stratégie", 360, "Geo-Score")


def fig_equity(dates: pd.Series, bh_curve: pd.Series, gq_curve: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=bh_curve, name="Buy & Hold", line=dict(color=BLEU, width=2.1)))
    fig.add_trace(go.Scatter(x=dates, y=gq_curve, name="Stratégie GeoQuant", line=dict(color=VERT, width=2.1)))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6d7d91", line_width=.8)
    return style_plot(fig, "Comparaison Buy & Hold vs stratégie GeoQuant", 390, "Capital normalisé")


def fig_signal(df: pd.DataFrame, col_prix: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=df["date"], y=df[col_prix], name="Cours", line=dict(color=BLEU, width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["position"], name="Exposition cible", marker_color=np.where(df["position"] > 0, "rgba(89,199,154,.78)", "rgba(217,108,108,.58)")), row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    ajouter_evenements(fig, df)
    return style_plot(fig, "Signal et exposition simulée", 430)


def fig_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(217,108,108,.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(89,199,154,.08)", line_width=0)
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI_14"], name="RSI 14", line=dict(color=OR, width=1.5)))
    fig.update_yaxes(range=[0, 100])
    return style_plot(fig, "RSI 14", 300)


def fig_drawdown(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Drawdown"] * 100, name="Drawdown", line=dict(color=ROUGE, width=1.5), fill="tozeroy", fillcolor="rgba(217,108,108,.16)"))
    return style_plot(fig, "Drawdown historique", 280, "Drawdown (%)")


def fig_returns(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["Returns"].dropna() * 100, nbinsx=60, name="Rendements (%)", marker_color=BLEU, opacity=.78))
    return style_plot(fig, "Distribution des rendements journaliers", 320, "Nombre d'observations", "Rendement journalier (%)")


def table_news(news_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["date", "headline", "geo_score_article", "label", "score", "source"] if c in news_df.columns]
    out = news_df[cols].copy().rename(columns={"date": "Date", "headline": "Actualite", "geo_score_article": "Score article", "label": "Label FinBERT", "score": "Confiance", "source": "Source"})
    if "Date" in out:
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%d %b %Y")
    if "Score article" in out:
        out["Score article"] = out["Score article"].map(lambda x: f"{x:+.3f}")
    if "Confiance" in out:
        out["Confiance"] = out["Confiance"].map(lambda x: f"{x:.3f}")
    return out


def table_signaux(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    out = df[["date", "geo_score", "geo_score_source", "decision", "position", "decision_reason"]].tail(rows).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%d/%m/%Y")
    out["geo_score"] = out["geo_score"].map(lambda x: f"{x:+.3f}")
    out["position"] = out["position"].map(lambda x: f"{x:.0%}")
    out["geo_score_source"] = out["geo_score_source"].map(lambda x: "news fraîches" if x == "fresh_news" else "score porté")
    return out.rename(columns={"date": "Date", "geo_score": "Geo-Score", "geo_score_source": "Source du score", "decision": "Décision", "position": "Exposition", "decision_reason": "Justification"})


def table_portefeuille(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    out = pd.DataFrame(history).rename(columns={"date": "Date", "action": "Action", "prix": "Prix", "valeur": "Valeur", "pnl_eur": "P&L EUR", "pnl_pct": "P&L %", "geo_score": "Geo-Score", "signal": "Signal"})
    if "Date" in out:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
    for col in ["Prix", "Valeur", "P&L EUR"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    if "Geo-Score" in out:
        out["Geo-Score"] = pd.to_numeric(out["Geo-Score"], errors="coerce").map(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    return out


def table_trades(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log is None or len(trade_log) == 0:
        return pd.DataFrame()
    out = trade_log.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%d/%m/%Y")
    if "Prix" in out.columns:
        out["Prix"] = pd.to_numeric(out["Prix"], errors="coerce").map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    return out


def snapshot_pipeline(df_strategy: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> dict[str, Any]:
    current = df_strategy.iloc[-1]
    fresh = df_strategy[df_strategy.get("has_fresh_news", pd.Series(False, index=df_strategy.index))]
    focus = fresh.iloc[-1] if len(fresh) > 0 else current
    focus_date = pd.Timestamp(focus["date"]).normalize()
    headline, article_score = "", None
    if news_df is not None and len(news_df) > 0:
        same_day = news_df[pd.to_datetime(news_df["date"]).dt.normalize() == focus_date].copy()
        if len(same_day) > 0:
            same_day["abs_score"] = same_day["geo_score_article"].abs()
            top = same_day.sort_values("abs_score", ascending=False).iloc[0]
            headline = str(top.get("headline", ""))
            article_score = float(top["geo_score_article"]) if pd.notna(top.get("geo_score_article")) else None
    return {"current_date": pd.Timestamp(current["date"]), "focus_date": focus_date, "headline": headline, "article_score": article_score, "geo_score": float(current["geo_score"]), "decision": str(current["decision"]), "position": float(current["position"]), "reason": str(current["decision_reason"]), "source": str(current.get("geo_score_source", ""))}


def demo_case(df_strategy: pd.DataFrame, news_df: Optional[pd.DataFrame]) -> dict[str, Any] | None:
    d0, d1 = pd.Timestamp("2022-02-24"), pd.Timestamp("2022-02-25")
    row0 = df_strategy[df_strategy["date"] == d0]
    row1 = df_strategy[df_strategy["date"] == d1]
    if row0.empty or row1.empty:
        return None
    row0, row1 = row0.iloc[0], row1.iloc[0]
    headline, score = None, None
    if news_df is not None and len(news_df) > 0:
        same_day = news_df[pd.to_datetime(news_df["date"]).dt.normalize() == d0]
        if len(same_day) > 0:
            top = same_day.sort_values("geo_score_article", ascending=True).iloc[0]
            headline = str(top["headline"])
            score = float(top["geo_score_article"]) if pd.notna(top.get("geo_score_article")) else None
    window = df_strategy[(df_strategy["date"] >= d1) & (df_strategy["date"] <= d1 + pd.Timedelta(days=30))].copy()
    if len(window) >= 2:
        bh_curve = (1 + window["Returns"].fillna(0.0)).cumprod()
        gq_curve = (1 + window["Returns"].fillna(0.0) * window["position"].shift(1).fillna(0.0)).cumprod()
        bh_dd = float((bh_curve / bh_curve.cummax() - 1).min())
        gq_dd = float((gq_curve / gq_curve.cummax() - 1).min())
    else:
        bh_dd = gq_dd = 0.0
    return {"headline": headline, "article_score": score, "fresh_geo": float(row0.get("geo_score_fresh", row0["geo_score"])), "daily_geo": float(row0["geo_score"]), "decision": str(row1["decision"]), "reason": str(row1["decision_reason"]), "position": float(row1["position"]), "bh_drawdown": bh_dd, "gq_drawdown": gq_dd}


dataset = charger_dataset()
if dataset is None or len(dataset) == 0:
    st.error("Dataset introuvable. Lancez : `python src/news_main.py`, puis `python src/main.py`, puis `python src/run_geo_scorer.py`.")
    st.stop()

news_scored = charger_news_scorees()

dmin, dmax = pd.Timestamp(dataset["date"].min()).date(), pd.Timestamp(dataset["date"].max()).date()

if "plage_dates" not in st.session_state:
    st.session_state["plage_dates"] = (dmin, dmax)
if "seuil_geo" not in st.session_state:
    st.session_state["seuil_geo"] = -0.60
if "costs_bps" not in st.session_state:
    st.session_state["costs_bps"] = 10
if "risk_free_pct" not in st.session_state:
    st.session_state["risk_free_pct"] = 0.0
if "afficher_mm" not in st.session_state:
    st.session_state["afficher_mm"] = True

with st.sidebar:
    st.markdown("## 🧭 Paramètres de simulation")
    st.markdown("")
    plage_dates = st.slider("📅 Période d'analyse", min_value=dmin, max_value=dmax, format="DD/MM/YY", key="plage_dates")
    seuil_geo = st.slider("📊 Seuil Geo-Score", -1.0, 1.0, step=0.05, format="%.2f", help="Sous ce seuil, la stratégie réduit fortement l'exposition.", key="seuil_geo")
    costs_bps = st.slider("💸 Coûts de transaction (bps)", 0, 50, step=5, key="costs_bps")
    risk_free_pct = st.slider("📈 Taux sans risque (%)", 0.0, 6.0, step=0.5, key="risk_free_pct")
    st.divider()
    st.markdown("## ⚙️ Options")
    afficher_mm = st.toggle("Afficher les moyennes mobiles", key="afficher_mm")
    st.divider()
    st.markdown("## 🔍 Transparence")
    st.markdown(
        "**Source des donnees**  \nNews financieres S&P 500, dataset historique local\n\n"
        "**Modele NLP**  \nFinBERT\n\n"
        "**Simulation**  \nPaper trading uniquement\n\n"
        "**Execution**  \nAucun trading reel"
    )
    st.caption(
        "Fichier source : news_sp500_news_2024_processed.csv\n\n"
        "Le score est aligne sur les dates disponibles et reste explicite lorsqu'il est porte d'une seance a l'autre."
    )

mask = (dataset["date"] >= pd.Timestamp(plage_dates[0])) & (dataset["date"] <= pd.Timestamp(plage_dates[1]))
df = dataset.loc[mask].reset_index(drop=True)
df_strategy = Strategy(base_dir=RACINE, seuil_geo=seuil_geo).apply(df)
col_prix = "Adj Close" if "Adj Close" in df_strategy.columns else "Close"
bh, gq = run_backtest(df_strategy, price_col=col_prix, costs_bps=float(costs_bps), risk_free_annual=risk_free_pct / 100.0)
snap = snapshot_pipeline(df_strategy, news_scored)
demo = demo_case(df_strategy, news_scored)

paper_trader = None
paper_portfolio = None
paper_error = None
if PaperTrader is not None:
    try:
        paper_trader = PaperTrader(base_dir=RACINE, seuil_geo=seuil_geo)
        paper_portfolio = paper_trader.load_portfolio()
    except Exception as exc:
        paper_error = str(exc)

st.markdown(
    f"""
<div class="hero">
  <div class="kicker">MVP de simulation geopolitique et financiere</div>
  <div class="title">GeoQuant AI</div>
  <div class="subtitle">Transformer les news geopolitiques en signal financier quantifie, puis en decision de strategie simulée. Période affichée : <b>{fmt_date(df_strategy["date"].min())}</b> à <b>{fmt_date(df_strategy["date"].max())}</b>.</div>
  <div class="pipe">
    <span class="step">News</span><span class="step">NLP / FinBERT</span><span class="step">Geo-Score</span><span class="step">Décision</span><span class="step">Backtest</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_market, tab_nlp, tab_backtest, tab_paper, tab_tech, tab_about = st.tabs(["Vue de marché", "Geo-Score & NLP", "Backtest", "Paper trading", "Analyse technique", "À propos"])

with tab_market:
    section("Vue de marche", "Lecture simple du contexte de marche, des moyennes mobiles et de l'activite des actualites sur la periode selectionnee.")
    st.markdown('<div class="market-wrap"></div>', unsafe_allow_html=True)
    st.plotly_chart(fig_marche(df_strategy, afficher_mm), width="stretch", key="market_context_chart")
    st.markdown('<div class="market-wrap"></div>', unsafe_allow_html=True)

with tab_nlp:
    section("Geo-Score & NLP", "Meme source d'actualites, meme scoring FinBERT, meme signal ensuite consomme par la strategie.")
    st.markdown('<div class="callout">Cet onglet affiche les actualites reellement utilisees pour calculer le Geo-Score. Il n\'y a plus de source de news separee entre la demo, le score et le backtest.</div>', unsafe_allow_html=True)
    source_label = "actualites fraiches" if snap["source"] == "fresh_news" else "score porte"
    impact_label = f"{(gq.max_drawdown - bh.max_drawdown):+.2%}"
    p1, p2, p3, p4 = st.columns([1.9, 1, 1, 1], gap="large")
    with p1:
        st.markdown(
            f'<div class="box focus-card focus-news"><div class="label">Actualite suivie</div><div class="headline">{snap["headline"] or "Aucune actualite fraiche exploitable sur la derniere seance affichee."}</div><div class="note">Date de reference : <b>{fmt_date(snap["focus_date"])}</b><br>Source du score : <b>{source_label}</b><br>Score article dominant : <b>{fmt_signed(snap["article_score"])}</b></div></div>',
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            focus_card("Geo-Score actuel", fmt_signed(snap["geo_score"]), "Signal quantifie utilise par la strategie.", "focus-score"),
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            focus_card("Decision simulee", snap["decision"], snap["reason"], "focus-decision"),
            unsafe_allow_html=True,
        )
    with p4:
        st.markdown(
            focus_card("Impact", f"{snap['position']:.0%}", f"Exposition cible | drawdown GeoQuant vs B&H : {impact_label}", "focus-impact"),
            unsafe_allow_html=True,
        )
    score_actuel = float(df_strategy["geo_score"].iloc[-1])
    label = "Stress" if score_actuel < seuil_geo else "Prudence" if score_actuel < 0 else "Confiance"
    color = ROUGE if score_actuel < seuil_geo else OR if score_actuel < 0 else VERT
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=score_actuel, domain=dict(x=[0, 1], y=[0, 1]), title=dict(text=f"Geo-Score du jour<br><span style='color:{color}'>{label}</span>"), number=dict(font=dict(size=34, color=color)), delta=dict(reference=0), gauge=dict(axis=dict(range=[-1, 1]), bar=dict(color=color), bgcolor="#16263b", steps=[dict(range=[-1, seuil_geo], color="rgba(217,108,108,.18)"), dict(range=[seuil_geo, 0], color="rgba(217,179,93,.13)"), dict(range=[0, 1], color="rgba(89,199,154,.13)")]))); gauge.update_layout(paper_bgcolor=FOND, font=dict(color=TEXTE), height=290, margin=dict(t=32, b=10, l=18, r=18)); st.plotly_chart(gauge, width="stretch", key="nlp_gauge")
    with c2:
        st.plotly_chart(fig_score(df_strategy, seuil_geo), width="stretch", key="nlp_geo_score_timeline")
    section("Actualites reellement utilisees", "Extraction article par article avec leur score FinBERT et leur score Geo-Score individuel.")
    if news_scored is not None and len(news_scored) > 0:
        news_mask = (news_scored["date"] >= pd.Timestamp(plage_dates[0])) & (news_scored["date"] <= pd.Timestamp(plage_dates[1]))
        st.dataframe(table_news(news_scored.loc[news_mask]).head(20), width="stretch", hide_index=True)
    else:
        st.markdown('<div class="callout warn">`news_scored.csv` est absent. Relancez `python src/run_geo_scorer.py` pour reconstruire la table des actualites scorees.</div>', unsafe_allow_html=True)
    if demo is not None:
        section("Exemple concret", "Lecture simple d'un cas réel de la chaine : actualite -> score -> decision -> impact.")
        st.markdown(
            f'<div class="callout good"><b>24 fevrier 2022 - choc Ukraine</b><br>'
            f'Une actualite geopolitique est scoree negativement. Le Geo-Score du jour devient <b>{fmt_signed(demo["fresh_geo"])}</b>, '
            f'puis la strategie applique le lendemain la decision <b>{demo["decision"]}</b> avec une exposition cible de <b>{demo["position"]:.0%}</b>. '
            f'Sur les 30 jours suivants, le drawdown passe de <b>{fmt_pct(demo["bh_drawdown"])}</b> en Buy & Hold a <b>{fmt_pct(demo["gq_drawdown"])}</b> pour GeoQuant.</div>',
            unsafe_allow_html=True,
        )
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(card("1. Actualite", "24/02/2022", demo["headline"] or "Actualite non disponible dans la selection."), unsafe_allow_html=True)
        with d2:
            st.markdown(card("2. Score article", fmt_signed(demo["article_score"]), "Score individuel de l'actualite cle."), unsafe_allow_html=True)
        with d3:
            st.markdown(card("3. Geo-Score", fmt_signed(demo["daily_geo"]), "Score journalier agrégé utilise par la strategie."), unsafe_allow_html=True)
        with d4:
            st.markdown(card("4. Decision", demo["decision"], f"Exposition cible : {demo['position']:.0%}"), unsafe_allow_html=True)
        st.markdown(
            f'<div class="box"><div class="label">Lecture du cas</div>'
            f'<div class="note">'
            f'<b>Actualite :</b> {demo["headline"] or "Actualite non disponible dans la selection courante."}<br>'
            f'<b>Signal produit :</b> score article {fmt_signed(demo["article_score"])} puis Geo-Score journalier {fmt_signed(demo["fresh_geo"])}<br>'
            f'<b>Reaction de la strategie :</b> {demo["decision"]} ({demo["reason"]})<br>'
            f'<b>Impact observe :</b> drawdown Buy & Hold {fmt_pct(demo["bh_drawdown"])} vs GeoQuant {fmt_pct(demo["gq_drawdown"])}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

with tab_backtest:
    section("Backtest comparatif", "Le backtest est calculé sur le dataset unifié et compare strictement Buy & Hold à la stratégie GeoQuant.")
    st.markdown('<div class="callout">Les métriques ci-dessous sont calculées sur la même période alignée entre prix, Geo-Score et stratégie. Aucun résultat n\'est décoratif : les chiffres viennent du moteur de backtest.</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(card("Performance GeoQuant", f"{gq.total_return:.2%}", "Rendement total de la stratégie."), unsafe_allow_html=True)
    with t2:
        st.markdown(card("Écart de drawdown", f"{(gq.max_drawdown - bh.max_drawdown):+.2%}", "GeoQuant vs Buy & Hold."), unsafe_allow_html=True)
    with t3:
        st.markdown(card("Nombre de trades", str(gq.nb_trades), "Trades aller-retour détectés."), unsafe_allow_html=True)
    with t4:
        st.markdown(card("Exposition moyenne", f"{float(df_strategy['position'].mean()):.0%}", "Part moyenne du capital engagée."), unsafe_allow_html=True)
    metrics = [("Rendement total", bh.total_return, gq.total_return, ".2%", True), ("Rendement annualisé", bh.annualised, gq.annualised, ".2%", True), ("Max drawdown", bh.max_drawdown, gq.max_drawdown, ".2%", False), ("Sharpe ratio", bh.sharpe, gq.sharpe, ".2f", True), ("Win rate", bh.win_rate, gq.win_rate, ".1%", True), ("Nombre de trades", float(bh.nb_trades), float(gq.nb_trades), ".0f", False)]
    cols = st.columns(len(metrics))
    for i, (label, bh_val, gq_val, fmt, higher_better) in enumerate(metrics):
        cols[i].metric(label, f"{gq_val:{fmt}}", delta=f"vs B&H {gq_val - bh_val:+{fmt}}", delta_color="normal" if higher_better else "inverse")
    st.plotly_chart(fig_equity(df_strategy["date"], bh.equity_curve, gq.equity_curve), width="stretch", key="backtest_equity_comparison")
    st.plotly_chart(fig_signal(df_strategy, col_prix), width="stretch", key="backtest_signal_exposure")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.dataframe(pd.DataFrame({"Indicateur": [m[0] for m in metrics], "Buy & Hold": [f"{m[1]:{m[3]}}" for m in metrics], "GeoQuant": [f"{m[2]:{m[3]}}" for m in metrics]}), width="stretch", hide_index=True)
    with right:
        if len(gq.trade_log) > 0:
            st.dataframe(table_trades(gq.trade_log), width="stretch", hide_index=True)
        else:
            st.markdown('<div class="callout warn">Aucun trade généré sur la période sélectionnée.</div>', unsafe_allow_html=True)

with tab_paper:
    section("Paper trading simule", "Un poste de pilotage sans broker ni execution reelle, pour suivre la strategie dans un cadre de demonstration.")
    st.markdown('<div class="callout warn"><b>Paper trading uniquement :</b> aucune exécution réelle, aucune connexion bancaire, aucune recommandation d\'investissement. Cette section sert à illustrer le suivi simulé de la stratégie.</div>', unsafe_allow_html=True)
    latest = df_strategy.iloc[-1]
    headlines = ""
    if news_scored is not None and len(news_scored) > 0:
        same_day = news_scored[pd.to_datetime(news_scored["date"]).dt.normalize() == snap["focus_date"]]
        if len(same_day) > 0:
            headlines = " | ".join(same_day.sort_values("geo_score_article").head(3)["headline"].astype(str).tolist())
    p1, p2 = st.columns([1.35, 1], gap="large")
    with p1:
        source_label = "actualites fraiches" if latest.get("geo_score_source", "") == "fresh_news" else "score porte"
        st.markdown(f'<div class="box"><div class="label">Signal simulé actuel</div><div class="headline">{badge(str(latest["decision"]))} &nbsp; <span class="badge wait">{source_label}</span></div><div class="note">Dernière séance visible : <b>{fmt_date(latest["date"])}</b><br>Geo-Score appliqué : <b>{fmt_signed(float(latest["geo_score"]))}</b><br>Exposition cible : <b>{float(latest["position"]):.0%}</b><br>Justification : <b>{latest["decision_reason"]}</b></div></div>', unsafe_allow_html=True)
        if headlines:
            st.markdown(f'<div class="callout"><b>Dernieres actualites prises en compte</b><br>{headlines}</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(card("Signal", str(latest["decision"]), "Décision simulée issue de la stratégie."), unsafe_allow_html=True)
        with s2:
            st.markdown(card("Exposition", f"{float(latest['position']):.0%}", "Niveau d'engagement cible."), unsafe_allow_html=True)
        with s3:
            st.markdown(card("Dernière mise à jour", fmt_date(latest["date"]), "Dernière date du dataset."), unsafe_allow_html=True)
    with p2:
        if paper_error is not None:
            st.markdown(f'<div class="callout warn">Impossible de charger le portefeuille virtuel : <code>{paper_error}</code></div>', unsafe_allow_html=True)
        elif paper_portfolio is not None:
            win_rate = paper_portfolio.trades_gagnants / paper_portfolio.nb_trades if paper_portfolio.nb_trades else 0.0
            q1, q2 = st.columns(2)
            q3, q4 = st.columns(2)
            with q1:
                st.markdown(card("Capital total", f"{paper_portfolio.valeur_totale:,.2f} EUR", "Simulation persistée localement."), unsafe_allow_html=True)
            with q2:
                st.markdown(card("Cash disponible", f"{paper_portfolio.capital_cash:,.2f} EUR", "Part non investie."), unsafe_allow_html=True)
            with q3:
                st.markdown(card("Capital investi", f"{paper_portfolio.capital_investi:,.2f} EUR", "Valeur de marché virtuelle."), unsafe_allow_html=True)
            with q4:
                st.markdown(card("Trades clos", str(paper_portfolio.nb_trades), f"Win rate virtuel : {win_rate:.0%}"), unsafe_allow_html=True)
        else:
            st.markdown('<div class="callout">Le module de paper trading n\'a pas fourni de portefeuille exploitable dans cette session.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1.1], gap="large")
    with left:
        st.dataframe(table_signaux(df_strategy, 10), width="stretch", hide_index=True)
    with right:
        if paper_portfolio is not None and paper_portfolio.historique:
            st.dataframe(table_portefeuille(paper_portfolio.historique[-8:]), width="stretch", hide_index=True)
        else:
            st.markdown('<div class="callout">Le portefeuille virtuel ne contient pas encore d\'historique exploitable. La section reste cohérente, mais encore partielle.</div>', unsafe_allow_html=True)
    if paper_trader is not None:
        with st.expander("Simulation live optionnelle"):
            use_live_finbert = st.toggle("Actualiser la simulation live avec FinBERT", value=False, help="Desactive par defaut pour garder une actualisation legere.")
            if st.button("Actualiser le signal live", type="primary"):
                with st.spinner("Actualisation du signal live simule..."):
                    try:
                        snap_live = paper_trader.get_live_snapshot(use_finbert=use_live_finbert)
                        st.session_state["paper_live"] = dict(snap_live.__dict__)
                        st.session_state.pop("paper_live_error", None)
                    except Exception as exc:
                        st.session_state["paper_live_error"] = str(exc)
            if st.session_state.get("paper_live_error"):
                st.markdown(f'<div class="callout warn">Impossible d\'actualiser la simulation live dans cet environnement : <code>{st.session_state["paper_live_error"]}</code></div>', unsafe_allow_html=True)
            if st.session_state.get("paper_live"):
                live = st.session_state["paper_live"]
                l1, l2, l3, l4 = st.columns(4)
                with l1:
                    st.markdown(card("Signal live", "BUY" if live["signal"] == 1 else "CASH", "Module `paper_trading.py`"), unsafe_allow_html=True)
                with l2:
                    st.markdown(card("Exposition live", f'{float(live["position"]):.0%}', "Exposition virtuelle instantanee."), unsafe_allow_html=True)
                with l3:
                    st.markdown(card("Geo-Score live", fmt_signed(float(live["geo_score"])), f'{int(live["nb_news"])} actualites live'), unsafe_allow_html=True)
                with l4:
                    st.markdown(card("Derniere actualisation", str(live["timestamp"]), "Aucune connexion broker."), unsafe_allow_html=True)
                if live.get("headlines_with_scores"):
                    live_df = pd.DataFrame(live["headlines_with_scores"], columns=["Actualite", "Score article"])
                    live_df["Score article"] = live_df["Score article"].map(lambda x: f"{float(x):+.3f}")
                    st.dataframe(live_df.head(8), width="stretch", hide_index=True)
                elif live.get("headlines"):
                    st.dataframe(pd.DataFrame({"Actualite": live["headlines"][:8]}), width="stretch", hide_index=True)

with tab_tech:
    section("Analyse technique", "Compléments de lecture pour relier régime de marché, drawdown et distribution des rendements.")
    if {"RSI_14", "Drawdown", "Returns"}.issubset(df_strategy.columns):
        a, b = st.columns(2, gap="large")
        with a:
            st.plotly_chart(fig_rsi(df_strategy), width="stretch", key="tech_rsi")
        with b:
            st.plotly_chart(fig_drawdown(df_strategy), width="stretch", key="tech_drawdown")
        st.plotly_chart(fig_returns(df_strategy), width="stretch", key="tech_return_distribution")
    else:
        st.markdown('<div class="callout warn">Certaines colonnes techniques sont absentes du dataset affiché.</div>', unsafe_allow_html=True)

with tab_about:
    section("Ce que montre réellement l'application", "Résumé honnête de la chaîne, du niveau de simulation et des fichiers reliés au dashboard.")
    st.markdown(f'<div class="box"><div class="label">Vue du pipeline</div><div class="note">Une seule source de news historique est utilisée du début à la fin : <code>news_sp500_news_2024_processed.csv</code>.<br>FinBERT score chaque actualite et sauvegarde les resultats article par article dans <code>news_scored.csv</code>.<br>Un Geo-Score quotidien est ensuite calcule, aligne sur le calendrier de marche, puis injecte dans <code>dataset_final.csv</code>.<br>La strategie GeoQuant consomme ce meme score pour ajuster l\'exposition et le backtest compare Buy & Hold a GeoQuant.<br>Le paper trading reste une simulation independante, sans execution reelle.</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="callout">Prototype de simulation uniquement : aucun trading reel, aucun conseil financier.<br>Le signal est decale d\'une seance par rapport a la news pour eviter le look-ahead.<br>Quand une seance n\'apporte pas de nouvelle actualite, le dernier Geo-Score connu est porte explicitement.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="callout good"><b>Fichiers relies au dashboard</b><br>Dataset aligne : <code>{CHEMIN_DATASET}</code><br>Actualites scorees : <code>{CHEMIN_NEWS}</code><br>Geo-Scores journaliers : <code>{CHEMIN_GEO}</code><br>Strategie : <code>src/strategy.py</code><br>Backtest : <code>src/backtest.py</code></div>', unsafe_allow_html=True)
