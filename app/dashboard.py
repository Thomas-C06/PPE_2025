"""GeoQuant AI -- Tableau de bord interactif (Bloc 4 -- Streamlit).

Lancement depuis la racine du projet :
    /c/Users/mathi/anaconda3/python.exe -m streamlit run app/dashboard.py

Ce fichier constitue l'interface principale du projet GeoQuant AI.
Il lit uniquement le dataset S&P 500 (dataset_final.csv) produit par le Bloc 1.
"""

from __future__ import annotations

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

    # Seuil Geo-Score (utilise par le Bloc 3)
    st.divider()
    st.markdown("**Seuil Geo-Score** (Bloc 3)")
    seuil_geo = st.slider(
        "Si Geo-Score < seuil -> Cash",
        min_value=-1.0, max_value=0.0, value=-0.5, step=0.05, format="%.2f",
        help="Seuil en dessous duquel la strategie Bloc 3 coupe la position.",
    )
    st.caption(f"Seuil actuel : **{seuil_geo:.2f}**")

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
    st.markdown('<span class="badge-attente">BLOC 2</span> NLP / FinBERT',  unsafe_allow_html=True)
    st.markdown('<span class="badge-attente">BLOC 3</span> Strategie',      unsafe_allow_html=True)
    st.markdown('<span class="badge-encours">BLOC 4</span> Dashboard',      unsafe_allow_html=True)


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
onglet_marche, onglet_nlp, onglet_backtest, onglet_tech, onglet_apropos = st.tabs([
    "📊 Vue Marche",
    "🧠 Sentiment & NLP",
    "⚔️ Backtest",
    "📈 Analyse Technique",
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
        score_actuel = float(df["geo_score"].dropna().iloc[-1])
        st.subheader("Geo-Score -- Sentiment du marche")

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
    st.markdown("""
<div class="bloc-placeholder">
    <h3>⚔️ Bloc 3 -- Moteur de Strategie en attente</h3>
    <p>Le backtesting comparatif <b>Buy &amp; Hold vs GeoQuant</b>
    sera implemente ici.</p>
    <br>
    <p style="color:#666; font-size:0.8rem;">
        Regle prevue : SI (MA50 > MA200) ET (Geo-Score > seuil) ALORS Long SINON Cash
    </p>
</div>""", unsafe_allow_html=True)

    st.divider()
    col_g, col_d = st.columns(2)

    with col_g:
        st.markdown("**Graphique comparatif *(a venir)***")
        fig_vide = go.Figure()
        fig_vide.add_annotation(
            text="Disponible apres Bloc 3",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16, color="#555"),
        )
        fig_vide.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0e1117", paper_bgcolor="#1e1e2e",
            height=280, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_vide, use_container_width=True)

    with col_d:
        st.markdown("**Metriques de performance *(a venir)***")
        st.dataframe(pd.DataFrame({
            "Metrique":   ["Rendement total", "Max Drawdown", "Sharpe Ratio", "Win Rate"],
            "Buy & Hold": ["?", "?", "?", "?"],
            "GeoQuant":   ["?", "?", "?", "?"],
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Journal des trades virtuels *(a venir)***")
    st.dataframe(pd.DataFrame({
        "Date":   ["--", "--", "--"],
        "Signal": ["--", "--", "--"],
        "Prix":   ["--", "--", "--"],
        "P&L":    ["--", "--", "--"],
    }), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# ONGLET 4 -- Analyse Technique
# ---------------------------------------------------------------------------
with onglet_tech:
    colonnes_requises = ["RSI_14", "Drawdown", "Volatility_20", "Returns"]
    manquantes = [c for c in colonnes_requises if c not in df.columns]

    if manquantes:
        st.warning(f"Colonnes manquantes : {manquantes}. Relancez `python src/main.py`.")
    else:
        st.plotly_chart(construire_graphique_rsi(df), use_container_width=True)
        st.divider()

        col_dd, col_vol = st.columns(2)
        with col_dd:
            st.plotly_chart(construire_graphique_drawdown(df),   use_container_width=True)
        with col_vol:
            st.plotly_chart(construire_graphique_volatilite(df), use_container_width=True)

        st.divider()
        st.plotly_chart(construire_distribution_rendements(df),  use_container_width=True)

        # Statistiques descriptives
        st.divider()
        st.subheader("Statistiques descriptives")
        rend_pct = df["Returns"].dropna() * 100
        st.dataframe(pd.DataFrame({
            "Metrique": [
                "Rendement moyen / jour", "Ecart-type",
                "Skewness", "Kurtosis",
                "VaR 5%", "CVaR 5%",
                "Pire journee", "Meilleure journee",
            ],
            "Valeur": [
                f"{rend_pct.mean():.4f}%",
                f"{rend_pct.std():.4f}%",
                f"{rend_pct.skew():.3f}",
                f"{rend_pct.kurtosis():.3f}",
                f"{rend_pct.quantile(0.05):.4f}%",
                f"{rend_pct[rend_pct <= rend_pct.quantile(0.05)].mean():.4f}%",
                f"{rend_pct.min():.4f}%",
                f"{rend_pct.max():.4f}%",
            ],
        }), use_container_width=False, hide_index=True, width=420)


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
| **2 -- NLP / FinBERT** | `nlp/` (a creer) | ⏳ A venir |
| **3 -- Strategie** | `strategy/` (a creer) | ⏳ A venir |
| **4 -- Dashboard** | `app/dashboard.py` | 🔄 En cours |
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
