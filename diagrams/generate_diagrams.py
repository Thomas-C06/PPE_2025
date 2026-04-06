"""
Diagrammes d'architecture GeoQuant AI - version claire et propre.
Regle de base : ZERO fleche diagonale, espacement genereux, labels non chevauchants.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT   = Path(__file__).parent
BG    = "#0e1117"
PANEL = "#161b22"
C_B1  = "#1f77b4"
C_B2  = "#2ca02c"
C_B3  = "#ff7f0e"
C_B4  = "#d62728"
C_EXT = "#9467bd"
TEXT  = "#e8e8e8"
MUTED = "#888888"
GRAY  = "#444444"


# ── helpers de base ──────────────────────────────────────────────────────────

def new_fig(w, h, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=14)
    return fig, ax


def card(ax, cx, cy, w, h, title, subtitle="",
         color=C_B1, title_fs=9, sub_fs=7.5):
    """Carte avec bande header coloree + corps sombre."""
    lx, by = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((lx, by), w, h,
                 boxstyle="round,pad=0,rounding_size=0.015",
                 facecolor=PANEL, edgecolor=color, linewidth=1.8))
    hh = h * 0.40
    ax.add_patch(FancyBboxPatch((lx, cy + h / 2 - hh), w, hh,
                 boxstyle="round,pad=0,rounding_size=0.015",
                 facecolor=color, alpha=0.48, edgecolor="none"))
    ax.text(cx, cy + h / 2 - hh / 2, title,
            ha="center", va="center", color="white",
            fontsize=title_fs, fontweight="bold")
    if subtitle:
        ax.text(cx, cy - h * 0.12, subtitle,
                ha="center", va="center", color=MUTED,
                fontsize=sub_fs, style="italic", family="monospace",
                multialignment="center")


def note_box(ax, cx, cy, w, h, text, color=GRAY, fs=8):
    """Boite simple sans header (pour fichiers CSV, notes)."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.012",
                 facecolor=color, alpha=0.15,
                 edgecolor=color, linewidth=1.2, linestyle="--"))
    ax.text(cx, cy, text, ha="center", va="center", color=MUTED,
            fontsize=fs, multialignment="center")


def diamond(ax, cx, cy, w, h, text, color=C_B1):
    pts = np.array([[cx, cy + h/2], [cx + w/2, cy],
                    [cx, cy - h/2], [cx - w/2, cy]])
    ax.add_patch(plt.Polygon(pts, facecolor=color, alpha=0.18,
                             edgecolor=color, linewidth=1.8))
    ax.text(cx, cy, text, ha="center", va="center", color=TEXT,
            fontsize=8.5, fontweight="bold", multialignment="center")


def h_arrow(ax, x1, x2, y, color=GRAY, label="", above=True):
    """Fleche horizontale stricte."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=14))
    if label:
        mx = (x1 + x2) / 2
        dy = 0.021 if above else -0.021
        va = "bottom" if above else "top"
        ax.text(mx, y + dy, label, ha="center", va=va,
                color=TEXT, fontsize=7.5, fontweight="bold",
                bbox=dict(facecolor=BG, edgecolor="none", pad=2))


def v_arrow(ax, x, y1, y2, color=GRAY, label="", right=True):
    """Fleche verticale stricte."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=13))
    if label:
        my = (y1 + y2) / 2
        dx = 0.015 if right else -0.015
        ha = "left" if right else "right"
        ax.text(x + dx, my, label, ha=ha, va="center",
                color=TEXT, fontsize=7.5, fontweight="bold",
                bbox=dict(facecolor=BG, edgecolor="none", pad=2))


def elbow_arrow(ax, x1, y1, x2, y2, color=GRAY):
    """Fleche en L (horizontal puis vertical) - 0 diagonal."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                connectionstyle="angle,angleA=0,angleB=90",
                                mutation_scale=13))


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMME 1  –  Vue globale (pipeline en colonnes, 0 croisement)
# ═════════════════════════════════════════════════════════════════════════════
def diag1():
    fig, ax = new_fig(16, 8.5, "GeoQuant AI  —  Architecture en 4 blocs")

    # ── Mise en page : 5 colonnes ────────────────────────────────────────────
    #   Col 0 : Sources historiques   (x=0.09)
    #   Col 1 : Bloc 1               (x=0.28)
    #   Col 2 : Bloc 2               (x=0.50)
    #   Col 3 : Bloc 3               (x=0.72)
    #   Col 4 : Bloc 4               (x=0.91)
    #   Col -1 : Sources live (bas)  independantes

    CX = [0.09, 0.28, 0.50, 0.72, 0.91]
    CW, CH = 0.155, 0.092          # largeur / hauteur d'une carte

    # ── Titres de colonnes ───────────────────────────────────────────────────
    for x, lbl, col in [
        (CX[0], "Sources\nhistoriques", C_EXT),
        (CX[1], "BLOC 1\nData Engineering", C_B1),
        (CX[2], "BLOC 2\nNLP & Geo-Score",  C_B2),
        (CX[3], "BLOC 3\nStrategie",         C_B3),
        (CX[4], "BLOC 4\nDashboard",         C_B4),
    ]:
        ax.add_patch(FancyBboxPatch((x - CW/2, 0.88), CW, 0.09,
                     boxstyle="round,pad=0,rounding_size=0.015",
                     facecolor=col, alpha=0.25, edgecolor=col, linewidth=1.5))
        ax.text(x, 0.925, lbl, ha="center", va="center",
                color=col, fontsize=8.5, fontweight="bold",
                multialignment="center")

    # ── Sources historiques (col 0) ──────────────────────────────────────────
    # 2 sources principales sur la meme ligne que les modules Bloc 1 qu'elles alimentent
    card(ax, CX[0], 0.74, CW, CH, "Yahoo Finance", "OHLCV  2022-2024", C_EXT)
    card(ax, CX[0], 0.58, CW, CH, "CSV Kaggle",    "S&P 500 News",     C_EXT)

    # ── Bloc 1 ───────────────────────────────────────────────────────────────
    card(ax, CX[1], 0.74, CW, CH, "PriceLoader",    "loader.py",        C_B1)
    card(ax, CX[1], 0.58, CW, CH, "NewsLoader",     "news_loader.py",   C_B1)
    card(ax, CX[1], 0.42, CW, CH, "DataMerger",     "MA50 · MA200 · RSI", C_B1, sub_fs=7)
    card(ax, CX[1], 0.26, CW, CH*0.85,
         "DataVisualizer", "4 graphiques PNG", C_B1, title_fs=8, sub_fs=7)

    # ── Bloc 2 ───────────────────────────────────────────────────────────────
    card(ax, CX[2], 0.74, CW, CH,
         "FinBertSentiment", "ProsusAI/finbert\np_pos · p_neg", C_B2, title_fs=8, sub_fs=7)
    card(ax, CX[2], 0.58, CW, CH, "SentimentCache", "JSON cache",       C_B2)
    card(ax, CX[2], 0.42, CW, CH, "GeoScorer",
         "geo = p_pos - p_neg\nrolling(3)", C_B2, sub_fs=7)

    # ── Bloc 3 ───────────────────────────────────────────────────────────────
    card(ax, CX[3], 0.72, CW, CH, "Strategy",
         "Golden Cross\n+ Geo-filter", C_B3, sub_fs=7)
    card(ax, CX[3], 0.54, CW, CH, "BacktestEngine",
         "Buy&Hold vs GeoQuant", C_B3, title_fs=8, sub_fs=7)
    card(ax, CX[3], 0.37, CW, CH*0.85, "BacktestResult",
         "CAGR · Sharpe · MaxDD\nWin Rate", C_B3, title_fs=8, sub_fs=7)

    # ── Bloc 4 ───────────────────────────────────────────────────────────────
    card(ax, CX[4], 0.72, CW, CH, "Dashboard",
         "Streamlit  5 onglets", C_B4, title_fs=8, sub_fs=7)
    card(ax, CX[4], 0.54, CW, CH, "PaperTrader",
         "simulation live", C_B4, title_fs=8)
    card(ax, CX[4], 0.37, CW, CH, "AlertManager",
         "st.toast + email", C_B4, title_fs=8, sub_fs=7)

    # ── Sources live (bas, independantes) ────────────────────────────────────
    card(ax, 0.26, 0.10, CW, CH*0.8, "Yahoo Finance", "cours live", C_EXT, title_fs=8, sub_fs=7)
    card(ax, 0.44, 0.10, CW, CH*0.8, "Yahoo RSS",     "news live",  C_EXT, title_fs=8, sub_fs=7)
    ax.text(0.35, 0.02, "Sources live (Paper Trading)",
            ha="center", va="bottom", color=C_EXT, fontsize=8)

    # ── FLECHES horizontales Sources hist → Bloc 1 ──────────────────────────
    h_arrow(ax, CX[0]+CW/2, CX[1]-CW/2, 0.74, C_EXT)
    h_arrow(ax, CX[0]+CW/2, CX[1]-CW/2, 0.58, C_EXT)

    # ── FLECHES verticales internes Bloc 1 ──────────────────────────────────
    v_arrow(ax, CX[1], 0.74-CH/2, 0.58+CH/2, C_B1)   # PriceLoader → DataMerger
    v_arrow(ax, CX[1], 0.58-CH/2, 0.42+CH/2, C_B1)   # NewsLoader  → DataMerger

    # ── FLECHE Bloc 1 → Bloc 2 ───────────────────────────────────────────────
    h_arrow(ax, CX[1]+CW/2, CX[2]-CW/2, 0.42, C_B1, "dataset_final.csv")

    # ── FLECHES verticales internes Bloc 2 ──────────────────────────────────
    v_arrow(ax, CX[2], 0.74-CH/2, 0.58+CH/2, C_B2)
    v_arrow(ax, CX[2], 0.58-CH/2, 0.42+CH/2, C_B2)

    # ── FLECHE Bloc 2 → Bloc 3 ───────────────────────────────────────────────
    h_arrow(ax, CX[2]+CW/2, CX[3]-CW/2, 0.54, C_B2, "geo_scores.csv")

    # ── FLECHES verticales internes Bloc 3 ──────────────────────────────────
    v_arrow(ax, CX[3], 0.72-CH/2, 0.54+CH/2, C_B3)
    v_arrow(ax, CX[3], 0.54-CH/2, 0.37+CH*0.42, C_B3)

    # ── FLECHE Bloc 3 → Bloc 4 ───────────────────────────────────────────────
    h_arrow(ax, CX[3]+CW/2, CX[4]-CW/2, 0.72, C_B3, "BacktestResult")

    # ── FLECHES sources live → PaperTrader (elbow propre) ────────────────────
    # Horizontale au niveau y=0.10, puis montee vers PaperTrader
    ax.annotate("", xy=(CX[4]-CW/2, 0.54), xytext=(0.44+CW/2, 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_EXT, lw=1.4,
                                connectionstyle="angle,angleA=0,angleB=-90",
                                mutation_scale=13))
    ax.annotate("", xy=(0.26+CW/2, 0.10), xytext=(0.44-CW/2, 0.10),
                arrowprops=dict(arrowstyle="-", color=C_EXT, lw=1.2))
    ax.text(0.58, 0.10, "flux live", ha="left", va="center",
            color=C_EXT, fontsize=7.5, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "diag1_vue_globale.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  OK  {p.name}")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMME 2  –  Flux de données (waterfall vertical, 0 croisement)
# ═════════════════════════════════════════════════════════════════════════════
def diag2():
    fig, ax = new_fig(12, 10, "Flux de donnees  —  GeoQuant AI")

    # Disposition : 4 blocs empiles verticalement (haut → bas)
    # Chaque bloc = bande horizontale coloree + modules a l'interieur
    # Les fichiers CSV intermediaires sont montres entre les blocs

    BW = 0.88    # largeur bande de bloc
    BX = 0.50    # centre x
    CARD_W = 0.18
    CARD_H = 0.08

    blocs = [
        # (y_centre_bande, hauteur_bande, couleur, label_gauche, y_modules)
        (0.875, 0.11, C_B1, "BLOC 1\nData Engineering",   []),
        (0.640, 0.11, C_B2, "BLOC 2\nNLP & Geo-Score",    []),
        (0.420, 0.11, C_B3, "BLOC 3\nStrategie & Backtest",[]),
        (0.200, 0.11, C_B4, "BLOC 4\nDashboard",          []),
    ]

    for yc, bh, col, lbl, _ in blocs:
        # Bande de fond
        ax.add_patch(FancyBboxPatch((BX - BW/2, yc - bh/2), BW, bh,
                     boxstyle="round,pad=0,rounding_size=0.02",
                     facecolor=col, alpha=0.07,
                     edgecolor=col, linewidth=1.5, linestyle="--"))
        # Etiquette a gauche
        ax.text(BX - BW/2 - 0.01, yc, lbl, ha="right", va="center",
                color=col, fontsize=9, fontweight="bold",
                multialignment="center")

    # ── BLOC 1 : modules ─────────────────────────────────────────────────────
    y1 = 0.875
    card(ax, 0.20, y1, CARD_W, CARD_H, "PriceLoader",
         "yfinance OHLCV", C_B1, title_fs=8, sub_fs=7)
    card(ax, 0.41, y1, CARD_W, CARD_H, "NewsLoader",
         "CSV Kaggle", C_B1, title_fs=8, sub_fs=7)
    card(ax, 0.62, y1, CARD_W, CARD_H, "DataMerger",
         "MA50 · MA200 · RSI", C_B1, title_fs=8, sub_fs=7)
    card(ax, 0.83, y1, CARD_W*0.85, CARD_H*0.85, "DataVisualizer",
         "4 PNG charts", C_B1, title_fs=8, sub_fs=7)

    # ── Fichier intermediaire B1 → B2 ────────────────────────────────────────
    note_box(ax, 0.45, 0.762, 0.28, 0.056,
             "dataset_final.csv  (prix + news + indicateurs)", GRAY, fs=7.5)
    v_arrow(ax, 0.45, 0.820, 0.790, C_B1)
    v_arrow(ax, 0.45, 0.735, 0.695, C_B2)

    # ── BLOC 2 : modules ─────────────────────────────────────────────────────
    y2 = 0.640
    card(ax, 0.20, y2, CARD_W, CARD_H, "FinBertSentiment",
         "ProsusAI/finbert\np_pos - p_neg", C_B2, title_fs=8, sub_fs=7)
    card(ax, 0.41, y2, CARD_W, CARD_H, "SentimentCache",
         "JSON  (evite recalcul)", C_B2, title_fs=8, sub_fs=7)
    card(ax, 0.62, y2, CARD_W, CARD_H, "GeoScorer",
         "geo = p_pos - p_neg\nrolling(3)", C_B2, title_fs=8, sub_fs=7)

    # Fleche interne B2
    h_arrow(ax, 0.20+CARD_W/2, 0.41-CARD_W/2, y2, C_B2)
    h_arrow(ax, 0.41+CARD_W/2, 0.62-CARD_W/2, y2, C_B2)

    # ── Fichier intermediaire B2 → B3 ────────────────────────────────────────
    note_box(ax, 0.45, 0.542, 0.28, 0.056,
             "geo_scores.csv  (score NLP par jour)", GRAY, fs=7.5)
    v_arrow(ax, 0.45, 0.595, 0.570, C_B2)
    v_arrow(ax, 0.45, 0.515, 0.475, C_B3)

    # ── BLOC 3 : modules ─────────────────────────────────────────────────────
    y3 = 0.420
    card(ax, 0.22, y3, CARD_W, CARD_H, "Strategy",
         "Golden Cross\n+ geo >= -0.5", C_B3, title_fs=8, sub_fs=7)
    card(ax, 0.45, y3, CARD_W, CARD_H, "BacktestEngine",
         "Buy&Hold vs GeoQuant\ncouts en bps", C_B3, title_fs=8, sub_fs=7)
    card(ax, 0.68, y3, CARD_W, CARD_H, "BacktestResult",
         "CAGR · Sharpe · MaxDD\nWin Rate · Trade Log", C_B3, title_fs=8, sub_fs=7)

    h_arrow(ax, 0.22+CARD_W/2, 0.45-CARD_W/2, y3, C_B3)
    h_arrow(ax, 0.45+CARD_W/2, 0.68-CARD_W/2, y3, C_B3)

    # ── Fleche B3 → B4 ───────────────────────────────────────────────────────
    v_arrow(ax, 0.58, 0.375, 0.258, C_B3, "BacktestResult")

    # ── BLOC 4 : modules ─────────────────────────────────────────────────────
    y4 = 0.200
    card(ax, 0.20, y4, CARD_W, CARD_H, "Dashboard",
         "5 onglets Streamlit", C_B4, title_fs=8, sub_fs=7)
    card(ax, 0.45, y4, CARD_W, CARD_H, "PaperTrader",
         "yfinance + Yahoo RSS\n+ FinBERT live", C_B4, title_fs=8, sub_fs=7)
    card(ax, 0.70, y4, CARD_W, CARD_H, "AlertManager",
         "st.toast + email\nsi signal change", C_B4, title_fs=8, sub_fs=7)

    # ── Sources live (sous Bloc 4) ────────────────────────────────────────────
    ax.text(BX, 0.055,
            "Sources live : Yahoo Finance (cours) · Yahoo RSS (news) · HuggingFace (FinBERT)",
            ha="center", va="center", color=C_EXT, fontsize=8,
            bbox=dict(facecolor=PANEL, edgecolor=C_EXT,
                      linewidth=0.8, pad=4, boxstyle="round,pad=0.4"))
    v_arrow(ax, 0.45, 0.080, 0.155, C_EXT)

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    p = OUT / "diag2_flux_donnees.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  OK  {p.name}")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMME 3  –  Logique de signal (flowchart propre - INCHANGE)
# ═════════════════════════════════════════════════════════════════════════════
def diag3():
    fig, ax = new_fig(9, 11, "Logique de signal  —  Golden Cross + Geo-Score")

    CX = 0.50
    W_BOX, H_BOX = 0.46, 0.08
    W_DIA, H_DIA = 0.52, 0.09

    Y = dict(
        start   = 0.93,
        d1      = 0.79,
        d2      = 0.61,
        long    = 0.46,
        sizing  = 0.31,
        exemple = 0.17,
        exec_   = 0.05,
        cash    = 0.70,
    )

    def s_box(cx, cy, w, h, text, color, fs=8.5):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0,rounding_size=0.015",
                     facecolor=color, alpha=0.18,
                     edgecolor=color, linewidth=1.6))
        ax.text(cx, cy, text, ha="center", va="center", color=TEXT,
                fontsize=fs, fontweight="bold", multialignment="center")

    s_box(CX, Y["start"], W_BOX*0.6, H_BOX*0.85, "Chaque jour t", "#555")
    diamond(ax, CX, Y["d1"], W_DIA, H_DIA, "MA50[t]  >  MA200[t] ?", C_B1)
    diamond(ax, CX, Y["d2"], W_DIA, H_DIA, "geo_score[t-1]  >=  -0.5 ?", C_B2)
    s_box(CX, Y["long"],   W_BOX*0.7, H_BOX,      "signal = 1   (LONG)", C_B2)
    s_box(CX, Y["sizing"], W_BOX*0.95, H_BOX*1.1,
          "Position sizing :\nscale = ( geo_score - (-0.5) ) / 0.5\nposition = clip(scale, 0, 1)", C_B3, fs=8)
    s_box(CX, Y["exemple"], W_BOX*0.9, H_BOX*1.0,
          "Ex : geo_score = -0.10\nscale = 0.40 / 0.50 = 0.80\nposition = 80%", "#333", fs=8)
    s_box(CX, Y["exec_"], W_BOX*0.75, H_BOX*0.85,
          "BacktestEngine  /  PaperTrader", C_B4)
    s_box(0.86, 0.70, 0.20, H_BOX, "signal = 0\n(CASH)", C_B4)

    v_arrow(ax, CX, Y["start"] - H_BOX*0.45, Y["d1"] + H_DIA/2, GRAY)
    v_arrow(ax, CX, Y["d1"] - H_DIA/2,       Y["d2"] + H_DIA/2, C_B1)
    v_arrow(ax, CX, Y["d2"] - H_DIA/2,       Y["long"] + H_BOX/2, C_B2)
    v_arrow(ax, CX, Y["long"] - H_BOX/2,     Y["sizing"] + H_BOX*0.60, C_B2)
    v_arrow(ax, CX, Y["sizing"] - H_BOX*0.58, Y["exemple"] + H_BOX*0.55, C_B3)
    v_arrow(ax, CX, Y["exemple"] - H_BOX*0.5, Y["exec_"] + H_BOX*0.45, GRAY)

    for yt in [Y["d1"], Y["d2"]]:
        ax.text(CX + 0.04, yt - 0.06, "Oui", color="#2ca02c",
                fontsize=8.5, fontweight="bold")

    ax.annotate("", xy=(0.86, Y["cash"] + H_BOX/2),
                xytext=(CX + W_DIA/2, Y["d1"]),
                arrowprops=dict(arrowstyle="-|>", color=C_B4, lw=1.5,
                                connectionstyle="arc3,rad=-0.2",
                                mutation_scale=13))
    ax.text(CX + W_DIA/2 + 0.02, Y["d1"] + 0.02, "Non",
            color=C_B4, fontsize=8.5, fontweight="bold")

    ax.annotate("", xy=(0.86, Y["cash"] - H_BOX/2),
                xytext=(CX + W_DIA/2, Y["d2"]),
                arrowprops=dict(arrowstyle="-|>", color=C_B4, lw=1.5,
                                connectionstyle="arc3,rad=-0.15",
                                mutation_scale=13))
    ax.text(CX + W_DIA/2 + 0.02, Y["d2"] + 0.02, "Non",
            color=C_B4, fontsize=8.5, fontweight="bold")

    ax.annotate("", xy=(CX + 0.05, Y["exec_"] + H_BOX*0.4),
                xytext=(0.86, Y["cash"] - H_BOX/2),
                arrowprops=dict(arrowstyle="-|>", color=C_B4, lw=1.2,
                                connectionstyle="arc3,rad=0.3",
                                mutation_scale=11))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "diag3_logique_signal.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  OK  {p.name}")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMME 4  –  Paper Trading – séquence (INCHANGE)
# ═════════════════════════════════════════════════════════════════════════════
def diag4():
    fig, ax = new_fig(14, 8.5, "Paper Trading  —  Boucle temps reel")

    actors = [
        ("Utilisateur",           0.08,  C_EXT),
        ("Dashboard\n(Streamlit)", 0.24,  C_B4),
        ("PaperTrader",            0.42,  C_B3),
        ("Yahoo Finance\n(yfinance)", 0.58, C_B1),
        ("Yahoo RSS",              0.72,  C_B1),
        ("FinBERT",                0.86,  C_B2),
    ]

    TOP = 0.93
    BOT = 0.04
    AW, AH = 0.12, 0.085

    for name, x, color in actors:
        card(ax, x, TOP, AW, AH, name, color=color, title_fs=7.5)
        ax.plot([x, x], [TOP - AH/2, BOT],
                color=color, lw=1, alpha=0.25, ls="--")

    def msg(x1, x2, y, label, color, dashed=False):
        ls = "--" if dashed else "-"
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                   linestyle=ls, mutation_scale=13))
        mid = (x1 + x2) / 2
        above = x2 > x1
        dy = 0.018 if above else -0.018
        va = "bottom" if above else "top"
        ax.text(mid, y + dy, label, ha="center", va=va,
                color=TEXT, fontsize=7.5,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

    def self_call(x, y, label, color):
        r = 0.04
        arc = mpatches.Arc((x + r, y), r*2, 0.06, angle=0,
                           theta1=90, theta2=270,
                           color=color, lw=1.3, ls="--")
        ax.add_patch(arc)
        ax.annotate("", xy=(x, y - 0.03), xytext=(x + r*2, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.3, mutation_scale=11))
        ax.text(x + r*2 + 0.015, y, label, va="center",
                color=MUTED, fontsize=7,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))

    ys = [0.83, 0.74, 0.66, 0.59, 0.51, 0.44, 0.36, 0.29, 0.21, 0.13]

    msg(0.08, 0.24, ys[0], 'Clic  "Rafraichir"',       C_EXT)
    msg(0.24, 0.42, ys[1], "get_live_snapshot()",       C_B4)
    msg(0.42, 0.58, ys[2], "download(^GSPC, 60d)",      C_B3)
    msg(0.58, 0.42, ys[3], "OHLCV + MA50 + MA200",      C_B1, dashed=True)
    msg(0.42, 0.72, ys[4], "fetch_yahoo_news_rss()",    C_B3)
    msg(0.72, 0.42, ys[5], "headlines du jour",         C_B1, dashed=True)
    msg(0.42, 0.86, ys[6], "predict(headlines)",        C_B3)
    msg(0.86, 0.42, ys[7], "SentimentResult[ ]",        C_B2, dashed=True)
    self_call(0.42, ys[8],
              "geo_score = mean(p_pos - p_neg)   |   signal = GoldenCross AND geo >= seuil",
              C_B3)
    msg(0.42, 0.24, ys[9],
        "LiveSnapshot (signal, position, prix, geo_score)", C_B3, dashed=True)

    ax.text(0.24, 0.07,
            "AlertManager : signal_changed() → st.toast() + email si changement",
            ha="center", va="center", color=MUTED, fontsize=7.5,
            bbox=dict(facecolor=PANEL, edgecolor=C_B4, linewidth=0.8,
                      pad=5, boxstyle="round,pad=0.4"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "diag4_paper_trading.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  OK  {p.name}")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAMME 5  –  Classes (4 colonnes, fleches seulement H ou V)
# ═════════════════════════════════════════════════════════════════════════════
def diag5():
    fig, ax = new_fig(16, 9, "Diagramme de classes  —  GeoQuant AI")

    # 4 colonnes = 4 blocs, chacun avec ses classes empilees
    # Fleches : horizontales entre colonnes, verticales dans une colonne
    # AUCUNE fleche ne croise une autre colonne

    CW, CH = 0.175, 0.145
    GAP_X  = 0.04    # espace entre colonnes
    COL_X  = [0.12, 0.37, 0.62, 0.88]   # centres des 4 colonnes

    # ── Labels de colonnes ────────────────────────────────────────────────────
    for x, lbl, col in [
        (COL_X[0], "BLOC 1\nData Engineering", C_B1),
        (COL_X[1], "BLOC 2\nNLP & Geo-Score",  C_B2),
        (COL_X[2], "BLOC 3\nStrategie",         C_B3),
        (COL_X[3], "BLOC 4\nDashboard",         C_B4),
    ]:
        ax.add_patch(FancyBboxPatch((x - CW/2, 0.88), CW, 0.09,
                     boxstyle="round,pad=0,rounding_size=0.015",
                     facecolor=col, alpha=0.25, edgecolor=col, linewidth=1.5))
        ax.text(x, 0.925, lbl, ha="center", va="center",
                color=col, fontsize=9, fontweight="bold",
                multialignment="center")

    def cls(cx, cy, name, file_, methods, color):
        lx, by = cx - CW/2, cy - CH/2
        ax.add_patch(FancyBboxPatch((lx, by), CW, CH,
                     boxstyle="round,pad=0,rounding_size=0.012",
                     facecolor=PANEL, edgecolor=color, linewidth=1.8))
        # Header
        hh = CH * 0.28
        ax.add_patch(FancyBboxPatch((lx, cy + CH/2 - hh), CW, hh,
                     boxstyle="round,pad=0,rounding_size=0.012",
                     facecolor=color, alpha=0.50, edgecolor="none"))
        ax.text(cx, cy + CH/2 - hh/2, name,
                ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        # Fichier
        ax.text(cx, cy + CH/2 - hh - 0.012, file_,
                ha="center", va="top", color=MUTED,
                fontsize=6.5, style="italic")
        # Separateur
        sep_y = cy - CH/2 + CH*0.42
        ax.plot([lx + 0.01, lx + CW - 0.01], [sep_y, sep_y],
                color=color, lw=0.6, alpha=0.4)
        # Methodes
        ax.text(cx, cy - CH/2 + CH*0.18, methods,
                ha="center", va="center", color=TEXT,
                fontsize=6.5, family="monospace",
                multialignment="center")

    # ── Colonne 0  –  Bloc 1 ──────────────────────────────────────────────────
    Y10 = 0.76  # PriceLoader
    Y11 = 0.56  # NewsLoader
    Y12 = 0.36  # DataMerger
    Y13 = 0.15  # DataVisualizer

    cls(COL_X[0], Y10, "PriceLoader", "loader.py",
        "fetch_data()\ncompute_returns()\nsave_data()", C_B1)
    cls(COL_X[0], Y11, "NewsLoader", "news_loader.py",
        "load_raw()  clean()\naggregate_by_day()\nrun()", C_B1)
    cls(COL_X[0], Y12, "DataMerger", "merger.py",
        "merge()\ncompute_technical_features()\nrun()  save_dataset()", C_B1)
    cls(COL_X[0], Y13, "DataVisualizer", "visualizer.py",
        "plot_price_vs_media()\nplot_drawdown()  plot_rsi()\nrun()", C_B1)

    v_arrow(ax, COL_X[0], Y10-CH/2, Y11+CH/2, C_B1)
    v_arrow(ax, COL_X[0], Y11-CH/2, Y12+CH/2, C_B1)
    v_arrow(ax, COL_X[0], Y12-CH/2, Y13+CH/2, C_B1)

    # ── Colonne 1  –  Bloc 2 ──────────────────────────────────────────────────
    Y20 = 0.76  # FinBertSentiment
    Y21 = 0.53  # SentimentCache
    Y22 = 0.28  # GeoScorer

    cls(COL_X[1], Y20, "FinBertSentiment", "finbert_sentiment.py",
        "predict(texts)\n→ List[SentimentResult]\nbatch processing", C_B2)
    cls(COL_X[1], Y21, "SentimentCache", "sentiment_cache.py",
        "load_cache()\nsave_cache()\n(JSON, evite recalcul)", C_B2)
    cls(COL_X[1], Y22, "GeoScorer", "geo_scorer.py",
        "score_headlines()\naggregate_by_day()\nsmooth()  run()", C_B2)

    v_arrow(ax, COL_X[1], Y20-CH/2, Y21+CH/2, C_B2)
    v_arrow(ax, COL_X[1], Y21-CH/2, Y22+CH/2, C_B2)

    # ── Colonne 2  –  Bloc 3 ──────────────────────────────────────────────────
    Y30 = 0.76  # Strategy
    Y31 = 0.53  # BacktestEngine
    Y32 = 0.28  # BacktestResult

    cls(COL_X[2], Y30, "Strategy", "strategy.py",
        "load()  apply()  run()\nGolden Cross + Geo-filter\nposition sizing", C_B3)
    cls(COL_X[2], Y31, "BacktestEngine", "backtest.py",
        "run(prices, signals)\n_equity_curve()\n_sharpe()  _max_drawdown()", C_B3)
    cls(COL_X[2], Y32, "BacktestResult", "backtest.py  (dataclass)",
        "total_return  cagr\nmax_drawdown  sharpe\nwin_rate  trade_log", C_B3)

    v_arrow(ax, COL_X[2], Y30-CH/2, Y31+CH/2, C_B3)
    v_arrow(ax, COL_X[2], Y31-CH/2, Y32+CH/2, C_B3)

    # ── Colonne 3  –  Bloc 4 ──────────────────────────────────────────────────
    Y40 = 0.76  # LiveSnapshot
    Y41 = 0.53  # PortfolioState
    Y42 = 0.28  # AlertManager

    cls(COL_X[3], Y40, "LiveSnapshot", "paper_trading.py",
        "prix_actuel  ma50  ma200\ngolden_cross  geo_score\nsignal  position  confidence", C_B4)
    cls(COL_X[3], Y41, "PortfolioState", "paper_trading.py",
        "capital_cash  nb_parts\nvaleur_totale()\nto_dict()  from_dict()", C_B4)
    cls(COL_X[3], Y42, "AlertManager", "alert_manager.py",
        "signal_changed()\nsend_email()\nsave_signal()", C_B4)

    v_arrow(ax, COL_X[3], Y40-CH/2, Y41+CH/2, C_B4)
    v_arrow(ax, COL_X[3], Y41-CH/2, Y42+CH/2, C_B4)

    # ── Fleches INTER-colonnes (horizontales strictes) ────────────────────────
    # B1 → B2 : DataMerger fournit le dataset a GeoScorer
    h_arrow(ax, COL_X[0]+CW/2, COL_X[1]-CW/2, Y12,
            C_B1, "dataset_final.csv", above=False)

    # B2 → B3 : GeoScorer fournit les geo_scores a Strategy
    h_arrow(ax, COL_X[1]+CW/2, COL_X[2]-CW/2, Y22,
            C_B2, "geo_scores.csv", above=False)

    # B3 → B4 : BacktestResult va au Dashboard
    h_arrow(ax, COL_X[2]+CW/2, COL_X[3]-CW/2, Y32,
            C_B3, "BacktestResult", above=False)

    # Legende
    handles = [
        mpatches.Rectangle((0,0),1,1, color=c, alpha=0.5, label=l)
        for l, c in [("Bloc 1 - Data Eng.", C_B1), ("Bloc 2 - NLP", C_B2),
                     ("Bloc 3 - Strategy",  C_B3), ("Bloc 4 - Dashboard", C_B4)]
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4,
              facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT,
              fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    p = OUT / "diag5_classes.png"
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  OK  {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generation des diagrammes...")
    diag1()
    diag2()
    diag3()
    diag4()
    diag5()
    print(f"\nDone - PNG dans : {OUT}")
