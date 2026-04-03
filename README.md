# GeoQuant AI

> **Problématique PPE 2025-2026**
> *"Comment l'intégration de l'analyse sémantique automatisée (NLP) des flux d'actualités
> permet-elle d'améliorer la résilience (Risk-Management) d'une stratégie d'investissement
> face aux chocs exogènes ?"*

GeoQuant AI est un système de backtesting qui combine données de marché, analyse de sentiment NLP (FinBERT) et indicateurs techniques pour piloter une stratégie d'investissement sur le S&P 500. Le projet est découpé en **4 blocs** indépendants qui s'enchaînent.

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [Installation](#2-installation)
3. [Bloc 1 — Data Engineering](#3-bloc-1--data-engineering)
4. [Bloc 2 — NLP & Geo-Score](#4-bloc-2--nlp--geo-score)
5. [Bloc 3 — Stratégie & Backtest](#5-bloc-3--stratégie--backtest)
6. [Bloc 4 — Dashboard Streamlit](#6-bloc-4--dashboard-streamlit)
7. [Structure des dossiers](#7-structure-des-dossiers)
8. [Données disponibles](#8-données-disponibles)
9. [Ordre d'exécution complet](#9-ordre-dexécution-complet)
10. [Contributeurs](#10-contributeurs)

---

## 1. Architecture générale

```
┌─────────────────────────────────────────────────────────────────────┐
│  BLOC 1 — Data Engineering                                          │
│  Yahoo Finance ──► loader.py ──► merger.py ──► dataset_final.csv   │
│                   (prix OHLCV)   (+ indicateurs techniques)         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ dataset_final.csv
┌───────────────────────────────▼─────────────────────────────────────┐
│  BLOC 2 — NLP / FinBERT                                             │
│  News Kaggle ──► kaggle_news_loader.py ──► geo_scorer.py            │
│  (S&P 500 headlines)   (nettoyage)       (FinBERT ► geo_scores.csv) │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ geo_scores.csv
┌───────────────────────────────▼─────────────────────────────────────┐
│  BLOC 3 — Stratégie & Backtest                                      │
│  strategy.py  ──►  backtest.py  ──►  résultats (rendement, Sharpe…) │
│  (signal MA + Geo-Score)  (simulation Buy&Hold vs GeoQuant)         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ résultats
┌───────────────────────────────▼─────────────────────────────────────┐
│  BLOC 4 — Dashboard Streamlit                                        │
│  app/dashboard.py  ──►  interface interactive (5 onglets)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Installation

### Prérequis

- Python 3.10 ou supérieur
- pip ou conda

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales

| Package | Version min | Usage |
|---------|------------|-------|
| pandas | 2.0 | Manipulation des données |
| numpy | 1.24 | Calculs numériques |
| yfinance | 0.2.36 | Téléchargement des prix (Yahoo Finance) |
| matplotlib / seaborn | 3.7 / 0.13 | Graphiques statiques |
| transformers | 4.40 | Modèle FinBERT (Hugging Face) |
| torch | 2.1 | Backend PyTorch pour FinBERT |
| tqdm | 4.66 | Barres de progression |
| streamlit | 1.30 | Dashboard interactif |
| plotly | 5.18 | Graphiques interactifs |
| aiohttp / requests | 3.8 / 2.31 | Scraping Yahoo News (optionnel) |
| python-dotenv | 1.0 | Variables d'environnement |

> **Note FinBERT** : la première exécution télécharge le modèle `ProsusAI/finbert`
> depuis Hugging Face (~500 Mo). Les exécutions suivantes utilisent le cache local.

---

## 3. Bloc 1 — Data Engineering

**Objectif** : télécharger les prix de marché, les fusionner avec des actualités et calculer les indicateurs techniques.

### Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `src/config.py` | Configuration centrale (tickers, dates, noms de fichiers) |
| `src/loader.py` | Télécharge les prix via Yahoo Finance (`yfinance`) |
| `src/news_loader.py` | Charge et agrège un fichier CSV de news local |
| `src/merger.py` | Fusionne prix + news, calcule les indicateurs techniques |
| `src/visualizer.py` | Génère 4 graphiques PNG dans `data/processed/graphiques/` |
| `src/main.py` | **Point d'entrée** — orchestre les étapes 1 à 4 |
| `src/fetch_prices.py` | Script de test pour vérifier le téléchargement |

### Lancement

```bash
cd PPE_2025/src
python main.py
```

### Ce que fait `main.py` (pipeline en 4 étapes)

```
STEP 1 — Téléchargement des prix
    PriceLoader télécharge 9 actifs (S&P 500, CAC 40, BTC, Gold, Oil, VIX…)
    via Yahoo Finance sur la période 2022-01-01 → 2024-12-31.
    Sauvegarde : data/raw/{ticker}_raw.csv

STEP 2 — Chargement des news
    NewsLoader lit data/raw/sample_news.csv (actualités financières/géopolitiques
    manuellement sélectionnées), agrège les titres par jour.

STEP 3 — Fusion prix + news + indicateurs techniques
    DataMerger joint les deux sources sur la date (left join sur le calendrier
    de trading). Calcule pour le S&P 500 :
      · MA20, MA50, MA200  (moyennes mobiles simples)
      · RSI_14             (Wilder EMA — 14 jours)
      · Volatility_20      (volatilité annualisée glissante 20j)
      · Drawdown           (perte depuis le plus haut courant)
    Sauvegarde : data/processed/dataset_final.csv

STEP 4 — Visualisations
    DataVisualizer génère 4 graphiques PNG :
      01_price_vs_media.png    — prix + activité news
      02_drawdown.png          — drawdown glissant
      03_rsi.png               — RSI 14j
      04_return_distribution.png — distribution des rendements
```

### Résultat

`data/processed/dataset_final.csv` — 752 lignes (jours de trading), 18 colonnes :

```
date | Adj Close | Close | High | Low | Open | Volume | Ticker |
Returns | Log_Returns | nb_articles | titles |
MA20 | MA50 | MA200 | RSI_14 | Volatility_20 | Drawdown
```

### Configuration (`src/config.py`)

```python
TICKERS = ["^GSPC", "^FCHI", "EURUSD=X", "BTC-USD", "GC=F", "CL=F", "^VIX", "^IXIC", "^DJI"]
PRIMARY_TICKER = "^GSPC"   # ticker principal pour dataset_final
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"
```

---

## 4. Bloc 2 — NLP & Geo-Score

**Objectif** : attribuer un score de sentiment quotidien entre -1 et +1 à chaque journée de trading en analysant les titres d'actualités avec FinBERT.

### Le Geo-Score

```
Geo-Score = P(positive) − P(negative)   ∈ [-1, +1]

  > 0  →  sentiment positif (calme / optimisme)
  = 0  →  sentiment neutre
  < 0  →  sentiment négatif (peur / panique)
```

### Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `src/finbert_sentiment.py` | Wrapper FinBERT — prédit label + probabilités pour un texte |
| `src/sentiment_cache.py` | Cache JSON sur disque pour éviter de rescorer les titres déjà traités |
| `src/news_config.py` | Mapping colonnes des datasets Kaggle supportés |
| `src/kaggle_news_loader.py` | Charge et normalise un CSV Kaggle (datasets spécialisés) |
| `src/news_main.py` | Pipeline de chargement des news Kaggle |
| `src/geo_scorer.py` | **Cœur du Bloc 2** — pipeline complet de scoring |
| `src/run_geo_scorer.py` | **Point d'entrée** Bloc 2 |

### Données d'entrée

Le Bloc 2 utilise le dataset Kaggle **"S&P 500 with Financial News Headlines (2008-2024)"**.

1. Télécharger le ZIP sur [Kaggle](https://www.kaggle.com/)
2. Extraire le CSV dans `data/raw/`
3. Vérifier que le nom correspond à `news_sp500_news_2024_processed.csv`
   (ou modifier `NEWS_PROCESSED_FILE` dans `src/config.py`)

### Lancement

```bash
cd PPE_2025/src
python run_geo_scorer.py
```

### Pipeline `geo_scorer.py` (5 étapes)

```
STEP 1 — Chargement des news
    Lit data/processed/news_sp500_news_2024_processed.csv
    Normalise les dates, filtre les titres vides.

STEP 2 — Scoring FinBERT
    Pour chaque titre : appel au modèle ProsusAI/finbert.
    Résultat : p_positive, p_neutral, p_negative (somme = 1).
    Cache JSON (sentiment_cache.json) : les titres déjà scorés sont récupérés
    instantanément sans re-passer dans le modèle.
    Agrégation par article : geo_score_article = p_positive − p_negative

STEP 3 — Agrégation journalière
    Groupe les articles par date.
    Moyenne pondérée par confidence = max(p_positive, p_neutral, p_negative).
    Lissage lookback 3 jours (rolling mean causal, center=False) pour
    réduire le bruit journalier.
    Résultat clampé dans [-1, +1].

STEP 4 — Sauvegarde
    data/processed/geo_scores.csv
    Colonnes : date | geo_score | geo_score_raw | nb_articles_scored

STEP 5 — Injection dans dataset_final.csv
    Ajoute la colonne geo_score dans dataset_final.csv par jointure sur date.
```

### Cache de sentiment

`data/processed/sentiment_cache.json` stocke tous les titres déjà analysés.
Cela permet de relancer le script sans re-scorer inutilement (~12 000 entrées).

### Module Yahoo News (Arthur — optionnel)

Pour enrichir les données avec des news en temps réel :

```bash
cd PPE_2025/src
python run_yahoo_finbert.py         # un seul ticker
python run_yahoo_finbert_many.py    # plusieurs tickers
```

Fichiers : `yahoo_news.py` (synchrone), `yahoo_news_async.py` (asynchrone via aiohttp).

---

## 5. Bloc 3 — Stratégie & Backtest

**Objectif** : générer des signaux de trading à partir des indicateurs techniques et du Geo-Score, puis simuler et comparer deux stratégies.

### La règle de trading GeoQuant

```
SI  (MA50 > MA200)          ← Golden Cross (tendance haussière)
ET  (geo_score[t-1] ≥ seuil) ← pas de panique géopolitique la veille
ALORS → Long  (position ∈ [0, 1])
SINON → Cash  (position = 0)
```

> **Correction look-ahead** : le geo_score du jour `t-1` est utilisé pour le signal
> du jour `t`. Le signal de `t` est lui-même décalé d'un jour pour l'exécution :
> la position effective est appliquée le jour `t+1`. Aucune donnée future n'est utilisée.

### Position sizing

La taille de position n'est pas binaire mais linéaire :

```
position = clip((geo_score[t-1] − seuil) / sizing_range, 0, 1)

Exemple (seuil = -0.5, sizing_range = 0.5) :
  geo_score = -0.50  →  position = 0%   (seuil atteint)
  geo_score = -0.25  →  position = 50%
  geo_score ≥  0.00  →  position = 100% (pleinement investi)
```

### Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `src/strategy.py` | Fusionne geo_scores + dataset_final, génère signal et position |
| `src/backtest.py` | Simule Buy&Hold et GeoQuant, calcule les métriques |

### Métriques calculées

| Métrique | Définition |
|----------|-----------|
| **Rendement total** | `(valeur_finale / valeur_initiale) − 1` |
| **CAGR** | Rendement annualisé sur la base de 252 jours de trading |
| **Max Drawdown** | Pire perte depuis un plus haut : `min((P − max_P) / max_P)` |
| **Sharpe Ratio** | `(rendement_moyen − Rf_journalier) / écart_type × √252` |
| **Win Rate** | % de trades complétés (entrée→sortie) avec P&L > 0 |
| **Nb trades** | Nombre de aller-retours complets |

### Coûts de transaction

10 bps (0.10 %) sont déduits par côté (entrée OU sortie) par défaut.
Le coût est proportionnel au changement de position (`|Δposition| × coût`).

### Utilisation programmatique

```python
from pathlib import Path
from strategy import Strategy
from backtest import run_backtest

base = Path(".")                           # racine du projet
strat = Strategy(base_dir=base, seuil_geo=-0.5)
df    = strat.run()                        # dataset avec signal + position

bh, gq = run_backtest(df, price_col="Adj Close",
                       costs_bps=10,
                       risk_free_annual=0.05)

print(f"Buy & Hold : {bh.total_return:+.2%}")
print(f"GeoQuant   : {gq.total_return:+.2%}  Sharpe={gq.sharpe:.2f}")
```

---

## 6. Bloc 4 — Dashboard Streamlit

**Objectif** : interface interactive pour explorer les données, visualiser les signaux et analyser les performances.

### Lancement

```bash
cd PPE_2025
python -m streamlit run app/dashboard.py
```

### Les 5 onglets

#### 📊 Vue Marché
- Cours de clôture S&P 500 + moyennes mobiles MA50 / MA200
- Barres d'activité médiatique (nb articles/jour)
- Annotations des événements géopolitiques clés
  (invasion Ukraine, hausses Fed, effondrement SVB, attaque Hamas…)
- Métriques : cours actuel, rendement total, Max Drawdown, volatilité

#### 🧠 Sentiment & NLP
- Jauge Geo-Score du dernier jour (CONFIANCE / PRUDENCE / PANIQUE)
- Timeline du Geo-Score avec zones colorées et ligne de seuil
- Tableau des dernières actualités
- **Pouvoir prédictif** : corrélation glissante 60j entre `geo_score[t-1]`
  et `rendement[t]` + nuage de points

#### ⚔️ Backtest
- Métriques comparatives Buy&Hold vs GeoQuant (6 indicateurs)
- Courbe de performance cumulée (base = 1)
- Graphique des zones Long / Cash sur le cours
- Tableau détaillé + journal des trades (date, action, prix, P&L)
- **Walk-Forward Test IS/OOS** : division 70% / 30% pour valider la robustesse
- **Analyse de sensibilité** : rendement et Sharpe selon le seuil Geo-Score
  (de -1.0 à 0.0) — permet de détecter le sur-ajustement

#### 📈 Analyse Technique
- RSI 14j avec zones surachat / survente
- Drawdown glissant depuis le plus haut
- Volatilité annualisée glissante 20j
- Distribution des rendements quotidiens (histogramme + loi normale + VaR 5%)
- Statistiques descriptives : moyenne, écart-type, skewness, kurtosis, CVaR

#### ℹ️ À Propos
- Description du projet et de la problématique
- Schéma d'architecture
- Disclaimer

### Paramètres de la sidebar

| Paramètre | Description |
|-----------|-------------|
| **Période** | Slider date de début / fin |
| **Seuil Geo-Score** | Valeur entre -1.0 et 0.0 (défaut -0.5) — réactualise instantanément tous les résultats |
| **Coûts de transaction** | Bps par côté (défaut 10 bps = 0.10%) |
| **Taux sans risque** | % annuel pour le Sharpe (défaut 0%) |
| **Afficher les MM** | Toggle MA50/MA200 |
| **Annoter les événements** | Toggle annotations géopolitiques |

---

## 7. Structure des dossiers

```
PPE_2025/
│
├── app/
│   ├── __init__.py
│   └── dashboard.py          # Tableau de bord Streamlit (Bloc 4)
│
├── src/
│   │
│   │  ── Bloc 1 : Data Engineering ──────────────────────────────────
│   ├── config.py             # Configuration centrale (tickers, dates, chemins)
│   ├── loader.py             # Téléchargement prix Yahoo Finance
│   ├── news_loader.py        # Chargement CSV news local + agrégation
│   ├── merger.py             # Fusion prix + news + indicateurs techniques
│   ├── visualizer.py         # Génération graphiques PNG
│   ├── main.py               # ★ Point d'entrée Bloc 1
│   ├── fetch_prices.py       # Script de test du téléchargement
│   │
│   │  ── Bloc 2 : NLP ──────────────────────────────────────────────
│   ├── finbert_sentiment.py  # Wrapper FinBERT (ProsusAI/finbert)
│   ├── sentiment_cache.py    # Cache JSON des prédictions FinBERT
│   ├── geo_scorer.py         # ★ Pipeline Geo-Score complet
│   ├── run_geo_scorer.py     # ★ Point d'entrée Bloc 2
│   ├── news_config.py        # Mapping colonnes datasets Kaggle
│   ├── kaggle_news_loader.py # Chargement datasets Kaggle news
│   ├── news_main.py          # Pipeline chargement news Kaggle
│   ├── convert_kaggle_json.py # Conversion JSON → CSV pour certains datasets
│   ├── generate_sample_news.py # Générateur de news d'exemple
│   ├── hello_world.py        # Script de test rapide FinBERT
│   │
│   │  ── Bloc 2 (branche Arthur) : Yahoo News ───────────────────────
│   ├── yahoo_news.py         # Scraping actualités Yahoo Finance
│   ├── yahoo_news_async.py   # Version asynchrone (aiohttp)
│   ├── run_yahoo_finbert.py  # FinBERT sur news Yahoo (1 ticker)
│   ├── run_yahoo_finbert_many.py # FinBERT sur news Yahoo (N tickers)
│   ├── test_finbert.py       # Tests unitaires FinBERT
│   │
│   │  ── Bloc 3 : Stratégie & Backtest ────────────────────────────
│   ├── strategy.py           # Règle Golden Cross + Geo-Score + position sizing
│   └── backtest.py           # Simulation Buy&Hold vs GeoQuant + métriques
│
├── data/
│   ├── raw/
│   │   ├── {ticker}_raw.csv          # Prix bruts par ticker
│   │   ├── sample_news.csv           # News géopolitiques/financières (manuel)
│   │   └── sp500_news_headlines.csv  # Dataset Kaggle headlines
│   │
│   └── processed/
│       ├── dataset_final.csv         # ★ Dataset principal (Bloc 1 → Bloc 3)
│       ├── geo_scores.csv            # ★ Geo-Scores quotidiens (Bloc 2)
│       ├── {ticker}_processed.csv    # Prix + rendements par ticker
│       ├── news_aggregated.csv       # News agrégées par jour
│       ├── news_sp500_news_2024_processed.csv  # News Kaggle nettoyées
│       ├── sentiment_cache.json      # Cache FinBERT (~12 000 entrées)
│       └── graphiques/
│           ├── 01_price_vs_media.png
│           ├── 02_drawdown.png
│           ├── 03_rsi.png
│           └── 04_return_distribution.png
│
├── notebooks/
│   └── hello_world.ipynb     # Notebook de test FinBERT + geo_scorer
│
├── requirements.txt
├── README.md
├── README_Prix.md            # Documentation spécifique au module Prix
└── JOURNAL_DE_BORD.md        # Détail technique de chaque fichier
```

---

## 8. Données disponibles

### Données de marché

| Ticker | Actif | Période |
|--------|-------|---------|
| `^GSPC` | S&P 500 (indice principal) | 2022–2024 |
| `^FCHI` | CAC 40 | 2022–2024 |
| `EURUSD=X` | EUR/USD | 2022–2024 |
| `BTC-USD` | Bitcoin | 2022–2024 |
| `GC=F` | Or (XAU/USD) | 2022–2024 |
| `CL=F` | Pétrole WTI | 2022–2024 |
| `^VIX` | Indice de volatilité CBOE | 2022–2024 |
| `^IXIC` | NASDAQ Composite | 2022–2024 |
| `^DJI` | Dow Jones | 2022–2024 |

### Événements géopolitiques annotés dans le dashboard

| Date | Événement |
|------|-----------|
| 2022-02-24 | Invasion de l'Ukraine |
| 2022-06-15 | Fed +75 bps (première hausse importante) |
| 2022-11-11 | Effondrement FTX |
| 2023-03-10 | Faillite Silicon Valley Bank |
| 2023-10-07 | Attaque du Hamas |
| 2024-08-05 | Krach japonais (Nikkei -12%) |
| 2024-09-18 | Fed -50 bps (pivot) |
| 2024-11-05 | Élection Trump |

---

## 9. Ordre d'exécution complet

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Bloc 1 — Télécharger les prix + créer dataset_final.csv
cd PPE_2025/src
python main.py

# 3. Bloc 2 — Calculer le Geo-Score avec FinBERT
#    (nécessite d'avoir placé le CSV Kaggle dans data/raw/)
python run_geo_scorer.py

# 4. Bloc 4 — Lancer le dashboard (Bloc 3 s'exécute dans le dashboard)
cd PPE_2025
python -m streamlit run app/dashboard.py
```

> **Raccourci** : si `data/processed/geo_scores.csv` existe déjà, vous pouvez
> lancer directement le dashboard sans relancer les blocs 1 et 2.

---

## 10. Contributeurs

| Membre | Branche | Contribution |
|--------|---------|--------------|
| **Mathias** | `DataFMat` | Bloc 1 (loader, merger, visualizer), Bloc 4 (dashboard Streamlit), JOURNAL_DE_BORD |
| **Elena** | `NewsElena` | Bloc 2 (geo_scorer, kaggle_news_loader, news_config, run_geo_scorer) |
| **Arthur** | `news_Arthur` | Bloc 2 (finbert_sentiment, yahoo_news, run_yahoo_finbert) |
| **Thomas** | `main` | Architecture initiale, configuration, Bloc 3 (strategy, backtest) |

---

*GeoQuant AI — PPE 2025-2026 — ECE Paris*
