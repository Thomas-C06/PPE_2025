# GeoQuant AI

> **Problématique PPE 2025-2026**
> *"Comment l'intégration de l'analyse sémantique automatisée (NLP) des flux d'actualités
> permet-elle d'améliorer la résilience (Risk-Management) d'une stratégie d'investissement
> face aux chocs exogènes ?"*

GeoQuant AI est un système de trading algorithmique qui combine données de marché, analyse de sentiment NLP (FinBERT) et indicateurs techniques pour piloter une stratégie d'investissement sur le S&P 500. Le projet est découpé en **4 blocs** indépendants qui s'enchaînent, et propose également un mode **Paper Trading en temps réel**.

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [Installation](#2-installation)
3. [Bloc 1 — Data Engineering](#3-bloc-1--data-engineering)
4. [Bloc 2 — NLP & Geo-Score](#4-bloc-2--nlp--geo-score)
5. [Bloc 3 — Stratégie & Backtest](#5-bloc-3--stratégie--backtest)
6. [Bloc 4 — Dashboard & Paper Trading](#6-bloc-4--dashboard--paper-trading)
7. [Guide d'utilisation pas à pas](#7-guide-dutilisation-pas-à-pas)
8. [Explication technique détaillée](#8-explication-technique-détaillée)
9. [Structure des dossiers](#9-structure-des-dossiers)
10. [Données disponibles](#10-données-disponibles)
11. [Ordre d'exécution complet](#11-ordre-dexécution-complet)
12. [Contributeurs](#12-contributeurs)

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
│  BLOC 4 — Dashboard Streamlit + Paper Trading                       │
│  app/dashboard.py  ──►  interface interactive (5 onglets)           │
│  src/paper_trading.py ──► signaux en temps réel (Yahoo RSS + yfinance) │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Dépendances principales

| Package | Usage |
|---------|-------|
| `pandas` | Manipulation des données tabulaires |
| `numpy` | Calculs numériques |
| `yfinance` | Téléchargement des prix (Yahoo Finance) |
| `transformers` | Modèle FinBERT (Hugging Face) |
| `torch` | Backend PyTorch pour FinBERT |
| `streamlit` | Dashboard interactif |
| `plotly` | Graphiques interactifs |
| `requests` | Récupération des flux RSS Yahoo |

> **Note FinBERT** : la première exécution télécharge le modèle `ProsusAI/finbert`
> depuis Hugging Face (~500 Mo). Les exécutions suivantes utilisent le cache local.

---

## 3. Bloc 1 — Data Engineering

**Objectif** : télécharger les prix de marché et calculer les indicateurs techniques.

### Fichiers concernés

| Fichier | Rôle |
|---------|------|
| `src/config.py` | Configuration centrale (tickers, dates, noms de fichiers) |
| `src/loader.py` | Télécharge les prix via Yahoo Finance (`yfinance`) |
| `src/news_loader.py` | Charge et agrège un fichier CSV de news local |
| `src/merger.py` | Fusionne prix + news, calcule les indicateurs techniques |
| `src/visualizer.py` | Génère 4 graphiques PNG dans `data/processed/graphiques/` |
| `src/main.py` | **Point d'entrée** — orchestre les étapes 1 à 4 |

### Lancement

```bash
cd PPE_2025/src
python main.py
```

### Ce que fait `main.py`

```
STEP 1 — Téléchargement des prix
    PriceLoader télécharge 9 actifs (S&P 500, CAC 40, BTC, Gold, Oil, VIX…)
    via Yahoo Finance sur la période 2022-01-01 → 2024-12-31.
    Sauvegarde : data/raw/{ticker}_raw.csv

STEP 2 — Chargement des news
    NewsLoader lit data/raw/sample_news.csv (actualités financières/géopolitiques).

STEP 3 — Fusion prix + news + indicateurs techniques
    DataMerger joint les deux sources sur la date. Calcule :
      · MA20, MA50, MA200  (moyennes mobiles simples)
      · RSI_14             (Relative Strength Index 14 jours)
      · Volatility_20      (volatilité annualisée glissante 20j)
      · Drawdown           (perte depuis le plus haut courant)
    Sauvegarde : data/processed/dataset_final.csv

STEP 4 — Visualisations
    DataVisualizer génère 4 graphiques PNG dans data/processed/graphiques/
```

### Résultat

`data/processed/dataset_final.csv` — colonnes principales :

```
date | Adj Close | Returns | Log_Returns | nb_articles |
MA20 | MA50 | MA200 | RSI_14 | Volatility_20 | Drawdown
```

### Configuration (`src/config.py`)

```python
TICKERS      = ["^GSPC", "^FCHI", "EURUSD=X", "BTC-USD", "GC=F", "CL=F", "^VIX", "^IXIC", "^DJI"]
PRIMARY_TICKER = "^GSPC"
START_DATE   = "2022-01-01"
END_DATE     = "2024-12-31"
```

---

## 4. Bloc 2 — NLP & Geo-Score

**Objectif** : attribuer un score de sentiment quotidien entre -1 et +1 à chaque journée de trading.

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
| `src/finbert_sentiment.py` | Wrapper FinBERT — prédit label + probabilités |
| `src/sentiment_cache.py` | Cache JSON — évite de rescorer les titres déjà traités |
| `src/geo_scorer.py` | **Cœur du Bloc 2** — pipeline complet de scoring |
| `src/kaggle_news_loader.py` | Charge et normalise un CSV Kaggle |
| `src/run_geo_scorer.py` | **Point d'entrée** Bloc 2 |
| `src/yahoo_news.py` | Scraping actualités Yahoo Finance (temps réel) |

### Données d'entrée

Le Bloc 2 utilise le dataset Kaggle **"S&P 500 with Financial News Headlines (2008-2024)"**.

1. Télécharger le ZIP sur [Kaggle](https://www.kaggle.com/)
2. Extraire le CSV dans `data/raw/`
3. Vérifier que le nom correspond à `news_sp500_news_2024_processed.csv`

### Lancement

```bash
cd PPE_2025/src
python run_geo_scorer.py
```

### Pipeline `geo_scorer.py`

```
STEP 1 — Chargement et nettoyage des news
    Lit le CSV Kaggle, normalise les dates, filtre les titres vides.

STEP 2 — Scoring FinBERT
    Pour chaque titre : appel au modèle ProsusAI/finbert.
    Résultat : p_positive, p_neutral, p_negative (somme = 1).
    Cache JSON : les titres déjà scorés sont récupérés instantanément.
    geo_score_article = p_positive − p_negative

STEP 3 — Agrégation journalière
    Groupe les articles par date.
    Moyenne pondérée par confidence = max(p_positive, p_neutral, p_negative).
    Lissage lookback 3 jours (rolling mean causal) pour réduire le bruit.
    Résultat clampé dans [-1, +1].

STEP 4 — Sauvegarde
    data/processed/geo_scores.csv
    Colonnes : date | geo_score | geo_score_raw | nb_articles_scored

STEP 5 — Injection dans dataset_final.csv
    Ajoute la colonne geo_score dans dataset_final.csv.
```

---

## 5. Bloc 3 — Stratégie & Backtest

**Objectif** : générer des signaux de trading, simuler et comparer deux stratégies.

### La règle de trading GeoQuant

```
SI  (MA50 > MA200)             ← Golden Cross (tendance haussière)
ET  (geo_score[t-1] ≥ seuil)  ← pas de panique géopolitique la veille
ALORS → Long  (position ∈ [0, 1])
SINON → Cash  (position = 0)
```

> **Correction look-ahead** : `geo_score[t-1]` est utilisé pour le signal du jour `t`.
> Le signal est lui-même décalé d'un jour à l'exécution. Aucune donnée future n'est utilisée.

### Position sizing

```
position = clip((geo_score[t-1] − seuil) / sizing_range, 0, 1)

Exemple (seuil = -0.5, sizing_range = 0.5) :
  geo_score = -0.50  →  position = 0%    (seuil atteint)
  geo_score = -0.25  →  position = 50%
  geo_score ≥  0.00  →  position = 100%  (pleinement investi)
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
| **CAGR** | Rendement annualisé sur 252 jours de trading |
| **Max Drawdown** | Pire perte depuis un plus haut |
| **Sharpe Ratio** | `(rendement_moyen − Rf/252) / écart_type × √252` |
| **Win Rate** | % de trades entrée→sortie avec P&L > 0 |
| **Nb trades** | Nombre d'allers-retours complets |

### Coûts de transaction

10 bps (0.10%) par côté par défaut. Proportionnels au changement de position (`|Δposition| × coût`).

---

## 6. Bloc 4 — Dashboard & Paper Trading

### Lancement

```bash
cd PPE_2025
python -m streamlit run app/dashboard.py
```

Ouvre automatiquement **http://localhost:8501**

### Les 5 onglets

| Onglet | Contenu |
|--------|---------|
| 📊 **Vue Marché** | Prix, MA50/MA200, zones Golden Cross, annotations géopolitiques |
| 🧠 **Sentiment & NLP** | Timeline Geo-Score, corrélation sentiment/rendements |
| ⚔️ **Backtest** | Métriques comparatives, Walk-Forward Test, analyse de sensibilité |
| 🟢 **Paper Trading** | Signaux en temps réel, portfolio virtuel, historique des trades |
| 📈 **Analyse Technique** | RSI, drawdown, volatilité, distribution des rendements |

### Paper Trading (`src/paper_trading.py`)

Le Paper Trading simule la stratégie avec des données **en direct** :

```
Yahoo RSS (SPY)           yfinance (^GSPC)
      ↓                         ↓
News du jour            Prix actuel + MA50/MA200
      ↓                         ↓
FinBERT (avec cache)    Golden Cross ?
      ↓                         ↓
Geo-Score du jour ──────────────┘
      ↓
Signal LONG / CASH
      ↓
Portfolio virtuel (10 000 € de départ)
```

---

## 7. Guide d'utilisation pas à pas

### Étape 1 — Lancer l'application

Ouvre **PowerShell** et exécute :

```powershell
cd C:\Users\mathi\.vscode\PPE_2026\PPE_2025
python -m streamlit run app/dashboard.py
```

L'application s'ouvre dans ton navigateur à l'adresse **http://localhost:8501**.

---

### Étape 2 — Configurer la sidebar (panneau gauche)

Avant d'explorer les onglets, règle les paramètres dans la barre latérale gauche.
Ces paramètres s'appliquent à **tous les onglets simultanément**.

| Paramètre | Valeur recommandée | Explication |
|-----------|-------------------|-------------|
| **Ticker** | `^GSPC` | Indice à analyser (S&P 500) |
| **Seuil Geo-Score** | `-0.5` | Sensibilité au sentiment négatif |
| **Coûts de transaction** | `10 bps` | Frais réalistes pour un ETF |
| **Taux sans risque** | `3%` | Correspond aux taux actuels |

> **Astuce seuil** : un seuil de `-0.5` signifie "je reste investi même si le sentiment est légèrement négatif". Un seuil de `0.0` exige un sentiment positif pour investir.

---

### Étape 3 — Explorer l'onglet "Vue Marché"

Cet onglet te donne une vue d'ensemble du S&P 500 sur la période sélectionnée.

**Ce que tu vois :**
- La courbe de prix avec MA50 (orange) et MA200 (rouge)
- Les **zones vertes** = périodes Golden Cross (MA50 > MA200) = tendance haussière
- Les **marqueurs** = événements géopolitiques majeurs
- Les métriques en haut : prix actuel, rendement, drawdown max, volatilité

**Ce qu'il faut retenir :**
Quand la MA50 passe **au-dessus** de la MA200 → signal technique positif = GeoQuant peut entrer en position si le Geo-Score est également favorable.

---

### Étape 4 — Explorer l'onglet "Sentiment & NLP"

Cet onglet montre comment le sentiment des news évolue dans le temps.

**Ce que tu vois :**
- La **jauge** en haut : état du sentiment aujourd'hui (CONFIANCE / PRUDENCE / PANIQUE)
- La **timeline** du Geo-Score avec la ligne de seuil
- Le tableau des **dernières actualités** analysées
- Le graphique de **corrélation** : est-ce que le sentiment d'hier prédit le rendement d'aujourd'hui ?

**Ce qu'il faut retenir :**
Quand le Geo-Score descend sous le seuil (ligne rouge), GeoQuant sort de sa position même si le Golden Cross est actif. C'est le mécanisme de protection contre les chocs géopolitiques.

---

### Étape 5 — Explorer l'onglet "Backtest"

C'est l'onglet le plus important pour **élaborer et valider ta stratégie**.

#### 5a. Lire le tableau comparatif

En haut de l'onglet, un tableau compare **GeoQuant AI** vs **Buy & Hold** sur 6 métriques.

| Si GeoQuant > Buy & Hold sur... | C'est un bon signe |
|---|---|
| Rendement annualisé | La stratégie est plus rentable |
| Sharpe Ratio | Elle est plus rentable **pour le risque pris** |
| Max Drawdown (valeur absolue plus faible) | Elle protège mieux lors des crises |
| Win Rate > 50% | Plus de trades gagnants que perdants |

#### 5b. Utiliser l'analyse de sensibilité

Descends jusqu'à la section **"Analyse de sensibilité"**. Ce graphique montre les performances pour chaque valeur de seuil entre `-1.0` et `0.0`.

**Comment trouver ton seuil optimal :**
1. Repère le **pic du Sharpe Ratio** sur le graphique
2. Note la valeur du seuil correspondante
3. Règle ce seuil dans la sidebar → les résultats se mettent à jour automatiquement

#### 5c. Valider avec le Walk-Forward Test

La section **"Walk-Forward Test"** divise les données en :
- **70% in-sample (IS)** : période d'entraînement
- **30% out-of-sample (OOS)** : période de validation

**Règle de décision :**

| Situation | Interprétation |
|---|---|
| OOS ≈ IS | Stratégie robuste → continuer |
| OOS bien inférieur à IS | Sur-ajustement → changer le seuil |
| OOS meilleur que IS | Excellent signe de robustesse |

#### 5d. Ajuster le seuil et itérer

```
Essaie seuil = -1.0  →  note Sharpe OOS
Essaie seuil = -0.5  →  note Sharpe OOS  (défaut)
Essaie seuil = -0.3  →  note Sharpe OOS
Essaie seuil =  0.0  →  note Sharpe OOS
              ↓
Garde le seuil avec le meilleur Sharpe OOS
```

---

### Étape 6 — Tester en temps réel (Paper Trading)

Une fois ton seuil validé en backtest, passe à l'onglet **"🟢 Paper Trading"**.

#### 6a. Premier lancement

1. Sélectionne le ticker (`^GSPC` par défaut)
2. Coche **"Utiliser FinBERT"** pour obtenir le vrai Geo-Score
3. Clique **"Rafraîchir"**

> La première fois, FinBERT télécharge ~500 Mo depuis Hugging Face. C'est normal, cela ne se produit qu'une seule fois.

#### 6b. Lire le signal

Après le rafraîchissement, la carte signal affiche :

```
┌─────────────────────────────────────┐
│  SIGNAL : LONG ✅                   │
│  Prix actuel : 5 234.18             │
│  MA50 : 5 180.42  MA200 : 4 920.15  │
│  Golden Cross : Oui                 │
│  Geo-Score : +0.32  (seuil : -0.50) │
│  Position : 100%                    │
└─────────────────────────────────────┘
```

**Interprétation :**
- **LONG** = les deux conditions sont remplies → investir
- **CASH** = une condition manque → rester liquide

#### 6c. Suivre le portfolio

Coche **"Exécuter automatiquement"** pour que le portfolio virtuel (10 000 €) suive les signaux automatiquement à chaque rafraîchissement.

Les métriques du portfolio s'affichent :
- **Valeur totale** : capital actuel (cash + investi)
- **Rendement** : performance depuis le début
- **Win Rate** : % de trades gagnants
- **Nb trades** : nombre d'opérations effectuées

#### 6d. Utilisation quotidienne recommandée

```
Chaque matin (avant l'ouverture des marchés US, 15h30 heure française) :
1. Cliquer "Rafraîchir"
2. Lire le signal
3. Si "Exécuter automatiquement" est coché → rien à faire
4. Sinon → appuyer manuellement sur "Exécuter le signal"
```

#### 6e. Réinitialiser le portfolio

Le bouton **"Réinitialiser"** remet le capital à 10 000 € et efface l'historique.
Utile pour tester un nouveau seuil depuis le début.

---

### Étape 7 — Workflow complet recommandé

```
1. PARAMÉTRER
   Régler le seuil dans la sidebar (commence à -0.5)
         ↓
2. BACKTESTER
   Onglet Backtest → analyser Sharpe Ratio et Max Drawdown
         ↓
3. OPTIMISER
   Analyse de sensibilité → trouver le seuil avec meilleur Sharpe
         ↓
4. VALIDER
   Walk-Forward Test → vérifier que OOS ≈ IS
         ↓
5. PAPER TRADING
   Appliquer le seuil validé → rafraîchir quotidiennement
         ↓
6. ÉVALUER
   Après 2-4 semaines → comparer les résultats Paper Trading
   avec les prédictions du backtest
```

---

## 8. Explication technique détaillée

### 8.1 Le modèle FinBERT

FinBERT est une version de BERT (Bidirectional Encoder Representations from Transformers) fine-tunée sur des textes financiers par ProsusAI. Il prend un titre en entrée et retourne trois probabilités :

```python
# Exemple d'appel FinBERT
clf = FinBertSentiment()
result = clf.predict("Fed raises interest rates by 75 basis points")
# → {"label": "negative", "p_positive": 0.08, "p_neutral": 0.22, "p_negative": 0.70}
```

**Pourquoi FinBERT et pas un modèle généraliste ?**
FinBERT a été entraîné sur des corpus financiers (Reuters, Financial Times, Seeking Alpha). Il comprend des nuances comme "rate hike" (négatif pour les marchés) ou "earnings beat" (positif) que BERT généraliste ne capte pas correctement.

### 8.2 Le calcul du Geo-Score

Le Geo-Score d'un jour `t` se calcule en trois étapes :

**1. Score par article :**
```python
geo_score_article = p_positive - p_negative   # ∈ [-1, +1]
```

**2. Agrégation journalière pondérée :**
```python
confidence = max(p_positive, p_neutral, p_negative)  # certitude du modèle
geo_score_jour = somme(score_i × confidence_i) / somme(confidence_i)
```

**3. Lissage anti-bruit (rolling 3 jours, causal) :**
```python
geo_score_lisse = geo_score_jour.rolling(3, min_periods=1, center=False).mean()
```
`center=False` est crucial : le lissage n'utilise que les jours passés (pas de look-ahead).

### 8.3 La correction look-ahead bias

Le look-ahead bias est l'erreur classique du backtesting : utiliser des informations du futur pour générer des signaux passés.

GeoQuant corrige ce biais à deux niveaux :

**Niveau 1 — Geo-Score :**
```python
# Dans strategy.py
geo_lag = df["geo_score"].shift(1).fillna(0.0)
# Le score d'hier (t-1) génère le signal d'aujourd'hui (t)
```

**Niveau 2 — Exécution :**
```python
# Dans backtest.py
position_shifted = df["position"].shift(1).fillna(0.0)
# Le signal de (t) est exécuté le lendemain (t+1)
# Représente le délai réel entre signal et passage d'ordre
```

Sans ces corrections, le backtest serait artificiellement gonflé car la stratégie "saurait à l'avance" ce qui va se passer.

### 8.4 Le position sizing

La position n'est pas binaire (0 ou 100%) mais proportionnelle à la force du signal :

```python
# Dans strategy.py
scale = (geo_lag - self.seuil_geo) / max(self.sizing_range, 1e-9)
scale = scale.clip(0.0, 1.0)
df["position"] = np.where(df["signal"] == 1, scale, 0.0)
```

**Exemple avec seuil = -0.5 et sizing_range = 0.5 :**

| Geo-Score | Position |
|-----------|----------|
| -0.50 | 0% (seuil atteint) |
| -0.40 | 20% |
| -0.25 | 50% |
| -0.10 | 80% |
| ≥ 0.00 | 100% (pleinement investi) |

**Avantage :** une news légèrement négative réduit la position progressivement au lieu de fermer brusquement.

### 8.5 Le calcul du Sharpe Ratio

```python
# Dans backtest.py
def _sharpe(daily_rets, risk_free_annual=0.0):
    rf_daily = risk_free_annual / 252         # taux journalier
    excess   = daily_rets - rf_daily          # rendements excédentaires
    if excess.std() < 1e-9:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)
```

Un Sharpe > 1.0 est généralement considéré comme bon. > 2.0 est excellent.

### 8.6 Le calcul du Win Rate par trade

Contrairement à une mesure par jour, GeoQuant calcule le Win Rate sur des **trades complets** (entrée → sortie) :

```python
# Dans backtest.py
# Un trade = période entre position > 0 et retour à 0
# P&L du trade = (prix_sortie / prix_entrée) - 1
# Win = P&L > 0
```

Cette définition est plus juste car elle mesure si chaque décision d'investissement a été profitable.

### 8.7 Le Walk-Forward Test

La division 70/30 est calculée sur les dates de trading :

```python
split_idx    = int(len(df) * 0.70)
df_is        = df.iloc[:split_idx]    # in-sample
df_oos       = df.iloc[split_idx:]    # out-of-sample
```

Le backtest est ensuite lancé indépendamment sur chaque période avec les **mêmes paramètres** (seuil Geo-Score). Si les performances out-of-sample sont comparables à l'in-sample, la stratégie n'est pas sur-ajustée aux données historiques.

### 8.8 L'analyse de sensibilité

```python
# Dans dashboard.py
seuils  = np.arange(-1.0, 0.05, 0.05)   # 21 valeurs entre -1.0 et 0.0
for seuil in seuils:
    strat = Strategy(base_dir, seuil_geo=seuil)
    df    = strat.run()
    _, gq = run_backtest(df, costs_bps=costs_bps)
    # enregistre Sharpe et rendement pour chaque seuil
```

Cela permet de visualiser si la stratégie est **robuste** (courbe lisse avec plateau) ou **sur-ajustée** (pic isolé avec chute rapide de chaque côté).

### 8.9 Le cache de sentiment

Pour éviter de passer chaque titre dans FinBERT à chaque lancement :

```python
# Dans sentiment_cache.py
cache = {
    "Fed raises rates by 75bps": {
        "p_positive": 0.08, "p_neutral": 0.22, "p_negative": 0.70
    },
    ...
}
```

Le cache est un dictionnaire JSON persisté sur disque. À chaque run, seuls les **nouveaux titres** sont envoyés au modèle.

### 8.10 Le Paper Trading en temps réel

**Récupération du prix :**
```python
# Dans paper_trading.py
hist = yf.download("^GSPC", period="1y", interval="1d")
prix  = hist["Adj Close"].iloc[-1]
ma50  = hist["Adj Close"].rolling(50).mean().iloc[-1]
ma200 = hist["Adj Close"].rolling(200).mean().iloc[-1]
```

**Récupération des news :**
```python
# Yahoo RSS ne supporte pas ^GSPC directement → mapping vers SPY
rss_ticker = "SPY"
news = fetch_yahoo_news(rss_ticker, count=30)
```

**Persistance du portfolio :**
```json
{
  "capital_initial": 10000.0,
  "capital_cash": 7500.0,
  "capital_investi": 2600.0,
  "nb_parts": 0.497,
  "prix_entree": 5230.0,
  "nb_trades": 3,
  "trades_gagnants": 2,
  "historique": [...]
}
```

---

## 9. Structure des dossiers

```
PPE_2025/
│
├── app/
│   └── dashboard.py              # Tableau de bord Streamlit (5 onglets)
│
├── src/
│   │
│   │  ── Bloc 1 : Data Engineering ──────────────────────────────────
│   ├── config.py                 # Configuration centrale
│   ├── loader.py                 # Téléchargement prix Yahoo Finance
│   ├── news_loader.py            # Chargement CSV news local
│   ├── merger.py                 # Fusion prix + news + indicateurs
│   ├── visualizer.py             # Graphiques PNG
│   ├── main.py                   # ★ Point d'entrée Bloc 1
│   │
│   │  ── Bloc 2 : NLP ──────────────────────────────────────────────
│   ├── finbert_sentiment.py      # Wrapper FinBERT
│   ├── sentiment_cache.py        # Cache JSON des scores FinBERT
│   ├── geo_scorer.py             # ★ Pipeline Geo-Score complet
│   ├── run_geo_scorer.py         # ★ Point d'entrée Bloc 2
│   ├── kaggle_news_loader.py     # Chargement datasets Kaggle
│   ├── yahoo_news.py             # Scraping Yahoo RSS (temps réel)
│   ├── yahoo_news_async.py       # Version asynchrone (aiohttp)
│   │
│   │  ── Bloc 3 : Stratégie & Backtest ──────────────────────────
│   ├── strategy.py               # Golden Cross + Geo-Score + sizing
│   ├── backtest.py               # Simulation + métriques
│   │
│   │  ── Bloc 4 : Paper Trading ───────────────────────────────────
│   └── paper_trading.py          # Engine temps réel (prix + news + signal)
│
├── data/
│   ├── raw/
│   │   ├── {ticker}_raw.csv              # Prix bruts
│   │   └── sample_news.csv              # News géopolitiques manuelles
│   │
│   └── processed/
│       ├── dataset_final.csv            # ★ Dataset principal
│       ├── geo_scores.csv               # ★ Geo-Scores quotidiens
│       ├── sentiment_cache.json         # Cache FinBERT
│       ├── paper_portfolio.json         # Portfolio virtuel Paper Trading
│       └── graphiques/
│           ├── 01_price_vs_media.png
│           ├── 02_drawdown.png
│           ├── 03_rsi.png
│           └── 04_return_distribution.png
│
├── requirements.txt
└── README.md
```

---

## 10. Données disponibles

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

### Événements géopolitiques annotés

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

## 11. Ordre d'exécution complet

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

> **Raccourci** : si `data/processed/geo_scores.csv` existe déjà, tu peux
> lancer directement le dashboard sans relancer les blocs 1 et 2.

---

## 12. Contributeurs

| Membre | Branche | Contribution |
|--------|---------|--------------|
| **Mathias** | `DataFMat` | Bloc 1 (loader, merger, visualizer), Bloc 4 (dashboard), intégration générale |
| **Elena** | `NewsElena` | Bloc 2 (geo_scorer, kaggle_news_loader, run_geo_scorer) |
| **Arthur** | `news_Arthur` | Bloc 2 (finbert_sentiment, yahoo_news, run_yahoo_finbert) |
| **Thomas** | `main` | Architecture initiale, Bloc 3 (strategy, backtest), Paper Trading |

---

*GeoQuant AI — PPE 2025-2026 — ECE Paris*
