# Journal de bord — GeoQuant AI

> Projet PPE 2025-2026
> Ce document décrit chaque fichier du projet, son rôle, ses fonctions principales et leur utilité.

---

## Problématique

> **"Comment l'intégration de l'analyse sémantique automatisée (NLP) des flux d'actualités
> permet-elle d'améliorer la résilience (Risk-Management) d'une stratégie d'investissement
> face aux chocs exogènes ?"**

L'idée centrale est de construire un système qui lit les news financières et géopolitiques,
leur attribue un score de sentiment, et utilise ce score pour piloter une stratégie
d'investissement sur le S&P 500.

---

## Architecture globale — Les 4 Blocs

```
Bloc 1 : Data Engineering     → télécharger, nettoyer, fusionner, sauvegarder
Bloc 2 : NLP / FinBERT        → lire les titres de news et leur attribuer un Geo-Score
Bloc 3 : Stratégie            → règle de décision basée sur les indicateurs + Geo-Score
Bloc 4 : Dashboard Streamlit  → interface interactive pour tout visualiser
```

---

## Structure des dossiers

```
PPE_2025/
├── src/                    # Code source Python (pipeline Bloc 1)
│   ├── config.py           # Configuration centrale (tickers, dates)
│   ├── fetch_prices.py     # Script de test de téléchargement
│   ├── loader.py           # Téléchargement et traitement des prix
│   ├── news_loader.py      # Chargement et nettoyage des news
│   ├── merger.py           # Fusion prix + news + indicateurs techniques
│   ├── visualizer.py       # Génération de graphiques PNG
│   └── main.py             # Orchestrateur du pipeline complet
├── app/
│   ├── __init__.py         # Fichier vide (marque app/ comme package Python)
│   └── dashboard.py        # Tableau de bord Streamlit interactif (Bloc 4)
├── data/
│   ├── raw/
│   │   └── sample_news.csv # Dataset d'actualités géopolitiques et financières
│   └── processed/
│       ├── dataset_final.csv         # Dataset principal (prix + news + indicateurs)
│       ├── news_aggregated.csv       # News agrégées par jour
│       ├── ^GSPC_processed.csv       # Prix SP500 avec rendements
│       ├── (autres tickers).csv      # Idem pour BTC, CAC40, Gold, etc.
│       └── graphiques/               # Graphiques PNG générés par visualizer.py
│           ├── 01_price_vs_media.png
│           ├── 02_drawdown.png
│           ├── 03_rsi.png
│           └── 04_return_distribution.png
├── requirements.txt        # Dépendances Python
├── README.md               # Description courte du projet
└── JOURNAL_DE_BORD.md      # Ce fichier
```

---

## Commandes de lancement

```bash
# Lancer le pipeline Bloc 1 (génère dataset_final.csv)
cd PPE_2025/src
/c/Users/mathi/anaconda3/python.exe main.py

# Lancer le tableau de bord Streamlit (Bloc 4)
cd PPE_2025
/c/Users/mathi/anaconda3/python.exe -m streamlit run app/dashboard.py

# Tester uniquement le téléchargement des prix
cd PPE_2025/src
/c/Users/mathi/anaconda3/python.exe fetch_prices.py
```

---

## Fichiers source — Bloc 1 Data Engineering

---

### `src/config.py`

**Rôle** : Fichier de configuration centrale. Contient toutes les constantes utilisées par
les autres modules. Aucune logique de traitement ici.

**Contenu** :

| Variable | Type | Description |
|----------|------|-------------|
| `TICKERS` | `list[str]` | Liste des 9 tickers Yahoo Finance téléchargés |
| `PRIMARY_TICKER` | `str` | Ticker principal pour le dataset final (`^GSPC` = S&P 500) |
| `TICKER_NAMES` | `dict[str, str]` | Correspondance ticker -> nom lisible (ex. `^GSPC` -> `"SP500"`) |
| `START_DATE` | `str` | Date de début des données : `"2022-01-01"` |
| `END_DATE` | `str` | Date de fin des données : `"2024-12-31"` |

**Pourquoi** : Centraliser la configuration évite de répéter les mêmes valeurs dans plusieurs
fichiers. Si on veut changer la période ou ajouter un ticker, on ne modifie que `config.py`.

---

### `src/fetch_prices.py`

**Rôle** : Script autonome pour tester que le téléchargement des prix fonctionne.
Ne fait pas partie du pipeline principal — s'exécute seul avec `python fetch_prices.py`.

**Fonction principale** :

| Fonction | Description |
|----------|-------------|
| `fetch_and_preview()` | Télécharge tous les tickers via `PriceLoader` et affiche un tableau récapitulatif dans le terminal : nombre de lignes, dates de début/fin, dernier cours de clôture. |

**Pourquoi** : Outil de diagnostic rapide. Permet de vérifier que Yahoo Finance répond
correctement avant de lancer le pipeline complet.

---

### `src/loader.py` — classe `PriceLoader`

**Rôle** : Télécharger les données de marché depuis Yahoo Finance, calculer les rendements
et les indicateurs techniques, sauvegarder les fichiers CSV.

**Attributs de la dataclass** :

| Attribut | Type | Description |
|----------|------|-------------|
| `tickers` | `list[str]` | Tickers à télécharger |
| `start_date` | `str` | Date de début (YYYY-MM-DD) |
| `end_date` | `str` | Date de fin (YYYY-MM-DD) |
| `base_dir` | `Path` | Dossier racine du projet |

**Méthodes** :

| Méthode | Description |
|---------|-------------|
| `fetch_data()` | Appelle `yfinance.download()` pour chaque ticker. Retourne un dictionnaire `{ticker: DataFrame}`. Lève une `RuntimeError` si le téléchargement échoue. |
| `compute_returns()` | Ajoute les colonnes `Returns` (rendement simple) et `Log_Returns` (log-rendement) à chaque DataFrame. Ces rendements sont calculés à partir du prix ajusté (`Adj Close`). |
| `compute_technical_features()` | Ajoute MA20, MA50, MA200, RSI_14, Volatility_20 et Drawdown. Utilisé dans le contexte des CSV individuels par ticker. |
| `save_data()` | Sauvegarde les DataFrames bruts dans `data/raw/` et les DataFrames traités dans `data/processed/`. |

**Pourquoi séparer `fetch_data` et `compute_returns`** : Cela permet de relancer uniquement
le calcul des rendements (rapide) sans re-télécharger les données depuis internet (lent).

---

### `src/news_loader.py` — classe `NewsLoader`

**Rôle** : Charger un fichier CSV d'actualités, le nettoyer, et agréger les news par jour.

**Attributs de la dataclass** :

| Attribut | Type | Description |
|----------|------|-------------|
| `base_dir` | `Path` | Dossier racine du projet |
| `news_file` | `Optional[Path]` | Chemin vers le CSV de news. Si `None`, utilise `data/raw/sample_news.csv` |
| `date_col` | `str` | Nom de la colonne date dans le CSV source (défaut : `"date"`) |
| `title_col` | `str` | Nom de la colonne titre dans le CSV source (défaut : `"title"`) |

**Méthodes** :

| Méthode | Description |
|---------|-------------|
| `load_raw()` | Lit le CSV de news. Si les colonnes `date` ou `title` n'existent pas avec ces noms, lance la détection automatique via `_detect_columns()`. |
| `_detect_columns()` | Parcourt une liste de noms de colonnes candidats pour identifier automatiquement les colonnes date et titre. Utile pour les datasets Kaggle avec des noms non standards. |
| `clean()` | Supprime les lignes avec des dates invalides (NaT), des titres vides ou des doublons exacts (même date + même titre). Normalise les dates à minuit. |
| `aggregate_by_day()` | Regroupe les news par date : compte le nombre d'articles (`nb_articles`) et concatène les titres avec ` \| ` comme séparateur (`titles`). |
| `save_aggregated()` | Sauvegarde le résultat dans `data/processed/news_aggregated.csv`. |
| `run()` | Orchestre les 4 étapes ci-dessus en un seul appel. Retourne le DataFrame agrégé. |

**Pourquoi agréger par jour** : Le S&P 500 a une granularité journalière. On doit donc
ramener les news (qui peuvent être plusieurs par jour) à une seule ligne par jour,
compatible avec le calendrier boursier.

---

### `data/raw/sample_news.csv`

**Rôle** : Dataset d'actualités géopolitiques et financières couvrant la période 2022-2024.
C'est la matière première du Bloc 2 (NLP).

**Format** :

| Colonne | Description |
|---------|-------------|
| `date` | Date de l'événement (YYYY-MM-DD) |
| `title` | Titre de l'actualité en anglais |
| `source` | Source (Reuters, BBC, AP, etc.) |
| `category` | Catégorie (Geopolitics, Macro, Finance, Energy, etc.) |

**Contenu** : 106 événements majeurs incluant :
- Invasion de l'Ukraine (24 fév. 2022)
- Hausses de taux de la Fed (série 2022-2023)
- Effondrement de FTX (nov. 2022)
- Faillite de SVB (mars 2023)
- Attaque du Hamas (oct. 2023)
- Krach japonais (août 2024)
- Élection de Trump (nov. 2024)

---

### `src/merger.py` — classe `DataMerger`

**Rôle** : Fusionner le DataFrame de prix avec les news agrégées, puis calculer tous les
indicateurs techniques. Produit le fichier `dataset_final.csv`.

**Attributs de la dataclass** :

| Attribut | Type | Description |
|----------|------|-------------|
| `base_dir` | `Path` | Dossier racine du projet |
| `primary_ticker` | `str` | Ticker de référence (pour les logs), défaut `"^GSPC"` |

**Méthodes** :

| Méthode | Description |
|---------|-------------|
| `merge()` | Jointure gauche (left join) du DataFrame de prix sur la colonne `date`. Les jours sans news reçoivent `nb_articles=0` et `titles=""`. La jointure est gauche car on veut garder **tous** les jours de trading, même sans news. |
| `_flatten_columns()` | Corrige le MultiIndex que yfinance génère parfois (ex. `("Close", "^GSPC")` -> `"Close"`). |
| `compute_technical_features()` | Calcule sur la colonne de prix (`Adj Close` ou `Close`) : MA20, MA50, MA200, RSI_14 (méthode EMA de Wilder), Volatility_20 (annualisée), Drawdown depuis le plus haut glissant. |
| `save_dataset()` | Sauvegarde dans `data/processed/dataset_final.csv`. |
| `run()` | Orchestre merge -> compute_technical_features -> save_dataset. |

**Indicateurs calculés** :

| Indicateur | Formule | Interprétation |
|------------|---------|----------------|
| MA20 / MA50 / MA200 | Moyenne glissante simple | Tendance à court/moyen/long terme |
| RSI_14 | EMA des gains / EMA des pertes sur 14j | Momentum : > 70 surachat, < 30 survente |
| Volatility_20 | σ(log-rendements 20j) × √252 | Risque annualisé sur 20 jours |
| Drawdown | (prix - max_glissant) / max_glissant | Perte depuis le sommet historique |

---

### `src/visualizer.py` — classe `DataVisualizer`

**Rôle** : Générer 4 graphiques PNG de diagnostic à partir du `dataset_final.csv`.
Ces graphiques sont sauvegardés dans `data/processed/graphiques/`.

**Attributs de la dataclass** :

| Attribut | Type | Description |
|----------|------|-------------|
| `base_dir` | `Path` | Dossier racine du projet |
| `ticker_name` | `str` | Nom lisible affiché dans les titres des graphiques |

**Méthodes** :

| Méthode | Fichier produit | Description |
|---------|-----------------|-------------|
| `plot_price_vs_media()` | `01_price_vs_media.png` | Courbe de prix (axe gauche) + barres de news (axe droit) + lignes verticales pour les 8 événements géopolitiques majeurs |
| `plot_drawdown()` | `02_drawdown.png` | Drawdown en % avec annotation du point de drawdown maximal |
| `plot_rsi()` | `03_rsi.png` | RSI 14 jours avec zones rouges (surachat ≥70) et vertes (survente ≤30) |
| `plot_return_distribution()` | `04_return_distribution.png` | Histogramme des rendements quotidiens + courbe normale théorique + marqueur VaR 5% |
| `run()` | (les 4 fichiers) | Appelle les 4 méthodes ci-dessus dans l'ordre |

**Pourquoi matplotlib et pas Plotly ici** : Ces graphiques sont des exports PNG statiques
pour les rapports ou présentations. Plotly est utilisé dans le dashboard (interactif).

---

### `src/main.py`

**Rôle** : Point d'entrée du pipeline Bloc 1. Orchestre les 4 étapes dans l'ordre
et affiche un résumé en fin d'exécution.

**Fonction `main()`** — les 4 étapes :

| Étape | Module utilisé | Ce qui se passe |
|-------|---------------|-----------------|
| STEP 1 | `PriceLoader` | Téléchargement via Yahoo Finance pour tous les tickers configurés dans `config.py`. Calcul des rendements. Sauvegarde des CSV bruts et traités. |
| STEP 2 | `NewsLoader` | Chargement de `sample_news.csv`, nettoyage, agrégation par jour, sauvegarde de `news_aggregated.csv`. |
| STEP 3 | `DataMerger` | Fusion du DataFrame S&P 500 avec les news agrégées. Calcul de tous les indicateurs techniques. Sauvegarde de `dataset_final.csv`. |
| STEP 4 | `DataVisualizer` | Génération des 4 graphiques PNG dans `data/processed/graphiques/`. |

**Résumé affiché en fin d'exécution** :
- Ticker principal et période couverte
- Nombre total de jours de trading
- Nombre et pourcentage de jours avec news
- Max drawdown sur la période
- Chemin du dataset final et des graphiques

---

## Fichier dashboard — Bloc 4

---

### `app/dashboard.py`

**Rôle** : Interface web interactive built avec Streamlit. Lit uniquement
`data/processed/dataset_final.csv` (S&P 500) et les visualise de façon interactive.

**Lancement** :
```bash
cd PPE_2025
/c/Users/mathi/anaconda3/python.exe -m streamlit run app/dashboard.py
# → http://localhost:8501
```

**Fonctions de chargement** :

| Fonction | Description |
|----------|-------------|
| `charger_dataset()` | Lit `dataset_final.csv`. Normalisé avec `@st.cache_data` pour ne pas relire le fichier à chaque interaction. Retourne `None` si le fichier est absent (affiche un message d'erreur). |
| `charger_news_brutes()` | Lit `sample_news.csv` pour le tableau des actualités dans l'onglet NLP. |

**Fonctions de graphiques (Plotly)** :

| Fonction | Graphique produit |
|----------|-------------------|
| `construire_graphique_prix()` | Cours de clôture + MA50/MA200 + barres de news + lignes d'événements géopolitiques |
| `construire_graphique_geo_score()` | Timeline du Geo-Score avec zones colorées rouge/orange/vert selon le seuil |
| `construire_graphique_rsi()` | RSI 14 jours avec zones surachat/survente |
| `construire_graphique_drawdown()` | Drawdown avec annotation du maximum |
| `construire_graphique_volatilite()` | Volatilité annualisée 20 jours |
| `construire_distribution_rendements()` | Histogramme des rendements + courbe normale + VaR 5% |
| `_ajouter_lignes_evenements()` | Helper interne : ajoute les lignes verticales d'événements sur n'importe quel graphique |

**Organisation des 5 onglets** :

| Onglet | Contenu |
|--------|---------|
| 📊 Vue Marché | 5 métriques clés + graphique prix/news |
| 🧠 Sentiment & NLP | Jauge Geo-Score + timeline (si Bloc 2 fait) ou placeholder + tableau des actualités |
| ⚔️ Backtest | Placeholder Bloc 3 avec emplacements réservés |
| 📈 Analyse Technique | RSI + Drawdown + Volatilité + Distribution des rendements + stats |
| ℹ️ À Propos | Description du projet, problématique, architecture, disclaimer |

---

## Données produites — `data/processed/dataset_final.csv`

Ce fichier est le résultat du Bloc 1. C'est lui que lit le dashboard.

| Colonne | Type | Description |
|---------|------|-------------|
| `date` | datetime | Date de trading |
| `Open` | float | Prix d'ouverture |
| `High` | float | Plus haut de la journée |
| `Low` | float | Plus bas de la journée |
| `Close` | float | Prix de clôture |
| `Adj Close` | float | Prix de clôture ajusté (dividendes + splits) |
| `Volume` | int | Volume échangé |
| `Ticker` | str | Symbole Yahoo Finance (`^GSPC`) |
| `Returns` | float | Rendement simple journalier |
| `Log_Returns` | float | Log-rendement journalier |
| `nb_articles` | int | Nombre d'articles de news ce jour |
| `titles` | str | Titres des articles séparés par ` \| ` |
| `MA20` | float | Moyenne mobile 20 jours |
| `MA50` | float | Moyenne mobile 50 jours |
| `MA200` | float | Moyenne mobile 200 jours |
| `RSI_14` | float | RSI 14 jours (0-100) |
| `Volatility_20` | float | Volatilité annualisée sur 20 jours |
| `Drawdown` | float | Drawdown depuis le plus haut (valeur négative) |
| `geo_score` | float | **(Bloc 2 — à venir)** Score NLP entre -1 et +1 |
| `signal` | int | **(Bloc 3 — à venir)** Signal de trading (1=Long, 0=Cash) |

---

## Blocs à venir

### Bloc 2 — NLP / FinBERT

**Objectif** : Attribuer un score de sentiment géopolitique à chaque journée de trading.

**Plan** :
1. Créer `nlp/geo_scorer.py` avec une classe `GeoScorer`
2. Pour chaque ligne de `dataset_final.csv` ayant des `titles` non vides,
   passer le texte dans FinBERT (`ProsusAI/finbert` sur Hugging Face)
3. FinBERT retourne pour chaque texte 3 probabilités : positive, negative, neutral
4. Calculer le Geo-Score = `P(positive) - P(negative)` → valeur entre -1 et +1
5. Réécrire `dataset_final.csv` avec la nouvelle colonne `geo_score`

### Bloc 3 — Stratégie de trading

**Objectif** : Implémenter et backtester une stratégie qui utilise le Geo-Score.

**Règle de trading envisagée** :
```
SI (MA50 > MA200)  ← tendance haussière (Golden Cross)
ET (geo_score > seuil_geo)  ← pas de panique géopolitique
ALORS → Long (on reste investi)
SINON → Cash (on coupe la position)
```

**Métriques de performance à calculer** :
- Rendement total vs Buy & Hold
- Max Drawdown
- Ratio de Sharpe
- Win Rate
- Nombre de trades

---

## Dépendances — `requirements.txt`

| Package | Version min | Usage |
|---------|-------------|-------|
| `pandas` | 2.0 | Manipulation des DataFrames |
| `numpy` | 1.24 | Calculs numériques (RSI, volatilité) |
| `yfinance` | 0.2.36 | Téléchargement des prix de marché |
| `matplotlib` | 3.7 | Graphiques PNG statiques (visualizer.py) |
| `seaborn` | 0.13 | Styles graphiques matplotlib |
| `transformers` | 4.35 | Modèle FinBERT (Bloc 2) |
| `torch` | 2.1 | Backend PyTorch pour FinBERT |
| `streamlit` | 1.30 | Interface web du dashboard |
| `plotly` | 5.18 | Graphiques interactifs dans le dashboard |

---

## Problèmes connus et solutions

| Problème | Solution |
|----------|---------|
| `CAC40.PA` retourne une erreur sur Yahoo Finance | Utiliser `^FCHI` à la place |
| `streamlit` command not found sur Windows | Lancer avec `python -m streamlit run` |
| Caractères unicode dans `print()` causent des erreurs sur Windows | La console Windows est cp1252 — utiliser uniquement des caractères ASCII dans les `print()`. Les chaînes dans les fichiers (UTF-8) ne sont pas affectées. |
| La colonne `titles` est lue comme `float` (NaN) par pandas | Utiliser `.fillna("").astype(str)` après la lecture du CSV |
| MultiIndex dans les colonnes yfinance | Utiliser `df.columns.get_level_values(0)` pour aplatir |

---

*Dernière mise à jour : mars 2026*
