# GeoQuant AI - Module Prix

Ce module telecharge, nettoie, enrichit et sauvegarde des donnees de prix de marche.

## Ce que fait le code, point par point

1. Charge la configuration depuis `src/config.py` :
   - `TICKERS` : liste des actifs a telecharger. Un "ticker" est le symbole boursier/identifiant utilise par Yahoo Finance pour un actif (ex: `^GSPC`, `^FCHI`, `EURUSD=X`).
   - `START_DATE` : date de debut.
   - `END_DATE` : date de fin (date du jour).
2. Initialise `PriceLoader` avec les tickers et les dates.
3. Verifie ou cree les dossiers de sortie :
   - `data/raw` pour les CSV bruts.
   - `data/processed` pour les CSV enrichis.
4. Telecharge les donnees via Yahoo Finance (yfinance) pour chaque ticker.
5. Ajoute une colonne `Ticker` a chaque DataFrame.
6. Calcule les rendements :
   - `Returns` : rendement simple en pourcentage.
   - `Log_Returns` : rendement logarithmique.
7. Sauvegarde les fichiers :
   - CSV brut dans `data/raw/{ticker}_raw.csv`.
   - CSV calcule dans `data/processed/{ticker}_processed.csv`.
8. Affiche un apercu (`head`) de chaque actif dans la console.

## Structure

- `data/raw` : CSV bruts telecharges.
- `data/processed` : CSV propres avec rendements.
- `src` : code source (config, loader, main).

## Installation

```powershell
pip install -r requirements.txt
```

## Execution

```powershell
python src/main.py
```

## Configuration

Modifie les tickers et dates dans `src/config.py`.
