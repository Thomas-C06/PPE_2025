# GeoQuant AI

Moteur de backtesting focuse sur la recuperation et le traitement des donnees de prix.

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
