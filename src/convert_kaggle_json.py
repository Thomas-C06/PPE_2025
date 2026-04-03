"""
Convert the Kaggle "US Financial News Articles" dataset to CSV.

The dataset contains monthly folders (2018_01_xxx, 2018_02_xxx, ...),
each holding hundreds of JSON files named blogs_XXXXXXX.json.

This script:
    1. Scans all those folders recursively.
    2. Extracts title, date, and source from each JSON.
    3. Saves a single clean CSV to data/raw/combined_news_data.csv

Usage:
    py -3 src/convert_kaggle_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ── CONFIGURE THIS PATH ──────────────────────────────────────────────────────
# Paste the path to the folder that contains the 2018_01_xxx subfolders.
# Example: r"C:\Users\elena\Desktop\ING4\PPE"
KAGGLE_FOLDER = r"C:\Users\elena\Desktop\ING4\PPE"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "data" / "raw" / "combined_news_data.csv"


def extract_fields(json_path: Path) -> dict | None:
    """Read one JSON file and return the three fields we need."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("title", "").strip()
        published = data.get("thread", {}).get("published", "")
        site = data.get("thread", {}).get("site", "")

        if not title or not published:
            return None

        return {
            "date": published[:10],   # keep only YYYY-MM-DD
            "headline": title,
            "publisher": site,
        }
    except Exception:
        return None


def main() -> None:
    kaggle_dir = Path(KAGGLE_FOLDER)
    if not kaggle_dir.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {KAGGLE_FOLDER}\n"
            "Verifie le chemin dans la variable KAGGLE_FOLDER."
        )

    # Find all JSON files in all subfolders
    json_files = list(kaggle_dir.rglob("blogs_*.json"))
    total = len(json_files)
    print(f"Fichiers JSON trouves : {total:,}")

    if total == 0:
        print("Aucun fichier trouve. Verifie le chemin KAGGLE_FOLDER.")
        return

    # Parse each file
    records = []
    for i, path in enumerate(json_files, 1):
        if i % 5000 == 0:
            print(f"  Traitement : {i:,} / {total:,} fichiers...")
        row = extract_fields(path)
        if row:
            records.append(row)

    print(f"Articles valides extraits : {len(records):,} / {total:,}")

    # Build DataFrame and save
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nCSV sauvegarde dans : {OUT_PATH}")
    print(f"Periode couverte    : {df['date'].min()} to {df['date'].max()}")
    print(f"Nombre d'articles   : {len(df):,}")
    print("\nPreview :")
    print(df[["date", "headline", "publisher"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
