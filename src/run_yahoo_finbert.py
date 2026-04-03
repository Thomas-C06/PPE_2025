from __future__ import annotations

import pandas as pd
from datetime import datetime

from yahoo_news import fetch_yahoo_news, news_to_text
from finbert_sentiment import FinBertSentiment

from dotenv import load_dotenv
load_dotenv()

def main():
    symbol = "TSLA"  # change ici
    n = 15           # nombre de news

    news = fetch_yahoo_news(symbol, count=n)

    keyword = symbol.lower()
    company_words = [symbol.lower(), "tesla"]

    news = [x for x in news if any(w in (x.get("title","").lower()) for w in company_words)]
    if not news:
        print("Aucune news récupérée (RSS vide ou bloqué).")
        return

    # transformer les news en texte pour FinBERT
    texts = [news_to_text(x) for x in news]

    # charger FinBERT
    clf = FinBertSentiment()

    # analyse
    df = clf.predict_df(texts)

    # score trading utile
    df["sentiment"] = df["p_positive"] - df["p_negative"]
    ticker_score = df["sentiment"].mean()
    print(f"\n{symbol} mean sentiment (n={len(df)}): {ticker_score:.3f}")

    # ajouter le titre de la news
    df["title"] = [x.get("title", "") for x in news]

    # sauvegarde
    filename = f"finbert_news_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)

    print(df[["title", "label", "sentiment"]].head())
    print("\nSaved:", filename)


if __name__ == "__main__":
    main()