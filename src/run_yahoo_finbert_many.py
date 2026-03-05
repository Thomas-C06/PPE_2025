from __future__ import annotations

import asyncio
from datetime import datetime
import pandas as pd

from yahoo_news_async import fetch_yahoo_news_rss_many, news_to_text
from finbert_sentiment import FinBertSentiment
from sentiment_cache import load_cache, save_cache

from dotenv import load_dotenv
load_dotenv()


def aggregate_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df["confidence"] = df[["p_positive", "p_negative", "p_neutral"]].max(axis=1)

    # weighted mean sentiment par ticker
    def wmean(g: pd.DataFrame) -> float:
        w = g["confidence"]
        if float(w.sum()) == 0.0:
            return float(g["sentiment"].mean())
        return float((g["sentiment"] * w).sum() / w.sum())

    agg = (
        df.groupby("symbol", as_index=False)
        .apply(lambda g: pd.Series({
            "n_news": int(len(g)),
            "mean_sentiment": float(g["sentiment"].mean()),
            "weighted_sentiment": wmean(g),
            "mean_confidence": float(g["confidence"].mean()),
        }))
        .reset_index(drop=True)
        .sort_values("weighted_sentiment", ascending=False)
    )
    return agg


def main():
    # ✅ Liste de tickers à analyser
    symbols = ["TSLA", "AAPL", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "AMD", "INTC"]

    # ✅ Paramètres
    count_per_symbol = 20          # ~20 news / ticker => ~200 news
    concurrency = 10               # nb de requêtes simultanées
    batch_size = 32                # ajuste si besoin (64 si ton CPU tient)
    max_length = 128               # plus rapide que 256 pour des titres

    # 1) Fetch async
    news = asyncio.run(
        fetch_yahoo_news_rss_many(symbols, count_per_symbol=count_per_symbol, concurrency=concurrency)
    )
    if not news:
        print("Aucune news récupérée.")
        return

    texts = [news_to_text(x) for x in news]

    # 2) Cache
    cache = load_cache()

    uncached_texts = []
    for t in texts:
        if t not in cache:
            uncached_texts.append(t)

    # 3) FinBERT (chargé 1 fois)
    clf = FinBertSentiment()

    if uncached_texts:
        df_new = clf.predict_df(uncached_texts, batch_size=batch_size, max_length=max_length)
        for i, t in enumerate(uncached_texts):
            cache[t] = df_new.iloc[i].to_dict()
        save_cache(cache)

    # reconstruire df_pred dans le même ordre
    df_pred = pd.DataFrame([cache[t] for t in texts])
    df_pred["sentiment"] = df_pred["p_positive"] - df_pred["p_negative"]

    # 4) Metadata + join
    meta = pd.DataFrame([{
        "symbol": x.get("symbol"),
        "title": x.get("title"),
        "link": x.get("link"),
        "published": x.get("published"),
    } for x in news])

    df = pd.concat([meta, df_pred.drop(columns=["text"], errors="ignore")], axis=1)

    # 5) Exports
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    details_file = f"finbert_news_many_details_{ts}.csv"
    agg_file = f"finbert_news_many_agg_{ts}.csv"

    df.to_csv(details_file, index=False)
    agg = aggregate_by_symbol(df.copy())
    agg.to_csv(agg_file, index=False)

    print("\n=== Aggregation par ticker ===")
    print(agg.to_string(index=False))

    print(f"\nSaved details: {details_file}")
    print(f"Saved agg:     {agg_file}")


if __name__ == "__main__":
    main()