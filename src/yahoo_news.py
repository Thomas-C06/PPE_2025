from __future__ import annotations

from typing import List, Dict, Any, Optional
import time
import requests
import xml.etree.ElementTree as ET


def fetch_yahoo_news_rss(symbol: str, count: int = 20, region: str = "US", lang: str = "en-US") -> List[Dict[str, Any]]:
    """
    Récupère les news via le RSS Yahoo Finance pour un ticker.
    Exemple de flux : https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={region}&lang={lang}"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) trading-ia-sentiment/1.0"
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = channel.findall("item")
    out: List[Dict[str, Any]] = []

    for it in items[:count]:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub_date = (it.findtext("pubDate") or "").strip()
        desc = (it.findtext("description") or "").strip()

        out.append({
            "title": title,
            "link": link,
            "published": pub_date,
            "summary": desc,
            "publisher": None,
        })

    return out


def fetch_yahoo_news(symbol: str, count: int = 20) -> List[Dict[str, Any]]:
    """
    Point d’entrée unique. Pour l’instant on utilise RSS (le plus robuste).
    """
    # petit retry simple
    last_err = None
    for _ in range(2):
        try:
            news = fetch_yahoo_news_rss(symbol, count=count)
            if news:
                return news
        except Exception as e:
            last_err = e
            time.sleep(0.7)
    # si échec: retourne vide (ou tu peux raise last_err)
    return []


def news_to_text(item: Dict[str, Any]) -> str:
    title = (item.get("title") or "").strip()
    summary = (item.get("summary") or "").strip()
    text = (title + ". " + summary).strip()
    return text if text else title