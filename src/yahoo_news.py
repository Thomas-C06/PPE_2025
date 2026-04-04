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


def _fetch_rss_generic(url: str, count: int, publisher: str) -> List[Dict[str, Any]]:
    """
    Recupere les news depuis n’importe quel flux RSS standard.
    """
    headers = {"User-Agent": "Mozilla/5.0 trading-ia-sentiment/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        channel = root.find("channel")
        if channel is None:
            return []
        items = channel.findall("item")
        out = []
        for it in items[:count]:
            title    = (it.findtext("title") or "").strip()
            pub_date = (it.findtext("pubDate") or "").strip()
            desc     = (it.findtext("description") or "").strip()
            link     = (it.findtext("link") or "").strip()
            if title:
                out.append({"title": title, "published": pub_date,
                            "summary": desc, "link": link, "publisher": publisher})
        return out
    except Exception:
        return []


def fetch_reuters_news(count: int = 15) -> List[Dict[str, Any]]:
    """
    Recupere les news financieres depuis le flux RSS Reuters Business.
    """
    url = "https://feeds.reuters.com/reuters/businessNews"
    return _fetch_rss_generic(url, count, "Reuters")


def fetch_google_finance_news(query: str = "stock market", count: int = 15) -> List[Dict[str, Any]]:
    """
    Recupere les news financieres depuis Google News RSS.
    """
    url = f"https://news.google.com/rss/search?q={query}+finance&hl=en-US&gl=US&ceid=US:en"
    return _fetch_rss_generic(url, count, "Google News")


def fetch_yahoo_news(symbol: str, count: int = 20) -> List[Dict[str, Any]]:
    """
    Point d’entree unique. Combine Yahoo Finance RSS + Reuters + Google Finance.
    Retourne jusqu’a `count` articles dedupliques par titre.
    """
    all_news: List[Dict[str, Any]] = []

    # 1. Yahoo Finance RSS (source principale)
    last_err = None
    for _ in range(2):
        try:
            news = fetch_yahoo_news_rss(symbol, count=count)
            if news:
                all_news.extend(news)
                break
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    # 2. Reuters Business RSS (source complementaire)
    reuters = fetch_reuters_news(count=10)
    all_news.extend(reuters)

    # 3. Google Finance RSS (source complementaire)
    google = fetch_google_finance_news(count=10)
    all_news.extend(google)

    # Dedupliquer par titre (insensible a la casse)
    seen_titles: set = set()
    unique_news: List[Dict[str, Any]] = []
    for item in all_news:
        key = (item.get("title") or "").lower().strip()
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique_news.append(item)

    return unique_news[:count]


def news_to_text(item: Dict[str, Any]) -> str:
    title = (item.get("title") or "").strip()
    summary = (item.get("summary") or "").strip()
    text = (title + ". " + summary).strip()
    return text if text else title