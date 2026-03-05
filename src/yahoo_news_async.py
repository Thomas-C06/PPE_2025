from __future__ import annotations

import asyncio
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

import aiohttp


def _parse_rss(xml_text: str, limit: int) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    out: List[Dict[str, Any]] = []
    for it in channel.findall("item")[:limit]:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub_date = (it.findtext("pubDate") or "").strip()
        desc = (it.findtext("description") or "").strip()

        out.append(
            {
                "title": title,
                "link": link,
                "published": pub_date,
                "summary": desc,
            }
        )
    return out


async def fetch_yahoo_news_rss_one(
    session: aiohttp.ClientSession,
    symbol: str,
    count: int = 20,
    region: str = "US",
    lang: str = "en-US",
) -> List[Dict[str, Any]]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={region}&lang={lang}"
    headers = {"User-Agent": "Mozilla/5.0 trading-ia-sentiment/1.0"}

    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as r:
        r.raise_for_status()
        text = await r.text()

    items = _parse_rss(text, limit=count)
    for it in items:
        it["symbol"] = symbol
    return items


async def fetch_yahoo_news_rss_many(
    symbols: List[str],
    count_per_symbol: int = 20,
    concurrency: int = 10,
) -> List[Dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_yahoo_news_rss_one(session, s, count=count_per_symbol) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    out: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        out.extend(res)

    # dédup par lien
    seen = set()
    dedup = []
    for it in out:
        link = it.get("link") or ""
        if link and link in seen:
            continue
        if link:
            seen.add(link)
        dedup.append(it)

    return dedup


def news_to_text(item: Dict[str, Any]) -> str:
    title = (item.get("title") or "").strip()
    summary = (item.get("summary") or "").strip()
    text = (title + ". " + summary).strip()
    return text if text else title