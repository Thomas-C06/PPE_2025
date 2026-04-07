"""Historical Google News RSS extension with real publication dates.

This loader fetches historical search feeds from Google News RSS and keeps only
articles with a reliable `pubDate`. It never assigns the query date as the
publication date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from time import sleep
from typing import Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET

import pandas as pd
import requests


@dataclass
class GoogleNewsHistoricalLoader:
    """Fetch and clean a historically dated Google News RSS extension."""

    base_dir: Path
    start_date: str
    end_date: str
    pause_seconds: float = 0.05
    region: str = "US"
    lang: str = "en-US"
    edition: str = "US:en"
    queries: dict[str, str] = field(
        default_factory=lambda: {
            "macro": (
                "(stock market OR federal reserve OR inflation OR earnings OR "
                "recession OR treasury OR oil prices OR sanctions)"
            ),
            "geopolitics": (
                "(ukraine OR russia OR israel OR hamas OR iran OR china OR "
                "taiwan OR opec OR middle east OR red sea)"
            ),
        }
    )

    def _raw_path(self) -> Path:
        return self.base_dir / "data" / "raw" / "news_google_historical_raw.csv"

    def _processed_path(self) -> Path:
        return self.base_dir / "data" / "processed" / "news_google_historical_processed.csv"

    def _build_url(self, query: str) -> str:
        return (
            "https://news.google.com/rss/search?"
            f"q={quote(query)}&hl={self.lang}&gl={self.region}&ceid={self.edition}"
        )

    @staticmethod
    def _clean_headline(title: str, source: Optional[str]) -> str:
        title = (title or "").strip()
        if source:
            suffix = f" - {source}"
            if title.endswith(suffix):
                return title[: -len(suffix)].strip()
        return title

    def _fetch_query_window(
        self,
        session: requests.Session,
        day: pd.Timestamp,
        query_name: str,
        query_base: str,
    ) -> list[dict]:
        next_day = day + pd.Timedelta(days=1)
        query = f"{query_base} after:{day.date()} before:{next_day.date()}"
        url = self._build_url(query)
        response = session.get(
            url,
            timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)

        rows: list[dict] = []
        for item in root.findall(".//item"):
            title_raw = (item.findtext("title") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            rss_link = (item.findtext("link") or "").strip() or None
            source_node = item.find("source")
            source = None
            source_url = None
            if source_node is not None:
                source = (source_node.text or "").strip() or None
                source_url = source_node.attrib.get("url")

            if not pub_raw or not title_raw:
                continue

            try:
                published_at = parsedate_to_datetime(pub_raw).astimezone(timezone.utc)
            except Exception:
                continue

            published_at = pd.Timestamp(published_at).tz_convert("UTC").tz_localize(None)
            headline = self._clean_headline(title_raw, source)
            if not headline:
                continue

            rows.append(
                {
                    "query_day": day.normalize(),
                    "query_name": query_name,
                    "published_at": published_at,
                    "date": published_at.normalize(),
                    "headline": headline,
                    "source": source,
                    "source_url": source_url,
                    "url": rss_link,
                }
            )
        return rows

    def fetch_raw(self) -> pd.DataFrame:
        """Fetch daily historical RSS windows and keep real publication dates."""
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()
        days = pd.date_range(start, end, freq="D")

        rows: list[dict] = []
        with requests.Session() as session:
            for i, day in enumerate(days, start=1):
                for query_name, query_base in self.queries.items():
                    rows.extend(self._fetch_query_window(session, day, query_name, query_base))
                if self.pause_seconds > 0:
                    sleep(self.pause_seconds)
                if i % 30 == 0 or i == len(days):
                    print(
                        f"  [GoogleNewsHistorical] fetched {i}/{len(days)} windows "
                        f"({len(rows):,} rows before cleaning)."
                    )

        raw = pd.DataFrame(rows)
        if raw.empty:
            return raw

        raw["published_at"] = pd.to_datetime(raw["published_at"], errors="coerce")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        return raw

    def clean(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Clean, validate and deduplicate the raw RSS extension."""
        if raw_df.empty:
            return raw_df.copy()

        df = raw_df.copy()
        start = pd.Timestamp(self.start_date).normalize()
        end = pd.Timestamp(self.end_date).normalize()

        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.normalize()
        df["headline"] = df["headline"].astype(str).str.strip()
        df["source"] = df["source"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})

        df = df.dropna(subset=["published_at", "date", "headline"])
        df = df[df["headline"] != ""]
        df = df[df["date"].between(start, end)]
        df = df.drop_duplicates(subset=["published_at", "headline", "source", "url"])
        df = df.sort_values(["published_at", "headline"]).reset_index(drop=True)
        return df

    def save(self, raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> tuple[Path, Path]:
        """Persist raw and processed extension CSV files."""
        raw_path = self._raw_path()
        processed_path = self._processed_path()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        processed_df.to_csv(processed_path, index=False)
        return raw_path, processed_path

    def run(self) -> pd.DataFrame:
        """Fetch, clean and save the historical extension."""
        print("[GoogleNewsHistorical] Building historical RSS extension...")
        raw_df = self.fetch_raw()
        processed_df = self.clean(raw_df)
        raw_path, processed_path = self.save(raw_df, processed_df)

        if processed_df.empty:
            print("  [GoogleNewsHistorical] No valid historical news retained.")
            return processed_df

        daily_counts = processed_df.groupby("date").size()
        print(
            f"  [GoogleNewsHistorical] Saved {len(processed_df):,} articles "
            f"across {daily_counts.size} days "
            f"({processed_df['date'].min().date()} -> {processed_df['date'].max().date()})."
        )
        print(
            f"  [GoogleNewsHistorical] Articles/day min={daily_counts.min()} "
            f"max={daily_counts.max()} mean={daily_counts.mean():.1f}"
        )
        print(f"  [GoogleNewsHistorical] Raw      -> {raw_path}")
        print(f"  [GoogleNewsHistorical] Processed -> {processed_path}")
        return processed_df
