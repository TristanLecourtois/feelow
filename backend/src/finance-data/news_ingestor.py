"""
Feelow Backend — News Ingestion Module
Yahoo Finance RSS + Finviz fallback.
No Streamlit dependency — uses cachetools.TTLCache.
"""

from __future__ import annotations
import os, sys, logging, time
from datetime import datetime
from typing import List

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RSS_URL_PATTERN, NEWS_CACHE_TTL

logger = logging.getLogger(__name__)

# Module-level cache: key=(ticker,) -> DataFrame, 300s TTL
_news_cache: TTLCache = TTLCache(maxsize=64, ttl=NEWS_CACHE_TTL)


class NewsIngestor:
    """Multi-source news ingestor for financial headlines."""

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.rss_url = RSS_URL_PATTERN.format(ticker=self.ticker)

    @staticmethod
    def _parse_date(entry) -> datetime:
        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    return datetime.fromtimestamp(time.mktime(parsed))
                except Exception:
                    pass
        return datetime.now()

    @staticmethod
    def _clean(title: str) -> str:
        return " ".join((title or "").split())

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=["title", "link", "published", "source"])

    def _fetch_rss(self) -> List[dict]:
        items = []
        try:
            feed = feedparser.parse(self.rss_url)
            for e in feed.entries:
                title = self._clean(getattr(e, "title", ""))
                if title:
                    items.append({
                        "title": title,
                        "link": getattr(e, "link", ""),
                        "published": self._parse_date(e),
                        "source": "Yahoo Finance",
                    })
        except Exception as exc:
            logger.warning(f"RSS error for {self.ticker}: {exc}")
        return items

    def _fetch_finviz(self) -> List[dict]:
        items = []
        try:
            url = f"https://finviz.com/quote.ashx?t={self.ticker}"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.find(id="news-table")
                if table:
                    for row in table.find_all("tr")[:20]:
                        a_tag = row.find("a")
                        if a_tag:
                            title = self._clean(a_tag.text)
                            link = a_tag.get("href", "")
                            items.append({
                                "title": title,
                                "link": link,
                                "published": datetime.now(),
                                "source": "Finviz",
                            })
        except Exception as exc:
            logger.debug(f"Finviz fallback failed for {self.ticker}: {exc}")
        return items

    def fetch_news(self) -> pd.DataFrame:
        cache_key = self.ticker
        if cache_key in _news_cache:
            return _news_cache[cache_key]

        items = self._fetch_rss()
        if len(items) < 5:
            items.extend(self._fetch_finviz())

        if not items:
            return self._empty()

        df = pd.DataFrame(items)
        df["published"] = pd.to_datetime(df["published"])
        df = df.drop_duplicates(subset=["title"]).sort_values("published", ascending=False).reset_index(drop=True)

        _news_cache[cache_key] = df
        return df
