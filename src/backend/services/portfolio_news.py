"""Bounded historical headline retrieval, plus optional curated JSON archives.

Google News RSS is a best-effort index, not a complete historical news archive.
Only the dated headline is used; publisher pages are never silently treated as
historical snapshots. Network errors must not break portfolio generation.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import os
from pathlib import Path
import re
from xml.etree import ElementTree

import requests

from src.gaengine.news_context import clean_text, news_cutoff, normalize_news_context

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _rss_items(ticker, cutoff, lookback_days):
    query = (f'"{ticker}" saham after:{(cutoff - timedelta(days=lookback_days)).date().isoformat()} '
             f'before:{cutoff.date().isoformat()}')
    response = requests.get("https://news.google.com/rss/search", params={
        "q": query, "hl": "id", "gl": "ID", "ceid": "ID:id"},
        timeout=(3, 6), allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200 or len(response.content) > 2_000_000:
        raise ValueError("Respons RSS tidak sesuai batas.")
    root = ElementTree.fromstring(response.content)
    items = []
    for item in root.findall("./channel/item")[:100]:
        title = clean_text(item.findtext("title"), 220)
        source = clean_text(item.findtext("source"), 80)
        # A ticker must actually appear in the headline, not merely in a broad
        # search result. Do not use RSS descriptions: often just lists of links.
        if not re.search(r"\b" + re.escape(ticker) + r"\b", title, re.I):
            continue
        # Index timestamps on evergreen quote/profile pages are not evidence
        # that their changing contents describe a historical corporate event.
        if (source.casefold() in {"tradingview", "investing.com", "yahoo finance"}
                or re.search(r"analisis teknikal dari|harga saham dan grafik|profil perusahaan|stock price.*quote", title, re.I)):
            continue
        if (re.search(r"analisis.*(?:teknikal|fundamental)|teknikal saham", title, re.I)
                and not re.search(r"naik|turun|menguat|melemah|bullish|bearish|laba|rugi|akuisisi|merger|dividen|asing|kredit", title, re.I)):
            # "Analisis teknikal, perhatikan catatan berikut" contains no event
            # or direction. The model must not invent a sentiment for it.
            continue
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
            if published.tzinfo is None:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        items.append({"ticker": ticker, "title": title, "source": source,
                      "url": item.findtext("link"), "published_at": published.isoformat()})
    return items


def load_portfolio_news(context):
    enabled = os.getenv("PORTFOLIO_NEWS_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return normalize_news_context(context, {"status": "disabled"})
    cutoff = news_cutoff(context)
    if cutoff is None or cutoff > datetime.now(timezone.utc):
        return normalize_news_context(context, {"reason": "Tanggal keputusan belum dapat digunakan untuk pencarian berita."})
    items, failed = [], False
    archive = os.getenv("PORTFOLIO_NEWS_ARCHIVE", "").strip()
    if archive:
        path = Path(archive)
        if not path.is_absolute():
            path = ROOT / path
        try:
            if path.stat().st_size > 10_000_000:
                raise ValueError("Arsip berita terlalu besar.")
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else payload.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Arsip berita harus berisi daftar items.")
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Arsip berita tidak dapat dibaca (%s).", type(exc).__name__)
            failed = True
    result = normalize_news_context(context, {"items": items})
    provider = os.getenv("PORTFOLIO_NEWS_PROVIDER", "google_rss").strip().lower()
    if provider not in {"google_rss", "archive"}:
        return normalize_news_context(context, {"items": items, "reason": "Penyedia berita belum dikonfigurasi dengan benar."})
    if provider == "google_rss":
        # Limit latency and search only selected issuers; prioritize allocations.
        tickers = list(dict.fromkeys(str(row["ticker"]).upper() for row in context.get("emiten_terpilih", [])
                                   if re.fullmatch(r"[A-Z]{4}", str(row["ticker"]).upper())))[:3]
        covered = {row["ticker"] for row in result["items"]}
        tickers = [ticker for ticker in tickers if ticker not in covered]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_rss_items, ticker, cutoff, 90) for ticker in tickers]
            for future in futures:
                try:
                    items.extend(future.result())
                except (requests.RequestException, ValueError, ElementTree.ParseError) as exc:
                    failed = True
                    logger.warning("Pencarian berita dilewati (%s).", type(exc).__name__)
    reason = ("Sebagian sumber berita tidak dapat diakses; cakupan berita belum lengkap." if failed
              else "Tidak ditemukan berita bertanggal yang relevan pada jendela pencarian.")
    result = normalize_news_context(context, {"items": items, "reason": reason})
    result["provider"] = provider
    result["retrieval_partial"] = failed
    return result
