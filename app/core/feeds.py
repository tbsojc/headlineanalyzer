# ============================
# 📁 app/core/feeds.py
# (RSS-Feeds einlesen und als Pydantic-Artikel zurückgeben)

from __future__ import annotations

import http.client
import feedparser
from dateutil import parser as date_parse
from datetime import datetime, timezone
from typing import List

from app.schemas import Article as ArticleSchema
from app.core.clean_utils import clean_html


FEEDS = {
    "Zeit": "https://newsfeed.zeit.de/politik/index",
    "Spiegel": "https://www.spiegel.de/politik/index.rss",
    "Süddeutsche": "https://rss.sueddeutsche.de/rss/Politik",
    "Handelsblatt": "https://www.handelsblatt.com/contentexport/feed/politik",
    "Deutschlandfunk": "https://www.deutschlandfunk.de/politikportal-100.rss",
    "Tagesschau": "https://www.tagesschau.de/inland/innenpolitik/index~rss2.xml",
    "FAZ": "https://www.faz.net/rss/aktuell/",
    "n-tv": "https://www.n-tv.de/rss",
    "Focus Politik": "https://www.focus.de/politik/rss.xml",
    "Welt": "https://www.welt.de/feeds/section/politik.rss",
    "RP Online": "https://rp-online.de/politik/feed.rss",
    "Stern": "https://www.stern.de/feed/standard/politik/",
    "Tichys Einblick": "https://www.tichyseinblick.de/feed/",
    "Reitschuster": "https://reitschuster.de/feed/",
    "Unzensuriert": "https://unzensuriert.at/feed/",
    "Achse des Guten": "https://www.achgut.com/rss2",
    "Anti-Spiegel": "https://anti-spiegel.com/feed/",
    "Telepolis Politik": "https://www.telepolis.de/news-atom.xml",
    "NachDenkSeiten": "https://www.nachdenkseiten.de/?feed=rss2",
    "Volksverpetzer": "https://www.volksverpetzer.de/feed",
    "Krautzone": "https://krautzone.de/feed",
    "Niggemeier/Übermedien": "https://uebermedien.de/feed/",
    "taz": "https://taz.de/!p4608;rss/",
    "junge Welt": "https://www.jungewelt.de/feeds/newsticker.rss",
    "NZZ Deutschland": "https://www.nzz.ch/startseite.rss",
    "Watson Deutschland": "https://www.watson.de/rss",
    "Frankfurter Rundschau": "https://www.fr.de/politik/rssfeed.rdf",
    "Bild": "https://www.bild.de/feed/politik.xml",
    "Nius": "https://www.nius.de/rss",
    "Deutsche Wirtschaftsnachrichten": "https://deutsche-wirtschafts-nachrichten.de/feed",
    "Manova": "https://www.manova.news/artikel.atom",
    "Auf1": "https://auf1.tv/feed",
    "RT DE": "https://de.rt.com/rss/",
    "Correctiv": "https://correctiv.org/feed/",
    "Netzpolitik.org": "https://netzpolitik.org/feed/",
    "Norbert Häring": "https://norberthaering.de/feed",
    "Overton-Magazin (Politik)": "https://overton-magazin.de/feed/?cat=2222,2398",
    "Berliner Zeitung (Politik)": "https://www.berliner-zeitung.de/feed.id_politik_und_gesellschaft.xml",
    "der Freitag (Politik)": "https://www.freitag.de/politik/@@RSS",
    "Makroskop": "https://makroskop.eu/feed",
    "German Foreign Policy": "https://www.german-foreign-policy.com/feed.xml",
    "Infosperber (Politik)": "https://infosperber.ch/shoutemfeed?category=5",
    "TauBlog (Politik)": "https://www.taublog.de/feed",
    "Jacobin Deutschland": "https://jacobin.de/rss.xml",
    "Multipolar": "https://multipolar-magazin.de/atom-meldungen.xml",
    "Junge Freiheit": "https://jungefreiheit.de/feed/",
    "neues deutschland": "https://www.nd-aktuell.de/rss/politik.xml",
    "Blackout News": "https://blackout-news.de/feed",
    "Cicero": "http://www.cicero.de/rss.xml",
    "rnd": "https://www.rnd.de/arc/outboundfeeds/rss/category/politik/",
    "Alexander Wallasch": "https://www.alexander-wallasch.de/share/rss-feed.xml",
    "Compact": "https://www.compact-online.de/feed/",
    "Epoch Times": "https://www.epochtimes.de/rss",
    "Blätter für deutsche und internationale Politik": "https://www.blaetter.de/rss.xml",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(dt_str: str | None) -> datetime | None:
    """Versucht, ein Datumsstring tz-aware zu parsen. Fällt sonst auf UTC-Jetzt zurück."""
    if not dt_str:
        return None
    try:
        dt = date_parse.parse(dt_str)
        if dt is None:
            return None
        # Wenn ohne tzinfo -> als UTC interpretieren
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _make_teaser(summary: str, limit: int = 200) -> str:
    text = clean_html(summary or "")
    if len(text) <= limit:
        return text
    return (text[:limit]).rstrip() + "…"


def _entry_published(entry) -> datetime | None:
    """Robustes Published-Datum aus einem feedparser-Entry ermitteln."""
    # Reihenfolge der Versuche
    for key in ("published", "updated", "pubDate"):
        dt = _parse_utc(entry.get(key))
        if dt:
            return dt
    # Einige Feeds haben strukturiertes Datum (z. B. published_parsed)
    try:
        pp = entry.get("published_parsed") or entry.get("updated_parsed")
        if pp:
            # time.struct_time -> naive datetime -> UTC
            return datetime(*pp[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def _to_article_schema(entry, source_name: str) -> ArticleSchema | None:
    """Konvertiert ein feedparser-Entry in ein ArticleSchema oder None, wenn unvollständig."""
    published_at = _entry_published(entry) or _utc_now()
    title = getattr(entry, "title", None)
    url = getattr(entry, "link", None)

    if not title or not url:
        # ohne Basisdaten überspringen
        return None

    return ArticleSchema(
        title=title,
        teaser=_make_teaser(entry.get("summary", "")),
        url=url,
        source=source_name,
        topic="Sonstiges",
        published_at=published_at,
    )


def fetch_articles(limit_per_feed: int = 250) -> List[ArticleSchema]:
    """
    Lädt alle FEEDS und gibt eine Liste von ArticleSchema zurück.
    Fehlerhafte Feeds/Einträge werden geloggt und übersprungen.
    """
    items: List[ArticleSchema] = []

    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except http.client.IncompleteRead:
            print(f"[ERROR] Feed von {source} konnte nicht vollständig geladen werden.")
            continue
        except Exception as e:
            print(f"[ERROR] Fehler beim Parsen von {source}: {e}")
            continue

        for entry in feed.entries[:limit_per_feed]:
            art = _to_article_schema(entry, source)
            if art is None:
                print(f"[INFO] Ungültiger Artikel übersprungen ({source})")
                continue
            items.append(art)

    return items


def fetch_articles_from_source(source_name: str, limit: int = 25) -> List[ArticleSchema]:
    """
    Lädt einen einzelnen Feed (per Name in FEEDS) und gibt ArticleSchema-Objekte zurück.
    """
    url = FEEDS.get(source_name)
    if not url:
        return []

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"[ERROR] Fehler beim Parsen von {source_name}: {e}")
        return []

    items: List[ArticleSchema] = []
    for entry in feed.entries[:limit]:
        art = _to_article_schema(entry, source_name)
        if art is None:
            continue
        items.append(art)

    return items
