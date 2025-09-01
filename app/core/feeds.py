import http.client
import sqlite3
import feedparser
import datetime
from dateutil import parser as date_parse
from datetime import datetime, timedelta, timezone

from app.models import Article
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
    "Anti‑Spiegel": "https://anti-spiegel.com/feed/",
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
    "Linksnet (Meta-Rechtslinks-Themen)": "https://www.linksnet.de/rss",
    "Norbert Häring": "https://norberthaering.de/feed",
    "Overton‑Magazin (Politik)": "https://overton-magazin.de/feed/politik",
    "Berliner Zeitung (Politik)": "https://www.berliner-zeitung.de/feed.id_politik_und_gesellschaft.xml",
    "der Freitag (Politik)": "https://www.freitag.de/politik/@@RSS",
    "Makroskop": "https://makroskop.eu/feed",
    "Lost in Europe": "https://lostineurope.eu/feed",
    "German Foreign Policy": "https://www.german-foreign-policy.com/feed.xml",
    "Antikrieg": "https://www.antikrieg.de/feed",
    "Infosperber (Politik)": "https://www.infosperber.ch/politik/feed",
    "Legal Tribune Online (Politik)": "https://www.lto.de/recht/rss",
    "TauBlog (Politik)": "https://taublog.de/feed",
    "Jacobin Deutschland": "https://jacobin.de/feed",
    "Multipolar": "https://multipolar-magazin.de/atom-meldungen.xml",
    "Junge Freiheit":"https://jungefreiheit.de/feed/",
    "neues deutschland": "https://www.nd-aktuell.de/rss/politik.xml"
}



def _utc_now():
    return datetime.now(timezone.utc)

def _parse_utc(dt_str: str | None):
    if not dt_str:
        return None
    try:
        dt = date_parse.parse(dt_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _make_teaser(summary: str, limit: int = 200) -> str:
    text = clean_html(summary or "")
    if len(text) <= limit:
        return text
    return (text[:limit]).rstrip() + "…"



def fetch_articles():
    articles = []
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except http.client.IncompleteRead:
            print(f"[ERROR] Feed von {source} konnte nicht vollständig geladen werden.")
            continue
        except Exception as e:
            print(f"[ERROR] Fehler beim Parsen von {source}: {e}")
            continue

        for entry in feed.entries[:25]:
            published_at = _parse_utc(entry.get("published") or entry.get("updated"))
            if not published_at:
                print(f"[INFO] Artikel ohne gültiges Datum übersprungen: {getattr(entry, 'title', '(ohne Titel)')} ({source})")
                continue

            try:
                articles.append(Article(
                    title=entry.title,
                    teaser=_make_teaser(entry.get("summary", "")),
                    url=entry.link,
                    source=source,
                    topic="Sonstiges",  # ehemals classify(...)
                    published_at=published_at
                ))
            except Exception as e:
                print(f"[ERROR] Fehler beim Erstellen eines Artikels: {e}")
                continue
    return articles


def fetch_articles_from_source(source_name: str):
    url = FEEDS.get(source_name)
    if not url:
        return []
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:5]:
        published_at = _parse_utc(entry.get("published")) or _utc_now()
        articles.append(Article(
            title=entry.title,
            teaser=_make_teaser(entry.get("summary", "")),
            url=entry.link,
            source=source_name,
            topic="Sonstiges",
            published_at=published_at
        ))
    return articles



def get_articles_last_hours(hours: int):
    conn = sqlite3.connect("app/db.sqlite3")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    since = _utc_now() - timedelta(hours=hours)
    cursor.execute("""
        SELECT * FROM articles
        WHERE published_at >= ?
        ORDER BY published_at DESC
    """, (since.isoformat(),))

    results = []
    for row in cursor.fetchall():
        data = dict(row)
        try:
            dt = date_parse.parse(data["published_at"])
            data["published_at"] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            data["published_at"] = _utc_now()
        try:
            results.append(Article(**data))
        except Exception as e:
            print(f"[ERROR] Artikel konnte nicht erstellt werden: {e}")
            continue
    conn.close()
    return results
