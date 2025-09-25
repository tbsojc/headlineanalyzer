# backend/app/api/perf.py

from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import text
import hashlib
from datetime import datetime, timezone
import time

# Zentrale Performance-Defaults
MAX_ROWS_PER_REQUEST = 5000       # harter Schutz
DEFAULT_PAGE_SIZE    = 500        # Vorgabe für Pagination
MAX_PAGE_SIZE        = 5000       # Obergrenze pro Seite
_CACHE: dict[str, tuple[float, object]] = {}
DEFAULT_TTL = 60.0


def _parse_int(val, default):
    try:
        return int(val)
    except Exception:
        return default

def pagination_params(request, *, unbounded: bool = False):
    """
    Liefert (limit, offset) für Listen-Endpunkte.
    Wenn unbounded=True, wird (None, None) zurückgegeben – Aggregationen
    dürfen dann die *gesamte* Datenmenge im Zeitfenster sehen.
    """
    if unbounded:
        return None, None

    page = _parse_int(request.query_params.get("page"), 1)
    page_size = _parse_int(request.query_params.get("page_size"), DEFAULT_PAGE_SIZE)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE, MAX_ROWS_PER_REQUEST))
    offset = max(0, (page - 1) * page_size)
    return page_size, offset


def ensure_indexes(db: Session) -> None:
    """
    Legt notwendige Indizes an, falls sie fehlen.
    SQLite: CREATE INDEX IF NOT EXISTS
    Passe Tabellennamen ggf. an eure ORM-Tabellennamen an.
    """
    # Index auf Veröffentlichungszeitpunkt
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_articles_published_at ON articles (published_at)"))
    # Optional: Quelle
    db.execute(text("CREATE INDEX IF NOT EXISTS ix_articles_source ON articles (lower(source))"))
    db.commit()



def make_etag(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"|")
    return '"' + h.hexdigest() + '"'

def http_cache_headers(response, items, start, end):
    """
    Setzt ETag und Last-Modified anhand des Intervalls und der Daten.
    Call im Endpunkt NACH der Ermittlung von items/start/end.
    """
    # Last-Modified: nutze Endzeit (Server-UTC) oder max(published_at)
    lm = end
    try:
        # falls vorhanden: max published_at
        mx = max((a.published_at for a in items if a.published_at), default=None)
        if mx and mx > lm:
            lm = mx
    except Exception:
        pass

    lm_http = lm.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    etag = make_etag(lm.isoformat(), str(len(items)))

    response.headers["ETag"] = etag
    response.headers["Last-Modified"] = lm_http
    response.headers["Cache-Control"] = "public, max-age=60"  # 60s



def cache_get(key: str):
    ent = _CACHE.get(key)
    if not ent:
        return None
    expires, value = ent
    if time.time() > expires:
        _CACHE.pop(key, None)
        return None
    return value

def cache_put(key: str, value: object, ttl: float = DEFAULT_TTL):
    _CACHE[key] = (time.time() + ttl, value)

def make_key(path: str, params: dict) -> str:
    parts = [path] + [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "|".join(parts)
