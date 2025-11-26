# ============================
# 📁 app/api/routes.py
# (hier sammelst du alle Endpunkte)

from typing import Optional
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import re
import pandas as pd

from fastapi import APIRouter, Query, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

# Bestehender In-Memory/Model-Store (bewusst noch genutzt für einige Endpunkte)
from app.models import get_articles

# Feeds einlesen (liefert Artikelobjekte; Speicherung jetzt via Repository)
from app.core.feeds import fetch_articles, fetch_articles_from_source, EXCLUDED_WORDS

# ORM/Repository – zentrale DB-Zugriffe
from app.database import get_db
from app.repositories.articles import get_articles_last_hours, bulk_upsert_articles, get_articles_between
from app.schemas import Article as ArticleSchema
from app.api.date_range_spec import (
    decide_time_window, bucket_for_range, ensure_utc,
    iter_slots, round_to_bucket, previous_window
)
from app.api.perf import ensure_indexes, MAX_ROWS_PER_REQUEST, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, pagination_params
from math import ceil
from fastapi import Response



router = APIRouter()

MEDIA_CSV_PATH = Path("app/static/Medien-Kompass__Systemn_he_.csv")



@lru_cache(maxsize=1)
def load_media_df() -> pd.DataFrame:
    df = pd.read_csv(MEDIA_CSV_PATH)
    df["norm_name"] = df["Medium"].str.strip().str.lower()
    return df


def norm_source(name: str) -> str:
    return (name or "").strip().lower()

def _time_window_or_400(hours: int | None, from_: str | None, to: str | None):
    try:
        return decide_time_window(hours, from_, to)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def filter_words(words):
    return [w for w in words if w.lower() not in EXCLUDED_WORDS]


def _select_articles(
    db: Session,
    hours: int | None,
    from_: str | None,
    to: str | None,
    *,
    unbounded: bool = False,          # <- Schalter
    order: str = "desc",              # "desc" für Listen, "asc" für Aggregationen ok
    request: Request | None = None,   # für page/page_size bei Listen
):
    tw = _time_window_or_400(hours, from_, to)
    if tw["mode"] == "range":
        start = ensure_utc(tw["from"])
        end   = ensure_utc(tw["to"])
        bucket = bucket_for_range(start, end)
        mode = "range"
    else:
        hours_eff = int(tw["hours"])
        end   = datetime.now(timezone.utc).replace(microsecond=0)
        start = end - timedelta(hours=hours_eff)
        bucket = "hour"
        mode = "hours"

    # Pagination nur für Listen
    limit, offset = pagination_params(request, unbounded=unbounded) if request else (None, None)

    articles = get_articles_between(db, start, end, limit=limit, offset=offset)

    if order == "asc":
        articles = list(sorted(articles, key=lambda a: a.published_at))

    return articles, start, end, mode, bucket




# -----------------------------
# Themen-Übersicht (nutzt noch get_articles)
# -----------------------------
@router.get("/topics")
def get_topics():
    articles = get_articles()
    topics = {}
    for art in articles:
        topics.setdefault(art.topic, 0)
        topics[art.topic] += 1
    return sorted(topics.items(), key=lambda x: x[1], reverse=True)


# === Länder-Helfer ============================================================

@lru_cache(maxsize=1)
def country_map() -> dict[str, str]:
    """
    Mappt normierte Mediennamen -> Ländercode (z.B. "zeit" -> "DE").
    Falls die CSV-Spalte 'Land' fehlt, wird 'UN' (unknown) gesetzt.
    """
    df = load_media_df()
    col = "Land" if "Land" in df.columns else None
    if not col:
        df["Land"] = "UN"
    return dict(zip(df["norm_name"], df["Land"].fillna("UN").astype(str).str.strip().str.upper()))

@lru_cache(maxsize=1)
def available_countries() -> list[str]:
    df = load_media_df()
    if "Land" not in df.columns:
        return []
    vals = df["Land"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    vals.sort()
    return vals



# -----------------------------
# Medienkompass (CSV)
# -----------------------------
@router.get("/media-positions")
def media_positions(request: Request):
    # Datei-Status für Caching
    stat = MEDIA_CSV_PATH.stat()
    mtime = stat.st_mtime  # Unix seconds
    last_mod_http = datetime.utcfromtimestamp(mtime).strftime("%a, %d %b %Y %H:%M:%S GMT")
    etag = f'W/"media-csv-{int(mtime)}-{stat.st_size}"'  # schwaches ETag reicht

    # Conditional GET: 304, wenn Client frisch hat
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")
    if (inm and inm == etag) or (ims and ims == last_mod_http):
        return Response(status_code=304)

    # Daten bauen (wie bisher)
    df = load_media_df()
    data = [
        {"medium": row["Medium"], "x": row["Systemnähe (X)"], "y": row["Globalismus (Y)"]}
        for _, row in df.iterrows()
    ]

    resp = JSONResponse(content=jsonable_encoder(data))
    # 24h Cache + Revalidation erlaubt
    resp.headers["Cache-Control"] = "public, max-age=86400, must-revalidate"
    resp.headers["ETag"] = etag
    resp.headers["Last-Modified"] = last_mod_http
    return resp



@router.get("/media-positions/by-keyword")
def media_positions_by_keyword(
    word: str,
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: Optional[str] = Query(None),
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    keyword = word.lower()
    pattern = rf"\b{re.escape(keyword)}\b"

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )

    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    # optional Quelle filtern
    if source:
        s = norm_source(source)
        articles = [a for a in articles if norm_source(a.source) == s]

    # Keyword-Match auf Titel/Teaser
    articles = [
        a
        for a in articles
        if (
            re.search(pattern, a.title.lower())
            or (teaser and a.teaser and re.search(pattern, a.teaser.lower()))
        )
    ]

    # Häufigkeiten je Medium
    from collections import Counter

    source_counts = Counter(norm_source(a.source) for a in articles)

    # Medienkompass joinen
    df = load_media_df()
    filtered = df[df["norm_name"].isin(source_counts.keys())]

    data = [
        {
            "medium": row["Medium"],
            "x": row["Systemnähe (X)"],
            "y": row["Globalismus (Y)"],
            "count": source_counts[row["norm_name"]],
        }
        for _, row in filtered.iterrows()
    ]
    return JSONResponse(content=jsonable_encoder(data))


# -----------------------------
# Headlines/Artikel (einige Endpunkte weiterhin auf get_articles)
# -----------------------------
@router.get("/articles")
def articles_by_topic(topic: str):
    return [a.dict() for a in get_articles() if a.topic == topic]


@router.get("/headlines")
def all_headlines(
    source: str | None = None,
    after: str | None = None,
):
    articles = get_articles()

    if source:
        s = norm_source(source)
        articles = [a for a in articles if norm_source(a.source) == s]

    if after:
        try:
            after_dt = datetime.fromisoformat(after)
            if after_dt.tzinfo is None:
                after_dt = after_dt.replace(tzinfo=timezone.utc)
            else:
                after_dt = after_dt.astimezone(timezone.utc)
            articles = [a for a in articles if a.published_at >= after_dt]
        except Exception:
            pass

    return JSONResponse(content=jsonable_encoder(articles))


def keyword_match(text, keyword):
    return re.search(rf"\b{re.escape(keyword.lower())}\b", text.lower()) is not None


@router.get("/headlines/by-topic")
def headlines_by_topic(topic: str):
    return [
        {"title": a.title, "url": a.url}
        for a in get_articles()
        if a.topic.lower() == topic.lower()
    ]


@router.get("/headlines/by-keyword")
def headlines_by_keyword(word: str):
    return [
        {
            "title": a.title,
            "url": a.url,
            "source": a.source,
            "published_at": a.published_at,
            "topic": a.topic,
        }
        for a in get_articles()
        if keyword_match(a.title, word) or keyword_match(a.teaser, word)
    ]


@router.get("/headlines/by-keyword-and-source")
def headlines_by_keyword_and_source(word: str, source: str):
    keyword = word.lower()
    source = source.strip().lower()
    articles = get_articles()

    pattern = rf"\b{re.escape(keyword)}\b"

    filtered = [
        a.dict()
        for a in articles
        if (
            a.source.strip().lower() == source
            and (re.search(pattern, a.title.lower()) or re.search(pattern, a.teaser.lower()))
        )
    ]
    return JSONResponse(filtered)


@router.get("/headlines/by-source")
def headlines_by_source(source: str):
    source = source.strip().lower()
    articles = get_articles()

    filtered = [
        a.dict()
        for a in articles
        if a.source.strip().lower() == source
    ]
    return JSONResponse(filtered)


# -----------------------------
# Feeds aktualisieren (jetzt via Repository speichern)
# -----------------------------
@router.get("/refresh")
def refresh(db: Session = Depends(get_db)):
    ensure_indexes(db)  # Indexe sicherstellen
    items = fetch_articles()
    count = bulk_upsert_articles(db, items)
    return {"status": "updated", "count": count}

@router.get("/fetch")
def fetch_source(source: str = Query(...), db: Session = Depends(get_db)):
    ensure_indexes(db)  # Indexe sicherstellen
    items = fetch_articles_from_source(source)
    count = bulk_upsert_articles(db, items)
    return {
        "source": source,
        "count": count,
        "articles": [a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in items],
    }


# -----------------------------
# Gefilterte Artikel (ORM + Repository)
# -----------------------------
@router.get("/articles/filtered", response_model=list[ArticleSchema])
def filtered_articles(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: list[str] | None = Query(None),   # ← Liste statt str
    keyword: str | None = Query(None),
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1),
    db: Session = Depends(get_db),
    request: Request = None,
):
    arts, start, end, mode, bucket = _select_articles(
        db, hours, from_, to,
        unbounded=True, order="desc", request=request
    )
    articles = arts

    articles = filter_articles_by_countries(articles, country)

    # Quellen-Filter (Mehrfach): akzeptiert mehrere ?source=… Vorkommen
    if source:
        wanted = {s.strip().lower() for s in source if s and s.strip()}
        if wanted:
            # Map Label→norm_name zulassen (wie zuvor)
            df = load_media_df()
            label2norm = dict(zip(df["Medium"].str.strip().str.lower(), df["norm_name"]))
            valid = wanted | {label2norm.get(s, s) for s in wanted}
            articles = [a for a in articles if (a.source or "").strip().lower() in valid]

    # Keyword (wie gehabt) …
    if keyword:
        import re
        terms = [t.strip().lower() for t in keyword.split('+') if t.strip()]
        if terms:
            patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

            def match(a):
                title = (a.title or "").lower()
                te = (a.teaser or "").lower() if (teaser and a.teaser) else ""
                txt = title + (" " + te if te else "")
                # Alle Terme müssen vorkommen (Titel ODER optional Teaser)
                return all(p.search(txt) for p in patterns)

            articles = [a for a in articles if match(a)]
    elif teaser:
        articles = [a for a in articles if a.teaser]

    return articles



# -----------------------------
# Medienkompass gefiltert (ORM + Repository)
# -----------------------------
@router.get("/media-positions/filtered")
def media_positions_filtered(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: list[str] | None = Query(None),   # Mehrfachquellen
    keyword: str = Query(None),
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
):
    from collections import Counter
    import re

    # Zeitfenster + Artikel holen
    tw = _time_window_or_400(hours, from_, to)
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    # Quellenfilter (Mehrfach)
    if source:
        wanted = { (s or "").strip().lower() for s in source if s and s.strip() }
        if wanted:
            articles = [a for a in articles if ((a.source or "").strip().lower() in wanted)]

    # Keyword-Filter: Unterstützung für "term1+term2"
    if keyword:
        terms = [t.strip().lower() for t in keyword.split('+') if t.strip()]
        patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

        def match(a):
            txt = (a.title or "").lower()
            if teaser and a.teaser:
                txt += " " + (a.teaser or "").lower()
            return all(p.search(txt) for p in patterns)

        articles = [a for a in articles if match(a)]

    # Häufigkeiten pro Medium sammeln
    matching_sources = [ (a.source or "").strip().lower() for a in articles if a.source ]
    source_counts = Counter(matching_sources)

    # ⚠️ Hier fehlte das DataFrame:
    df = load_media_df()  # <-- hinzufügen
    filtered = df[df["norm_name"].isin(source_counts.keys())]

    # Rückgabeformat für den Scatter
    return [
        {
            "medium": row["Medium"],
            "x": row["Systemnähe (X)"],
            "y": row["Globalismus (Y)"],
            "count": int(source_counts.get(row["norm_name"], 0)),
        }
        for _, row in filtered.iterrows()
    ]


# -----------------------------
# Keyword-Analysen (alle via Repository/Session)
# -----------------------------
@router.get("/keywords/trending")
def keyword_trends(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: list[str] | None = Query(None),
    ngram: int = Query(1, ge=1, le=3),
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """
    Dynamische Keyword-Trends:
    - Basis ist das Ende des gewählten Zeitfensters (Date-Range oder 'jetzt')
    - Für jede Stufe (24h, 72h, 7d, 30d):
        * aktueller Zeitraum vs. vorheriger gleicher Zeitraum
        * Filter: country, source, ngram, teaser
    - Aufsteiger: starke relative Zunahme
    - Absteiger: einfach die stärksten Verluste (Delta am negativsten)
    """
    from collections import Counter
    from itertools import combinations
    from datetime import datetime, timedelta, timezone
    from app.core.clean_utils import extract_relevant_words

    # 1) Zeitfenster bestimmen
    tw = _time_window_or_400(hours, from_, to)

    # base_to: immer ein UTC-aware datetime
    if tw["mode"] == "range":
        base_to = ensure_utc(tw["to"])
    else:
        # Stundenmodus -> "jetzt" in UTC
        base_to = datetime.now(timezone.utc).replace(microsecond=0)

    # Sicherheit: base_to auf UTC normalisieren
    if base_to.tzinfo is None:
        base_to = base_to.replace(tzinfo=timezone.utc)
    else:
        base_to = base_to.astimezone(timezone.utc)

    timeframes = {
        "24h": 24,
        "72h": 72,
        "7d": 24 * 7,
        "30d": 24 * 30,
    }

    def norm_source(x: str | None) -> str:
        return (x or "").strip().lower()

    # Einheiten (1er/2er/3er-Kombis) pro Artikel
    def units_from_article(a):
        tokens = set(extract_relevant_words(a.title or ""))
        if teaser and getattr(a, "teaser", None):
            tokens |= set(extract_relevant_words(a.teaser or ""))
        toks = sorted(tokens)
        if not toks:
            return set()
        if ngram == 1:
            return set(toks)
        return {" + ".join(c) for c in combinations(toks, ngram)}

    # Hilfssfunktion: beliebiges datetime nach UTC normalisieren
    def to_utc(dt_raw):
        if not isinstance(dt_raw, datetime):
            return None
        if dt_raw.tzinfo is None:
            return dt_raw.replace(tzinfo=timezone.utc)
        return dt_raw.astimezone(timezone.utc)

    result: dict[str, dict[str, list[dict]]] = {}

    for label, h in timeframes.items():
        # 2h-Fenster (vorher + aktuell) in einem Rutsch holen
        window_end = base_to
        window_start = base_to - timedelta(hours=h * 2)

        articles, start, end, mode, bucket = _select_articles(
            db,
            hours=None,
            from_=window_start.isoformat(),
            to=window_end.isoformat(),
            unbounded=True,
            order="asc",
            request=request,
        )

        # Länderfilter
        articles = filter_articles_by_countries(articles, country)

        # Quellenfilter (Mehrfachauswahl, UI-Label ODER norm_name)
        if source:
            # Rohwerte aus der Query
            raw = {(s or "").strip().lower() for s in source if s and s.strip()}
            if raw:
                # Mapping aus Medienkompass laden
                df = load_media_df()
                label2norm = dict(zip(
                    df["Medium"].str.strip().str.lower(),
                    df["norm_name"]          # normierter Name in der CSV/DB
                ))

                # Ziel-Menge: erlaubte normierte Namen
                wanted = set()
                for s in raw:
                    wanted.add(label2norm.get(s, s))  # Label -> norm_name, sonst roh

                articles = [
                    a for a in articles
                    if norm_source(getattr(a, "source", None)) in wanted
                ]


        now_words = Counter()
        past_words = Counter()

        cutoff = base_to - timedelta(hours=h)
        # cutoff ebenfalls explizit UTC
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        else:
            cutoff = cutoff.astimezone(timezone.utc)

        for a in articles:
            dt = getattr(a, "published_at", None) or getattr(a, "created_at", None)
            dt = to_utc(dt)
            if not dt:
                continue

            units = units_from_article(a)
            if not units:
                continue

            if dt < cutoff:
                past_words.update(units)
            else:
                now_words.update(units)

        # Wenn nichts da ist, leere Listen liefern
        if not now_words and not past_words:
            result[label] = {"top": [], "flop": []}
            continue

        # Veränderungen berechnen
        changes = []
        all_terms = set(now_words.keys()) | set(past_words.keys())
        for term in all_terms:
            now_c = now_words.get(term, 0)
            prev_c = past_words.get(term, 0)
            if now_c == 0 and prev_c == 0:
                continue
            delta = now_c - prev_c
            if prev_c > 0:
                rel_change = delta / prev_c
            else:
                # neu aufgetaucht
                rel_change = float("inf") if now_c > 0 else 0.0
            changes.append((term, delta, rel_change, now_c, prev_c))

        # Aufsteiger: wie vorher (rel. Veränderung, dann absolute Häufigkeit)
        changes_sorted_top = sorted(
            changes,
            key=lambda x: (x[2], x[3]),  # rel_change, dann now_c
            reverse=True,
        )
        top = changes_sorted_top[:5]

        # Absteiger: ganz simpel – stärkster Verlust (Delta am negativsten)
        changes_sorted_flop = sorted(
            changes,
            key=lambda x: x[1],  # Delta
        )
        flop = [c for c in changes_sorted_flop if c[1] < 0][:5]

        def serialize(entries):
            out = []
            for w, delta, rel_change, now_c, prev_c in entries:
                # relative Änderung in Prozent; "neu" wenn kein Vorwert
                if prev_c == 0 or rel_change in (float("inf"), float("-inf")):
                    pct = None   # Frontend zeigt "neu"
                else:
                    pct = round(rel_change * 100.0, 2)

                out.append(
                    {
                        "word": w,
                        "delta": int(delta),
                        "change_pct": pct,   # ⬅ wichtig: Feldname
                        "now": int(now_c),
                        "prev": int(prev_c),
                    }
                )
            return out


        result[label] = {
            "top": serialize(top),
            "flop": serialize(flop),
        }

    return result


@router.get("/keywords/extreme-bubble")
def extreme_keywords(
    hours: int = 72,
    country: list[str] | None = Query(None),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    from collections import defaultdict, Counter
    from app.core.clean_utils import extract_relevant_words

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    # Medienkompass laden
    df = load_media_df()
    bias_map = dict(zip(df["norm_name"], df["Systemnähe (X)"]))


    words_extreme = Counter()
    words_other = Counter()
    word_sources = defaultdict(list)

    # Schwelle X-Achse (hier aktuell >= 0.0, ggf. anpassen)
    for a in articles:
        words = extract_relevant_words(a.title)
        norm = a.source.strip().lower()
        if norm not in bias_map:
            continue

        score = bias_map[norm]
        if abs(score) >= 0.0:
            words_extreme.update(words)
            for w in words:
                word_sources[w].append(score)
        else:
            words_other.update(words)

    result = []
    for w, count in words_extreme.items():
        if w in words_other or count < 3:
            continue
        scores = word_sources.get(w, [])
        if scores:
            avg_score = sum(scores) / len(scores)
            result.append((w, count, round(avg_score, 3)))

    return sorted(result, key=lambda x: abs(x[2]), reverse=True)


@router.get("/keywords/bias-score")
def keyword_bias_scores(
    hours: int = 72,
    country: list[str] | None = Query(None),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    from collections import defaultdict
    from app.core.clean_utils import extract_relevant_words

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    df = load_media_df()
    media_bias = dict(zip(df["norm_name"], df["Systemnähe (X)"]))

    keyword_positions = defaultdict(list)

    for a in articles:
        norm = a.source.strip().lower()
        if norm not in media_bias:
            continue
        bias_score = media_bias[norm]
        words = extract_relevant_words(a.title)
        for w in words:
            keyword_positions[w].append(bias_score)

    result = {}
    for w, scores in keyword_positions.items():
        if len(scores) >= 3:
            avg = sum(scores) / len(scores)
            result[w] = round(avg, 3)

    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))


@router.get("/keywords/bias-vector")
def keyword_bias_vector(
    hours: int = 72,
    country: list[str] | None = Query(None),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    from collections import defaultdict
    from app.core.clean_utils import extract_relevant_words



    df = load_media_df()
    bias_map = {
        row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
        for _, row in df.iterrows()
    }

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    keyword_coords = defaultdict(list)

    for a in articles:
        norm = a.source.strip().lower()
        if norm not in bias_map:
            continue
        x, y = bias_map[norm]
        words = extract_relevant_words(a.title)
        for w in words:
            keyword_coords[w].append((x, y))

    result = {}
    for w, coords in keyword_coords.items():
        if len(coords) < 3:
            continue
        avg_x = sum(x for x, _ in coords) / len(coords)
        avg_y = sum(y for _, y in coords) / len(coords)
        result[w] = {"x": round(avg_x, 3), "y": round(avg_y, 3)}

    return result

@router.get("/keywords/timeline")
def keyword_timeline(
    word: str,
    country: list[str] | None = Query(None),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    teaser: bool = Query(False),
    source: list[str] | None = Query(None),
    db: Session = Depends(get_db),
):
    try:
        from datetime import datetime, timedelta, timezone
        from collections import defaultdict
        import re, traceback
        from fastapi import HTTPException
        from app.core.clean_utils import extract_relevant_words

        # ---- Datumsparser (ISO mit/ohne TZ, oder YYYY-MM-DD) ----
        def parse_any_dt(s: str | None):
            if not s:
                return None
            s = s.strip()
            try:
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                dt = datetime.fromisoformat(s)  # mit/ohne TZ
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except Exception:
                pass
            try:
                d = datetime.strptime(s[:10], "%Y-%m-%d")
                return d.replace(tzinfo=timezone.utc)
            except Exception:
                raise HTTPException(status_code=400, detail={"where": "parse_dates", "error": f"Invalid date/datetime: {s}"})

        now = datetime.now(timezone.utc).replace(microsecond=0)
        to_dt = parse_any_dt(to) or now
        from_dt = parse_any_dt(from_) or (to_dt - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        if to_dt <= from_dt:
            to_dt = from_dt + timedelta(days=1)

        # ---- Artikel laden (nutzt euren Helper in DIESER Datei) ----
        try:
            # Signatur wie in euren anderen Stellen: hours=None, from_, to, unbounded=True, order="asc"
            articles, start, end, _mode, _bucket = _select_articles(
                db,
                hours=None,
                from_=from_dt.isoformat(),
                to=to_dt.isoformat(),  # exklusiv
                unbounded=True,
                order="asc",
            )

            articles = filter_articles_by_countries(articles, country)

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            raise HTTPException(status_code=400, detail={"where": "_select_articles", "error": str(e), "trace": tb})

        # ---- Optional: Quellen filtern ----
        if source:
            wanted = {s.strip().lower() for s in source if s and s.strip()}
            if wanted:
                def norm(x): return (x or "").strip().lower()
                articles = [a for a in articles if norm(getattr(a, "source", None) or getattr(a, "medium", None)) in wanted]

        # ---- Suchterme vorbereiten ('a+b' = UND) ----
        raw = (word or "").strip().lower()
        terms = [t for t in (p.strip() for p in raw.split("+")) if t]
        if not terms:
            return []
        patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

        # ---- Zählen pro Tag (Regex ODER Token-Match) ----
        counts = defaultdict(int)

        for a in articles:
            d = getattr(a, "published_at", None) or getattr(a, "created_at", None)
            if not d:
                continue
            if isinstance(d, datetime):
                d = d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
            else:
                d = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

            if not (from_dt <= d < to_dt):
                continue

            title_l  = (getattr(a, "title", "") or "").lower()
            teaser_l = (getattr(a, "teaser", "") or "").lower() if (teaser and getattr(a, "teaser", None)) else ""

            try:
                toks_title  = set(extract_relevant_words(title_l))
                toks_teaser = set(extract_relevant_words(teaser_l)) if (teaser and teaser_l) else set()
            except Exception:
                toks_title, toks_teaser = set(), set()

            ok = True
            for i, term in enumerate(terms):
                if not (
                    patterns[i].search(title_l)
                    or (teaser and patterns[i].search(teaser_l))
                    or (term in toks_title)
                    or (term in toks_teaser)
                ):
                    ok = False
                    break
            if ok:
                counts[d.date()] += 1

        # ---- lückenlose Tagesachse ----
        first_day = from_dt.date()
        last_day  = (to_dt - timedelta(microseconds=1)).date()
        out = []
        cur = first_day
        while cur <= last_day:
            out.append({"time": f"{cur.isoformat()}T00:00:00Z", "count": int(counts.get(cur, 0))})
            cur += timedelta(days=1)

        return out

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=400, detail={"where": "keyword_timeline", "error": str(e), "trace": tb})



@router.get("/keywords/top-absolute")
def keywords_top_absolute(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    ngram: int = Query(1, ge=1, le=3),
    teaser: bool = Query(False),
    compare_prev: bool = Query(False),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    source: list[str] | None = Query(None),
    top_n: int = Query(25, ge=1, le=200),
    spark_days: int = Query(0, ge=0, le=30),   # ← NEU: 0 = aus, 7 = letzte 7 Tage
    db: Session = Depends(get_db),
    request: Request = None,
):
    from collections import Counter, defaultdict
    from itertools import combinations
    from datetime import timedelta
    from app.core.clean_utils import extract_relevant_words

    # 1) Zeitfenster & Artikel (wie gehabt)
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)
    articles = filter_articles_by_countries(articles, country)

    # Quelle(n) filtern
    source_set = None
    if source:
      source_set = { (s or "").strip().lower() for s in source }
      def norm(x): return (x or "").strip().lower()
      articles = [a for a in articles if norm(getattr(a, "source", None)) in source_set]

    # Ngram-Units je Artikel
    def units_from_article(a):
      tokens = set(extract_relevant_words(a.title or ""))
      if teaser and getattr(a, "teaser", None):
        tokens |= set(extract_relevant_words(a.teaser or ""))
      toks = sorted(tokens)
      if ngram == 1:
        return set(toks)
      return {" + ".join(c) for c in combinations(toks, ngram)}

    # 2) Zählen: aktuelle und (optional) vorige Periode
    now_words = Counter()
    for a in articles:
      now_words.update(units_from_article(a))

    past_words = Counter()
    if compare_prev:
      p_start, p_end = previous_window(start, end)
      prev_articles = get_articles_between(db, p_start, p_end)
      if source_set:
        prev_articles = [
          a for a in prev_articles
          if ((getattr(a, "source", "") or "").strip().lower() in source_set)
        ]
      for a in prev_articles:
        past_words.update(units_from_article(a))

    # 3) Ränge
    rank_current = {term: i+1 for i, (term, _) in enumerate(now_words.most_common())}
    rank_prev    = {term: i+1 for i, (term, _) in enumerate(past_words.most_common())}

    # 4) Top N Ergebnis (ohne Spark)
    top_terms = [term for term, _ in now_words.most_common(top_n)]
    result = []
    for term in top_terms:
      current_count = now_words.get(term, 0)
      prev_count = past_words.get(term, 0)
      delta = current_count - prev_count
      change_pct = (delta / prev_count * 100.0) if prev_count > 0 else (100.0 if current_count>0 and prev_count==0 else 0.0)
      result.append({
        "word": term,
        "current": int(current_count),
        "previous": int(prev_count),
        "delta": int(delta),
        "change_pct": round(change_pct, 2),
        "rank_current": rank_current.get(term),
        "rank_prev":    rank_prev.get(term),
      })

    # 5) Optional: 7-Tage Sparkline für die Top-Begriffe
    if spark_days and top_terms:
      # Fenster: letzte spark_days bis zum aktuellen 'end'
      spark_end = end
      spark_start = (spark_end - timedelta(days=spark_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
      )
      spark_articles = get_articles_between(db, spark_start, spark_end)
      if country:
        spark_articles = filter_articles_by_countries(spark_articles, country)
      if source_set:
        spark_articles = [
          a for a in spark_articles
          if ((getattr(a, "source", "") or "").strip().lower() in source_set)
        ]

      # Tag-Index vorbereiten
      day_index = lambda dt: (dt.date() - spark_start.date()).days
      series = {t: [0]*spark_days for t in top_terms}

      for a in spark_articles:
        try:
          idx = day_index(getattr(a, "published_at"))
        except Exception:
          continue
        if not (0 <= idx < spark_days):
          continue
        units = units_from_article(a)
        # Nur Top-Begriffe zählen
        for u in units:
          if u in series:
            series[u][idx] += 1

      # In Ergebnis mappen
      lookup = {row["word"]: row for row in result}
      for term, vals in series.items():
        # Sicherheit: immer Länge spark_days liefern
        lookup[term]["spark"] = list(vals)

    return result



@router.get("/headlines/words")
def headlines_words(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: list[str] | None = Query(None),    # ← Liste statt str
    keyword: str | None = Query(None),
    teaser: bool = Query(False),
    ngram: int = Query(1, ge=1, le=3),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
    request: Request = None
):
    from collections import Counter
    from itertools import combinations
    from app.core.clean_utils import extract_relevant_words

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )

    arts, start, end, mode, bucket = _select_articles(db, hours, from_, to, unbounded=True, order="asc", request=request)
    arts = filter_articles_by_countries(arts, country)
    # ... Rest unverändert ...
    articles = arts


    # Quelle filtern
    if source:
        wanted = {s.strip().lower() for s in source if s and s.strip()}
        if wanted:
            arts = [a for a in arts if (a.source or "").strip().lower() in wanted]


    # optional: Keyword-Filter (wie /articles/filtered)
    import re
    if keyword:
        # Mehrfachbegriffe mit '+' (ukraine + frieden) → alle müssen matchen
        terms = [t.strip().lower() for t in keyword.split('+') if t.strip()]
        def match(a):
            txt = a.title.lower()
            if teaser and a.teaser:
                txt += " " + a.teaser.lower()
            return all(re.search(rf"\b{re.escape(t)}\b", txt) for t in terms)
        arts = [a for a in arts if match(a)]

    # Zählen: pro Headline jede Einheit nur 1×
    c = Counter()
    for a in arts:
        tokens = list(extract_relevant_words(a.title))  # Stopwords bleiben draußen
        if teaser and a.teaser:
            # optional auch Teaser berücksichtigen, falls du es möchtest:
            # tokens += list(extract_relevant_words(a.teaser))
            pass

        units: set[str]
        if ngram == 1:
            units = set(tokens)
        else:
            # Reihenfolgeunabhängige Kombis: alphabetisch sortieren und mit " + " verbinden
            combos = set(" + ".join(sorted(tup)) for tup in combinations(tokens, ngram))
            units = combos

        c.update(units)

    # Array-Form für die UI
    return sorted([[w, n] for w, n in c.items()], key=lambda x: x[1], reverse=True)


@router.get("/keywords/sides")
def keyword_sides(
    word: str,
    hours: int = Query(72),
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    """
    Verteilung (X/Y) für 1..3 Terme.
    Mehrere Terme mit '+' trennen, z. B.:
      - 'ukraine+frieden'
      - 'ukraine+selensky+frieden'
    Ein Artikel zählt, wenn **alle** Terme im Titel (oder optional Teaser) vorkommen.
    Matching: Wortgrenzen-Regex ODER Token-Set aus extract_relevant_words (Stopwörter bleiben ausgeschlossen).
    """
    import re
    from collections import Counter
    from app.core.clean_utils import extract_relevant_words

    # Schwelle für "Seiten" vs. Neutral (wie bisher)
    T = 0.33

    # Terme
    terms = [t.strip().lower() for t in (word or "").split("+") if t.strip()]
    if not terms:
        return {"error": "word required"}
    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    # 🔽 NEU: Zeitfenster entscheiden
    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )

    # Artikel holen (vorerst stundenbasiert; echte Range in WP-C)
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    # Artikel-Matching: alle Terme müssen getroffen werden
    matched = []
    for a in articles:
        title_l  = (a.title or "").lower()
        teaser_l = (a.teaser or "").lower() if (teaser and a.teaser) else ""

        tokens_title  = set(extract_relevant_words(a.title))
        tokens_teaser = set(extract_relevant_words(a.teaser)) if (teaser and a.teaser) else set()

        ok = True
        for i, term in enumerate(terms):
            hit_title  = bool(patterns[i].search(title_l))  or (term in tokens_title)
            hit_teaser = bool(patterns[i].search(teaser_l)) or (term in tokens_teaser) if teaser else False
            if not (hit_title or hit_teaser):
                ok = False
                break
        if ok:
            matched.append(a)

    if not matched:
        return {
            "word": word, "hours": hours, "t": T,
            "counts": {"x": {"kritisch":0,"neutral":0,"nah":0,"total":0},
                       "y": {"national":0,"neutral":0,"global":0,"total":0}},
            "blindspots": {"x": None, "y": None}
        }

    # Medienkompass laden und auf normierte Namen mappen
    df = load_media_df()
    bias_map = {row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
                for _, row in df.iterrows()}

    # Zählen je Achse
    x_counts = Counter(); y_counts = Counter()
    for a in matched:
        norm = norm_source(a.source)
        if norm not in bias_map:
            continue
        x, y = bias_map[norm]
        # X-Achse
        if x <= -T: x_counts["kritisch"] += 1
        elif x >=  T: x_counts["nah"] += 1
        else: x_counts["neutral"] += 1
        # Y-Achse
        if y <= -T: y_counts["national"] += 1
        elif y >=  T: y_counts["global"] += 1
        else: y_counts["neutral"] += 1

    counts = {
        "x": {"kritisch": x_counts["kritisch"], "neutral": x_counts["neutral"], "nah": x_counts["nah"], "total": sum(x_counts.values())},
        "y": {"national": y_counts["national"], "neutral": y_counts["neutral"], "global": y_counts["global"], "total": sum(y_counts.values())},
    }

    # Einfache Blindspot-Heuristik (wie bisher)
    MIN_COUNT = 3
    RATIO_MAX = 0.1
    def blind(axis_counts, left_key, right_key):
        L = axis_counts[left_key]; R = axis_counts[right_key]; total = axis_counts["total"]
        if total < MIN_COUNT: return None
        if R == 0 and L >= MIN_COUNT: return f"{right_key} fehlt"
        if L == 0 and R >= MIN_COUNT: return f"{left_key} fehlt"
        if L > 0 and R > 0:
            if R/(L+1e-6) <= RATIO_MAX and L >= MIN_COUNT: return f"{right_key} extrem selten"
            if L/(R+1e-6) <= RATIO_MAX and R >= MIN_COUNT: return f"{left_key} extrem selten"
        return None

    blindspots = {
        "x": blind(counts["x"], "kritisch", "nah"),
        "y": blind(counts["y"], "national", "global"),
    }

    return {"word": word, "hours": hours, "t": T, "counts": counts, "blindspots": blindspots}


@router.get("/blindspots/keywords-feed")
def blindspot_keywords_feed(
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    min_sources_min: int = Query(1),
    min_sources_max: int = Query(50),
    min_total_min:   int = Query(1),
    min_total_max:   int = Query(50),
    ratio_min: float = Query(0.0),     # 0.00 .. 1.00
    ratio_max: float = Query(0.05),    # 0.00 .. 1.00
    top_n: int = Query(25),
    teaser: bool = Query(False),
    ngram: int = Query(1, ge=1, le=3),
    from_: str | None = Query(None, alias="from"),   # NEU
    to: str | None = Query(None),                     # NEU
    db: Session = Depends(get_db),
):
    """
    Vier Listen: kaum systemkritisch / systemnah / nationalistisch / globalistisch.
    Zählung je Headline: pro (n=1/2/3)-Einheit höchstens 1×.
    """
    from collections import defaultdict
    from itertools import combinations
    from app.core.clean_utils import extract_relevant_words

    T = 0.20  # Bucket-Schwelle wie gehabt

    # Medien-Bias laden
    df = load_media_df()
    bias_map = {
        row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
        for _, row in df.iterrows()
    }

    tw = _time_window_or_400(hours, from_, to)
    hours_eff = tw["hours"] if tw["mode"] == "hours" else max(
        int((tw["to"] - tw["from"]).total_seconds() // 3600), 1
    )
    articles, start, end, mode, bucket = _select_articles(db, hours, from_, to)

    articles = filter_articles_by_countries(articles, country)

    # je "word"/Kombi: alle (x,y)-Punkte + Quellenset sammeln
    coords_by_word = defaultdict(list)
    sources_by_word = defaultdict(set)

    for a in articles:
        norm = norm_source(a.source)
        if norm not in bias_map:
            continue
        x, y = bias_map[norm]

        # Tokens ohne Stopwörter, optional inkl. Teaser
        toks = set(extract_relevant_words(a.title))
        if teaser and a.teaser:
            toks |= set(extract_relevant_words(a.teaser))
        toks = sorted(toks)

        # 1er/2er/3er-Einheiten pro Headline (reihenfolgeunabhängig, dedupliziert)
        units = set(toks) if ngram == 1 else {" + ".join(c) for c in combinations(toks, ngram)}

        for w in units:
            coords_by_word[w].append((x, y))
            sources_by_word[w].add(norm)

    # Buckets zählen (X/Y)
    def counts_for(coords):
        xk = xn = xr = 0
        yn = yc = yg = 0
        for x, y in coords:
            if x <= -T: xk += 1
            elif x >=  T: xr += 1
            else: xn += 1
            if y <= -T: yn += 1
            elif y >=  T: yg += 1
            else: yc += 1
        total = len(coords)
        return (
            {"kritisch": xk, "neutral": xn, "nah": xr, "total": total},
            {"national": yn, "neutral": yc, "global": yg, "total": total},
        )

    L_kritisch, L_nah, L_national, L_global = [], [], [], []

    for w, coords in coords_by_word.items():
        total = len(coords)
        num_sources = len(sources_by_word[w])

        # Bereiche prüfen
        if not (min_total_min <= total <= min_total_max):
            continue
        if not (min_sources_min <= num_sources <= min_sources_max):
            continue

        # Raten berechnen (wie gehabt)
        x_counts, y_counts = counts_for(coords)
        den = x_counts["total"] or 1
        p_krit = x_counts["kritisch"] / den
        p_nah  = x_counts["nah"]      / den
        p_nat  = y_counts["national"] / den
        p_glo  = y_counts["global"]   / den

        # In Range? -> dann aufnehmen. (Vorher: nur ≤ ratio_max)
        def in_range(p): return (ratio_min <= p <= ratio_max)

        item = {
            "word": w,
            "counts": {"x": x_counts, "y": y_counts},
            "sources": num_sources,
            "total": total,
            "ratios": {
                "kritisch": round(p_krit, 3),
                "nah":      round(p_nah, 3),
                "national": round(p_nat, 3),
                "global":   round(p_glo, 3),
            },
            "zero_badge": {
                "kritisch": x_counts["kritisch"] == 0,
                "nah":      x_counts["nah"]      == 0,
                "national": y_counts["national"] == 0,
                "global":   y_counts["global"]   == 0,
            }
        }

        # Items in jene Liste, wo die jeweilige Achse in der Range liegt
        if in_range(p_krit): L_kritisch.append(item)
        if in_range(p_nah):  L_nah.append(item)
        if in_range(p_nat):  L_national.append(item)
        if in_range(p_glo):  L_global.append(item)


    def sort_list(lst, key_name):
        return sorted(lst, key=lambda it: (it["ratios"][key_name], -it["total"]))[:top_n]

    return {
        "params": {
            "hours": hours,
            "min_sources_min": min_sources_min,
            "min_sources_max": min_sources_max,
            "min_total_min":   min_total_min,
            "min_total_max":   min_total_max,
            "ratio_min":       ratio_min,
            "ratio_max":       ratio_max,
            "top_n":           top_n,
            "ngram":           ngram,
            "teaser":          teaser,
        },
        "items": {
            "systemkritisch":  sort_list(L_kritisch, "kritisch"),
            "systemnah":       sort_list(L_nah,      "nah"),
            "nationalistisch": sort_list(L_national, "national"),
            "globalistisch":   sort_list(L_global,   "global"),
        }
    }

@router.get("/chronicle/weekly-top3")
def chronicle_weekly_top3(
    weeks: int = Query(12, ge=1, le=104),                # Anzahl zurückzugebender Wochen (rückwärts)
    country: list[str] | None = Query(None),
    teaser: bool = Query(False),                         # Teaser berücksichtigen?
    source: list[str] | None = Query(None),              # Mehrfachquellen (?source=…)
    ngram: int = Query(1, ge=1, le=3),                   # 1er/2er/3er-Kombis (optional)
    from_: str | None = Query(None, alias="from"),       # optionales Zeitfenster
    to: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Liefert für jede ISO-Kalenderwoche im gewählten Zeitraum die Top-3 Keywords.
    Falls kein from/to übergeben wird, wird aus dem aktuellen Fenster (/_select_articles)
    der effektive Zeitraum genommen und dann maximal `weeks` Wochen rückwärts ausgegeben.
    """
    from collections import Counter, defaultdict
    from itertools import combinations
    from datetime import datetime, timedelta, timezone
    import math
    import re
    from app.core.clean_utils import extract_relevant_words

    # Artikel + Zeitraum bestimmen (nutzt eure zentrale Hilfsfunktion)
    arts, start, end, _mode, _bucket = _select_articles(
        db, hours=None, from_=from_, to=to, unbounded=True, order="asc"
    )

    # Quellenfilter
    if source:
        wanted = { (s or "").strip().lower() for s in source if s and s.strip() }
        if wanted:
            arts = [a for a in arts if (a.source or "").strip().lower() in wanted]

    # Ngram-Extractor je Artikel
    def units_from_article(a):
        tokens = set(extract_relevant_words(a.title or ""))
        if teaser and getattr(a, "teaser", None):
            tokens |= set(extract_relevant_words(a.teaser or ""))
        toks = sorted(tokens)
        if ngram == 1:
            return set(toks)
        return {" + ".join(c) for c in combinations(toks, ngram)}

    # Gruppierung nach ISO-Jahr/ISO-Woche
    # Wir mappen Woche -> Counter und merken uns das reale Wochenintervall (Mo..So)
    week_counts: dict[tuple[int,int], Counter] = defaultdict(Counter)
    week_bounds: dict[tuple[int,int], tuple[datetime, datetime]] = {}

    def iso_week_bounds(d_utc: datetime):
        # ISO: Woche beginnt am Montag
        # Normalisiere auf 00:00 UTC
        d0 = d_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        wd = (d0.isoweekday() % 7)  # Mo=1..So=7 -> 0..6 (So=0)
        monday = d0 - timedelta(days=(wd-1 if wd != 0 else 6))
        if wd == 0:  # Sonntag
            monday = d0 - timedelta(days=6)
        sunday_end_exclusive = monday + timedelta(days=7)
        return monday, sunday_end_exclusive

    for a in arts:
        dt = getattr(a, "published_at", None)
        if not isinstance(dt, datetime):
            continue
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)

        # Einmalig Bounds berechnen/merken
        if key not in week_bounds:
            w_start, w_end_ex = iso_week_bounds(dt)
            week_bounds[key] = (w_start, w_end_ex)

        # Einheiten zählen
        units = units_from_article(a)
        week_counts[key].update(units)

    if not week_counts:
        return []

    # Wochen sortieren (neueste zuerst), auf gewünschte Anzahl begrenzen
    keys_sorted = sorted(week_counts.keys(), key=lambda k: (k[0], k[1]), reverse=True)
    if weeks:
        keys_sorted = keys_sorted[:weeks]

    # Ausgabeformat
    out = []
    for key in keys_sorted:
        iso_year, iso_week = key
        w_start, w_end_ex = week_bounds[key]  # [Mo 00:00, Mo(+7) 00:00)
        # „Sonntag“ = letzter inklusiver Tag ist w_end_ex - 1 Sekunde
        end_inclusive = (w_end_ex - timedelta(seconds=1))

        top3 = [
            {"word": term, "count": int(cnt)}
            for term, cnt in week_counts[key].most_common(3)
        ]

        out.append({
            "iso_year": iso_year,
            "iso_week": iso_week,
            "week_label": f"KW {iso_week}",
            "start_date": w_start.date().isoformat(),
            "end_date": end_inclusive.date().isoformat(),
            "top": top3,
            "ngram": ngram,
        })

    return out

@router.get("/chronicle/weekly-top3-all")
def chronicle_weekly_top3_all(
    weeks: int = Query(5, ge=1, le=104),               # Default: 5 Wochen
    country: list[str] | None = Query(None),
    teaser: bool = Query(False),
    source: list[str] | None = Query(None),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Gibt pro ISO-Woche Top-3 für n=1/2/3 zurück.
    WICHTIG: Wenn 'from'/'to' NICHT übergeben werden, wird der Zeitraum
    explizit auf die letzten (weeks * 7) Tage gesetzt (Default 5 Wochen),
    statt ein beliebiges Standardfenster zu verwenden.
    """
    from collections import defaultdict, Counter
    from itertools import combinations
    from datetime import datetime, timedelta, timezone
    from app.core.clean_utils import extract_relevant_words

    # ---- Zeitraum festziehen -----------------------------------------------
    # Falls die UI KEIN from/to mitsendet, erzwingen wir hier das 5-Wochen-Fenster.
    if not from_ and not to:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = (now - timedelta(days=weeks * 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        # mit explizitem Range über _select_articles laden
        arts, start, end, _mode, _bucket = _select_articles(
            db,
            hours=None,
            from_=start.isoformat(),
            to=now.isoformat(),           # exklusiv
            unbounded=True,
            order="asc",
        )
    else:
        # Wenn from/to gesetzt ist, respektieren wir die UI-Auswahl
        arts, start, end, _mode, _bucket = _select_articles(
            db, hours=None, from_=from_, to=to, unbounded=True, order="asc"
        )
        arts = filter_articles_by_countries(arts, country)

    # ---- Quellenfilter ------------------------------------------------------
    if source:
        wanted = { (s or "").strip().lower() for s in source if s and s.strip() }
        if wanted:
            def norm(x): return (x or "").strip().lower()
            arts = [a for a in arts if norm(getattr(a, "source", None)) in wanted]

    # ---- Pro ISO-Woche Zähler für n=1/2/3 + Bounds --------------------------
    week_counts_1: dict[tuple[int,int], Counter] = defaultdict(Counter)
    week_counts_2: dict[tuple[int,int], Counter] = defaultdict(Counter)
    week_counts_3: dict[tuple[int,int], Counter] = defaultdict(Counter)
    week_bounds: dict[tuple[int,int], tuple[datetime, datetime]] = {}

    def iso_week_bounds(d_utc: datetime):
        d0 = d_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        wd = (d0.isoweekday() % 7)            # Mo=1..So=7 -> 0..6 (So=0)
        monday = d0 - timedelta(days=(wd-1 if wd != 0 else 6))
        if wd == 0:                            # Sonntag
            monday = d0 - timedelta(days=6)
        end_ex = monday + timedelta(days=7)    # exklusiv
        return monday, end_ex

    for a in arts:
        dt = getattr(a, "published_at", None)
        if not isinstance(dt, datetime):
            continue
        dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)

        if key not in week_bounds:
            w_start, w_end_ex = iso_week_bounds(dt)
            week_bounds[key] = (w_start, w_end_ex)

        tokens = set(extract_relevant_words(a.title or ""))
        if teaser and getattr(a, "teaser", None):
            tokens |= set(extract_relevant_words(a.teaser or ""))
        if not tokens:
            continue
        toks_sorted = sorted(tokens)

        # n=1
        week_counts_1[key].update(tokens)
        # n=2
        if len(toks_sorted) >= 2:
            combos2 = set(" + ".join(c) for c in combinations(toks_sorted, 2))
            if combos2:
                week_counts_2[key].update(combos2)
        # n=3
        if len(toks_sorted) >= 3:
            combos3 = set(" + ".join(c) for c in combinations(toks_sorted, 3))
            if combos3:
                week_counts_3[key].update(combos3)

    if not week_bounds:
        return []

    # Neueste Wochen zuerst; auf 'weeks' begrenzen
    weeks_sorted = sorted(week_bounds.keys(), key=lambda k: week_bounds[k][0], reverse=True)[:weeks]

    def top3(counter: Counter):
        return [{"word": term, "count": int(cnt)} for term, cnt in counter.most_common(3)]

    out = []
    for key in weeks_sorted:
        iso_year, iso_week = key
        w_start, w_end_ex = week_bounds[key]
        end_inclusive = (w_end_ex - timedelta(seconds=1))
        out.append({
            "iso_year": iso_year,
            "iso_week": iso_week,
            "week_label": f"KW {iso_week}",
            "start_date": w_start.date().isoformat(),
            "end_date": end_inclusive.date().isoformat(),
            "top1": top3(week_counts_1.get(key, Counter())),
            "top2": top3(week_counts_2.get(key, Counter())),
            "top3": top3(week_counts_3.get(key, Counter())),
        })
    return out


@router.get("/countries")
def list_countries():
    """Liefert die in der CSV gepflegten Ländercodes (DE, AT, CH, …)."""
    return {"countries": available_countries()}


def filter_articles_by_countries(articles, countries: list[str] | None):
    """Filtert Artikel anhand des Ländercodes des Mediums (aus der CSV)."""
    if not countries:
        return articles
    wanted = {c.strip().upper() for c in countries if c and c.strip()}
    cmap = country_map()
    return [a for a in articles if cmap.get(norm_source(getattr(a, "source", "")), "UN") in wanted]


@router.get("/countries/compare")
def countries_compare(
    metric: str = Query("articles"),     # später: "keywords_top", "ngrams2", ...
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
):
    articles, start, end, mode, bucket = _select_articles(db, hours=None, from_=from_, to=to, unbounded=True, order="asc")
    cmap = country_map()
    from collections import Counter
    c = Counter()
    for a in articles:
        c[cmap.get(norm_source(getattr(a, "source", "")), "UN")] += 1
    # Ausgabe: [{country:"DE", value: 123}, ...] – gut für Charts
    return [{"country": k, "value": int(v)} for k, v in sorted(c.items())]


@router.get("/keywords/sankey")
def keyword_sankey(
    word: str,
    hours: int = Query(72),
    country: list[str] | None = Query(None),
    source: list[str] | None = Query(None),   # Mehrfachquellen
    teaser: bool = Query(False),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """
    Liefert für das Sankey je Bereich:
      - total_sources: Anzahl unterschiedlicher Medien im Bereich (aus Medienliste/CSV, gefiltert)
      - hit_sources:   Anzahl dieser Medien, die im Zeitfenster das Keyword ≥1× verwendet haben
    Bereiche: systemkritisch, systemnah (X); nationalistisch, globalistisch (Y)
    """
    import re

    # 1) Zeitfenster & Artikel holen (für Treffer)
    arts, start, end, _mode, _bucket = _select_articles(
        db, hours=hours, from_=from_, to=to, unbounded=True, order="asc"
    )
    arts = filter_articles_by_countries(arts, country)

    # Wenn Quellenfilter gesetzt ist: Artikel auf diese Quellen begrenzen (nur Performance)
    if source:
        wanted = {(s or "").strip().lower() for s in source if s and s.strip()}
        if wanted:
            arts = [a for a in arts if (a.source or "").strip().lower() in wanted]

    # 2) Medienkompass & Bucket-Zuordnung (aus CSV)
    df = load_media_df()
    bias = {
        row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
        for _, row in df.iterrows()
    }
    T = 0.01  # Schwellen wie in /keywords/sides

    def buckets_for(norm_name: str):
        x, y = bias[norm_name]
        bx = "systemkritisch" if x <= -T else ("systemnah" if x >= T else None)
        by = "nationalistisch" if y <= -T else ("globalistisch" if y >= T else None)
        return bx, by

    # 3) Grundgesamtheit: aus Medienliste (CSV), nur durch Country/Source-Filter einschränken
    #    -> unabhängig davon, ob im Zeitraum Artikel vorhanden sind
    # Basis-Kandidaten: alle norm_name aus df
    allowed_sources = {row["norm_name"] for _, row in df.iterrows()}

    # Country-Filter (falls df eine Country-Spalte hat)
    if country:
        wanted_countries = {c.upper() for c in country}
        country_cols = [c for c in df.columns if c.lower() in ("country", "country_code", "land", "laendercode")]
        if country_cols:
            col = country_cols[0]
            keep = set(
                df[df[col].astype(str).str.upper().isin(wanted_countries)]["norm_name"]
            )
            allowed_sources &= keep

    # Source-Filter (über norm_name, lower-normalisiert)
    if source:
        wanted = {(s or "").strip().lower() for s in source if s and s.strip()}
        keep = {nm for nm in allowed_sources if nm.lower() in wanted}
        allowed_sources = keep

    # totals je Bucket aus allowed_sources (X- und Y-Achse getrennt zählen)
    totals = {
        "systemkritisch": set(),
        "systemnah": set(),
        "nationalistisch": set(),
        "globalistisch": set(),
    }
    for n in allowed_sources:
        if n not in bias:
            continue
        bx, by = buckets_for(n)
        if bx:
            totals[bx].add(n)
        if by:
            totals[by].add(n)

    # 4) Keyword-Matches: pro Medium merken, ob es das Keyword verwendet hat (im Zeitraum)
    raw = (word or "").strip().lower()
    terms = [t for t in (p.strip() for p in raw.split("+")) if t]
    if not terms:
        return {"error": "word required"}

    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    def match_txt(title, teaser_txt):
        title = (title or "").lower()
        te = (teaser_txt or "").lower() if teaser and teaser_txt else ""
        # UND-Verknüpfung der Teilbegriffe: alle müssen in Titel oder (optional) Teaser vorkommen
        return all(p.search(title) or (te and p.search(te)) for p in patterns)

    used_by_source = set()
    for a in arts:
        if not a.source:
            continue
        n = norm_source(a.source)
        # Nur Medien zählen, die in der Grundgesamtheit sind
        if n not in allowed_sources or n not in bias:
            continue
        if match_txt(a.title, a.teaser):
            used_by_source.add(n)

    hits = {
        "systemkritisch": set(),
        "systemnah": set(),
        "nationalistisch": set(),
        "globalistisch": set(),
    }
    for n in used_by_source:
        bx, by = buckets_for(n)
        if bx:
            hits[bx].add(n)
        if by:
            hits[by].add(n)

    # 5) Antwortstruktur
    out = []
    order = ["systemkritisch", "systemnah", "nationalistisch", "globalistisch"]
    for key in order:
        total = len(totals[key])
        cnt = len(hits[key])
        pct = round((cnt / total * 100.0), 1) if total else 0.0
        out.append(
            {
                "bucket": key,
                "total_sources": total,
                "hit_sources": cnt,
                "pct": pct,
            }
        )

    return {
        "word": word,
        "range": {"from": start.isoformat(), "to": end.isoformat()},
        "items": out,
    }

