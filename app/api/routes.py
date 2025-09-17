# ============================
# 📁 app/api/routes.py
# (hier sammelst du alle Endpunkte)

from typing import Optional
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import re
import pandas as pd

from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

# Bestehender In-Memory/Model-Store (bewusst noch genutzt für einige Endpunkte)
from app.models import get_articles

# Feeds einlesen (liefert Artikelobjekte; Speicherung jetzt via Repository)
from app.core.feeds import fetch_articles, fetch_articles_from_source

# ORM/Repository – zentrale DB-Zugriffe
from app.database import get_db
from app.repositories.articles import get_articles_last_hours, bulk_upsert_articles
from app.schemas import Article as ArticleSchema

router = APIRouter()

MEDIA_CSV_PATH = Path("app/static/Medien-Kompass__Systemn_he_.csv")


@lru_cache(maxsize=1)
def load_media_df() -> pd.DataFrame:
    df = pd.read_csv(MEDIA_CSV_PATH)
    df["norm_name"] = df["Medium"].str.strip().str.lower()
    return df


def norm_source(name: str) -> str:
    return (name or "").strip().lower()


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


# -----------------------------
# Medienkompass (CSV)
# -----------------------------
@router.get("/media-positions")
def media_positions():
    df = load_media_df()
    data = [
        {"medium": row["Medium"], "x": row["Systemnähe (X)"], "y": row["Globalismus (Y)"]}
        for _, row in df.iterrows()
    ]
    return JSONResponse(content=jsonable_encoder(data))


@router.get("/media-positions/by-keyword")
def media_positions_by_keyword(
    word: str,
    hours: int = Query(72),
    source: Optional[str] = Query(None),
    teaser: bool = Query(False),
    db: Session = Depends(get_db),
):
    keyword = word.lower()
    pattern = rf"\b{re.escape(keyword)}\b"

    # Zeitraum
    articles = get_articles_last_hours(db, hours)

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
    items = fetch_articles()  # liefert Pydantic-ähnliche Artikel (siehe feeds.py Anpassung)
    count = bulk_upsert_articles(db, items)
    return {"status": "updated", "count": count}


@router.get("/fetch")
def fetch_source(source: str = Query(...), db: Session = Depends(get_db)):
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
    source: str | None = Query(None),
    keyword: str | None = Query(None),
    teaser: bool = Query(False),
    db: Session = Depends(get_db),
):
    articles = get_articles_last_hours(db, hours)

    if source:
        s = source.strip().lower()
        articles = [a for a in articles if a.source.strip().lower() == s]

    if keyword:
        # "ukraine + frieden" => beide Begriffe müssen vorkommen
        terms = [t.strip().lower() for t in keyword.split('+') if t.strip()]

        def match(a):
            txt = a.title.lower()
            if teaser and a.teaser:
                txt += " " + a.teaser.lower()
            import re
            return all(re.search(rf"\b{re.escape(t)}\b", txt) for t in terms)

        articles = [a for a in articles if match(a)]


    return [ArticleSchema.model_validate(a) for a in articles]


# -----------------------------
# Medienkompass gefiltert (ORM + Repository)
# -----------------------------
@router.get("/media-positions/filtered")
def media_positions_filtered(
    hours: int = Query(72),
    source: str = Query(None),
    keyword: str = Query(None),
    teaser: bool = Query(False),
    db: Session = Depends(get_db),
):
    from collections import Counter

    df = load_media_df()
    articles = get_articles_last_hours(db, hours)

    if source:
        articles = [a for a in articles if a.source.strip().lower() == source.strip().lower()]

    if keyword:
        terms = [t.strip().lower() for t in keyword.split('+') if t.strip()]

        def match(a):
            txt = a.title.lower()
            if teaser and a.teaser:
                txt += " " + a.teaser.lower()
            import re
            return all(re.search(rf"\b{re.escape(t)}\b", txt) for t in terms)

        articles = [a for a in articles if match(a)]


    matching_sources = [a.source.strip().lower() for a in articles]
    source_counts = Counter(matching_sources)

    filtered = df[df["norm_name"].isin(source_counts.keys())]

    return [
        {
            "medium": row["Medium"],
            "x": row["Systemnähe (X)"],
            "y": row["Globalismus (Y)"],
            "count": source_counts[row["norm_name"]],
        }
        for _, row in filtered.iterrows()
    ]


# -----------------------------
# Keyword-Analysen (alle via Repository/Session)
# -----------------------------
@router.get("/keywords/trending")
def keyword_trends(db: Session = Depends(get_db)):
    from collections import Counter
    from app.core.clean_utils import extract_relevant_words

    timeframes = {"24h": 24, "72h": 72, "7d": 168, "30d": 720}
    result = {}

    for label, hours in timeframes.items():
        now_words = Counter()
        past_words = Counter()

        # Aktueller Zeitraum
        now_articles = get_articles_last_hours(db, hours)
        for a in now_articles:
            now_words.update(extract_relevant_words(a.title))

        # Vergleichszeitraum (davor)
        past_articles = get_articles_last_hours(db, hours * 2)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        for a in past_articles:
            if a.published_at < cutoff:
                past_words.update(extract_relevant_words(a.title))

        # Veränderung berechnen
        changes = []
        for word in now_words:
            delta = now_words[word] - past_words.get(word, 0)
            rel_change = delta / max(1, past_words.get(word, 1))
            changes.append((word, delta, rel_change))

        top = sorted(changes, key=lambda x: x[2], reverse=True)[:5]
        flop = sorted(changes, key=lambda x: x[2])[:5]

        result[label] = {
            "top": [{"word": w, "delta": d, "change": round(r, 2)} for w, d, r in top],
            "flop": [{"word": w, "delta": d, "change": round(r, 2)} for w, d, r in flop],
        }

    return result


@router.get("/keywords/extreme-bubble")
def extreme_keywords(hours: int = 72, db: Session = Depends(get_db)):
    from collections import defaultdict, Counter
    from app.core.clean_utils import extract_relevant_words

    # Medienkompass laden
    df = load_media_df()
    bias_map = dict(zip(df["norm_name"], df["Systemnähe (X)"]))

    articles = get_articles_last_hours(db, hours)

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
def keyword_bias_scores(hours: int = 72, db: Session = Depends(get_db)):
    from collections import defaultdict
    from app.core.clean_utils import extract_relevant_words

    df = load_media_df()
    media_bias = dict(zip(df["norm_name"], df["Systemnähe (X)"]))

    articles = get_articles_last_hours(db, hours)
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
def keyword_bias_vector(hours: int = 72, db: Session = Depends(get_db)):
    from collections import defaultdict
    from app.core.clean_utils import extract_relevant_words

    df = load_media_df()
    bias_map = {
        row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
        for _, row in df.iterrows()
    }

    articles = get_articles_last_hours(db, hours)
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
    hours: int = 72,
    teaser: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Zählt pro Stunde die Headlines, die ALLE Terme aus `word` enthalten.
    Mehrere Terme mit '+' trennen, z. B.:
      - 'ukraine+frieden'  (2)
      - 'ukraine+selensky+frieden' (3)
    Stopwords bleiben draußen (extract_relevant_words).
    Antwort enthält je Bucket eine deduplizierte Quellenliste (sources) für Tooltips.
    """
    from collections import defaultdict
    from app.core.clean_utils import extract_relevant_words
    import re

    # Zeitfenster (volle Stunden)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours)
    slots = [start + timedelta(hours=i) for i in range(hours + 1)]

    # Terme vorbereiten
    raw = (word or "").strip().lower()
    terms = [t.strip() for t in raw.split("+") if t.strip()]
    if not terms:
        return [{"time": t.isoformat(), "count": 0, "sources": []} for t in slots]

    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    articles = get_articles_last_hours(db, hours)

    # Buckets
    counts = defaultdict(int)
    sources = defaultdict(set)

    def matches_all(a) -> bool:
        title_l = a.title.lower()
        teaser_l = (a.teaser or "").lower()
        tokens_title = set(extract_relevant_words(a.title))
        tokens_teaser = set(extract_relevant_words(a.teaser)) if (teaser and a.teaser) else set()

        for i, term in enumerate(terms):
            in_title = bool(patterns[i].search(title_l)) or (term in tokens_title)
            in_teasr = bool(patterns[i].search(teaser_l)) or (term in tokens_teaser) if teaser else False
            if not (in_title or in_teasr):
                return False
        return True

    for a in articles:
        if not matches_all(a):
            continue
        ts = a.published_at.replace(minute=0, second=0, microsecond=0)
        counts[ts] += 1
        if a.source:
            sources[ts].add(a.source.strip())

    return [
        {
            "time": t.isoformat(),
            "count": counts.get(t, 0),
            "sources": sorted(list(sources.get(t, set()))),
        }
        for t in slots
    ]


@router.get("/keywords/top-absolute")
def keywords_top_absolute(
    hours: int = Query(72),
    ngram: int = Query(1, ge=1, le=3),   # 1, 2 oder 3-Wort-Kombis
    teaser: bool = Query(False),          # optional Teaser einbeziehen
    db: Session = Depends(get_db),
):
    from collections import Counter
    from itertools import combinations
    from app.core.clean_utils import extract_relevant_words

    def units_from_article(a):
        # Stopwords bleiben draußen (extract_relevant_words)
        tokens = set(extract_relevant_words(a.title))
        if teaser and a.teaser:
            tokens |= set(extract_relevant_words(a.teaser))
        toks = sorted(tokens)
        if ngram == 1:
            return set(toks)
        # Reihenfolgeunabhängige Kombis, pro Headline nur 1× zählen
        return {" + ".join(c) for c in combinations(toks, ngram)}

    now_words = Counter()
    past_words = Counter()

    # Aktueller Zeitraum
    now_articles = get_articles_last_hours(db, hours)
    for a in now_articles:
        now_words.update(units_from_article(a))

    # Vergleichszeitraum (vorherige gleich lange Periode)
    past_articles = get_articles_last_hours(db, hours * 2)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    for a in past_articles:
        if a.published_at < cutoff:
            past_words.update(units_from_article(a))

    # Top 30 + Veränderung
    result = []
    for term, current_count in now_words.most_common(30):
        prev_count = past_words.get(term, 0)
        delta = current_count - prev_count
        change_pct = (delta / prev_count * 100) if prev_count > 0 else 0.0
        result.append({
            "word": term,
            "current": current_count,
            "previous": prev_count,
            "delta": delta,
            "change_pct": round(change_pct, 2),
        })
    return result


# ➕ in app/api/routes.py ersetzen
@router.get("/headlines/words")
def headlines_words(
    hours: int = Query(72),
    source: str | None = Query(None),
    keyword: str | None = Query(None),
    teaser: bool = Query(False),
    ngram: int = Query(1, ge=1, le=3),   # 👈 NEU: 1, 2 oder 3
    db: Session = Depends(get_db),
):
    from collections import Counter
    from itertools import combinations
    from app.core.clean_utils import extract_relevant_words

    arts = get_articles_last_hours(db, hours)

    # Quelle filtern
    if source:
        s = source.strip().lower()
        arts = [a for a in arts if a.source.strip().lower() == s]

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

    # Terme vorbereiten
    terms = [t.strip().lower() for t in (word or "").split("+") if t.strip()]
    if not terms:
        return {"error": "word required"}
    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]

    # Zeitraum
    articles = get_articles_last_hours(db, hours)

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
    min_total: int = Query(10),
    ratio_max: float = Query(0.05),
    top_n: int = Query(25),
    teaser: bool = Query(False),
    ngram: int = Query(1, ge=1, le=3),   # 👈 NEU: 1, 2 oder 3
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

    articles = get_articles_last_hours(db, hours)

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
        if len(coords) < min_total:
            continue
        x_counts, y_counts = counts_for(coords)
        total = x_counts["total"] or 1

        p_krit = x_counts["kritisch"] / total
        p_nah  = x_counts["nah"]      / total
        p_nat  = y_counts["national"] / total
        p_glo  = y_counts["global"]   / total

        item = {
            "word": w,
            "counts": {"x": x_counts, "y": y_counts},
            "sources": len(sources_by_word[w]),
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

        if p_krit <= ratio_max: L_kritisch.append(item)
        if p_nah  <= ratio_max: L_nah.append(item)
        if p_nat  <= ratio_max: L_national.append(item)
        if p_glo  <= ratio_max: L_global.append(item)

    def sort_list(lst, key_name):
        return sorted(lst, key=lambda it: (it["ratios"][key_name], -it["total"]))[:top_n]

    return {
        "params": {
            "hours": hours, "min_total": min_total, "ratio_max": ratio_max,
            "top_n": top_n, "ngram": ngram, "teaser": teaser
        },
        "items": {
            "systemkritisch":  sort_list(L_kritisch, "kritisch"),
            "systemnah":       sort_list(L_nah,      "nah"),
            "nationalistisch": sort_list(L_national, "national"),
            "globalistisch":   sort_list(L_global,   "global"),
        }
    }
