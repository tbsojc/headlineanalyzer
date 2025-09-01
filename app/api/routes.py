# ============================
# 📁 app/api/routes.py
# (hier sammelst du alle Endpunkte)
import pandas as pd
from typing import Optional
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta, timezone
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Query
from app.models import get_articles, save_articles, get_articles_last_hours
from app.core.feeds import fetch_articles, fetch_articles_from_source
from app.core.trending import get_trending_keywords, get_multi_period_trending
from functools import lru_cache
from pathlib import Path
import re

router = APIRouter()

MEDIA_CSV_PATH = Path("app/static/Medien-Kompass__Systemn_he_.csv")

@lru_cache(maxsize=1)
def load_media_df() -> pd.DataFrame:
    df = pd.read_csv(MEDIA_CSV_PATH)
    df["norm_name"] = df["Medium"].str.strip().str.lower()
    return df

def norm_source(name: str) -> str:
    return (name or "").strip().lower()

@router.get("/topics")
def get_topics():
    articles = get_articles()
    topics = {}
    for art in articles:
        topics.setdefault(art.topic, 0)
        topics[art.topic] += 1
    return sorted(topics.items(), key=lambda x: x[1], reverse=True)

@router.get("/media-positions")
def media_positions():
    df = load_media_df()  # <— zuweisen!
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
):
    keyword = word.lower()
    pattern = rf"\b{re.escape(keyword)}\b"

    # im Zeitraum filtern
    articles = get_articles_last_hours(hours)

    # optional Quelle filtern
    if source:
        s = norm_source(source)
        articles = [a for a in articles if norm_source(a.source) == s]

    # Keyword-Match auf Title/Teaser
    articles = [
        a for a in articles
        if (
            re.search(pattern, a.title.lower())
            or (teaser and re.search(pattern, a.teaser.lower()))
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



@router.get("/articles")
def articles_by_topic(topic: str):
    return [a.dict() for a in get_articles() if a.topic == topic]

@router.get("/refresh")
def refresh():
    articles = fetch_articles()
    save_articles(articles)
    return {"status": "updated", "count": len(articles)}

@router.get("/trending")
def trending():
    return get_trending_keywords(hours=24, top_n=100)

@router.get("/trending-multi")
def trending_multi():
    return get_multi_period_trending()

@router.get("/headlines")
def all_headlines(
    source: str | None = None,
    after: str | None = None
):
    # Konsistent aus dem Store lesen:
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
            # Optional: fail-soft – einfach ignorieren oder 422 raisen
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
            "topic": a.topic
        }
        for a in get_articles()
        if keyword_match(a.title, word) or keyword_match(a.teaser, word)
    ]

@router.get("/headlines/by-keyword-and-source")
def headlines_by_keyword_and_source(word: str, source: str):
    from app.models import get_articles

    keyword = word.lower()
    source = source.strip().lower()
    articles = get_articles()

    pattern = rf"\b{re.escape(keyword)}\b"

    filtered = [
        a.dict()
        for a in articles
        if (
            a.source.strip().lower() == source and
            (re.search(pattern, a.title.lower()) or re.search(pattern, a.teaser.lower()))
        )
    ]
    return JSONResponse(filtered)

@router.get("/headlines/by-source")
def headlines_by_source(source: str):
    from app.models import get_articles
    source = source.strip().lower()
    articles = get_articles()

    filtered = [
        a.dict()
        for a in articles
        if a.source.strip().lower() == source
    ]
    return JSONResponse(filtered)


@router.get("/fetch")
def fetch_source(source: str = Query(...)):
    articles = fetch_articles_from_source(source)
    save_articles(articles)
    return {
        "source": source,
        "count": len(articles),
        "articles": [a.dict() for a in articles]
    }

@router.get("/articles/filtered")
def filtered_articles(
    hours: int = Query(72, description="Zeitraum in Stunden"),
    source: Optional[str] = Query(None),
    keyword: Optional[str] = Query(None),
    teaser: bool = Query(False, description="Auch Teasertext nach Keyword durchsuchen")
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    keyword = keyword.lower() if keyword else None

    articles = [
        a for a in get_articles()
        if a.published_at >= since
    ]

    if source:
        articles = [a for a in articles if a.source.strip().lower() == source.strip().lower()]

    if keyword:
        import re
        pattern = rf"\b{re.escape(keyword)}\b"
        articles = [
            a for a in articles
            if (
                re.search(pattern, a.title.lower()) or
                (teaser and re.search(pattern, a.teaser.lower()))
            )
        ]

    return [a.dict() for a in articles]


@router.get("/media-positions/filtered")
def media_positions_filtered(
    hours: int = Query(72),
    source: str = Query(None),
    keyword: str = Query(None),
    teaser: bool = Query(False)
):
    from collections import Counter
    import re
    df = load_media_df()

    articles = get_articles_last_hours(hours)

    if source:
        articles = [a for a in articles if a.source.strip().lower() == source.strip().lower()]

    if keyword:
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        articles = [
            a for a in articles
            if (
                re.search(pattern, a.title.lower()) or
                (teaser and re.search(pattern, a.teaser.lower()))
            )
        ]

    matching_sources = [a.source.strip().lower() for a in articles]
    source_counts = Counter(matching_sources)

    filtered = df[df["norm_name"].isin(source_counts.keys())]

    return [
        {
            "medium": row["Medium"],
            "x": row["Systemnähe (X)"],
            "y": row["Globalismus (Y)"],
            "count": source_counts[row["norm_name"]]
        }
        for _, row in filtered.iterrows()
    ]

@router.get("/keywords/trending")
def keyword_trends():
    from collections import Counter
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words
    from datetime import datetime, timedelta, timezone

    timeframes = {
        "24h": 24,
        "72h": 72,
        "7d": 168,
        "30d": 720
    }

    result = {}

    for label, hours in timeframes.items():
        now_words = Counter()
        past_words = Counter()

        # Aktueller Zeitraum
        now_articles = get_articles_last_hours(hours)
        for a in now_articles:
            now_words.update(extract_relevant_words(a.title))

        # Vergleichszeitraum (davor)
        past_articles = get_articles_last_hours(hours * 2)
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
            "flop": [{"word": w, "delta": d, "change": round(r, 2)} for w, d, r in flop]
        }

    return result


@router.get("/keywords/extreme-bubble")
def extreme_keywords(hours: int = 72):
    from collections import defaultdict, Counter
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words

    # Medienkompass laden
    df = load_media_df()
    bias_map = dict(zip(df["norm_name"], df["Systemnähe (X)"]))

    articles = get_articles_last_hours(hours)

    words_extreme = Counter()
    words_other = Counter()
    word_sources = defaultdict(list)

    # Schwelle X-Achse ±0.5
    for a in articles:
        words = extract_relevant_words(a.title)
        norm_source = a.source.strip().lower()

        if norm_source not in bias_map:
            continue

        score = bias_map[norm_source]

        if abs(score) >= 0.0:
            words_extreme.update(words)
            for word in words:
                word_sources[word].append(score)
        else:
            words_other.update(words)

    result = []
    for word, count in words_extreme.items():
        if word in words_other or count < 3:
            continue
        scores = word_sources.get(word, [])
        if scores:
            avg_score = sum(scores) / len(scores)
            result.append((word, count, round(avg_score, 3)))

    return sorted(result, key=lambda x: abs(x[2]), reverse=True)


@router.get("/keywords/bias-score")
def keyword_bias_scores(hours: int = 72):
    from collections import defaultdict
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words

    df = load_media_df()
    media_bias = dict(zip(df["norm_name"], df["Systemnähe (X)"]))

    articles = get_articles_last_hours(hours)
    keyword_positions = defaultdict(list)

    for a in articles:
        norm_source = a.source.strip().lower()
        if norm_source not in media_bias:
            continue
        bias_score = media_bias[norm_source]
        words = extract_relevant_words(a.title)
        for word in words:
            keyword_positions[word].append(bias_score)

    # Berechne Mittelwert pro Keyword
    result = {}
    for word, scores in keyword_positions.items():
        if len(scores) >= 3:
            avg_score = sum(scores) / len(scores)
            result[word] = round(avg_score, 3)

    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))

@router.get("/keywords/bias-vector")
def keyword_bias_vector(hours: int = 72):
    from collections import defaultdict
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words

    df = load_media_df()
    bias_map = {
        row["norm_name"]: (row["Systemnähe (X)"], row["Globalismus (Y)"])
        for _, row in df.iterrows()
    }

    articles = get_articles_last_hours(hours)
    keyword_coords = defaultdict(list)

    for a in articles:
        norm_source = a.source.strip().lower()
        if norm_source not in bias_map:
            continue
        x, y = bias_map[norm_source]
        words = extract_relevant_words(a.title)
        for word in words:
            keyword_coords[word].append((x, y))

    result = {}
    for word, coords in keyword_coords.items():
        if len(coords) < 3:
            continue
        avg_x = sum(x for x, _ in coords) / len(coords)
        avg_y = sum(y for _, y in coords) / len(coords)
        result[word] = {"x": round(avg_x, 3), "y": round(avg_y, 3)}

    return result

@router.get("/keywords/timeline")
def keyword_timeline(word: str, hours: int = 72):
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words

    # Aktueller Zeitpunkt (abgerundet zur vollen Stunde)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=hours)

    # Stündliche Zeitintervalle vorbereiten
    time_slots = [start + timedelta(hours=i) for i in range(hours + 1)]

    # Keyword vorbereiten
    word = word.lower()
    articles = get_articles_last_hours(hours)

    # Zähle Treffer pro Stunde
    bins = defaultdict(int)
    for a in articles:
        ts = a.published_at.replace(minute=0, second=0, microsecond=0)
        words = extract_relevant_words(a.title)
        if word in words:
            bins[ts] += 1

    # Ergebnis: alle Zeitslots mit count (auch 0)
    result = [{"time": t.isoformat(), "count": bins.get(t, 0)} for t in time_slots]
    return result

@router.get("/keywords/top-absolute")
def keywords_top_absolute(hours: int = 72):
    from collections import Counter
    from datetime import datetime, timedelta, timezone
    from app.models import get_articles_last_hours
    from app.core.clean_utils import extract_relevant_words

    now_words = Counter()
    past_words = Counter()

    # Aktueller Zeitraum
    now_articles = get_articles_last_hours(hours)
    for a in now_articles:
        now_words.update(extract_relevant_words(a.title))

    # Vergleichszeitraum
    past_articles = get_articles_last_hours(hours * 2)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    for a in past_articles:
        if a.published_at < cutoff:
            past_words.update(extract_relevant_words(a.title))

    # Top 50 absolute Werte + Veränderung
    result = []
    for word, current_count in now_words.most_common(30):
        prev_count = past_words.get(word, 0)
        delta = current_count - prev_count
        change_pct = (delta / prev_count * 100) if prev_count > 0 else 0
        result.append({
            "word": word,
            "current": current_count,
            "previous": prev_count,
            "delta": delta,
            "change_pct": round(change_pct, 2)
        })

    return result
