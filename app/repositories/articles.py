from sqlalchemy.orm import Session
from sqlalchemy import select, func
from datetime import datetime, timedelta, timezone
from app.models_sql import ArticleORM
from app.schemas import Article as ArticleSchema
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from app.services.tagging import classify_topic, extract_tags

def _utc_aware(dt):
    """Hilfsfunktion: published_at konsistent in UTC speichern."""
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def get_articles_last_hours(db: Session, hours: int) -> list[ArticleORM]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = (
        select(ArticleORM)
        .where(ArticleORM.published_at >= since)
        .order_by(ArticleORM.published_at.desc())
    )
    items = list(db.execute(stmt).scalars())
    # 👇 Hier einmalig „aware“ machen (für alte, naive Einträge)
    for a in items:
        a.published_at = _utc_aware(a.published_at)
    return items


def bulk_upsert_articles(db: Session, items) -> int:
    """
    Persistiert neue Artikel (aus Feeds) in der SQLite-Datenbank.
    Neue Felder: topic (Single-Topic), tags (Multi-Topic-Labels).
    Bei Konflikt (gleiche URL) wird Artikel übersprungen.
    """
    # URL-Entduplizierung
    by_url = {}
    for a in items:
        if not getattr(a, "url", None):
            continue
        by_url[a.url] = a

    rows = []

    for a in by_url.values():
        title = getattr(a, "title", "") or ""
        teaser = getattr(a, "teaser", "") or ""
        source = getattr(a, "source", "") or ""
        published = getattr(a, "published_at", None)

        # Thema bestimmen (falls Feed nichts gegeben hat)
        topic = getattr(a, "topic", None)
        if not topic or topic.strip() == "":
            topic = classify_topic(title, teaser)

        # Tags bestimmen (falls Feed keine enthält)
        tag_list = getattr(a, "tags", None)
        if not tag_list:
            tag_list = extract_tags(title, teaser)

        rows.append({
            "title": title,
            "teaser": teaser,
            "url": a.url,
            "source": source,
            "topic": topic,
            "tags": tag_list,  # <-- NEU
            "published_at": _utc_aware(published),
        })

    if not rows:
        return 0

    table = ArticleORM.__table__
    stmt = sqlite_insert(table).values(rows).on_conflict_do_nothing(
        index_elements=["url"]
    )

    res = db.execute(stmt)
    db.commit()
    return res.rowcount or 0

def get_articles_between(db: Session, start_utc: datetime, end_utc: datetime, limit: int | None = None, offset: int = 0):
    q = (
        db.query(ArticleORM)
          .filter(ArticleORM.published_at >= start_utc)
          .filter(ArticleORM.published_at <  end_utc)
          .order_by(ArticleORM.published_at.desc())
    )
    if offset: q = q.offset(offset)
    if limit:  q = q.limit(limit)
    return q.all()
