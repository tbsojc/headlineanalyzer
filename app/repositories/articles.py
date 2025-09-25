from sqlalchemy.orm import Session
from sqlalchemy import select, func
from datetime import datetime, timedelta, timezone
from app.models_sql import ArticleORM
from app.schemas import Article as ArticleSchema
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

def _utc_aware(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

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
    Fügt Artikel in Bulk ein; doppelte URLs werden ignoriert.
    Gibt die Anzahl *neu* eingefügter Zeilen zurück.
    """

    # 1) lokale Dedupe nach URL (falls dieselbe URL mehrfach im Fetch ist)
    by_url = {}
    for a in items:
        if not getattr(a, "url", None):
            continue
        by_url[a.url] = a  # last-one-wins

    rows = [{
        "title": a.title,
        "teaser": (a.teaser or ""),
        "url": a.url,
        "source": a.source,
        "topic": (a.topic or "Sonstiges"),
        "published_at": _utc_aware(a.published_at),
    } for a in by_url.values()]

    if not rows:
        return 0

    # 2) SQLite: ON CONFLICT DO NOTHING auf der Unique-URL
    table = ArticleORM.__table__
    stmt = sqlite_insert(table).values(rows).on_conflict_do_nothing(
        index_elements=["url"]  # entspricht deinem UniqueConstraint auf url
    )

    res = db.execute(stmt)
    db.commit()

    # rowcount = Anzahl wirklich eingefügter Zeilen (Konflikte zählen nicht)
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
