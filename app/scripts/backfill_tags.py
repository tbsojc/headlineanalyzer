# app/scripts/backfill_topics_and_tags.py

from app.database import SessionLocal
from app.models_sql import ArticleORM
from app.services.tagging import classify_topic, extract_tags


def backfill_topics_and_tags() -> None:
    """
    Geht einmal über alle Artikel und überschreibt konsequent:

      - topic  -> wird immer neu mit classify_topic(...) gesetzt
      - tags   -> werden immer neu mit extract_tags(...) gesetzt

    Damit kannst du dein KEYWORDS-Dict in app/services/tagging.py ändern
    und anschließend die komplette Datenbank mit den neuen Regeln
    neu beschriften, ohne Artikel zu verlieren.
    """
    db = SessionLocal()
    try:
        query = db.query(ArticleORM)
        total = query.count()
        updated = 0

        print(f"Starte Backfill für {total} Artikel mit neuen Topics/Tags...")

        # In Batches iterieren, damit es bei vielen Artikeln nicht zu viel Speicher frisst
        for idx, art in enumerate(query.yield_per(500), start=1):
            title = art.title or ""
            teaser = art.teaser or ""

            # Neue Topic- und Tag-Werte nach aktuellen Regeln berechnen
            new_topic = classify_topic(title, teaser)
            new_tags = extract_tags(title, teaser)

            # Nur für Zählung prüfen, ob sich etwas ändert
            changed = (new_topic != art.topic) or (new_tags != art.tags)

            # In jedem Fall überschreiben wir die Felder mit den neuen Werten
            art.topic = new_topic
            art.tags = new_tags

            if changed:
                updated += 1

            # Zwischendurch committen und Status ausgeben
            if idx % 500 == 0:
                db.commit()
                print(f"... {idx}/{total} verarbeitet, {updated} aktualisiert")

        db.commit()
        print(f"Fertig: {updated} von {total} Artikeln mit neuen Topics/Tags aktualisiert.")

    finally:
        db.close()


if __name__ == "__main__":
    backfill_topics_and_tags()
