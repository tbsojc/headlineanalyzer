from fastapi import APIRouter
from collections import defaultdict, Counter
from app.models import get_articles
from app.services.tagging import extract_tags
from app.core.clean_utils import extract_relevant_words
from app.models import get_articles_last_hours
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

router = APIRouter()





@router.get("/headlines/words")
def headline_word_counts(top_n: int = 100, hours: int = 72, source: str = None, keyword: str = None, teaser: bool = False):
    counter = Counter()
    articles = get_articles_last_hours(hours)

    if source:
        articles = [a for a in articles if a.source.strip().lower() == source.strip().lower()]

    if keyword:
        import re
        pattern = rf"\b{re.escape(keyword.lower())}\b"
        articles = [
            a for a in articles
            if (
                re.search(pattern, a.title.lower()) or
                (teaser and re.search(pattern, a.teaser.lower()))
            )
        ]

    for a in articles:
        counter.update(extract_relevant_words(a.title))

    return counter.most_common(top_n)
