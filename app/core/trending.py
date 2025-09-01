from collections import Counter
import re
import nltk
from app.models import get_articles_last_hours
from app.core.clean_utils import clean_html

nltk.download("stopwords")
from nltk.corpus import stopwords

EXTRA_STOPWORDS = {"alignleft", "amp", "nbsp"}
STOPWORDS = set(stopwords.words("english") + stopwords.words("german"))
STOPWORDS.update(EXTRA_STOPWORDS)

def clean_text(text):
    text = clean_html(text)
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 3]

def get_trending_keywords(hours=24, top_n=15):
    articles = get_articles_last_hours(hours)
    counter = Counter()
    for art in articles:
        words = clean_text(art.title + " " + art.teaser)
        counter.update(words)
    return counter.most_common(top_n)

def get_multi_period_trending():
    return {
        "12h": get_trending_keywords(12, top_n=5),
        "24h": get_trending_keywords(24, top_n=5),
        "3d": get_trending_keywords(72, top_n=5),
        "7d": get_trending_keywords(168, top_n=5),
        "30d": get_trending_keywords(720, top_n=5)
    }
