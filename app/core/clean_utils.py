from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("german") + stopwords.words("english"))


def clean_html(raw_html: str) -> str:
    """
    Entfernt HTML-Tags aus einem gegebenen HTML-Text.

    Args:
        raw_html (str): Eingabetext mit HTML.

    Returns:
        str: Nur noch der sichtbare Text ohne HTML-Tags.
    """
    return BeautifulSoup(raw_html, "html.parser").get_text()


def extract_relevant_words(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]
