# app/services/tagging.py

from typing import List, Dict, Optional


# Zentrale Definition aller Tags:
# - key: Tag-Name (so wird er in Article.tags gespeichert)
# - category: Oberkategorie (z.B. Partei, Person, Nation, Institution, Thema, Krise, Konflikt)
# - keywords: Schlagwörter, nach denen in Titel/Teaser gesucht wird
TAG_DEFINITIONS: Dict[str, Dict[str, object]] = {
    # Nationen
    "Ukraine": {
        "category": "Nation",
        "keywords": ["ukraine", "kiew", "selenskyj", "donezk", "lwiw"],
    },
    "Russland": {
        "category": "Nation",
        "keywords": ["russland", "moskau", "putin", "kreml"],
    },
    "USA": {
        "category": "Nation",
        "keywords": ["usa", "biden", "trump", "washington"],
    },
    "China": {
        "category": "Nation",
        "keywords": ["china", "beijing", "xi jinping", "taiwan", "shanghai"],
    },
    "Deutschland": {
        "category": "Nation",
        "keywords": [
            "deutschland",
            "bundesrepublik",
            "berlin",
            "bundesregierung",
            "kabinet",
            "minister",
            "ministerium",
            "bundestag",
        ],
    },
    "Frankreich": {
        "category": "Nation",
        "keywords": ["frankreich", "paris", "macron"],
    },
    "Großbritannien": {
        "category": "Nation",
        "keywords": ["briten", "london", "westminster", "starmer"],
    },

    # Institutionen / Organisationen
    "EU": {
        "category": "Institution",
        "keywords": [" EU ", "EZB", "vdl", "von der leyen", "komission", "kallas", "brüssel"],
    },
    "NATO": {
        "category": "Institution",
        "keywords": ["nato"],
    },

    # Parteien
    "CDU": {
        "category": "Partei",
        "keywords": ["cdu", "merz", "union", "csu", "soeder", "weimer"],
    },
    "SPD": {
        "category": "Partei",
        "keywords": ["spd", "scholz", "esken", "klingbeil", "bas", "pistorius"],
    },
    "Grüne": {
        "category": "Partei",
        "keywords": ["grüne", "habeck", "baerbock"],
    },
    "FDP": {
        "category": "Partei",
        "keywords": ["fdp", "lindner"],
    },
    "AfD": {
        "category": "Partei",
        "keywords": ["afd", "weidel", "chrupalla", "höcke"],
    },
    "BSW": {
        "category": "Partei",
        "keywords": ["bsw", "wagenknecht", "dagdalen"],
    },
    "Linke": {
        "category": "Partei",
        "keywords": ["linke", "wissler", "bartsch", "reichineck", "van aken"],
    },

    # Konflikte
    "Nahostkonflikt": {
        "category": "Konflikt",
        "keywords": [
            "nahost",
            "gaza",
            "israel",
            "hamas",
            "idf",
            "netanjahu",
            "tel aviv",
            "jerusalem",
            "iran",
            "westjordanland",
        ],
    },

    # Krisen / wirtschaftliche Themen
    "Energiekrise": {
        "category": "Krise",
        "keywords": ["energiekrise", "gaspreis", "strompreis", "versorgung"],
    },
    "Inflation": {
        "category": "Krise",
        "keywords": ["inflation", "teuerung", "preissteigerung"],
    },
    "Wirtschaftskrise": {
        "category": "Krise",
        "keywords": ["rezession", "wirtschaftskrise", "insolvenz"],
    },

    # Gesellschaftliche / übergreifende Themen
    "Migration": {
        "category": "Thema",
        "keywords": ["migration", "flüchtling", "asyl", "grenze", "syrer", "afghane", "iraker", "somalier"],
    },
    "Umweltkatastrophen": {
        "category": "Thema",
        "keywords": ["überschwemmung", "sturm", "flut", "waldbrand"],
    },
    "Klima": {
        "category": "Thema",
        "keywords": ["klima", "klimawandel", "co2", "hitze"],
    },

    # Personen
    "Merz": {
        "category": "Person",
        "keywords": ["merz"],
    },
    "Weidel": {
        "category": "Person",
        "keywords": ["weidel"],
    },
    "Trump": {
        "category": "Person",
        "keywords": ["trump", "donald"],
    },
    "Selenskyj": {
        "category": "Person",
        "keywords": ["selenskyj"],
    },
    "Putin": {
        "category": "Person",
        "keywords": ["putin"],
    },
    "Netanjahu": {
        "category": "Person",
        "keywords": ["netanjahu"],
    },
}


# Optional: für Rückwärtskompatibilität – wenn irgendwo noch KEYWORDS importiert wird
KEYWORDS: Dict[str, List[str]] = {
    tag: list(defn.get("keywords", [])) for tag, defn in TAG_DEFINITIONS.items()
}


def classify_topic(title: str, teaser: str) -> str:
    """
    Gibt ein Hauptthema (Topic) zurück.
    Aktuell: erster passender Tag aus TAG_DEFINITIONS.
    """
    content = f"{title} {teaser}".lower()
    for tag, meta in TAG_DEFINITIONS.items():
        keywords: List[str] = meta.get("keywords", [])  # type: ignore[assignment]
        if any(k.lower() in content for k in keywords):
            return tag
    return "Sonstiges"


def extract_tags(title: str, teaser: str) -> List[str]:
    """
    Gibt alle Tags zurück, deren Keywords in Titel/Teaser vorkommen.
    """
    content = f"{title} {teaser}".lower()
    tags: List[str] = []
    for tag, meta in TAG_DEFINITIONS.items():
        keywords: List[str] = meta.get("keywords", [])  # type: ignore[assignment]
        if any(k.lower() in content for k in keywords):
            tags.append(tag)
    return tags


def get_tag_category(tag: str) -> Optional[str]:
    """
    Liefert die Kategorie eines Tags (z.B. 'Partei', 'Person', 'Nation', ...),
    oder None, falls das Tag unbekannt ist.
    """
    meta = TAG_DEFINITIONS.get(tag)
    if not meta:
        return None
    category = meta.get("category")
    return str(category) if category is not None else None


def list_categories() -> List[str]:
    """
    Liefert eine sortierte Liste aller verwendeten Kategorien.
    """
    cats = {str(meta.get("category")) for meta in TAG_DEFINITIONS.values() if meta.get("category")}
    return sorted(cats)
