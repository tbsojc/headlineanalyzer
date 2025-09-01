# app/services/tagging.py

from typing import List, Dict

KEYWORDS = {
    "Migration": ["migration", "flüchtling", "asyl", "grenzschutz"],
    "Wahl": ["wahl", "election", "stimmen", "urne"],
    "Krieg": ["krieg", "konflikt", "waffen", "angriff"],
    "Wirtschaft": ["wirtschaft", "konjunktur", "inflation", "arbeitsmarkt"],
    "Klima": ["klimawandel", "hitzewelle", "co2", "umwelt"],
    "Gesundheit": ["gesundheit", "krankheit", "impfung", "virus"],
    "Bildung": ["schule", "bildung", "lehrermangel", "abi"],
    "Digitalisierung": ["digital", "ki", "internet", "cybersicherheit"],
    "Künstliche Intelligenz": ["ki", "gpt", "algorithmus", "maschinelles lernen"],
    "Energie": ["energie", "strom", "gas", "erneuerbare"],
    "Verkehr": ["verkehr", "bahn", "flug", "auto"],
    "Wohnen": ["miete", "wohnen", "immobilien", "bau"],
    "Soziales": ["rente", "pflege", "sozialhilfe", "armut"],
    "Finanzen": ["finanzen", "aktien", "börse", "zins"],
    "Justiz": ["gericht", "urteil", "verfassung", "prozess"],
    "Innenpolitik": ["bundestag", "gesetz", "regierung", "minister"],
    "Außenpolitik": ["diplomatie", "botschaft", "vereinte nationen", "auswärtiges amt"],
    "Sport": ["olympia", "em", "fußball", "medaille"],
    "Kultur": ["film", "kunst", "theater", "festival"],
    "Technologie": ["tech", "robotik", "innovation", "startup"],
    "Arbeit": ["arbeitslosigkeit", "tarif", "streik", "gehalt"],
    "Gender": ["gender", "gleichstellung", "frauenquote", "lgbtq"],
    "Kriminalität": ["verbrechen", "polizei", "ermittlung", "raub"],
    "Naturkatastrophen": ["überschwemmung", "erdbeben", "sturm", "waldbrand"],
    "Bildungspolitik": ["schulreform", "bildungsminister", "pädagogik", "lehrplan"]
}

def classify_topic(title: str, teaser: str) -> str:
    content = f"{title} {teaser}".lower()
    for topic, keywords in KEYWORDS.items():
        if any(k in content for k in keywords):
            return topic
    return "Sonstiges"

def extract_tags(title: str, teaser: str) -> List[str]:
    content = f"{title} {teaser}".lower()
    tags = []
    for topic, keywords in KEYWORDS.items():
        if any(k in content for k in keywords):
            tags.append(topic)
    return tags
