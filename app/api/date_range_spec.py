# app/api/date_range_spec.py
"""
Zentrale Spezifikation für Date-Range-Handling.

Diese Datei enthält nur Konstanten und kleine Hilfsfunktionen und wird
von den Flask-Routen (C/E) sowie den Tests (J) importiert.

Designentscheidungen (A):
- from/to haben Vorrang vor hours
- 'to' ist EXKLUSIV (Halboffen [from, to) )
- Server rechnet intern in UTC
- Maximal erlaubte Spannweite begrenzen (DoS/teure Scans vermeiden)
- Bucket-Regel: <= 14 Tage = 'hour', > 14 Tage = 'day'
"""

# === Bucketing Helpers (E) ====================================================
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable, Literal

# ====== Presets ======
DEFAULT_PRESET_HOURS: int = 72
ALLOWED_PRESET_HOURS: tuple[int, ...] = (24, 72, 7 * 24, 30 * 24)

# ====== Range-Policy ======
TO_IS_EXCLUSIVE: bool = True
MAX_RANGE_DAYS: int = 180  # harte Obergrenze für freie Ranges

# ====== Bucketing ======
SMALL_RANGE_CUTOFF_DAYS: int = 14  # bis einschließlich => 'hour', darüber => 'day'

Bucket = Literal["hour", "day", "week"]

def ensure_utc(dt: datetime) -> datetime:
    """Erzwingt UTC-Awareness; naive Werte werden als UTC interpretiert."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def validate_range(dt_from: datetime, dt_to: datetime) -> None:
    """Validiert [from, to) gemäß Policy A; wirft ValueError bei Verstößen."""
    f = ensure_utc(dt_from)
    t = ensure_utc(dt_to)

    if t <= f:
        raise ValueError("Invalid range: 'to' must be greater than 'from' (exclusive).")

    # harte Obergrenze
    if t - f > timedelta(days=MAX_RANGE_DAYS):
        raise ValueError(f"Range too large. Max {MAX_RANGE_DAYS} days.")

    # keine Zukunft
    now_utc = datetime.now(timezone.utc)
    if f > now_utc or t > now_utc:
        raise ValueError("Range cannot be in the future.")

def bucket_for_range(dt_from: datetime, dt_to: datetime) -> str:
    """Liefert 'hour' oder 'day' je nach Spannweite."""
    f = ensure_utc(dt_from)
    t = ensure_utc(dt_to)
    if (t - f) <= timedelta(days=SMALL_RANGE_CUTOFF_DAYS):
        return "hour"
    return "day"

def parse_iso_to_utc(value: str) -> datetime:
    """
    Parst ISO-8601 (mit/ohne Offset) und liefert einen UTC-aware datetime.
    Wirf ValueError bei ungültigem Format.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def decide_time_window(
    hours: Optional[int],
    from_str: Optional[str],
    to_str: Optional[str],
) -> dict:
    """
    Vertrags-Helfer: entscheidet, ob 'hours' oder 'from/to' gilt.
    - Wenn from_str UND to_str gesetzt → validiere & gib {'mode':'range', 'from':dt, 'to':dt}
    - Sonst → gib {'mode':'hours', 'hours':int}
    Wirft ValueError bei invalidem Range.
    """
    if from_str and to_str:
        f = parse_iso_to_utc(from_str)
        t = parse_iso_to_utc(to_str)
        validate_range(f, t)
        return {"mode": "range", "from": f, "to": t}
    # Fallback auf Preset-Hours (inkl. Default)
    h = int(hours or DEFAULT_PRESET_HOURS)
    if h not in ALLOWED_PRESET_HOURS:
        # erlaub auch „freie“ Werte, aber zumindest positiv
        if h <= 0:
            h = DEFAULT_PRESET_HOURS
    return {"mode": "hours", "hours": h}



def iter_slots(start: datetime, end: datetime, bucket: Bucket) -> list[datetime]:
    """
    Erzeugt Slot-Grenzen inkl. Endpunkt, also [start, ..., end] für Gruppierung.
    Start/Ende werden passend gerundet.
    """
    f = ensure_utc(start)
    t = ensure_utc(end)
    if bucket == "hour":
        f = f.replace(minute=0, second=0, microsecond=0)
        t = t.replace(minute=0, second=0, microsecond=0)
        step = timedelta(hours=1)
        steps = int((t - f).total_seconds() // step.total_seconds())
        return [f + step * i for i in range(steps + 1)]
    elif bucket == "day":
        f = f.replace(hour=0, minute=0, second=0, microsecond=0)
        t = t.replace(hour=0, minute=0, second=0, microsecond=0)
        days = (t - f).days
        return [f + timedelta(days=i) for i in range(days + 1)]
    elif bucket == "week":
        # ISO-Woche: montags 00:00
        dow = (f.weekday() + 7) % 7  # 0=Mo
        f = (f - timedelta(days=dow)).replace(hour=0, minute=0, second=0, microsecond=0)
        dow_t = (t.weekday() + 7) % 7
        t = (t - timedelta(days=dow_t)).replace(hour=0, minute=0, second=0, microsecond=0)
        weeks = int((t - f).days // 7)
        return [f + timedelta(weeks=i) for i in range(weeks + 1)]
    else:
        raise ValueError("unknown bucket")

def round_to_bucket(dt: datetime, bucket: Bucket) -> datetime:
    dt = ensure_utc(dt)
    if bucket == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if bucket == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket == "week":
        dow = (dt.weekday() + 7) % 7
        base = dt - timedelta(days=dow)
        return base.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError("unknown bucket")

def previous_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Gibt die Vorperiode gleicher Länge direkt vor [start, end) zurück."""
    f = ensure_utc(start); t = ensure_utc(end)
    span = t - f
    return f - span, f
