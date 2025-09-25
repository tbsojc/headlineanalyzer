# Date-Range Spec (A)

## Regeln
- `from`/`to` haben Vorrang vor `hours`.
- Zeitintervall ist halboffen: **[from, to)** — `to` ist exklusiv.
- Server arbeitet in UTC; Anzeige im UI in Browser-Zeitzone.
- **Maximale Spannweite:** 180 Tage (konfigurierbar).
- **Buckets:** ≤ 14 Tage → stündlich; > 14 Tage → täglich.

## Presets
- 24h, 72h, 7d, 30d — Standard: 72h.

## Fehlerfälle
- `to ≤ from`, Zukunft, Überschreitung Max-Range → HTTP 400.
- UI zeigt passende Meldungen an (siehe `date-range-spec.js`).

Diese Datei spiegelt `app/api/date_range_spec.py` und `app/ui/date-range-spec.js`
wider und dient als Referenz bei Code-Reviews.


## API-Vertrag (B)
- Alle Zeit-basierten Endpunkte akzeptieren zusätzlich `from`/`to` (ISO-8601, inkl. TZ).
- Wenn `from` **und** `to` gesetzt → Range-Mode; sonst `hours`.
- In B wird intern noch `hours` genutzt; echte Range-Abfragen folgen in C.
- Fehler: ungültiges Format, `to ≤ from`, Zukunft, > MAX_RANGE_DAYS → HTTP 400.

## Implementierung (C)
- Repository-Funktion `get_articles_between(db, start, end)` nutzt das halboffene Intervall [start, end).
- Alle Routen wählen via `_select_articles()` automatisch Range- vs. Hours-Modus.
- Timeline: Bucket-Regel aus Spec (`hour` bis 14 Tage, sonst `day`).
- Top-Absolute: Vorperiode derzeit nur im Hours-Modus; Range-Modus ohne Vorperiode (wird in E optional erweitert).
