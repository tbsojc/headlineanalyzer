<!-- app/ui/date-range-spec.js -->
<script>
  // Globale Spezifikation für das UI (keine API-Calls, nur Regeln/Labels)
  window.DATE_RANGE_SPEC = Object.freeze({
    DEFAULT_PRESET_HOURS: 72,
    ALLOWED_PRESET_HOURS: [24, 72, 7 * 24, 30 * 24],
    TO_IS_EXCLUSIVE: true,
    MAX_RANGE_DAYS: 180,
    SMALL_RANGE_CUTOFF_DAYS: 14,
    TIMEZONE_DISPLAY: "auto", // "auto" = Browser-TZ für Anzeige; Server rechnet UTC
    I18N: {
      customButton: "Benutzerdefiniert…",
      apply: "Übernehmen",
      cancel: "Abbrechen",
      startLabel: "Start",
      endLabel: "Ende",
      invalidRange: "Ungültiger Zeitraum. Prüfe Start/Ende.",
      tooLarge: "Zeitraum zu groß (max. 180 Tage).",
      futureNotAllowed: "Zeiten in der Zukunft sind nicht erlaubt.",
      badgePrefix: "📅",
    }
  });
</script>
