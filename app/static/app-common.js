
    let activeBiasFilter = null;
    let MEDIA_POS_CACHE = null;
    let keywordScatterChart = null;
    let keywordTimelineChart = null;
    let mediaCompassChart = null;
    let selectedKeyword = "";
    let _lastCompassKey = '';
    let USER_TIME_OVERRIDE = false;

    // Zentrale Filter – eine Quelle der Wahrheit
    const FILTERS = {
      from: null,
      to: null,
      source: null,
      keyword: null,
      teaser: false,
      ngram: 1
    };

    const ACTIVE_SOURCES = new Set();        // ← neue Mehrfachauswahl
    const ACTIVE_COUNTRIES = new Set();
    const activeBiasFilters = new Set();     // ← ersetzt Single-Bias
    const BLINDSPOT_UI = {
      min_sources: 2,        // (Legacy – bleibt vorerst)
      min_total: 10,         // (Legacy – bleibt vorerst)
      ratio_max: 0.05,       // (Legacy – bleibt vorerst)
      // NEU: echte Bereiche
      min_sources_range: [2, 10],
      min_total_range:   [10, 20],
      ratio_range:       [0.01, 0.05]   // Dezimal (1–5 %)
    };
    const WORDCLOUD_CFG = {
      maxWords: 120,   // 10..250
      minCount: 7      // 1..50
    };


    // Keyword aus /ui/<keyword> extrahieren
    const path = window.location.pathname;
    const parts = path.split("/").filter(Boolean);
    let pageKeyword = null;
    if (parts.length === 2 && parts[0] === "ui") {
      pageKeyword = decodeURIComponent(parts[1]);
    }




    // Kompatibilität: alter Code darf noch "filters" benutzen
    Object.defineProperty(window, 'filters', {
      get() { return FILTERS; },
      set(v) {
        if (v && typeof v === 'object') {
          FILTERS.from    = v.from    ?? FILTERS.from;
          FILTERS.to      = v.to      ?? FILTERS.to;
          FILTERS.source  = v.source  ?? FILTERS.source;
          FILTERS.keyword = v.keyword ?? FILTERS.keyword;
          FILTERS.teaser  = v.teaser  ?? FILTERS.teaser;
          FILTERS.ngram   = v.ngram   ?? FILTERS.ngram;
        }
      }
    });

    // === Lazy-Mount Helper (einmalig) ===
    function lazyMount(id, loaderFn, rootMargin = '200px') {
      const el = document.getElementById(id);
      if (!el) return;
      const io = new IntersectionObserver((entries) => {
        if (entries.some(e => e.isIntersecting)) {
          try { loaderFn(); } finally { io.disconnect(); }
        }
      }, { rootMargin });
      io.observe(el);
    }

window.addEventListener('load', () => {
  // URL -> State (Zeitraum, Quellen, Länder, evtl. keyword aus Query)
  getFiltersFromURL();

  // URL normalisieren (exactly one of hours / from-to)
  updateURLFromFilters(true);

  // Wenn wir auf /ui/<keyword> sind, Keyword aus Pfad in Filter übernehmen
  if (pageKeyword) {
    FILTERS.keyword = pageKeyword;
    selectedKeyword = pageKeyword;

    const kwInput = document.getElementById("keywordInput");
    if (kwInput) kwInput.value = pageKeyword;
  }

  // Badge initial setzen
  updateDateRangeBadge();

  // UI sync
  const srcSel = document.getElementById("sourceSelect");
  if (srcSel) srcSel.value = FILTERS.source || "";

  const kwInput2 = document.getElementById("keywordInput");
  if (kwInput2 && !pageKeyword) {
    // nur dann aus Query übernehmen, wenn kein Pfad-Keyword gesetzt ist
    kwInput2.value = FILTERS.keyword || "";
  }

  const teaserToggle = document.getElementById("teaserToggle");
  if (teaserToggle) teaserToggle.checked = !!FILTERS.teaser;

  initCountryPanel();

  // Initiale Datenladungen (wie früher)
  loadWordcloud();
  loadTopAbsoluteKeywords();
  loadWeeklyChronicle();
  loadFilteredArticles();

  // Rest, sobald der Browser Leerlauf hat (Fallback auf setTimeout)
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 150));
  idle(async () => {
    // Sofort im Idle (leicht / wichtig)
    await loadMediaCompass();
    await loadFilteredArticles();
    await loadBlindspotFeed();


    // Zusatzlogik NUR für /ui/<keyword>
    if (pageKeyword) {
      // so, als hätte der Nutzer das Keyword gesucht
      await searchByKeyword(pageKeyword);

      // Kompass gefiltert nach Keyword neu laden (falls vorhanden)
      if (typeof loadFilteredMediaCompass === "function") {
        await loadFilteredMediaCompass();
      }

      const box = document.getElementById("keywordSidesBox");
      if (box) {
        box.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    // Schwere Blöcke erst, wenn im Viewport
    lazyMount('keywordTrends',   loadKeywordTrends);
    lazyMount('extremeBubbles',  loadExtremeBubbles);
  });
});


function initSourcePanelFromSelect() {
  const sel = document.getElementById('sourceSelect');
  const panel = document.getElementById('sourcePanel');
  if (!sel || !panel) return;

  // Aus vorhandenen <option>-Werten Checkboxen bauen
  const frag = document.createDocumentFragment();
  Array.from(sel.options).forEach(opt => {
    if (!opt.value) return; // "-- Alle Quellen --"
    const id = `src_${opt.value.replace(/\s+/g,'_')}`;
    const label = document.createElement('label');
    label.setAttribute('for', id);
    label.innerHTML = `<input type="checkbox" id="${id}" class="source-box" value="${opt.value}">
                       ${opt.textContent}`;
    frag.appendChild(label);
  });
  panel.replaceChildren(frag);
}

function updateSourceBtnLabel(){
  const btn = document.getElementById('sourceMultiBtn');
  const n = ACTIVE_SOURCES.size;
  btn.textContent = n === 0 ? 'Alle Quellen' : `${n} ausgewählt`;
}

function appendSources(qs){
  // mehrere ?source=… Parameter anhängen
  if (ACTIVE_SOURCES.size === 0) return;
  for (const s of ACTIVE_SOURCES) qs.append('source', s);
}


// Panel togglen
document.getElementById('sourceMultiBtn').addEventListener('click', () => {
  const p = document.getElementById('sourcePanel');
  p.hidden = !p.hidden;
});
document.addEventListener('click', (e) => {
  const btn = document.getElementById('sourceMultiBtn');
  const panel = document.getElementById('sourcePanel');
  if (!panel.contains(e.target) && e.target !== btn) panel.hidden = true;
});

// Checkbox-Änderungen
document.getElementById('sourcePanel').addEventListener('change', (e) => {
  if (!e.target.classList.contains('source-box')) return;
  const val = e.target.value;
  if (e.target.checked) ACTIVE_SOURCES.add(val);
  else ACTIVE_SOURCES.delete(val);
  updateSourceBtnLabel();
  updateURLFromFilters(false);  // URL sync (s. 2.5)
  applyFilters();
});

// Init beim Laden: aus <select> Checkboxen bauen
window.addEventListener('load', () => {
  initSourcePanelFromSelect();
  // Vorbelegung aus URL (s. 2.5) setzt ACTIVE_SOURCES; danach UI labeln & Häkchen setzen:
  const panel = document.getElementById('sourcePanel');
  Array.from(panel.querySelectorAll('.source-box')).forEach(cb => {
    cb.checked = ACTIVE_SOURCES.has(cb.value);
  });
  updateSourceBtnLabel();
});

window.addEventListener('load', () => {
  const elCount = document.getElementById('wcCount');
  const elCountVal = document.getElementById('wcCountVal');
  const elMin = document.getElementById('wcMin');
  const elMinVal = document.getElementById('wcMinVal');

  if (elCount && elCountVal) {
    const syncCount = (commit=false) => {
      WORDCLOUD_CFG.maxWords = parseInt(elCount.value, 10);
      elCountVal.textContent = WORDCLOUD_CFG.maxWords;
      if (commit) { loadWordcloud(); renderActiveFilterChips(); }
    };
    elCount.addEventListener('input', () => syncCount(false));
    ['change','mouseup','touchend'].forEach(ev => elCount.addEventListener(ev, () => syncCount(true)));
    syncCount(false);
  }

  if (elMin && elMinVal) {
    const syncMin = (commit=false) => {
      WORDCLOUD_CFG.minCount = parseInt(elMin.value, 10);
      elMinVal.textContent = `≥ ${WORDCLOUD_CFG.minCount}×`;
      if (commit) { loadWordcloud(); renderActiveFilterChips(); }
    };
    elMin.addEventListener('input', () => syncMin(false));
    ['change','mouseup','touchend'].forEach(ev => elMin.addEventListener(ev, () => syncMin(true)));
    syncMin(false);
  }
});


async function initCountryPanel(){
  const panel = document.getElementById('countryPanel');
  const btn   = document.getElementById('countryMultiBtn');
  if (!panel || !btn) return;

  try {
    const res = await fetch('/countries', { cache: 'no-store' });
    const data = await res.json();
    const codes = Array.isArray(data.countries) ? data.countries : [];

    const frag = document.createDocumentFragment();
    codes.forEach(code => {
      const id = `cty_${code}`;
      const label = document.createElement('label');
      label.setAttribute('for', id);
      label.innerHTML = `<input type="checkbox" id="${id}" class="country-box" value="${code}"> ${code}`;
      frag.appendChild(label);
    });
    panel.replaceChildren(frag);

    // Vorbelegung aus URL
    const params = new URLSearchParams(location.search);
    const pre = params.getAll('country');
    ACTIVE_COUNTRIES.clear();
    for (const c of pre) if (c && c.trim()) ACTIVE_COUNTRIES.add(c.trim().toUpperCase());

    // Häkchen setzen
    panel.querySelectorAll('.country-box').forEach(cb => {
      cb.checked = ACTIVE_COUNTRIES.has(cb.value);
    });

    updateCountryBtnLabel();
  } catch (e) {
    console.error('countries init failed', e);
  }
}

function updateCountryBtnLabel(){
  const btn = document.getElementById('countryMultiBtn');
  const n = ACTIVE_COUNTRIES.size;
  btn.textContent = n === 0 ? 'Alle Länder' : `${n} ausgewählt`;
}

function appendCountries(qs){
  if (ACTIVE_COUNTRIES.size === 0) return;
  for (const c of ACTIVE_COUNTRIES) qs.append('country', c);
}

// Panel togglen & Outside-Click
document.getElementById('countryMultiBtn').addEventListener('click', () => {
  const p = document.getElementById('countryPanel');
  p.hidden = !p.hidden;
});
document.addEventListener('click', (e) => {
  const btn = document.getElementById('countryMultiBtn');
  const panel = document.getElementById('countryPanel');
  if (!panel.contains(e.target) && e.target !== btn) panel.hidden = true;
});

// Checkbox-Änderungen
document.getElementById('countryPanel').addEventListener('change', (e) => {
  if (!e.target.classList.contains('country-box')) return;
  const val = e.target.value.toUpperCase();
  if (e.target.checked) ACTIVE_COUNTRIES.add(val);
  else ACTIVE_COUNTRIES.delete(val);
  updateCountryBtnLabel();
  updateURLFromFilters(false);
  applyFilters();
});


  function getFiltersFromURL() {
    const endEx = nowExclusive();
    FILTERS.from = toLocalIsoWithTZ(startOfDay(new Date(endEx.getTime() - 6 * 86400000)));
    FILTERS.to   = toLocalIsoWithTZ(endEx);
    FILTERS.source  = null;
    FILTERS.keyword = null;
    FILTERS.teaser  = false;

    ACTIVE_SOURCES.clear();
    ACTIVE_COUNTRIES.clear();
  }


  function updateURLFromFilters(replace=false) {
  // Keine Synchronisation mehr in die URL.
  // Alle Aufrufer bleiben bestehen, tun aber nichts mehr an location/history.
  }


  function buildTimeQuery() {
    if (FILTERS.from && FILTERS.to) {
      return `from=${encodeURIComponent(FILTERS.from)}&to=${encodeURIComponent(FILTERS.to)}`;
    }
    // Fallback: 7 Kalendertage bis JETZT (exklusiv)
    const endEx = nowExclusive();
    const from  = startOfDay(new Date(endEx.getTime() - 6*86400000));
    FILTERS.from = toLocalIsoWithTZ(from);
    FILTERS.to   = toLocalIsoWithTZ(endEx);
    return `from=${encodeURIComponent(FILTERS.from)}&to=${encodeURIComponent(FILTERS.to)}`;
  }


function withTime(urlBase, extraParams = "") {
  const time = buildTimeQuery();
  const sep = urlBase.includes("?") ? "&" : "?";
  return `${urlBase}${sep}${time}${extraParams ? "&" + extraParams : ""}`;
}



async function searchByKeyword(kw = null) {
  const keyword = kw || document.getElementById("keywordInput").value.trim();
  if (!keyword) return;

  filters.keyword = keyword;
  selectedKeyword = keyword;
  updateURLFromFilters(false);

  loadFilteredArticles();
  loadExtremeBubbles();
  renderKeywordTimeline(keyword);
  loadBlindspotFeed();
  loadWordcloud();
  updateFilterDisplay();

  // --- NEU/robust ---
  const box = document.getElementById("keywordSides");
  if (!box) return;                  // falls der Container im HTML noch fehlt
  box.innerHTML = "";                // alten Inhalt leeren

  try {
    const sides = await loadKeywordSides(keyword);
    if (sides) {
      renderKeywordSidesBars(box, sides);
      await loadKeywordDensityMini(keyword);   // ← Mini-Heatmap rechts
      // optional: box.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      box.textContent = "Keine Daten für dieses Keyword im aktuellen Zeitfenster.";
    }
  } catch (err) {
    console.error(err);
    box.textContent = "Fehler beim Laden der Verteilung.";
    showMiniEmpty("Fehler beim Laden");
  }
}


    function renderHeadlineList(articles) {
      const container = document.getElementById("filteredArticles");
      if (articles.length === 0) {
        container.innerHTML = "<p>Keine Artikel gefunden.</p>";
        return;
      }

      container.innerHTML = `
        <h3>${selectedKeyword ? `Artikel zu "${selectedKeyword}"` : "Artikel"} vom ausgewählten Medium</h3>
        <ul>
          ${articles.map(a => `<li><a href="${a.url}" target="_blank">${a.title}</a></li>`).join("")}
        </ul>
      `;
    }

async function loadFilteredArticles(page = 1) {
  const qs = new URLSearchParams();
  appendSources(qs);
  appendCountries(qs);
  if (FILTERS.keyword) qs.set("keyword", FILTERS.keyword);
  if (FILTERS.teaser)  qs.set("teaser", "true");
  qs.set("page", String(page));
  // page_size kannst du später paginieren; Standard reicht hier

  const url = withTime("/articles/filtered", qs.toString());
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    document.getElementById("filteredArticles").innerHTML =
      `<div class="empty-state">Fehler beim Laden (${res.status}).</div>`;
    return;
  }
  const data = await res.json();
  renderFilteredArticles(data);
}

function renderFilteredArticles(list) {
  const el = document.getElementById("filteredArticles");
  if (!el) return;

  if (!list || list.length === 0) {
    el.innerHTML = `
      <div class="empty-state">
        Keine Artikel im gewählten Zeitraum.
        <span class="hint">Tipp: Erhöhe den Zeitraum oder entferne Filter.</span>
      </div>`;
    return;
  }

  el.innerHTML = `
    <div class="article-list">
      ${list.map(a => `
        <article class="article-card">
          <header class="article-header">
            <h3 class="article-title">
              <a href="${a.url}" target="_blank" rel="noopener">${a.title ?? ""}</a>
            </h3>
          </header>
          ${a.teaser ? `<p class="article-teaser">${a.teaser}</p>` : ""}
          <footer class="article-meta">
            <span class="article-source">${a.source ?? "Unbekannte Quelle"}</span>
            <span class="article-date">${new Date(a.published_at).toLocaleString("de-DE")}</span>
          </footer>
        </article>
      `).join("")}
    </div>
  `;
}

async function loadKeywordTrends() {
  const qs = new URLSearchParams();

  // Kombilänge
  if (FILTERS.ngram) qs.set("ngram", String(FILTERS.ngram));

  // Teaser
  if (FILTERS.teaser) qs.set("teaser", "true");

  // Quellen-Mehrfachauswahl
  appendSources(qs);

  // Länder-Mehrfachauswahl
  appendCountries(qs);

  const url = withTime("/keywords/trending", qs.toString());
  const res = await fetch(url, { cache: "no-store" });

  const grid = document.getElementById("trendGrid");
  if (!res.ok) {
    grid.innerHTML = `<div class="empty-state">Fehler beim Laden (${res.status}).</div>`;
    return;
  }

  const data = await res.json();
  grid.innerHTML = "";

  for (const label of ["24h", "72h", "7d", "30d"]) {
    const entry = data[label];
    if (!entry) continue;

    const group = document.createElement("div");
    group.className = "trend-group";

    const heading = document.createElement("h4");
    heading.textContent = label;
    group.appendChild(heading);

    for (const type of ["top", "flop"]) {
      const sublist = document.createElement("div");
      sublist.className = "trend-sublist";

      const h5 = document.createElement("h5");
      h5.textContent = type === "top" ? "🔼 Aufsteiger" : "🔽 Absteiger";
      sublist.appendChild(h5);

      const ul = document.createElement("ul");
      for (const item of entry[type]) {
        const li = document.createElement("li");

        // Backend liefert:
        // now, prev, delta, change_pct (in % oder null = "neu")
        let pctLabel;
        let cls;

        if (item.change_pct === null || typeof item.change_pct === "undefined") {
          pctLabel = "neu";
          cls = "trend-up";
        } else {
          const val = item.change_pct;
          const sign = val >= 0 ? "+" : "";
          pctLabel = `${sign}${Math.round(val)}%`;
          cls = val >= 0 ? "trend-up" : "trend-down";
        }

        li.innerHTML = `
          ${item.word}
          <span class="${cls}">
            (${item.delta >= 0 ? "+" : ""}${item.delta} | ${pctLabel})
          </span>
        `;
        li.onclick = () => searchByKeyword(item.word);
        ul.appendChild(li);
      }

      sublist.appendChild(ul);
      group.appendChild(sublist);
    }

    grid.appendChild(group);
  }
}


async function loadTopAbsoluteKeywords() {
  // ➜ Tage aus aktuellem Zeitraum ableiten (min 1, max 30 – Serverlimit)
  const spanDays = (() => {
    try {
      const from = new Date(FILTERS.from);
      const to   = new Date(FILTERS.to);
      const diff = Math.ceil((to - from) / 86400000);
      return Math.min(Math.max(diff, 1), 30);
    } catch { return 7; }
  })();

  const qs = new URLSearchParams({
    ngram: String(FILTERS.ngram ?? 1),
    compare_prev: "1",
    top_n: "25",
    spark_days: String(spanDays)
  });
  if (FILTERS.teaser) qs.set("teaser", "true");
  appendSources(qs);
  appendCountries(qs);

  const url = withTime("/keywords/top-absolute", qs.toString());
  const res = await fetch(url, { cache: "no-store" });
  const wrap = document.getElementById("keywordTopAbsoluteTableWrap");
  if (!wrap) return;

  if (!res.ok) {
    wrap.innerHTML = `<div class="empty-state">Fehler beim Laden (${res.status}).</div>`;
    return;
  }
  const data = await res.json();

  if (!Array.isArray(data) || data.length === 0) {
    wrap.innerHTML = `<div class="empty-state">Keine Daten im gewählten Zeitraum.</div>`;
    return;
  }

  // Mini-SVG Sparkline (unverändert)
  function svgColumnSpark(values = [], {w=90, h=18, gap=2} = {}) {
    if (!values.length) return "—";
    const max = Math.max(...values, 1);
    const n = values.length;
    const barW = Math.max(1, Math.floor((w - gap*(n-1)) / n));
    let x = 0;
    const bars = values.map(v => {
      const bh = Math.max(1, Math.round((v / max) * (h - 1)));
      const y = h - bh;
      const rect = `<rect x="${x}" y="${y}" width="${barW}" height="${bh}" rx="1" ry="1"></rect>`;
      x += barW + gap;
      return rect;
    }).join("");
    return `<svg class="spark spark--cols" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${bars}</svg>`;
  }

  // 🆕 Hilfsfunktion: Rang-Vergleich mit Pfeil
  function renderRankChange(current, previous) {
    if (!previous || previous === 0) return `${current} <span class="rank-dash">–</span>`;
    const diff = previous - current;
    let arrow = '<span class="rank-dash">–</span>';
    if (diff > 0) arrow = '<span class="rank-up">▲</span>';
    else if (diff < 0) arrow = '<span class="rank-down">▼</span>';
    return `${current} ${arrow} <span class="rank-prev">(${previous})</span>`;
  }

  // Tabelle rendern – Spaltenkopf „7-Tage“ → dynamisch
  const table = document.createElement("table");
  table.className = "topabs-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>#</th>
        <th>Keyword</th>
        <th>Aktuell</th>
        <th>${spanDays}-Tage</th>
        <th>Rang (vorher)</th>
        <th>Vorher</th>
        <th>Δ abs</th>
        <th>Δ %</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");

  data.forEach((row, idx) => {
    const tr = document.createElement("tr");
    const pctDisplay = (row.previous === 0 && row.current > 0)
      ? "neu"
      : `${row.change_pct >= 0 ? "+" : ""}${Math.round(row.change_pct)}%`;

    const sparkHTML = Array.isArray(row.spark) && row.spark.length
      ? svgColumnSpark(row.spark)
      : "—";

    tr.innerHTML = `
      <td>${row.rank_current ?? (idx + 1)}</td>
      <td><button class="linklike" data-kw="${row.word}">${row.word}</button></td>
      <td>${row.current}</td>
      <td class="sparkcell">${sparkHTML}</td>
      <td class="rank-cell">
        ${renderRankChange(row.rank_current, row.rank_prev)}
      </td>
      <td>${row.previous}</td>
      <td class="${row.delta >= 0 ? 'trend-up':'trend-down'}">
        ${row.delta >= 0 ? "+" : ""}${row.delta}
      </td>
      <td class="${row.delta >= 0 ? 'trend-up':'trend-down'}">${pctDisplay}</td>
    `;
    tr.querySelector('button[data-kw]').addEventListener('click', () => searchByKeyword(row.word));
    tbody.appendChild(tr);
  });

  wrap.innerHTML = "";
  wrap.appendChild(table);
}



  // zentrale Refresh-Funktion: alles neu laden, was vom Bias abhängt
  function applyFilters() {
    loadExtremeBubbles();
    loadTopAbsoluteKeywords();
    loadFilteredArticles();
    loadKeywordTrends();
    loadWordcloud();
    loadFilteredMediaCompass();
    loadBlindspotFeed();
    loadWeeklyChronicle();
    updateFilterDisplay();
    if (FILTERS.keyword) renderKeywordTimeline(FILTERS.keyword); // ← NEU

  }

async function getMediaPositions() {
  if (MEDIA_POS_CACHE) return MEDIA_POS_CACHE;
  const res = await fetch('/media-positions', { cache: 'no-store' });
  if (!res.ok) {
    console.error('media-positions failed', res.status);
    return [];
  }
  const data = await res.json(); // [{medium, x, y}]
  MEDIA_POS_CACHE = Array.isArray(data) ? data : [];
  return MEDIA_POS_CACHE;
}

// 2) Quellen anhand eines Bias-Schalters bestimmen
async function pickSourcesForBias(biasName) {
  const THRESH = 0.01;                // Bereichs-Schwelle
  const data = await getMediaPositions(); // [{ medium, x, y }]

  const hits = [];
  for (const d of data) {
    const x = Number(d.x);
    const y = Number(d.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    if (biasName === 'systemkritisch'   && x <= -THRESH) hits.push(d.medium);
    if (biasName === 'systemnah'        && x >=  THRESH) hits.push(d.medium);
    if (biasName === 'globalistisch'    && y >=  THRESH) hits.push(d.medium);
    if (biasName === 'nationalistisch'  && y <= -THRESH) hits.push(d.medium);
  }
  return hits;
}


// 3) UI-Checkboxen im Quellen-Panel an ACTIVE_SOURCES anpassen
function syncSourcePanelChecks() {
  const panel = document.getElementById('sourcePanel');
  if (!panel) return;
  Array.from(panel.querySelectorAll('.source-box')).forEach(cb => {
    cb.checked = ACTIVE_SOURCES.has(cb.value);
  });
  updateSourceBtnLabel();
}

// 4) Bias setzen: Quellen aus CSV selektieren und Filter anwenden
async function setBiasAndSelectSources(biasName) {
  // "Alle" -> alles zurücksetzen
  if (!biasName || biasName === 'alle') {
    activeBiasFilters.clear();
    ACTIVE_SOURCES.clear();
    syncSourcePanelChecks();
    updateURLFromFilters(false);
    applyFilters();
    return;
  }

  // activeBiasFilters als Single-Choice (wenn du Multi willst, kannst du das ändern)
  activeBiasFilters.clear();
  activeBiasFilters.add(biasName);

  // Quellen laut CSV auswählen
  const picked = await pickSourcesForBias(biasName);

  ACTIVE_SOURCES.clear();
  for (const s of picked) ACTIVE_SOURCES.add(s);

  // Für Alt-Code: ersten Eintrag spiegeln
  FILTERS.source = ACTIVE_SOURCES.size ? Array.from(ACTIVE_SOURCES)[0] : null;

  // Buttons optisch markieren
  document.querySelectorAll('#biasFilters button[data-bias]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.bias === biasName);
  });

  syncSourcePanelChecks();
  updateURLFromFilters(false);
  applyFilters();
}


// Klicks auf beliebige Bias-Buttons abfangen (egal wo sie im DOM stehen)
document.addEventListener('click', (e) => {
  const btnBias = e.target.closest('button[data-bias]');
  if (btnBias) {
    e.preventDefault();
    setBiasAndSelectSources(btnBias.dataset.bias);
    return;
  }
  const clearBtn = e.target.closest('[data-role="bias-clear"]');
  if (clearBtn) {
    e.preventDefault();
    setBiasAndSelectSources('alle');
  }
});



// Setter für den Bias-Filter: setzt/toggelt den Filter und triggert die Aktualisierung
function setBiasFilter(filterName) {
  if (!filterName || filterName === 'alle') {
    activeBiasFilters.clear();
  } else {
    if (activeBiasFilters.has(filterName)) {
      activeBiasFilters.delete(filterName);
    } else {
      activeBiasFilters.add(filterName);
    }
  }

  // Buttons visuell markieren
  document.querySelectorAll('#biasFilters button[data-bias]').forEach(btn => {
    const key = btn.dataset.bias;
    btn.classList.toggle('active', activeBiasFilters.has(key));
  });

  applyFilters();
}

// EINZIGE Version der Prüf-Funktion (Multi-Auswahl, ODER-Logik)
function biasPassesFilters(bias) {
  // Keine aktiven Filter → alles zulassen
  if (activeBiasFilters.size === 0) return true;

  // Ungültiger Bias → blockieren
  if (!bias || !Number.isFinite(bias.x) || !Number.isFinite(bias.y)) return false;

  const { x, y } = bias;

  // ODER-Logik: reicht, wenn eine Bedingung passt
  for (const f of activeBiasFilters) {
    if (f === 'systemkritisch'   && x <= -0.1) return true;
    if (f === 'systemnah'        && x >=  0.1) return true;
    if (f === 'globalistisch'    && y >=  0.1) return true;
    if (f === 'nationalistisch'  && y <= -0.1) return true;
  }
  return false;
}


function setNgram(n){
  filters.ngram = n;
  loadWordcloud();
  loadFilteredMediaCompass();
  loadTopAbsoluteKeywords();
  loadBlindspotFeed();
  updateFilterDisplay && updateFilterDisplay();
}

const ngramBtns = () => Array.from(document.querySelectorAll('#ngramButtons button'));
function markActiveNgram(n){
  try {
    ngramBtns().forEach(b => b.classList.remove('is-active'));
    const labels = {1:'1 Wort', 2:'2er-Kombis', 3:'3er-Kombis'};
    const btn = ngramBtns().find(b => b.textContent.trim() === labels[n]);
    if (btn) btn.classList.add('is-active');
  } catch(_) {}
}

// Beim Setzen auch markieren
const _setNgram = window.setNgram || setNgram;
window.setNgram = function(n){
  _setNgram(n);
  markActiveNgram(n);
};

// Initial markieren nach Load
window.addEventListener('load', () => markActiveNgram(filters.ngram));


  async function loadKeywordSides(word) {
    const extra = `word=${encodeURIComponent(word)}` + (FILTERS.teaser ? "&teaser=true" : "");
    const url = withTime("/keywords/sides", extra);
    const res = await fetch(url);
    if (!res.ok) return null;
    return res.json();
  }



  function renderKeywordSidesBars(container, data) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "spectrum";

  const makeBar = (axisCounts, leftKey, rightKey, labelText) => {
    const total = axisCounts.total || 1;

    const row   = document.createElement("div"); row.className = "spectrum__row";
    const label = document.createElement("div"); label.className = "spectrum__label";
    label.textContent = labelText;

    const bar   = document.createElement("div"); bar.className = "spectrum__bar";

    const parts = [
      { cls: "spectrum__seg spectrum__seg--left",   key: leftKey,  val: axisCounts[leftKey] },
      { cls: "spectrum__seg spectrum__seg--center", key: "neutral", val: axisCounts.neutral },
      { cls: "spectrum__seg spectrum__seg--right",  key: rightKey, val: axisCounts[rightKey] },
    ];

    parts.forEach(p => {
      const seg = document.createElement("div");
      seg.className = p.cls;
      seg.style.flex = String(p.val);                        // Proportionen
      seg.style.minWidth = p.val === 0 ? "0px" : "2px";      // dünne Segmente sichtbar halten
      const pct = Math.round((p.val / total) * 100);
      seg.textContent = `${p.key === "neutral" ? "C" : p.key} ${pct}%`;
      bar.appendChild(seg);
    });

    row.appendChild(label);
    row.appendChild(bar);
    return row;
  };

  wrap.appendChild(makeBar(data.counts.x, "kritisch", "nah", "Systemkritisch — Neutral — Systemnah"));
  if (data.blindspots?.x) {
    const badge = document.createElement("div");
    badge.className = "spectrum__badge";
    badge.textContent = `Blindspot (X): ${data.blindspots.x}`;
    wrap.appendChild(badge);
  }

  wrap.appendChild(makeBar(data.counts.y, "national", "global", "Nationalistisch — Neutral — Globalistisch"));
  if (data.blindspots?.y) {
    const badge = document.createElement("div");
    badge.className = "spectrum__badge";
    badge.textContent = `Blindspot (Y): ${data.blindspots.y}`;
    wrap.appendChild(badge);
  }

  container.appendChild(wrap);
}



async function loadExtremeBubbles() {
  const bubblesRes = await fetch(withTime("/keywords/extreme-bubble"));
  const bubbleData = await bubblesRes.json();
  const biasRes = await fetch(withTime("/keywords/bias-vector"));
  const biasData = await biasRes.json();

  const container = document.getElementById("extremeKeywordBubbles");
  container.innerHTML = "";

  const sortMode = getSelectedBubbleSort(); // z. B. "bias-desc"

  // Daten kombinieren
  let bubbles = bubbleData.map(([word, count]) => {
    const bias = biasData[word];
    const strength = bias ? Math.sqrt(bias.x ** 2 + bias.y ** 2) : 0;
    return { word, count, bias, strength };
  });

  // Filter anwenden
  bubbles = bubbles.filter(({ bias }) => biasPassesFilters(bias));

  // Sortieren
  bubbles.sort((a, b) => {
    switch (sortMode) {
      case "count-asc": return a.count - b.count;
      case "count-desc": return b.count - a.count;
      case "bias-asc": return a.strength - b.strength;
      case "bias-desc":
      default: return b.strength - a.strength;
    }
  });

  renderKeywordScatterChart(bubbles);

  if (filters.keyword) {
    const match = bubbles.find(b => b.word === filters.keyword);
    container.innerHTML = "";

    if (match) {
      renderSingleBubble(match.word, match.count, match.bias);
    } else {
      container.innerHTML = "<p>Kein passendes Bubble-Keyword gefunden.</p>";
    }

    return;
  }



  // Anzeige
  for (const { word, count, bias } of bubbles) {
    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const label = bias
      ? `${word} (${count}) | x: ${bias.x}, y: ${bias.y}`
      : `${word} (${count})`;

    if (bias) {
      bubble.style.backgroundColor = getBiasGradientColor(bias.x, bias.y);
      bubble.style.color = "#fff";
      bubble.title = `x: ${bias.x}, y: ${bias.y}`;
    }

    bubble.textContent = label;
    bubble.onclick = () => {
      container.innerHTML = ""; // Alle anderen Bubbles entfernen
      renderSingleBubble(word, count, bias); // Nur diese Bubble anzeigen
      searchByKeyword(word); // Artikel laden wie gewohnt
    };

    container.appendChild(bubble);
  }
}

function renderSingleBubble(word, count, bias) {
  const container = document.getElementById("extremeKeywordBubbles");
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const label = bias
    ? `${word} (${count}) | x: ${bias.x}, y: ${bias.y}`
    : `${word} (${count})`;

  if (bias) {
    bubble.style.backgroundColor = getBiasGradientColor(bias.x, bias.y);
    bubble.style.color = "#fff";
    bubble.title = `x: ${bias.x}, y: ${bias.y}`;
  }

  bubble.textContent = label;
  container.appendChild(bubble);
}


function getBiasGradientColor(x, y) {
  const angle = Math.atan2(y, x);
  const hue = ((angle * 180 / Math.PI) + 360) % 360;

  const r = Math.min(Math.sqrt(x * x + y * y), 1); // 0 (zentral) bis 1 (extrem)
  const saturation = Math.round(r * 100);          // 0–100%
  const lightness = 60;                            // optional: 65 - 20 * r

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}






async function loadWordcloud() {
  const qs = new URLSearchParams();
  appendSources(qs);
  appendCountries(qs);
  if (FILTERS.keyword) qs.set("keyword", FILTERS.keyword);
  if (FILTERS.teaser)  qs.set("teaser", "true");
  if (FILTERS.ngram)   qs.set("ngram", String(FILTERS.ngram));
  const res = await fetch(withTime("/headlines/words", qs.toString()));
  const data = await res.json();

  const words = data
    .filter(([_, count]) => count >= WORDCLOUD_CFG.minCount)
    .map(([text, count]) => ({ text, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, WORDCLOUD_CFG.maxWords);   // Top-N gemäß Slider



  d3.select("#wordcloud").html(""); // alte SVG löschen

  let currentWordcloudData = words;

  d3.layout.cloud()
    .spiral("archimedean") // oder "rectangular"
    .size([window.innerWidth, window.innerHeight])
    .words(words)
    .rotate(() => ~~(Math.random() * 2) * 90)
    .font("Impact")
    .fontSize(d => scaleFontSize(d.count))
    .on("end", draw)
    .start();

  function scaleFontSize(count) {
    const minSize = 24;
    const maxSize = 68;

    const counts = currentWordcloudData.map(w => w.count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    if (maxCount === minCount) return minSize;

    return minSize + ((count - minCount) / (maxCount - minCount)) * (maxSize - minSize);
  }


  function scaleFontWeight(count) {
    const minWeight = 300;
    const maxWeight = 900;

    const counts = currentWordcloudData.map(w => w.count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    if (maxCount === minCount) return minWeight;

    return minWeight + ((count - minCount) / (maxCount - minCount)) * (maxWeight - minWeight);
  }



function draw(words) {
  const svg = d3.select("#wordcloud").append("svg")
    .attr("width", window.innerWidth)
    .attr("height", window.innerHeight);

  const g = svg.append("g")
    .attr("transform", `translate(${window.innerWidth / 2},${window.innerHeight / 2})`);

  // 1) Counts auslesen
  const counts = words.map(w => w.count);
  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);

  // 2) Farbskala: wenig -> #434343, viel -> #000000 (wie zuvor gewünscht)
  const colorScale = d3.scaleLinear()
    .domain([minCount, maxCount])
    .range(["#434343", "#000000"]);

  // 3) Fontsize mit leichtem Boost (pow > 1) betont hohe Counts → bleiben zentral
  const sizeScale = d3.scalePow()
    .exponent(1.35)                        // 1.2–1.5 ist ein guter Bereich
    .domain([minCount, maxCount])
    .range([24, 80]);                      // ggf. anpassen

  // 4) Layout: größte zuerst, keine Rotation, archimedische Spirale
  d3.layout.cloud()
    .size([window.innerWidth, window.innerHeight])
    .words(words.sort((a, b) => b.count - a.count))  // größte zuerst = landen näher am Zentrum
    .rotate(() => 0)                                 // kompakter, besseres Zentrieren
    .padding(d => (d.count >= (minCount + (maxCount-minCount)*0.6)) ? 4 : 1) // mehr Platz um Top-Wörter
    .font("Lato")
    .fontSize(d => sizeScale(d.count))
    .spiral("archimedean")
    .on("end", place)
    .start();

  function place(placed) {
    const texts = g.selectAll("text")
      .data(placed)
      .enter().append("text")
      .style("font-size", d => `${d.size}px`)
      .style("font-family", "Lato")
      .style("fill", d => colorScale(d.count))        // monochromer Verlauf
      .style("font-weight", d => scaleFontWeight(d.count))
      .attr("text-anchor", "middle")
      .attr("transform", d => `translate(${d.x},${d.y})rotate(${d.rotate || 0})`)
      .text(d => d.text);

    texts.append("title").text(d => `${d.text}: ${d.count}x`);

    texts.on("click", (_, d) => {
      searchByKeyword(d.text);
      loadFilteredMediaCompass(d.text);
      renderKeywordTimeline(d.text);
      loadWordcloud();
    });
  }
}


}




async function loadMediaCompass() {
  const res = await fetch(withTime("/media-positions/filtered"));
  const data = await res.json();
  drawMediaCompass(data, "Medien");
}

async function loadFilteredMediaCompass() {
  const qs = new URLSearchParams();

  if (FILTERS.keyword) qs.set("keyword", FILTERS.keyword);
  if (FILTERS.teaser) qs.set("teaser", "true");

  appendSources(qs);
  appendCountries(qs);

  const url = withTime("/media-positions/filtered", qs.toString());

  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    console.error("Fehler beim Laden des gefilterten Medien-Kompasses", res.status);
    return;
  }

  const data = await res.json();

  const label = FILTERS.keyword
    ? `Medien zu „${FILTERS.keyword}“`
    : "Medien";

  drawMediaCompass(data, label);
}


function drawMediaCompass(data, label = "Medien") {
  const ctx = document.getElementById("mediaCompassChart").getContext("2d");

  if (mediaCompassChart instanceof Chart) {
    mediaCompassChart.destroy();
  }

  const maxCount = Math.max(...data.map(d => d.count || 1));

  mediaCompassChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [{
        label: label,
        data: data.map(d => ({
          x: d.x,
          y: d.y,
          label: d.medium,
          count: d.count
        })),
        backgroundColor: "#3b82f6",
        pointRadius: ctx => {
          const count = ctx.raw.count || 1;
          return Math.max(4, count / maxCount * 20);
        }
      }]
    },

    options: {
      onClick: async (evt, elements) => {
        if (elements.length === 0) return;

        const point = elements[0];
        const medium = mediaCompassChart.data.datasets[point.datasetIndex].data[point.index].label;
        // Toggle der Quelle in der Mehrfachauswahl
        if (ACTIVE_SOURCES.has(medium)) ACTIVE_SOURCES.delete(medium);
        else ACTIVE_SOURCES.add(medium);

        // Optional: ersten Eintrag nach FILTERS.source spiegeln (Abwärtskompatibilität)
        FILTERS.source = ACTIVE_SOURCES.size ? Array.from(ACTIVE_SOURCES)[0] : null;

        // UI-Checkbox synchronisieren
        const cb = document.querySelector(`#sourcePanel .source-box[value="${CSS.escape(medium)}"]`);
        if (cb) cb.checked = ACTIVE_SOURCES.has(medium);

        updateSourceBtnLabel();
        updateURLFromFilters(false);
        applyFilters();
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.raw.label} (${ctx.raw.count} Artikel)`
          }
        }
      },
      scales: {
        x: {
          min: -1, max: 1,
          title: { display: true, text: "Systemkritisch (–1) → Systemnah (+1)" }
        },
        y: {
          min: -1, max: 1,
          title: { display: true, text: "Nationalistisch (–1) → Globalistisch (+1)" }
        }
      }
    }
  });
}


function resetFilters() {
  USER_TIME_OVERRIDE = false;  // ← zurück auf Default-5-Wochen-Logik für die Chronik

  const endEx = nowExclusive();
  FILTERS.from = toLocalIsoWithTZ(startOfDay(new Date(endEx.getTime() - 6*86400000)));
  FILTERS.to   = toLocalIsoWithTZ(endEx);
  FILTERS.source=""; FILTERS.keyword=""; FILTERS.teaser=false; FILTERS.ngram=1;

  ACTIVE_SOURCES.clear();
  updateSourceBtnLabel();
  const panel = document.getElementById('sourcePanel');
  if (panel) panel.querySelectorAll('.source-box').forEach(cb => cb.checked = false);

  // LÄNDER-Reset  <<<<<<<<<<<<<<<< NEU
  ACTIVE_COUNTRIES.clear();
  updateCountryBtnLabel();
  const cpanel = document.getElementById('countryPanel');
  if (cpanel) cpanel.querySelectorAll('.country-box').forEach(cb => cb.checked = false);

  document.getElementById("sourceSelect").value = "";
  document.getElementById("keywordInput").value = "";
  document.getElementById("teaserToggle").checked = false;

  selectedKeyword = "";
  markActiveDays(7);
  updateDateRangeBadge();
  updateURLFromFilters(false);
  applyFilters();
}



function getSelectedBubbleSort() {
  const select = document.getElementById("bubbleSort");
  return select ? select.value : "bias-desc";
}

function renderKeywordScatterChart(bubbles) {
  const canvas = document.getElementById("keywordScatterChart");
  const ctx = canvas.getContext("2d");

  // 🔧 Kritisch: echte Canvas-Größe setzen!
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const data = bubbles
    .filter(b => b.bias && b.count >= 15)
    .map(b => ({
      x: b.bias.x,
      y: b.bias.y,
      r: Math.min(3 + Math.sqrt(b.count), 15),
      label: b.word,
      count: b.count,
      backgroundColor: b.word === filters.keyword ? "#ef4444" : "#2563eb"
    }));

  if (keywordScatterChart instanceof Chart) {
    keywordScatterChart.destroy();
  }

  keywordScatterChart = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [{
        label: "Keywords",
        data: data.map(d => ({ x: d.x, y: d.y, r: d.r, label: d.label })),
        backgroundColor: data.map(d => d.backgroundColor),
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      onClick: (evt, elements) => {
        if (elements.length === 0) return;
        const point = elements[0];
        const word = keywordScatterChart.data.datasets[0].data[point.index].label;
        loadKeywordDensity(word);
        renderKeywordTimeline(word);
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const raw = ctx.raw || {};
              // statt r (Radius) die echte Anzahl anzeigen
              const count = raw.count ?? raw.value ?? 0;
              return `Artikel: ${count}`;
            },
          }
        }
      },
      scales: {
        x: {
          min: -1, max: 1,
          title: { display: true, text: "Systemkritisch → Systemnah" }
        },
        y: {
          min: -1, max: 1,
          title: { display: true, text: "Nationalistisch → Globalistisch" }
        }
      }
    }
  });
}

function renderKeywordDensityHeatmap(rawData, keyword = "") {
  const svg = d3.select("#keywordHeatmapOverlay");
  svg.selectAll("*").remove();

  const container = document.getElementById("keywordChartOverlayContainer");
  const rect = container.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  svg
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  const xScale = d3.scaleLinear().domain([-1, 1]).range([0, width]);
  const yScale = d3.scaleLinear().domain([-1, 1]).range([height, 0]);

  const points = rawData.flatMap(d => {
    const c = Math.min(30, Math.round(Math.log1p(d.count) * 5));
    if (!isFinite(d.x) || !isFinite(d.y)) return [];
    return Array(c).fill({ x: xScale(d.x), y: yScale(d.y) });
  });

  if (!points.length) {
    console.warn("⚠️ Keine Punkte für Heatmap generiert");
    return;
  }

  const densityData = d3.contourDensity()
    .x(d => d.x)
    .y(d => d.y)
    .size([width, height])
    .bandwidth(50)
    .thresholds(20)(points);

  const color = d3.scaleSequential(d3.interpolateYlOrRd)
    .domain([0, d3.max(densityData, d => d.value)]);

  svg.selectAll("path")
    .data(densityData)
    .join("path")
    .attr("d", d3.geoPath())
    .attr("fill", d => color(d.value))
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.3);
}


async function loadKeywordDensity(keyword) {
  const res = await fetch(withTime("/media-positions/by-keyword", `word=${encodeURIComponent(keyword)}`));
  const data = await res.json();
  renderKeywordDensityHeatmap(data, keyword);
}
async function showKeywordHeatOverlay(keyword) {
  const res = await fetch(withTime("/media-positions/by-keyword", `word=${encodeURIComponent(keyword)}`));
  const heatData = await res.json();
  renderKeywordDensityHeatmap(heatData);
  FILTERS.keyword = keyword;
  selectedKeyword = keyword;
  loadExtremeBubbles();
}

async function loadKeywordDensityMini(keyword) {
  const res = await fetch(withTime("/media-positions/by-keyword",
                   `word=${encodeURIComponent(keyword)}`));
  if (!res.ok) { showMiniEmpty("Fehler beim Laden"); return; }
  const data = await res.json();
  renderKeywordDensityHeatmapIn("keywordHeatMini", "keywordHeatMiniWrap", data);
}

function showMiniEmpty(msg){
  const svg = document.getElementById("keywordHeatMini");
  if (svg) svg.innerHTML = "";
  const ov = document.getElementById("keywordHeatMiniEmpty");
  if (ov){ ov.style.display = "grid"; ov.textContent = msg || "Keine Heatmap-Daten"; }
}
function hideMiniEmpty(){
  const ov = document.getElementById("keywordHeatMiniEmpty");
  if (ov) ov.style.display = "none";
}

function renderKeywordDensityHeatmapIn(svgId, wrapId, rawData){
  const svg = d3.select(`#${svgId}`);
  svg.selectAll("*").remove();

  const wrap = document.getElementById(wrapId);
  if (!wrap || !rawData || !rawData.length){ showMiniEmpty(); return; }
  hideMiniEmpty();

  const width = wrap.clientWidth, height = wrap.clientHeight;
  svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);

  const xScale = d3.scaleLinear().domain([-1, 1]).range([0, width]);
  const yScale = d3.scaleLinear().domain([-1, 1]).range([height, 0]);

  const points = rawData.flatMap(d => {
    const c = Math.min(30, Math.round(Math.log1p(d.count) * 5));
    if (!Number.isFinite(d.x) || !Number.isFinite(d.y)) return [];
    return Array(c).fill({ x: xScale(d.x), y: yScale(d.y) });
  });
  if (!points.length){ showMiniEmpty(); return; }

  const densityData = d3.contourDensity()
    .x(d => d.x).y(d => d.y)
    .size([width, height]).bandwidth(40).thresholds(16)(points);

  const color = d3.scaleSequential(d3.interpolateYlOrRd)
    .domain([0, d3.max(densityData, d => d.value)]);

  svg.selectAll("path")
    .data(densityData)
    .join("path")
    .attr("d", d3.geoPath())
    .attr("fill", d => color(d.value))
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.3);
}

// Säulendiagramm-Version (kopierfähig)
async function renderKeywordTimeline(keyword) {
  if (!keyword) return;

  const qs = new URLSearchParams({ word: keyword });
  if (FILTERS.teaser) qs.set("teaser", "true");
  appendSources(qs);
  appendCountries(qs);

  const res = await fetch(withTime("/keywords/timeline", qs.toString()));
  if (!res.ok) return;
  const data = await res.json();

  const chartWrap = document.getElementById('timelineChartWrap');
  const canvas    = document.getElementById('timelineChart');
  if (!chartWrap || !canvas) return;

  // Empty-State
  if (!Array.isArray(data) || data.length === 0 || data.every(d => (d.count||0) === 0)) {
    mountChartEmptyOverlay(chartWrap, "Keine Timeline-Daten im gewählten Zeitraum");
    const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height);
    if (window.keywordTimelineChart instanceof Chart) {
      window.keywordTimelineChart.destroy();
      window.keywordTimelineChart = null;
    }
    return;
  } else {
    hideChartEmptyOverlay(chartWrap);
  }

  const labels = data.map(d => {
    const dt = new Date(d.time); // ISO, Mitternacht UTC
    return new Intl.DateTimeFormat('de-DE', { weekday:'short', day:'2-digit', month:'2-digit' }).format(dt);
  });
  const values = data.map(d => d.count);

  const ctx = canvas.getContext('2d');

  // Wenn bereits vorhanden, aber als anderer Typ → neu erstellen
  if (window.keywordTimelineChart instanceof Chart && window.keywordTimelineChart.config.type !== 'bar') {
    window.keywordTimelineChart.destroy();
    window.keywordTimelineChart = null;
  }

  if (!(window.keywordTimelineChart instanceof Chart)) {
    window.keywordTimelineChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Anzahl Artikel pro Tag',
          data: values,
          borderWidth: 1,
          borderRadius: 6,         // abgerundete Balken
          maxBarThickness: 42,     // verhindert zu breite Balken
          categoryPercentage: 0.8,
          barPercentage: 0.9
          // Farbe: Chart.js nimmt Standardfarben/Theming; bei Bedarf hier backgroundColor setzen
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: {
            ticks: { autoSkip: true, maxTicksLimit: 10 },
            grid: { display: false }
          },
          y: {
            beginAtZero: true,
            ticks: { precision: 0 }, // nur ganze Zahlen
            title: { display: false }
          }
        }
      }
    });
  } else {
    const ch = window.keywordTimelineChart;
    ch.data.labels = labels;
    ch.data.datasets[0].data = values;
    ch.update();
  }
}

async function loadKeywordSankey(word) {
  const wrap = document.getElementById('keywordSankeyWrap');
  const svg = d3.select('#keywordSankey');
  const ov = document.getElementById('keywordSankeyEmpty');
  if (!wrap || !svg.node()) return;

  // Kein Keyword -> Empty anzeigen
  if (!word) {
    svg.selectAll('*').remove();
    if (ov) {
      ov.classList.remove('is-hidden');
      ov.textContent = 'Kein Sankey – bitte ein Keyword wählen';
    }
    return;
  }

  // Query bauen (inkl. from/to, country/source, teaser)
  const qs = new URLSearchParams({ word });
  if (FILTERS?.teaser) qs.set('teaser', 'true');
  appendSources(qs);
  appendCountries(qs);

  // Daten abrufen
  const res = await fetch(withTime('/keywords/sankey', qs.toString()), { cache: 'no-store' });
  if (!res.ok) {
    svg.selectAll('*').remove();
    if (ov) {
      ov.classList.remove('is-hidden');
      ov.textContent = `Fehler (${res.status})`;
    }
    return;
  }

  const data = await res.json();
  const items = data?.items || [];
  const totalSum = d3.sum(items, d => d.total_sources);

  if (!items.length || totalSum === 0) {
    svg.selectAll('*').remove();
    if (ov) {
      ov.classList.remove('is-hidden');
      ov.textContent = 'Keine Daten im gewählten Zeitraum';
    }
    return;
  }

  if (ov) ov.classList.add('is-hidden');

  // --- D3-Sankey vorbereiten ---
  svg.selectAll('*').remove();
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  svg.attr('viewBox', `0 0 ${width} ${height}`).attr('width', width).attr('height', height);

  // Knoten: 4 Bereiche + Keyword + versteckter Rest-Sammelknoten
  const LEFT = ["systemkritisch", "systemnah", "globalistisch", "nationalistisch"];
  const nodes = [...LEFT.map(n => ({ name: n })), { name: word }, { name: "__rest__" }];

  // Links: Treffer → Keyword, Rest → __rest__
  const links = [];
  for (const it of items) {
    const srcIdx = LEFT.indexOf(it.bucket);
    if (srcIdx < 0) continue;
    const used = Math.max(0, it.hit_sources);
    const rest = Math.max(0, it.total_sources - it.hit_sources);

    links.push({
      source: srcIdx,
      target: LEFT.length, // Keyword-Index
      value: used,
      meta: { ...it, type: "used" }
    });

    if (rest > 0) {
      links.push({
        source: srcIdx,
        target: LEFT.length + 1, // __rest__
        value: rest,
        meta: { ...it, type: "rest" }
      });
    }
  }

  // Layout berechnen
  const sankey = d3.sankey()
    .nodeWidth(18)
    .nodePadding(18)
    .extent([[8, 8], [width - 8, height - 8]]);

  const graph = sankey({
    nodes: nodes.map(d => ({ ...d })),
    links: links.map(d => ({ ...d }))
  });

  // Farbskala
  const color = d3.scaleOrdinal()
    .domain(LEFT)
    .range(["#cbd5e1", "#93c5fd", "#a5b4fc", "#9bd3ff"]);

  // --- Links zeichnen ---
  svg.append("g")
    .attr("fill", "none")
    .selectAll("path")
    .data(graph.links)
    .join("path")
      .attr("d", d3.sankeyLinkHorizontal())
      .attr("class", d =>
        d.meta.type === "rest"
          ? "sankey-link sankey-link--rest"
          : "sankey-link sankey-link--used"
      )
      .attr("stroke", d => color(d.meta.bucket))
      .attr("stroke-width", d => Math.max(1, d.width))
      .append("title")
        .text(d => {
          const m = d.meta;
          return (m.type === "rest")
            ? `${m.bucket} → (ohne Keyword)\nMedien: ${m.total_sources - m.hit_sources} von ${m.total_sources}`
            : `${m.bucket} → ${word}\nMedien: ${m.hit_sources} von ${m.total_sources} (${m.pct}%)`;
        });

  // --- Nodes zeichnen (REST-Node auslassen) ---
  const node = svg.append("g")
    .selectAll("g")
    .data(graph.nodes.filter(n => n.name !== "__rest__"))
    .join("g");

  node.append("rect")
    .attr("x", d => d.x0)
    .attr("y", d => d.y0)
    .attr("height", d => Math.max(2, d.y1 - d.y0))
    .attr("width", d => d.x1 - d.x0)
    .attr("rx", 6)
    .attr("ry", 6)
    .attr("class", d =>
      LEFT.includes(d.name)
        ? "sankey-node sankey-node--left"
        : "sankey-node sankey-node--right"
    )
    .attr("fill", d =>
      LEFT.includes(d.name) ? color(d.name) : "#4f6fd6"
    );

  // --- Labels ---
  node.append("text")
    .attr("x", d => (LEFT.includes(d.name) ? (d.x0 - 8) : (d.x1 + 8)))
    .attr("y", d => (d.y0 + d.y1) / 2)
    .attr("dy", "0.35em")
    .attr("text-anchor", d => (LEFT.includes(d.name) ? "end" : "start"))
    .attr("class", "sankey-label")
    .text(d => {
      if (!LEFT.includes(d.name)) return word;
      const m = items.find(it => it.bucket === d.name) || { total_sources: 0, hit_sources: 0, pct: 0 };
      return `${d.name}  ${m.hit_sources}/${m.total_sources} (${m.pct}%)`;
    });

  // --- Pfeil vorm Keyword ---
  const kw = graph.nodes.find(n => n.name === word);
  if (kw) {
    svg.append("text")
      .attr("x", kw.x0 - 14)
      .attr("y", (kw.y0 + kw.y1) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .attr("class", "sankey-arrow")
      .text("→");
  }
}


// Beim Keyword-Suchen mit ausführen:
const _searchByKeyword_Sankey = searchByKeyword;
searchByKeyword = async function(kw){
  await _searchByKeyword_Sankey(kw);
  // danach Sankey aktualisieren
  const k = kw || (document.getElementById('keywordInput')?.value || '').trim();
  loadKeywordSankey(k);
};

// Erstmaliger Zustand (ohne Keyword -> Empty)
window.addEventListener('load', () => loadKeywordSankey(''));



async function loadWeeklyChronicle() {
  const weeks = 8; // zum Start 5 Wochen
  const body = document.getElementById("weeklyChronicleBody");
  body.innerHTML = `<div class="empty-state">Wird geladen…</div>`;

  const qs = new URLSearchParams({ weeks: String(weeks) });
  appendSources(qs);
  appendCountries(qs);
  if (FILTERS.teaser) qs.set("teaser", "true");

  // WICHTIG: KEIN withTime() verwenden, damit from/to NICHT angehängt werden.
  const url = `/chronicle/weekly-top3-all?${qs.toString()}`;

  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      body.innerHTML = `<div class="empty-state">Fehler beim Laden (${res.status}).</div>`;
      return;
    }
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) {
      body.innerHTML = `<div class="empty-state">Keine Daten im gewählten Zeitraum.</div>`;
      return;
    }

    body.innerHTML = data.map(w => `
      <details class="chronicle-accordion" open>
        <summary><strong>${w.week_label}</strong> (${w.start_date} – ${w.end_date})</summary>
        <div class="chronicle-combos" style="display:grid;grid-template-columns:1fr;gap:8px;margin-top:6px">
          <div class="chronicle-combo">
            <h4 style="margin:.2rem 0 .4rem 0">Top 3 (Einzel)</h4>
            <div>
              ${(w.top1||[]).map((t,i)=>`
                <button class="linklike" data-kw="${t.word}">${t.word} (${t.count})</button>
              `).join(" · ") || "—"}
            </div>
          </div>
          <div class="chronicle-combo">
            <h4 style="margin:.2rem 0 .4rem 0">Top 3 (2er-Kombis)</h4>
            <div>
              ${(w.top2||[]).map((t,i)=>`
                <button class="linklike" data-kw="${t.word}">${t.word} (${t.count})</button>
              `).join(" · ") || "—"}
            </div>
          </div>
          <div class="chronicle-combo">
            <h4 style="margin:.2rem 0 .4rem 0">Top 3 (3er-Kombis)</h4>
            <div>
              ${(w.top3||[]).map((t,i)=>`
                <button class="linklike" data-kw="${t.word}">${t.word} (${t.count})</button>
              `).join(" · ") || "—"}
            </div>
          </div>
        </div>
      </details>
    `).join("");

    body.querySelectorAll('button[data-kw]').forEach(btn => {
      btn.addEventListener('click', () => searchByKeyword(btn.getAttribute('data-kw')));
    });
  } catch (err) {
    console.error(err);
    body.innerHTML = `<div class="empty-state">Unbekannter Fehler beim Laden.</div>`;
  }
}


async function loadBlindspotFeed() {
  const qs = new URLSearchParams({
    min_sources_min: String(BLINDSPOT_UI.min_sources_range?.[0] ?? 1),
    min_sources_max: String(BLINDSPOT_UI.min_sources_range?.[1] ?? 50),
    min_total_min:   String(BLINDSPOT_UI.min_total_range?.[0]   ?? 1),
    min_total_max:   String(BLINDSPOT_UI.min_total_range?.[1]   ?? 50),
    ratio_min:       String(BLINDSPOT_UI.ratio_range?.[0]       ?? 0.01), // Dezimal
    ratio_max:       String(BLINDSPOT_UI.ratio_range?.[1]       ?? 0.05), // Dezimal
    ngram:           String(FILTERS.ngram ?? 1)
  });
  if (FILTERS.teaser) qs.set("teaser", "true");
  appendCountries(qs);
  const url = withTime("/blindspots/keywords-feed", qs.toString());
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return;
  const data = await res.json();
  renderBlindspotFeed(data);
}


function renderBlindspotFeed(data) {
  const cfg = [
    { key: "systemkritisch",  el: "#feedKritisch",  ratioKey: "kritisch" },
    { key: "systemnah",       el: "#feedNah",       ratioKey: "nah" },
    { key: "nationalistisch", el: "#feedNational",  ratioKey: "national" },
    { key: "globalistisch",   el: "#feedGlobal",    ratioKey: "global" },
  ];

  for (const { key, el, ratioKey } of cfg) {
    const col = document.querySelector(`${el} ul`);
    if (!col) continue;
    col.innerHTML = "";

    const items = data?.items?.[key] || [];
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "— keine Blindspots im Fenster —";
      col.appendChild(li);
      continue;
    }

    for (const it of items) {
      const li = document.createElement("li");
      li.className = "feed-item";

      // links: Titel + Meta
      const left = document.createElement("div");
      left.className = "blindspot-item-left";

      const titleBtn = document.createElement("button");
      titleBtn.className = "linklike blindspot-item-title";
      titleBtn.textContent = it.word;
      titleBtn.onclick = () => searchByKeyword(it.word);

      const meta = document.createElement("div");
      meta.className = "blindspot-item-meta";
      meta.textContent = `(${it.total} Erwähnungen, ${it.sources} Quellen)`;

      left.appendChild(titleBtn);
      left.appendChild(meta);

      // rechts: schwarze Chip-Badge
      const chip = document.createElement("div");
      chip.className = "ratio-chip";
      const ratio = Number(it.ratios?.[ratioKey]) || 0;
      const pct   = Math.round(ratio * 100);
      const zero  = it.zero_badge?.[ratioKey] === true;
      chip.textContent = zero ? "0" : `≤ ${pct}%`;

      li.appendChild(left);
      li.appendChild(chip);
      col.appendChild(li);
    }
  }
}


function initBlindspotControls(){
  // Quellen
  const elSrcMin = document.getElementById('bsMinSourcesMin');
  const elSrcMax = document.getElementById('bsMinSourcesMax');
  const elSrcVal = document.getElementById('bsMinSourcesVal');

  // Erwähnungen
  const elTotMin = document.getElementById('bsMinTotalMin');
  const elTotMax = document.getElementById('bsMinTotalMax');
  const elTotVal = document.getElementById('bsMinTotalVal');

  // Ratio (in Prozent am Slider; im State in Dezimal)
  const elRatMin = document.getElementById('bsRatioMin');
  const elRatMax = document.getElementById('bsRatioMax');
  const elRatVal = document.getElementById('bsRatioVal');

  function clampPair(aEl, bEl){
    let a = parseInt(aEl.value,10), b = parseInt(bEl.value,10);
    if (a > b) [a, b] = [b, a];
    aEl.value = String(a); bEl.value = String(b);
    return [a, b];
  }

  function syncSources(){
    const [a,b] = clampPair(elSrcMin, elSrcMax);
    BLINDSPOT_UI.min_sources_range = [a,b];
    elSrcVal.textContent = `${a} – ${b}`;
    updateBlindspotInfo();
  }
  function syncTotals(){
    const [a,b] = clampPair(elTotMin, elTotMax);
    BLINDSPOT_UI.min_total_range = [a,b];
    elTotVal.textContent = `${a} – ${b}`;
    updateBlindspotInfo();
  }
  function syncRatios(){
    const [a,b] = clampPair(elRatMin, elRatMax);
    // im State als Dezimalzahlen (0–1)
    BLINDSPOT_UI.ratio_range = [a/100, b/100];
    elRatVal.textContent = `${a}% – ${b}%`;
    updateBlindspotInfo();
  }

  // Initial
  syncSources(); syncTotals(); syncRatios();

    // NEU: einmal mit den Presets laden
  loadBlindspotFeed();

  // Live-Update bei Input …
  [elSrcMin, elSrcMax].forEach(el => el.addEventListener('input', syncSources));
  [elTotMin, elTotMax].forEach(el => el.addEventListener('input', syncTotals));
  [elRatMin, elRatMax].forEach(el => el.addEventListener('input', syncRatios));

  // … und Laden beim Loslassen
  ['change','mouseup','touchend'].forEach(evt => {
    [elSrcMin, elSrcMax, elTotMin, elTotMax, elRatMin, elRatMax]
      .forEach(el => el.addEventListener(evt, loadBlindspotFeed));
  });
}

// malt den "einzigen" Track als Farbverlauf auf das obere (max-)Input
function bindDualRange(minId, maxId, displayId, {min, max, suffix = ''}, onCommit) {
  const a = document.getElementById(minId);
  const b = document.getElementById(maxId);
  const out = document.getElementById(displayId);
  const wrap = a.closest('.dual-lite') || b.closest('.dual-lite');

  const pct = v => (v - min) / (max - min);

  function sync(commit=false){
    let v1 = +a.value, v2 = +b.value;
    if (v1 > v2) [v1, v2] = [v2, v1];
    a.value = v1; b.value = v2;

    // Pixelgenaue Progress-Position: Track = wrapperBreite - ThumbBreite
    const wrapW = wrap.clientWidth;
    const thumbW = parseFloat(getComputedStyle(wrap).getPropertyValue('--thumb')) || 16;
    const trackW = Math.max(0, wrapW - thumbW);
    const leftPx = (thumbW / 2) + pct(v1) * trackW;
    const rightPx = (thumbW / 2) + pct(v2) * trackW;

    wrap.style.setProperty('--start-px', `${leftPx}px`);
    wrap.style.setProperty('--end-px', `${rightPx}px`);

    out.textContent = `${v1}${suffix} – ${v2}${suffix}`;
    if (commit && typeof onCommit === 'function') onCommit(v1, v2);
  }

  a.addEventListener('input', () => sync(false));
  b.addEventListener('input', () => sync(false));
  ['change','mouseup','touchend'].forEach(ev => {
    a.addEventListener(ev, () => sync(true));
    b.addEventListener(ev, () => sync(true));
  });

  // bei Resize neu berechnen (responsive)
  window.addEventListener('resize', () => sync(false));

  sync();
}


// Initialisieren (einmal nach DOM-Load)
window.addEventListener('load', () => {
  bindDualRange('bsMinSourcesMin','bsMinSourcesMax','bsMinSourcesVal',
    {min:1, max:50}, () => {           // Commit → State updaten + Feed neu
      const lo = +bsMinSourcesMin.value, hi = +bsMinSourcesMax.value;
      BLINDSPOT_UI.min_sources_range = [Math.min(lo,hi), Math.max(lo,hi)];
      updateBlindspotInfo(); loadBlindspotFeed();
    });

  bindDualRange('bsMinTotalMin','bsMinTotalMax','bsMinTotalVal',
    {min:1, max:50}, () => {
      const lo = +bsMinTotalMin.value, hi = +bsMinTotalMax.value;
      BLINDSPOT_UI.min_total_range = [Math.min(lo,hi), Math.max(lo,hi)];
      updateBlindspotInfo(); loadBlindspotFeed();
    });

  bindDualRange('bsRatioMin','bsRatioMax','bsRatioVal',
    {min:0, max:100, suffix:'%'}, () => {
      const lo = +bsRatioMin.value, hi = +bsRatioMax.value;
      BLINDSPOT_UI.ratio_range = [Math.min(lo,hi)/100, Math.max(lo,hi)/100]; // Dezimal
      updateBlindspotInfo(); loadBlindspotFeed();
    });
});



window.addEventListener('load', initBlindspotControls);

function updateBlindspotInfo() {
  const info = document.getElementById('blindspotInfo');
  if (!info) return;
  const [s1,s2] = BLINDSPOT_UI.min_sources_range || [2,10];
  const [t1,t2] = BLINDSPOT_UI.min_total_range   || [10,20];
  const [r1,r2] = BLINDSPOT_UI.ratio_range       || [0.01,0.05];
  info.textContent = `Keywords mit ${s1}–${s2} Quellen, ${t1}–${t2} Erwähnungen und einem Blindspot-Wert zwischen ${Math.round(r1*100)} % und ${Math.round(r2*100)} %.`;
}


  async function loadFilteredMediaCompass() {
    const qs = new URLSearchParams();
    appendSources(qs);
    appendCountries(qs);
    if (FILTERS.keyword) qs.set("keyword", FILTERS.keyword);
    if (FILTERS.teaser)  qs.set("teaser", "true");
    const key = qs.toString();
    if (key === _lastCompassKey) return;  // nichts Neues → keine Arbeit
    _lastCompassKey = key;
    const res = await fetch(withTime("/media-positions/filtered", key));
    const data = await res.json();
    drawMediaCompass(data, "Gefilterte Medien");
  }



    // Event-Listener für die Quellen-Auswahl
    document.getElementById("sourceSelect").addEventListener("change", (e) => {
      filters.source = e.target.value;
      loadFilteredArticles();
      loadKeywordTrends();
      loadExtremeBubbles();
      loadTopAbsoluteKeywords();
      loadFilteredMediaCompass();
      updateFilterDisplay();
      loadBlindspotFeed();
      updateURLFromFilters(false);
    });


    document.getElementById("teaserToggle").addEventListener("change", (e) => {
      filters.teaser = e.target.checked;
      loadFilteredArticles();
      loadKeywordTrends();      // 🔄 aktualisieren
      loadExtremeBubbles();     // 🔄 aktualisieren
      loadWordcloud();
      loadTopAbsoluteKeywords();
      loadFilteredMediaCompass();
      updateFilterDisplay();
      loadBlindspotFeed();
      updateURLFromFilters(false);



    });






/* === Drawer öffnen/schließen === */
const drawer = document.getElementById('filterDrawer');
const scrim  = document.getElementById('scrim');
const btnOpen  = document.getElementById('filterOpen');
const btnClose = document.getElementById('drawerClose');
const btnDone  = document.getElementById('drawerDone');

function openDrawer(){
  drawer.classList.add('open'); drawer.setAttribute('aria-hidden','false');
  scrim.classList.add('show');  scrim.setAttribute('aria-hidden','false');
}
function closeDrawer(){
  drawer.classList.remove('open'); drawer.setAttribute('aria-hidden','true');
  scrim.classList.remove('show');  scrim.setAttribute('aria-hidden','true');
}
btnOpen.addEventListener('click', openDrawer);
btnClose.addEventListener('click', closeDrawer);
btnDone.addEventListener('click', closeDrawer);
scrim.addEventListener('click', closeDrawer);

/* === Aktive Filter: Chips darstellen === */
function renderActiveFilterChips(){
  const bar = document.getElementById('activeFiltersBar');
  if (!bar) return;
  bar.innerHTML = ''; // reset

  // Helper zum Erstellen eines Chips
  const mk = (label, onRemove, {removable=true, ghost=false} = {})=>{
    const pill = document.createElement('span');
    pill.className = 'pill' + (ghost ? ' pill--ghost' : '');
    pill.textContent = label;

    if (removable){
      const x = document.createElement('button');
      x.className = 'pill-x';
      x.type = 'button';
      x.setAttribute('aria-label', label + ' entfernen');
      x.textContent = '×';
      x.onclick = onRemove;
      pill.appendChild(x);
    }
    bar.appendChild(pill);
  };

  // Zeitfenster (ganze Tage)
  if (FILTERS.from && FILTERS.to) {
    const f = new Date(FILTERS.from);
    const t = new Date(new Date(FILTERS.to).getTime() - 1);
    const fmt = new Intl.DateTimeFormat('de-DE', { day:'2-digit', month:'2-digit', year:'2-digit' });
    mk(`📅 ${fmt.format(f)} – ${fmt.format(t)}`, () => {
      const today = new Date();
      FILTERS.from = toLocalIsoWithTZ(startOfDay(new Date(today.getTime() - 6*86400000)));
      FILTERS.to   = toLocalIsoWithTZ(endOfDayExclusive(today));
      markActiveDays(7);
      updateDateRangeBadge();
      updateURLFromFilters(false);
      applyFilters();
    });
  }

  // Quelle
  if (ACTIVE_SOURCES.size > 0) {
    for (const s of Array.from(ACTIVE_SOURCES).sort()) {
      mk(`📰 ${s}`, () => {
        ACTIVE_SOURCES.delete(s);
        // Checkbox sync
        const cb = document.querySelector(`#sourcePanel .source-box[value="${CSS.escape(s)}"]`);
        if (cb) cb.checked = false;
        // Abwärtskompatibilität: FILTERS.source auf ersten Eintrag setzen/entfernen
        FILTERS.source = ACTIVE_SOURCES.size ? Array.from(ACTIVE_SOURCES)[0] : null;
        updateSourceBtnLabel();
        updateURLFromFilters(false);
        applyFilters();
      });
    }
  } else {
    mk('📰 Alle Quellen', null, { removable:false, ghost:true });
  }


  // Keyword
  if (filters.keyword){
    mk(`🔍 ${filters.keyword}`, () => {
      filters.keyword = '';
      if (typeof selectedKeyword !== 'undefined') selectedKeyword = '';
      const inp = document.getElementById('keywordInput'); if (inp) inp.value = '';
      loadFilteredArticles();
      loadExtremeBubbles();
      loadTopAbsoluteKeywords();
      loadFilteredMediaCompass();
      loadBlindspotFeed();
      updateURLFromFilters(false);
      renderActiveFilterChips();
    });
  }

  // Teaser
  if (filters.teaser){
    mk('📝 Teaser', () => {
      filters.teaser = false;
      const cb = document.getElementById('teaserToggle'); if (cb) cb.checked = false;
      loadFilteredArticles();
      loadExtremeBubbles();
      loadTopAbsoluteKeywords();
      loadFilteredMediaCompass();
      renderActiveFilterChips();
      loadBlindspotFeed();
    });
  }

  // Ngram
  if (filters.ngram && filters.ngram !== 1){
    const label = filters.ngram === 2 ? '🔗 2er-Kombis' : '🔗 3er-Kombis';
    mk(label, () => {
      filters.ngram = 1;
      loadWordcloud();
      loadFilteredMediaCompass();
      renderActiveFilterChips();
    });
  }

  if (ACTIVE_COUNTRIES.size > 0) {
    for (const c of Array.from(ACTIVE_COUNTRIES).sort()) {
      mk(`🌍 ${c}`, () => {
        ACTIVE_COUNTRIES.delete(c);
        updateCountryBtnLabel();
        updateURLFromFilters(false);
        applyFilters();
      });
    }
  } else {
    mk('🌍 Alle Länder', null, { removable:false, ghost:true });
  }


  // Alle löschen (schneller Reset)
  const dPreset = presetDaysFromRange();
  if (filters.source || filters.keyword || filters.teaser || dPreset !== 7){
    const all = document.createElement('button');
    all.className = 'pill';
    all.type = 'button';
    all.textContent = 'Alle ×';
    all.onclick = () => resetFilters(); // deine bestehende Funktion
    bar.appendChild(all);
  }
}

/* === Bestehende Anzeige-Funktion an neue Chips binden === */
const _updateFilterDisplay = window.updateFilterDisplay || (()=>{});
window.updateFilterDisplay = function(){
  // Alte Textanzeige (falls noch vorhanden) aktualisieren
  try { _updateFilterDisplay(); } catch {}
  // Neue Chips rendern
  renderActiveFilterChips();
};

/* === Nach Initialisierung & bei Filter-Events Chips aktualisieren === */
window.addEventListener('load', renderActiveFilterChips);

// Falls du setTimeFilter / resetFilters überschreibst, Chips dort erneut rendern:
// Beispiel (optional), nur wenn du sie "wrappen" möchtest:
/*
const _setTimeFilter = window.setTimeFilter;
window.setTimeFilter = function(h){
  _setTimeFilter(h);
  renderActiveFilterChips();
};
const _resetFilters = window.resetFilters;
window.resetFilters = function(){
  _resetFilters();
  renderActiveFilterChips();
};
*/

const dayBtns = Array.from(document.querySelectorAll('.time-presets [data-days]'));

function markActiveDays(d) {
  dayBtns.forEach(b => b.classList.remove('is-active'));
  const btn = dayBtns.find(b => parseInt(b.getAttribute('data-days'),10) === Number(d));
  if (btn) btn.classList.add('is-active');
}

document.addEventListener('click', (e) => {
  const btn = e.target.closest('.time-presets [data-days]');
  if (!btn) return;
  const days = parseInt(btn.getAttribute('data-days'), 10);
  if (Number.isNaN(days)) return;
  onPresetDaysClick(days);
}, true);

function onPresetDaysClick(days){
  USER_TIME_OVERRIDE = true;  // ← Nutzer hat aktiv einen Zeitraum gewählt
  const endEx = nowExclusive(); // JETZT (exklusiv)
  const start = startOfDay(new Date(endEx.getTime() - days*86400000));

  FILTERS.from = toLocalIsoWithTZ(start);
  FILTERS.to   = toLocalIsoWithTZ(endEx);

  markActiveDays(days);
  updateDateRangeBadge();
  updateURLFromFilters(false);
  applyFilters();
}



// Bias-Buttons markieren
const biasBtns = Array.from(document.querySelectorAll('#biasFilters button'));
function markActiveBias(nameOrNull){
  biasBtns.forEach(b => b.classList.remove('is-active'));
  let label = 'Alle';
  if (nameOrNull === 'systemkritisch') label = 'Systemkritisch';
  else if (nameOrNull === 'systemnah')  label = 'Systemnah';
  else if (nameOrNull === 'globalistisch') label = 'Globalistisch';
  else if (nameOrNull === 'nationalistisch') label = 'Nationalistisch';
  const btn = biasBtns.find(b => b.textContent.trim() === label);
  if (btn) btn.classList.add('is-active');
}
const _setBiasFilter = window.setBiasFilter;
window.setBiasFilter = function(name){
  _setBiasFilter(name);
  markActiveBias(name || null);
};

// Initiale Markierung (nach load)
window.addEventListener('load', () => {
  const d = presetDaysFromRange();   // ermittelt 1/3/7/30 aus from/to, sonst null
  if (d) markActiveDays(d);
  markActiveBias(window.activeBiasFilter || null);
});


  function inferBucketFromTimeline(data){
    // Nimmt Timeline-Daten [{time, count}] und erkennt hour/day/week
    if (!data || data.length < 2) return "hour";
    const a = new Date(data[0].time).getTime();
    const b = new Date(data[1].time).getTime();
    const diffH = Math.round((b - a) / 3_600_000);
    if (diffH === 1) return "hour";
    if (diffH >= 24*7) return "week";
    if (diffH >= 24) return "day";
    return "hour";
  }

  function formatTickLabel(iso, bucket){
    const d = new Date(iso);
    if (bucket === "hour") {
      const df = new Intl.DateTimeFormat('de-DE',{day:'2-digit',month:'2-digit'});
      const tf = new Intl.DateTimeFormat('de-DE',{hour:'2-digit'});
      return `${df.format(d)} ${tf.format(d)}h`;
    }
    if (bucket === "day") {
      return new Intl.DateTimeFormat('de-DE',{weekday:'short', day:'2-digit', month:'2-digit'}).format(d);
    }
    if (bucket === "week") {
      // Kalenderwoche grob als Montagsdatum anzeigen
      return "KW " + weekOfYear(d);
    }
    return new Intl.DateTimeFormat('de-DE').format(d);
  }

  function weekOfYear(date){
    // ISO-Woche (Mo-So)
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    const dayNum = (d.getUTCDay() + 6) % 7; // 0=Mo
    d.setUTCDate(d.getUTCDate() - dayNum + 3);
    const firstThursday = new Date(Date.UTC(d.getUTCFullYear(),0,4));
    const week = 1 + Math.round(((d - firstThursday) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7);
    return week;
  }

  // --- H: Empty-State Helpers -------------------------------------------------
  function showEmptyState(container, message, hint){
    if (!container) return;
    container.innerHTML = `
      <div class="empty-state">
        ${message}
        ${hint ? `<span class="hint">${hint}</span>` : ""}
      </div>`;
  }
  function clearEmptyState(container){
    if (!container) return;
    container.innerHTML = "";
  }

  // Für Charts: Overlay auf einem umschließenden Relativ-Container
  function mountChartEmptyOverlay(container, text="Keine Daten im gewählten Zeitraum"){
    if (!container) return;
    container.style.position = container.style.position || "relative";
    let overlay = container.querySelector('.chart-overlay-empty');
    if (!overlay){
      overlay = document.createElement('div');
      overlay.className = 'chart-overlay-empty';
      overlay.textContent = text;
      container.appendChild(overlay);
    } else {
      overlay.style.display = 'grid';
      overlay.textContent = text;
    }
  }
  function hideChartEmptyOverlay(container){
    const ov = container?.querySelector?.('.chart-overlay-empty');
    if (ov) ov.style.display = 'none';
  }



  function presetDaysFromRange() {
    if (!(FILTERS.from && FILTERS.to)) return null;
    const n = Math.round((new Date(FILTERS.to) - new Date(FILTERS.from)) / 86400000);
    return [1,3,7,30].includes(n) ? n : null;
  }

  // 1) ganz oben irgendwo nach den Preset-Funktionen:
  window.addEventListener('load', () => {
    const d = presetDaysFromRange();
    if (d) markActiveDays(d);
  });

  // 2) und weiter unten bei der kombinierten Initial-Markierung:
  window.addEventListener('load', () => {
    const d = presetDaysFromRange();
    if (d) markActiveDays(d);
    markActiveBias(window.activeBiasFilter || null);
  });





  function formatBadgeLabel(fromIso, toIso) {
    try {
      const f = new Date(fromIso);
      const t = new Date(new Date(toIso).getTime() - 1);
      const df = new Intl.DateTimeFormat('de-DE',{day:'2-digit',month:'2-digit',year:'2-digit'});
      return `${(window.DATE_RANGE_SPEC?.I18N?.badgePrefix)||'📅'} ${df.format(f)} – ${df.format(t)}`;
    } catch { return `${(window.DATE_RANGE_SPEC?.I18N?.badgePrefix)||'📅'} benutzerdefiniert`; }
  }

  function updateDateRangeBadge() {
    const el = document.getElementById('date-range-badge');
    if (!el) return;
    if (FILTERS.from && FILTERS.to) {
      el.style.display = 'inline-flex';
      const label = el.querySelector('.label');
      if (label) label.textContent = formatBadgeLabel(FILTERS.from, FILTERS.to);
    } else {
      el.style.display = 'none';
    }
  }

  document.addEventListener('click', (e) => {
    const rm = e.target.closest('#date-range-badge .remove');
    if (!rm) return;

    USER_TIME_OVERRIDE = false;  // ← zurück auf Default-5-Wochen-Logik

    const endEx = nowExclusive();
    FILTERS.from = toLocalIsoWithTZ(startOfDay(new Date(endEx.getTime() - 6*86400000)));
    FILTERS.to   = toLocalIsoWithTZ(endEx);
    markActiveDays(7);
    updateDateRangeBadge();
    updateURLFromFilters(false);
    applyFilters();
  }, true);



  // ---- APPLY FILTERS: ALLE FETCHES SYNCHRON MIT from/to ODER hours ----------
  // Suche in deinem Code alle Stellen, wo URLs gebaut werden, und hänge buildTimeQuery() an.
  // Unten zwei typische Beispiele; passe sie an deine tatsächlichen Funktionen an.

  async function fetchFilteredArticles() {
    const base = '/articles/filtered';
    const time = buildTimeQuery();
    const qs = new URLSearchParams();
    if (FILTERS.source)  qs.set('source', FILTERS.source);
    if (FILTERS.keyword) qs.set('keyword', FILTERS.keyword);
    if (FILTERS.teaser)  qs.set('teaser', '1');
    const url = `${base}?${time}${qs.toString() ? '&' + qs.toString() : ''}`;
    const res = await fetch(url);
    return res.json();
  }

  async function fetchKeywordTimeline(word) {
    const base = '/keywords/timeline';
    const time = buildTimeQuery();
    const qs = new URLSearchParams({ word });
    // Optional: bucket=auto|hour|day|week (Standard: auto)
    // qs.set('bucket', 'auto');
    const url = `${base}?${time}&${qs.toString()}`;
    const res = await fetch(url);
    return res.json();
  }



  // --- G: Flatpickr Range-Picker mit Uhrzeit ---------------------------------
  let fp = null;

  function openCustomRangeModal() {
    const modal = document.getElementById('customRangeModal');
    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');
    modal.setAttribute('aria-modal', 'true');
    // Fokus in das Eingabefeld
    setTimeout(() => document.getElementById('fpRange').focus(), 0);
  }

  function closeCustomRangeModal() {
    const modal = document.getElementById('customRangeModal');
    modal.style.display = 'none';
    modal.setAttribute('aria-hidden', 'true');
    modal.setAttribute('aria-modal', 'false');
    document.getElementById('btnCustomRange').focus();
  }


  function startOfDay(d) {
    const x = new Date(d);
    x.setHours(0,0,0,0);
    return x;
  }
  function endOfDayExclusive(d) {
    // exklusives Ende: 00:00 des FOLGETAGES
    const x = new Date(d);
    x.setHours(0,0,0,0);
    x.setDate(x.getDate() + 1);
    return x;
  }
  function toLocalIsoWithTZ(d) {
    const pad = n => String(n).padStart(2,'0');
    const offMin = -d.getTimezoneOffset();
    const sign = offMin >= 0 ? '+' : '-';
    const hh = pad(Math.floor(Math.abs(offMin)/60));
    const mm = pad(Math.abs(offMin)%60);
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}${sign}${hh}:${mm}`;
  }

  // Liefert "jetzt" als exklusive Obergrenze (Sekunden=00 für Konsistenz)
  function nowExclusive(){
    const n = new Date();
    n.setSeconds(0, 0);
    return n;
  }

  // Klemmt einen [from, to)-Bereich so, dass 'to' niemals in der Zukunft liegt.
  // Gibt {from:Date, to:Date} zurück (beide lokale Date-Objekte).
  function clampRangeToNow(fromDate, toDate){
    const now = nowExclusive();
    // to darf nicht in der Zukunft liegen
    const toClamped = (toDate > now) ? now : toDate;
    // und muss strikt > from bleiben (sonst min. +1 Minute)
    const minTo = new Date(fromDate.getTime() + 60*1000);
    return {
      from: fromDate,
      to: (toClamped <= fromDate) ? minTo : toClamped
    };
  }

  // Max-Range aus Spec holen (Fallback 180 Tage)
  function getMaxRangeDays() {
    return (window.DATE_RANGE_SPEC?.MAX_RANGE_DAYS ?? 180);
  }

  // Validierung: [from, to) und Max-Range
  function validateRange(fDate, tDate) {
    if (!fDate || !tDate) return { ok:false, msg: "Bitte Start und Ende wählen." };
    if (tDate <= fDate)   return { ok:false, msg: "Ende muss nach Start liegen." };
    const maxDays = getMaxRangeDays();
    const diffDays = (tDate - fDate) / (1000 * 60 * 60 * 24);
    if (diffDays > maxDays) return { ok:false, msg: `Zeitraum zu groß (max. ${maxDays} Tage).` };
    const now = new Date();
    if (fDate > now || tDate > now) return { ok:false, msg: "Zeiten in der Zukunft sind nicht erlaubt." };
    return { ok:true };
  }

function initFlatpickr() {
  const input = document.getElementById('fpRange');
  if (!input) return;

  if (window.flatpickr && window.flatpickr.l10ns && window.flatpickr.l10ns.de) {
    flatpickr.localize(flatpickr.l10ns.de);
  }

  fp = flatpickr(input, {
    mode: "range",
    enableTime: false,          // <— KEINE Uhrzeit
    time_24hr: true,            // egal, da keine Zeit
    altInput: true,
    altFormat: "d.m.Y",
    dateFormat: "Y-m-d",        // nur Datum
    defaultDate: (() => {
      if (FILTERS.from && FILTERS.to) {
        // FILTERS.to ist exklusiv -> -1 Tag für Anzeige
        const f = startOfDay(new Date(FILTERS.from));
        const tExclusive = new Date(FILTERS.to);
        const tDisplay = new Date(tExclusive.getTime() - 1); // 1ms zurück in den Vortag
        return [f, tDisplay];
      }
      // Fallback: letzte 7 Tage inkl. heute
      const today = new Date();
      const from  = startOfDay(new Date(today.getTime() - 6*86400000));
      return [from, today];
    })(),
    maxDate: new Date(),
    onOpen: () => {
      if (FILTERS.from && FILTERS.to) {
        const f = startOfDay(new Date(FILTERS.from));
        const tDisplay = new Date(new Date(FILTERS.to).getTime() - 1);
        fp.setDate([f, tDisplay], false);
      }
    }
  });
}

// Modal-Buttons
document.addEventListener('click', (e) => {
  if (e.target.matches('#btnCustomRange')) {
    e.preventDefault();
    openCustomRangeModal();
    if (!fp) initFlatpickr();
  }
  if (e.target.matches('#btnRangeClose') || e.target.matches('#btnRangeCancel')) {
    e.preventDefault();
    closeCustomRangeModal();
  }
  // … innerhalb deines Click-Handlers für #btnRangeApply:
  if (e.target.matches('#btnRangeApply')) {
    e.preventDefault();
    if (!fp) { closeCustomRangeModal(); return; }
    const sel = fp.selectedDates || [];
    if (sel.length < 2) { alert("Bitte Start und Ende wählen."); return; }

    const f = startOfDay(sel[0]);
    const chosenEndExclusive = endOfDayExclusive(sel[1]); // 00:00 Folgetag
    const endEx = nowExclusive();
    const toEx  = (chosenEndExclusive > endEx) ? endEx : chosenEndExclusive;

    if (toEx <= f) { alert("Ende muss nach Start liegen."); return; }

    USER_TIME_OVERRIDE = true;   // ← Nutzer hat aktiv gesetzt
    FILTERS.from = toLocalIsoWithTZ(f);
    FILTERS.to   = toLocalIsoWithTZ(toEx);

    const cb = document.getElementById('cbIncludeTeaserInRange');
    if (cb) FILTERS.teaser = !!cb.checked;

    markActiveDays(null);
    updateDateRangeBadge();
    updateURLFromFilters(false);
    applyFilters();
    closeCustomRangeModal();
  }


}, true);


// ESC schließt Modal
document.addEventListener('keydown', (e) => {
  const modal = document.getElementById('customRangeModal');
  if (modal?.style.display === 'flex' && e.key === 'Escape') {
    closeCustomRangeModal();
  }
});

// Bei erstem Load Picker vorbereiten (nicht öffnen)
window.addEventListener('DOMContentLoaded', () => {
  // Nur initialisieren, wenn du willst dass der Picker sofort bereit ist:
  // initFlatpickr();
});



