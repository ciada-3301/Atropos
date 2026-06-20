/* ============================================================
   Atropos Explorer — App logic
   ------------------------------------------------------------
   Fetches the reference cloud + region data once on load, wires
   up the control panel, and calls /api/predict when the user
   maps a sequence.
   ============================================================ */

(function () {
  const els = {
    canvas: document.getElementById("viewportCanvas"),
    loadingOverlay: document.getElementById("loadingOverlay"),
    statPoints: document.getElementById("statPoints"),
    statRegions: document.getElementById("statRegions"),
    sequenceInput: document.getElementById("sequenceInput"),
    kSlider: document.getElementById("kSlider"),
    kValue: document.getElementById("kValue"),
    mapBtn: document.getElementById("mapBtn"),
    errorBanner: document.getElementById("errorBanner"),
    resultBlock: document.getElementById("resultBlock"),
    topMatchName: document.getElementById("topMatchName"),
    topMatchMeta: document.getElementById("topMatchMeta"),
    topMatchBar: document.getElementById("topMatchBar"),
    topMatchDistance: document.getElementById("topMatchDistance"),
    bestGenus: document.getElementById("bestGenus"),
    confidencePill: document.getElementById("confidencePill"),
    matchesBlock: document.getElementById("matchesBlock"),
    matchesList: document.getElementById("matchesList"),
    regionsList: document.getElementById("regionsList"),
    hoverInfo: document.getElementById("hoverInfo"),
    toggleRegionsBtn: document.getElementById("toggleRegionsBtn"),
    resetCameraBtn: document.getElementById("resetCameraBtn"),
  };

  let regionsByPhylumId = {};

  function init() {
    AtroposViewport.init(els.canvas);
    AtroposViewport.onHover(handleHover);
    loadViewportData();
    wireControls();
  }

  async function loadViewportData() {
    try {
      const res = await fetch("/api/viewport-data");
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();

      AtroposViewport.buildPointCloud(data.points);
      AtroposViewport.buildRegionLabels(data.regions);

      data.regions.forEach((r) => { regionsByPhylumId[r.phylum_id] = r; });
      renderRegionsList(data.regions);

      els.statPoints.textContent = data.points.length.toLocaleString();
      els.statRegions.textContent = data.num_phyla.toLocaleString();

      els.loadingOverlay.classList.add("hidden");
    } catch (err) {
      els.loadingOverlay.innerHTML =
        `<span style="color:#b5564a;">Failed to load reference space.<br>${escapeHtml(err.message)}</span>`;
    }
  }

  function renderRegionsList(regions) {
    if (!regions.length) {
      els.regionsList.innerHTML = `<div class="regions-empty">No labeled regions.</div>`;
      return;
    }
    els.regionsList.innerHTML = regions
      .slice(0, 20)
      .map((r) => {
        const color = AtroposViewport.colorForPhylumId(r.phylum_id);
        return `
          <div class="region-row">
            <span class="region-swatch" style="background:${color}"></span>
            <span class="region-name">${escapeHtml(r.phylum)}</span>
            <span class="region-count">${r.count.toLocaleString()}</span>
          </div>`;
      })
      .join("");
  }

  function wireControls() {
    els.kSlider.addEventListener("input", () => {
      els.kValue.textContent = els.kSlider.value;
    });

    els.mapBtn.addEventListener("click", handleMapSequence);

    els.sequenceInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        handleMapSequence();
      }
    });

    els.toggleRegionsBtn.addEventListener("click", () => {
      const isPressed = els.toggleRegionsBtn.getAttribute("aria-pressed") === "true";
      const next = !isPressed;
      els.toggleRegionsBtn.setAttribute("aria-pressed", String(next));
      AtroposViewport.setRegionsVisible(next);
    });

    els.resetCameraBtn.addEventListener("click", () => {
      AtroposViewport.resetCamera();
    });
  }

  async function handleMapSequence() {
    const sequence = els.sequenceInput.value.trim();
    hideError();

    if (sequence.length < 30) {
      showError("Paste at least ~30bp of raw sequence (or a FASTA record) to map.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence, k: Number(els.kSlider.value) }),
      });

      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Prediction failed.");
        return;
      }

      renderResult(data);
      AtroposViewport.showQueryPoint(data.query_position);
    } catch (err) {
      showError(`Could not reach the server: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function renderResult(data) {
    const matches = data.matches || [];
    if (matches.length === 0) {
      showError("No matches returned.");
      return;
    }

    const top = matches[0];
    const maxDistance = Math.max(...matches.map((m) => m.distance), 0.0001);

    els.topMatchName.textContent = formatSpeciesName(top);
    els.topMatchMeta.textContent = `${top.family} · ${top.phylum}`;
    els.topMatchBar.style.width = `${(top.distance / maxDistance) * 100}%`;
    els.topMatchDistance.textContent = top.distance.toFixed(3);

    els.bestGenus.textContent = data.best_guess_genus;
    els.confidencePill.textContent = `${Math.round(data.confidence * 100)}% confidence`;

    els.matchesList.innerHTML = matches
      .map((m, i) => {
        const pct = (m.distance / maxDistance) * 100;
        return `
          <div class="match-row">
            <span class="match-rank">${String(i + 1).padStart(2, "0")}</span>
            <span class="match-name" title="${escapeHtml(formatSpeciesName(m))}">${escapeHtml(formatSpeciesName(m))}</span>
            <div class="match-bar-track"><div class="match-bar-fill" style="width:${pct}%"></div></div>
            <span class="match-distance">${m.distance.toFixed(3)}</span>
          </div>`;
      })
      .join("");

    els.resultBlock.hidden = false;
    els.matchesBlock.hidden = false;
  }

  function formatSpeciesName(m) {
    if (m.species && m.species !== "(unknown)") return m.species;
    if (m.genus && m.genus !== "(unknown)") return `${m.genus} sp.`;
    return m.family !== "(unknown)" ? `${m.family} (unident.)` : "Unidentified";
  }

  function handleHover(point) {
    if (!point) {
      els.hoverInfo.textContent = "Hover a point to inspect it";
      return;
    }
    const region = regionsByPhylumId[point.phylum_id];
    const label = region ? region.phylum : `taxon group ${point.phylum_id}`;
    els.hoverInfo.textContent = `${label}  ·  x:${point.x.toFixed(2)} y:${point.y.toFixed(2)} z:${point.z.toFixed(2)}`;
  }

  function setLoading(isLoading) {
    els.mapBtn.disabled = isLoading;
    els.mapBtn.textContent = isLoading ? "Mapping…" : "Map sequence";
  }

  function showError(msg) {
    els.errorBanner.textContent = msg;
    els.errorBanner.hidden = false;
  }

  function hideError() {
    els.errorBanner.hidden = true;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  document.addEventListener("DOMContentLoaded", init);
})();