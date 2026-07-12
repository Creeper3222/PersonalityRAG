export function createSystemController({ $, state, t, toast, libraryApi, statCards, formatVersionTag, escapeHtml }) {
async function loadSystem() {
  try {
    const data = await libraryApi("/stats");
    state.stats = data;
    statCards($("system-version-stats"), [
      [t("serviceVersion"), formatVersionTag(data.service_version)],
      [t("livingMemoryDbVersion"), formatVersionTag(data.livingmemory_database_version)],
    ]);
    statCards($("system-stats"), [
      [t("statsMemories"), data.total_memories],
      [t("statsNodes"), data.graph_nodes],
      [t("statsRelations"), data.graph_edges],
      [t("statsGraphEntries"), data.graph_entries],
      [t("statsAtoms"), data.atom_count],
      [t("statsActiveSessions"), Object.keys(data.sessions || {}).length],
      [t("statsMessages"), data.conversation_counts?.messages || 0],
    ]);
    $("provider-status").textContent = JSON.stringify(
      {
        library: data.library,
        provider: data.provider,
        status: data.provider_status,
      },
      null,
      2,
    );
    $("index-status").textContent = JSON.stringify(data.indexes, null, 2);
    scheduleSystemPanelsRefresh();
    drawBars($("importance-chart"), data.importance_distribution || {});
    drawBars($("atom-chart"), data.atom_breakdown || {});
    $("backup-list").innerHTML =
      (data.backups || [])
        .map(
          (item) =>
            `<div class="backup-item"><b>${escapeHtml(item.name)}</b><small>${item.file_count} ${escapeHtml(t("filesUnit"))}</small><small>${formatBytes(item.size_bytes)}</small></div>`,
        )
        .join("") || t("noBackups");
  } catch (error) {
    toast(error.message, true);
  }
}

function systemCollapsedHeight() {
  return Math.max(280, Math.round(window.innerHeight * 0.5));
}

function applySystemPanelCollapseState({ panelId, contentId, toggleId, expanded }) {
  const panel = $(panelId);
  const content = $(contentId);
  const toggle = $(toggleId);
  if (!panel || !content || !toggle) return;
  const collapsedHeight = systemCollapsedHeight();
  panel.style.setProperty("--system-panel-collapsed-height", `${collapsedHeight}px`);
  const overflowing = content.scrollHeight > collapsedHeight + 24;
  if (!overflowing) {
    toggle.classList.add("hidden");
    toggle.classList.remove("expanded");
    panel.classList.remove("system-panel-collapsed", "system-panel-expanded");
    return;
  }
  panel.classList.toggle("system-panel-collapsed", !expanded);
  panel.classList.toggle("system-panel-expanded", expanded);
  toggle.classList.toggle("expanded", expanded);
  toggle.classList.remove("hidden");
  const title = expanded ? t("collapseLibraryCard") : t("expandLibraryCard");
  toggle.title = title;
  toggle.setAttribute("aria-label", title);
  toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function applySystemProviderCollapseState() {
  applySystemPanelCollapseState({
    panelId: "system-provider-panel",
    contentId: "provider-status",
    toggleId: "system-provider-toggle",
    expanded: state.systemProviderExpanded,
  });
}

function applySystemIndexCollapseState() {
  applySystemPanelCollapseState({
    panelId: "system-index-panel",
    contentId: "index-status",
    toggleId: "system-index-toggle",
    expanded: state.systemIndexExpanded,
  });
}

function scheduleSystemPanelsRefresh() {
  if (state.systemPanelRefreshFrame) {
    cancelAnimationFrame(state.systemPanelRefreshFrame);
  }
  state.systemPanelRefreshFrame = requestAnimationFrame(() => {
    state.systemPanelRefreshFrame = 0;
    applySystemProviderCollapseState();
    applySystemIndexCollapseState();
  });
}

$("system-provider-toggle")?.addEventListener("click", () => {
  state.systemProviderExpanded = !state.systemProviderExpanded;
  applySystemProviderCollapseState();
});

$("system-index-toggle")?.addEventListener("click", () => {
  state.systemIndexExpanded = !state.systemIndexExpanded;
  applySystemIndexCollapseState();
});

function drawBars(target, data) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map((entry) => entry[1]));
  target.innerHTML = entries
    .map(
      ([key, value]) =>
        `<div class="bar-row"><span>${escapeHtml(key)}</span><div class="bar-track"><div class="bar-fill" style="width:${(value / max) * 100}%"></div></div><b>${value}</b></div>`,
    )
    .join("");
}

function formatBytes(size) {
  if (size < 1024) return size + " B";
  if (size < 1048576) return (size / 1024).toFixed(1) + " KB";
  if (size < 1073741824) return (size / 1048576).toFixed(1) + " MB";
  return (size / 1073741824).toFixed(2) + " GB";
}

  return { loadSystem, scheduleSystemPanelsRefresh, formatBytes };
}
