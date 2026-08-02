export function createLibrariesController(deps) { const { $, state, t, toast, api, selectedDatabaseApi, selectedDatabase, selectDatabase, refreshDatabaseContext, escapeHtml, confirmDialog, addOptimisticTask, removeOptimisticTask, trackQueuedJob, validateIdentifierInput, loadProviders, fillProviderSelect, activatePage, closeOverlay, joinLocalizedList, LIBRARY_EXPAND_ICON, DEFAULT_LIBRARY_CONVERSATION_SETTINGS, DEFAULT_LIBRARY_RECALL_SETTINGS, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS, asyncGuard, databaseApiPath, databaseCollectionApiPath, databaseRefKey, DEFAULT_DATABASE_TYPE, databaseUiRegistry, databasePageRoute, databaseTypePageRoute } = deps;
const CATALOG_CACHE_MS = 750;
let catalogLoadedAt = 0;
let catalogLoadPromise = null;
function databaseTypeForId(databaseId) {
  return state.databases.find((item) => (
    item.id === databaseId
    && (item.database_type || DEFAULT_DATABASE_TYPE) === state.selectedDatabaseType
  ))?.database_type || DEFAULT_DATABASE_TYPE;
}

function databaseForRef(databaseType, databaseId) {
  return state.databases.find((item) => (
    item.id === databaseId
    && (item.database_type || DEFAULT_DATABASE_TYPE) === databaseType
  )) || null;
}

function ensureCategorySelections() {
  state.selectedDatabaseRefByCategory ||= { memory: null, knowledge: null };
  return state.selectedDatabaseRefByCategory;
}

function databaseRefForRecord(library) {
  return library ? {
    id: library.id,
    databaseType: library.database_type || DEFAULT_DATABASE_TYPE,
  } : null;
}

function fallbackSelectionForCategory(category) {
  const library = state.databases.find((item) => item.database_category === category && item.is_default)
    || state.databases.find((item) => item.database_category === category);
  if (library) return databaseRefForRecord(library);
  const databaseType = state.databaseTypes.find((item) => item.category === category)?.id
    || (category === "memory" ? DEFAULT_DATABASE_TYPE : "");
  return { id: "", databaseType };
}

function rememberCategorySelection(library) {
  if (!library?.database_category) return;
  ensureCategorySelections()[library.database_category] = databaseRefForRecord(library);
}

function selectionForCategory(category) {
  const saved = ensureCategorySelections()[category];
  if (saved?.id) {
    const library = databaseForRef(saved.databaseType, saved.id);
    if (library?.database_category === category) return saved;
  }
  const fallback = fallbackSelectionForCategory(category);
  if (fallback?.id) {
    ensureCategorySelections()[category] = fallback;
  }
  return fallback;
}

function selectedDatabaseNameForCategory(category) {
  const selection = selectionForCategory(category);
  if (!selection?.id) return "";
  const library = databaseForRef(selection.databaseType, selection.id);
  return library?.name || "";
}

function applySelection(selection) {
  state.selectedDatabaseId = selection?.id || "";
  state.selectedDatabaseType = selection?.databaseType || state.selectedDatabaseType || DEFAULT_DATABASE_TYPE;
}

function updateDatabaseCategoryTabs() {
  document.querySelectorAll(".database-category-tab").forEach((item) => {
    const active = item.dataset.databaseCategory === state.databaseCategory;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", active ? "true" : "false");
    const hint = item.querySelector(".database-category-tab-current");
    if (hint) {
      const currentName = selectedDatabaseNameForCategory(item.dataset.databaseCategory || "");
      hint.textContent = currentName ? `· ${currentName}` : "";
      hint.classList.toggle("hidden", !currentName);
    }
  });
}

function setDatabaseCategory(category, options = {}) {
  if (!["memory", "knowledge"].includes(category)) return;
  state.databaseCategory = category;
  updateDatabaseCategoryTabs();
  if (options.restoreSelection !== false) {
    applySelection(selectionForCategory(category));
  }
}

async function fetchCatalog(options = {}) {
  const now = Date.now();
  const fresh = !options.force
    && state.databases.length
    && state.databaseTypes.length
    && now - catalogLoadedAt < CATALOG_CACHE_MS;
  if (fresh) return;
  if (!options.force && catalogLoadPromise) return catalogLoadPromise;
  catalogLoadPromise = Promise.all([
    api("/databases?stats_mode=summary"),
    api("/database-types"),
  ]).then(([data, typeData]) => {
    state.databases = data.items || [];
    state.databaseTypes = typeData.items || [];
    catalogLoadedAt = Date.now();
  }).finally(() => {
    catalogLoadPromise = null;
  });
  return catalogLoadPromise;
}

function databaseTypeMeta(databaseType) {
  return state.databaseTypes.find((item) => item.id === databaseType) || null;
}

function databaseTypeIconPath(databaseType) {
  const fallback = databaseUiRegistry.get(databaseType)?.icon || "";
  return databaseTypeMeta(databaseType)?.icon || fallback;
}

function databaseTypeIconMarkup(databaseType, iconPath = "") {
  const resolvedIconPath = escapeHtml(String(iconPath || databaseTypeIconPath(databaseType)));
  return `<span class="library-logo" aria-hidden="true" style="--library-icon-mask:url('${resolvedIconPath}')"></span>`;
}

function libraryAccessKeyIconMarkup() {
  return `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M701.248 0A323.2 323.2 0 0 0 393.6 420.8L6.528 807.872A22.4 22.4 0 0 0 0 823.68v178.048c0 12.288 9.984 22.272 22.272 22.272h155.84c5.888 0 11.52-2.368 15.744-6.528l44.8-44.8a22.336 22.336 0 0 0 6.528-15.744v-70.848h70.848a22.272 22.272 0 0 0 22.272-22.272v-62.4h62.4c5.888 0 11.52-2.368 15.744-6.528l170.24-170.24A323.2 323.2 0 0 0 1024 322.752 323.2 323.2 0 0 0 701.248 0z m-225.28 496.96l-379.072 379.072a11.136 11.136 0 1 1-15.68-15.744L460.288 481.28a11.136 11.136 0 0 1 15.744 15.744z m385.472-35.328l-15.744 15.744L546.56 178.304l15.744-15.744a210.112 210.112 0 0 1 149.504-61.888c56.512 0 109.632 21.952 149.568 61.888a211.712 211.712 0 0 1 0 299.072z"/></svg>`;
}

async function startDatabaseIndexRebuild(databaseId, providerId = "", reason = "manual", options = {}) {
  const databaseType = options.databaseType || databaseTypeForId(databaseId);
  const refKey = databaseRefKey(databaseType, databaseId);
  return asyncGuard.run(`database:${refKey}:index-rebuild`, async () => {
    const payload = { reason };
    if (providerId) {
      payload.provider_id = providerId;
    }
    const tempJobId = addOptimisticTask({
      kind: "index_rebuild",
      databaseId,
      databaseType,
      markLibraryConflict: Boolean(options.markLibraryConflict),
    });
    try {
      const result = await api(databaseApiPath(databaseType, databaseId, "/indexes/rebuild"), {
        method: "POST",
        body: JSON.stringify(payload),
      });
      trackQueuedJob(result, {
        kind: "index_rebuild",
        databaseId,
        databaseType,
        tempId: tempJobId,
      });
      if (!result.job_id && options.markLibraryConflict) {
        clearDatabaseIndexConflict(databaseId, databaseType);
      }
      toast(t("indexRebuildQueued", { library: databaseId }));
      return result.job_id;
    } catch (error) {
      removeOptimisticTask(tempJobId);
      if (options.markLibraryConflict) {
        clearDatabaseIndexConflict(databaseId, databaseType);
      }
      throw error;
    }
  }, {
    button: options.button,
    busyText: t("taskSubmitting"),
    toast,
    startMessage: options.startMessage,
  });
}

function latestProvider(providerId) {
  return state.providers.find((provider) => provider.id === providerId) || null;
}

function providerMaxContextTokens(provider) {
  const value = Number(provider?.max_context_tokens || 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

function libraryEditNeedsRebuild(library, provider) {
  if (!library || !provider) {
    return "";
  }
  const indexes = library.indexes || {};
  const manifest = indexes.manifest || {};
  const currentProvider = library.provider || {};
  if (provider.id && provider.id !== library.provider_id) {
    return "library_edit_provider_changed";
  }
  const providerUsage = (provider.used_by || []).find(
    (item) => item.usage_kind !== "rerank"
      && (
        item.memory_store_id
        || item.knowledge_base_id
        || item.database_id
        || item.library_id
      ) === library.id,
  );
  const providerRevisionNeedsRebuild = providerUsage
    ? Boolean(providerUsage.needs_rebuild)
    : Number(provider.revision || 0) !== Number(library.provider_revision || 0);
  if (providerRevisionNeedsRebuild) {
    return "library_edit_provider_revision_changed";
  }
  if (provider.model && currentProvider.model && provider.model !== currentProvider.model) {
    return "library_edit_provider_model_changed";
  }
  if (provider.model && manifest.configured_model && provider.model !== manifest.configured_model) {
    return "library_edit_index_model_mismatch";
  }
  if (!indexes.generation) {
    return "library_edit_missing_index";
  }
  return "";
}

function markDatabaseIndexConflict(databaseId, databaseType = DEFAULT_DATABASE_TYPE) {
  if (!databaseId) return;
  state.optimisticIndexConflicts.add(databaseRefKey(databaseType, databaseId));
  if (state.page === "libraries") {
    renderDatabaseCards();
  }
  refreshDatabaseContext();
}

function clearDatabaseIndexConflict(databaseId, databaseType = DEFAULT_DATABASE_TYPE) {
  const refKey = databaseRefKey(databaseType, databaseId);
  if (!databaseId || !state.optimisticIndexConflicts.has(refKey)) return;
  state.optimisticIndexConflicts.delete(refKey);
  if (state.page === "libraries") {
    renderDatabaseCards();
  }
  refreshDatabaseContext();
}

function libraryIndexState(library) {
  const stats = library?.stats || {};
  const indexes = library?.indexes || {};
  if (!indexes.generation) {
    return "pending";
  }
  if (state.optimisticIndexConflicts.has(databaseRefKey(
    library.database_type || DEFAULT_DATABASE_TYPE,
    library.id,
  ))) {
    return "conflict";
  }
  const provider = latestProvider(library?.provider_id) || library?.provider || null;
  const needsRebuild = provider ? Boolean(libraryEditNeedsRebuild(library, provider)) : false;
  const manifest = indexes.manifest || {};
  const activeMemories = Number(stats.active_memories ?? stats.total_memories ?? 0);
  const graphGranularity = String(manifest.graph_vector_granularity || "entry");
  const expectedGraphVectors = graphGranularity === "memory"
    ? Number(manifest.graph_source_memory_count ?? manifest.graph_vector_count ?? 0)
    : Number(stats.graph_entries || 0);
  const countHealthy = Number(indexes.document_vectors || 0) === activeMemories
    && Number(indexes.graph_vectors || 0) === expectedGraphVectors;
  if (needsRebuild || !countHealthy) {
    return "conflict";
  }
  return "healthy";
}

function knowledgeLibraryIndexState(library) {
  const stats = library?.stats || {};
  const indexes = library?.indexes || {};
  if (!indexes.generation) {
    return "pending";
  }
  if (library.status !== "ready") {
    return "conflict";
  }
  if (state.optimisticIndexConflicts.has(databaseRefKey(
    library.database_type || DEFAULT_DATABASE_TYPE,
    library.id,
  ))) {
    return "conflict";
  }
  const provider = latestProvider(library?.provider_id) || library?.provider || null;
  const needsRebuild = provider ? Boolean(libraryEditNeedsRebuild(library, provider)) : false;
  const countHealthy = Number(indexes.vector_count || 0) === Number(stats.chunks || 0);
  return needsRebuild || !countHealthy ? "conflict" : "healthy";
}

function indexHealthMarkup(indexState) {
  if (indexState === "healthy") {
    return `<span class="pill success">${escapeHtml(t("indexHealthy"))}</span>`;
  }
  if (indexState === "pending") {
    return `<span class="pill warning">${escapeHtml(t("indexPending"))}</span>`;
  }
  return `<span class="pill danger">${escapeHtml(t("indexConflict"))}</span>`;
}

function connectedAdaptersMarkup(library, adapterConnections) {
  const adapterButton = (item, extraClass = "") => {
    const label = item.adapter_id || item.adapter_type || "Adapter";
    const title = [
      item.adapter_type ? `type=${item.adapter_type}` : "",
      item.instance_id ? `instance=${String(item.instance_id).slice(0, 12)}` : "",
    ].filter(Boolean).join(" · ");
    return `<button type="button" class="${["used-lib-jump", "disconnect-adapter", extraClass].filter(Boolean).join(" ")}" data-database-id="${escapeHtml(library.id)}" data-adapter-id="${escapeHtml(item.adapter_id || "")}" data-instance-id="${escapeHtml(item.instance_id || "")}" title="${escapeHtml(title)}">${escapeHtml(label)}</button>`;
  };
  if (adapterConnections.length === 0) return escapeHtml(t("none"));
  if (adapterConnections.length === 1) return adapterButton(adapterConnections[0], "used-lib-plain");
  return `<div class="used-lib-row">${adapterButton(adapterConnections[0], "used-lib-first-btn")}<details class="used-lib-dd"><summary><span class="used-lib-count">＋${adapterConnections.length - 1}</span><span class="used-lib-caret" aria-hidden="true">▾</span></summary><ul class="used-lib-list used-lib-list-embedding">${adapterConnections.slice(1).map((item) => `<li>${adapterButton(item)}</li>`).join("")}</ul></details></div>`;
}

function libraryIsEmpty(library) {
  const stats = library?.stats || {};
  return [
    stats.total_memories,
    stats.graph_nodes,
    stats.graph_edges,
    stats.graph_entries,
    stats.atom_count,
    stats.conversation_counts?.sessions,
  ].every((value) => Number(value || 0) === 0);
}

$("integrity-check").onclick = async () => {
  try {
    const data = await selectedDatabaseApi("/integrity");
    $("integrity-result").textContent = JSON.stringify(data, null, 2);
    $("integrity-result").classList.remove("hidden");
    toast(t("integrityReady"));
  } catch (error) {
    toast(error.message, true);
  }
};

function navigate(page) {
  return activatePage(page);
}

$("library-context").onclick = () => navigate("libraries");
$("sidebar-current-library").onclick = () => navigate("libraries");
$("go-providers").onclick = () => navigate("providers");

async function ensureDatabaseSelection() {
  if (!state.databases.length) {
    await fetchCatalog();
  }
  let library = selectedDatabase();
  if (library && library.database_category !== state.databaseCategory) {
    applySelection(selectionForCategory(state.databaseCategory));
    library = selectedDatabase();
  }
  if (!library) {
    const route = String(state.route || "");
    const keepTypeContext = route.startsWith(`database-type:${state.selectedDatabaseType}:`)
      && databaseUiRegistry.availableTypePages(state.selectedDatabaseType).length;
    if (keepTypeContext) {
      state.selectedDatabaseId = "";
    } else {
      applySelection(selectionForCategory(state.databaseCategory));
      library = selectedDatabase();
    }
  }
  rememberCategorySelection(library);
  refreshDatabaseContext();
}

function renderDatabaseCards() {
  updateDatabaseCategoryTabs();
  const visibleLibraries = state.databases.filter(
    (library) => library.database_category === state.databaseCategory,
  );
  $("library-cards").innerHTML =
    visibleLibraries
      .map((library) => {
        const databaseType = library.database_type || DEFAULT_DATABASE_TYPE;
        const refKey = databaseRefKey(databaseType, library.id);
        const capabilities = new Set(library.capabilities || []);
        const stats = library.stats || {};
        const provider = library.provider || {};
        const indexes = library.indexes || {};
        const isSelected = library.id === state.selectedDatabaseId
          && databaseType === state.selectedDatabaseType;
        const isEmpty = libraryIsEmpty(library);
        const isExpanded = state.expandedDatabaseRefs.has(refKey);
        const uiDriver = databaseUiRegistry.get(databaseType);
        const adapterConnections = [...(library.adapter_connections || [])].sort((left, right) => {
          return String(left.adapter_id || "").localeCompare(
            String(right.adapter_id || ""),
            "zh-Hans-CN",
          );
        });
        const hasAdapters = adapterConnections.length > 0;
        if (!uiDriver) {
          return `<article class="management-card library-card" data-id="${escapeHtml(library.id)}" data-database-type="${escapeHtml(databaseType)}">
            <div class="library-card-primary">
              <header><div class="provider-card-head library-card-head">${databaseTypeIconMarkup(databaseType, library.type_metadata?.icon)}<div><h3>${escapeHtml(library.name || library.id)}</h3><span class="subtle">${escapeHtml(library.id)}</span></div></div><span class="pill warning">${escapeHtml(t("unsupportedWebUiType"))}</span></header>
              <p class="card-description">${escapeHtml(library.description || t("noDescription"))}</p>
              <dl class="provider-meta"><dt>${escapeHtml(t("databaseType"))}</dt><dd><code>${escapeHtml(databaseType)}</code></dd></dl>
            </div>
          </article>`;
        }
        if (uiDriver.cardLayout === "knowledge") {
          const indexHealth = indexHealthMarkup(knowledgeLibraryIndexState(library));
          const connectedAdaptersHtml = capabilities.has("adapter_access")
            ? connectedAdaptersMarkup(library, adapterConnections)
            : escapeHtml(t("none"));
          const modelDimension = indexes.dimensions || provider.dimensions || t("autoDetect");
          const expandTitle = isExpanded ? t("collapseLibraryCard") : t("expandLibraryCard");
          return `<article class="management-card library-card text-media-library-card ${isSelected ? "active-card" : ""} ${isExpanded ? "library-card-expanded" : "library-card-collapsed"}" data-id="${escapeHtml(library.id)}" data-database-type="${escapeHtml(databaseType)}" role="button" tabindex="0" aria-pressed="${isSelected ? "true" : "false"}">
            <div class="library-card-primary">
              <header><div class="provider-card-head library-card-head">${databaseTypeIconMarkup(databaseType, library.type_metadata?.icon)}<div><h3>${escapeHtml(library.name)}</h3><span class="subtle">${escapeHtml(library.id)}</span></div></div><span class="pill success selected-library-badge ${isSelected ? "" : "hidden"}">${escapeHtml(t("selectedBadge"))}</span></header>
              <p class="card-description">${escapeHtml(library.description || t("noDescription"))}</p>
              <div class="card-metrics text-media-card-metrics">
                <div class="mini-metric"><strong>${Number(stats.documents || 0)}</strong><span>${escapeHtml(t("documents"))}</span></div>
                <div class="mini-metric"><strong>${Number(stats.entries || 0)}</strong><span>${escapeHtml(t("knowledgeEntries"))}</span></div>
                <div class="mini-metric"><strong>${Number(stats.chunks || 0)}</strong><span>${escapeHtml(t("chunks"))}</span></div>
                <div class="mini-metric"><strong>${Number(stats.images || 0)}</strong><span>${escapeHtml(t("images"))}</span></div>
              </div>
              <dl class="provider-meta library-card-summary-meta">
                <dt>${escapeHtml(t("knowledgeLibraryType"))}</dt><dd>${escapeHtml(library.type_metadata?.display_name || t("textMediaTypeName"))}</dd>
                <dt>${escapeHtml(t("providerLabel"))}</dt><dd>${escapeHtml(provider.display_name || provider.id || "—")}</dd>
                <dt>${escapeHtml(t("modelDimension"))}</dt><dd>${escapeHtml(provider.model || provider.id || "—")} / ${escapeHtml(modelDimension)}</dd>
              </dl>
              <div class="library-card-extra">
                <div class="library-card-extra-inner">
                  <dl class="provider-meta">
                    <dt>${escapeHtml(t("generationLabel"))}</dt><dd>${escapeHtml(indexes.generation || t("indexPending"))}</dd>
                    <dt>${escapeHtml(t("indexStatus"))}</dt><dd>${indexHealth}</dd>
                    <dt>${escapeHtml(t("rerankProvider"))}</dt><dd>${escapeHtml(library.rerank_provider?.display_name || library.rerank_provider_id || t("none"))}${library.rerank_binding?.needs_recalibration ? ` · ${escapeHtml(t("rerankNeedsRecalibration"))}` : ""}</dd>
                    <dt>${escapeHtml(t("connectedAdapters"))}</dt><dd>${connectedAdaptersHtml}</dd>
                  </dl>
                </div>
              </div>
            </div>
            <div class="card-actions library-card-actions">
              <button class="primary manage-text-media" data-id="${escapeHtml(library.id)}">${escapeHtml(t("manageKnowledge"))}</button>
              <button class="ghost edit-text-media" data-id="${escapeHtml(library.id)}">${escapeHtml(t("edit"))}</button>
              ${capabilities.has("copy") ? `<button class="ghost copy-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("copyLibrary"))}</button>` : ""}
              ${capabilities.has("index_rebuild") ? `<button class="ghost rebuild-library-index" data-id="${escapeHtml(library.id)}" data-provider="${escapeHtml(provider.id || library.provider_id || "")}">${escapeHtml(t("rebuildIndex"))}</button>` : ""}
              ${capabilities.has("backup") ? `<button class="ghost backup-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("backupNow"))}</button>` : ""}
              ${hasAdapters ? `<button class="ghost danger adapter-delete-blocked" data-id="${escapeHtml(library.id)}" title="${escapeHtml(t("adapterDeleteBlocked"))}">${escapeHtml(t("delete"))}</button>` : `<button class="ghost danger delete-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("delete"))}</button>`}
            </div>
            ${capabilities.has("access_key") ? `<button class="ghost key-library key-library-fab" data-id="${escapeHtml(library.id)}" title="${escapeHtml(t("databaseAccessKey"))}" aria-label="${escapeHtml(t("databaseAccessKey"))}">${libraryAccessKeyIconMarkup()}</button>` : ""}
            <button class="library-expand-toggle ${isExpanded ? "expanded" : ""}" data-id="${escapeHtml(library.id)}" title="${escapeHtml(expandTitle)}" aria-label="${escapeHtml(expandTitle)}" aria-expanded="${isExpanded ? "true" : "false"}">
              ${LIBRARY_EXPAND_ICON}
            </button>
          </article>`;
        }
        const adapterButton = (item, extraClass = "") => {
          const label = item.adapter_id || item.adapter_type || "Adapter";
          const title = [
            item.adapter_type ? `type=${item.adapter_type}` : "",
            item.instance_id ? `instance=${String(item.instance_id).slice(0, 12)}` : "",
          ].filter(Boolean).join(" · ");
          return `<button type="button" class="${["used-lib-jump", "disconnect-adapter", extraClass].filter(Boolean).join(" ")}" data-database-id="${escapeHtml(library.id)}" data-adapter-id="${escapeHtml(item.adapter_id || "")}" data-instance-id="${escapeHtml(item.instance_id || "")}" title="${escapeHtml(title)}">${escapeHtml(label)}</button>`;
        };
        const connectedAdaptersHtml = adapterConnections.length === 0
          ? escapeHtml(t("none"))
          : adapterConnections.length === 1
            ? adapterButton(adapterConnections[0], "used-lib-plain")
            : `<div class="used-lib-row">${adapterButton(adapterConnections[0], "used-lib-first-btn")}<details class="used-lib-dd"><summary><span class="used-lib-count">＋${adapterConnections.length - 1}</span><span class="used-lib-caret" aria-hidden="true">▾</span></summary><ul class="used-lib-list used-lib-list-embedding">${adapterConnections.slice(1).map((item) => `<li>${adapterButton(item)}</li>`).join("")}</ul></details></div>`;
        const activeSessionCount = Number(
          stats.session_count ?? Object.keys(stats.sessions || {}).length,
        );
        const conversationCounts = stats.conversation_counts || {};
        const conversationBufferText = [
          `${Number(conversationCounts.sessions || 0)} ${t("shortSessionsUnit")}`,
          `${Number(conversationCounts.messages || 0)} ${t("messagesUnit")}`,
          `${Number(conversationCounts.pending_messages || 0)} ${t("pendingMessagesUnit")}`,
        ].join(" / ");
        const indexState = libraryIndexState(library);
        const indexHealth = indexState === "healthy"
          ? `<span class="pill success">${escapeHtml(t("indexHealthy"))}</span>`
          : indexState === "pending"
            ? `<span class="pill warning">${escapeHtml(t("indexPending"))}</span>`
            : `<span class="pill danger">${escapeHtml(t("indexConflict"))}</span>`;
        const expandTitle = isExpanded ? t("collapseLibraryCard") : t("expandLibraryCard");
        return `<article class="management-card library-card ${isSelected ? "active-card" : ""} ${isExpanded ? "library-card-expanded" : "library-card-collapsed"}" data-id="${escapeHtml(library.id)}" data-database-type="${escapeHtml(databaseType)}" role="button" tabindex="0" aria-pressed="${isSelected ? "true" : "false"}">
          <div class="library-card-primary">
            <header>
              <div class="provider-card-head library-card-head">
                ${databaseTypeIconMarkup(databaseType, library.type_metadata?.icon)}
                <div>
                <h3>${escapeHtml(library.name)} ${library.is_default ? `<span class="pill success">${escapeHtml(t("defaultBadge"))}</span>` : ""}</h3>
                <span class="subtle">${escapeHtml(library.id)}</span>
                </div>
              </div>
              <span class="pill success selected-library-badge ${isSelected ? "" : "hidden"}">${escapeHtml(t("selectedBadge"))}</span>
            </header>
            <p class="card-description">${escapeHtml(library.description || t("noDescription"))}</p>
            <div class="card-metrics">
              <div class="mini-metric"><strong>${stats.total_memories || 0}</strong><span>${escapeHtml(t("statsMemories"))}</span></div>
              <div class="mini-metric"><strong>${stats.graph_nodes || 0}</strong><span>${escapeHtml(t("statsNodes"))}</span></div>
              <div class="mini-metric"><strong>${stats.graph_edges || 0}</strong><span>${escapeHtml(t("statsRelations"))}</span></div>
              <div class="mini-metric"><strong>${stats.graph_entries || 0}</strong><span>${escapeHtml(t("statsGraphEntries"))}</span></div>
              <div class="mini-metric"><strong>${stats.atom_count || 0}</strong><span>${escapeHtml(t("statsAtoms"))}</span></div>
              <div class="mini-metric"><strong>${activeSessionCount}</strong><span>${escapeHtml(t("statsSessions"))}</span></div>
            </div>
            <dl class="provider-meta library-card-summary-meta">
              <dt>${escapeHtml(t("memoryLibraryType"))}</dt><dd>${escapeHtml(library.type_metadata?.display_name || "LivingMemory v8")}</dd>
              <dt>${escapeHtml(t("providerLabel"))}</dt><dd>${escapeHtml(provider.display_name || provider.id || "—")}</dd>
              <dt>${escapeHtml(t("modelDimension"))}</dt><dd>${escapeHtml(provider.model || "—")} / ${provider.dimensions || indexes.manifest?.dimension || t("autoDetect")}</dd>
            </dl>
            <div class="library-card-extra">
              <div class="library-card-extra-inner">
                <dl class="provider-meta">
                  <dt>${escapeHtml(t("generationLabel"))}</dt><dd>${escapeHtml(indexes.generation || t("indexPending"))}</dd>
                  <dt>${escapeHtml(t("indexStatus"))}</dt><dd>${indexHealth}</dd>
                  <dt>${escapeHtml(t("rerankProvider"))}</dt><dd>${escapeHtml(library.rerank_provider?.display_name || library.rerank_provider_id || t("none"))}${library.rerank_binding?.needs_recalibration ? ` · ${escapeHtml(t("rerankNeedsRecalibration"))}` : ""}</dd>
                  <dt>${escapeHtml(t("defaultPersona"))}</dt><dd>${escapeHtml(library.default_persona_id || t("noLimit"))}</dd>
                  <dt>${escapeHtml(t("connectedAdapters"))}</dt><dd>${connectedAdaptersHtml}</dd>
                  <dt>${escapeHtml(t("conversationBuffer"))}</dt><dd>${escapeHtml(conversationBufferText)}</dd>
                </dl>
              </div>
            </div>
          </div>
          <div class="card-actions library-card-actions">
            ${capabilities.has("recall") ? `<button class="primary enter-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("enterLibrary"))}</button>` : ""}
            <button class="ghost edit-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("edit"))}</button>
            ${capabilities.has("copy") ? `<button class="ghost copy-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("copyLibrary"))}</button>` : ""}
            ${isEmpty && capabilities.has("livingmemory_import") ? `<button class="ghost import-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("importMemory"))}</button>` : ""}
            ${capabilities.has("index_rebuild") ? `<button class="ghost rebuild-library-index" data-id="${escapeHtml(library.id)}" data-provider="${escapeHtml(provider.id || library.provider_id || "")}">${escapeHtml(t("rebuildIndex"))}</button>` : ""}
            ${capabilities.has("backup") ? `<button class="ghost backup-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("backupNow"))}</button>` : ""}
            ${library.is_default ? "" : `<button class="ghost default-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("setDefault"))}</button>${hasAdapters ? `<button class="ghost danger adapter-delete-blocked" data-id="${escapeHtml(library.id)}" title="${escapeHtml(t("adapterDeleteBlocked"))}">${escapeHtml(t("delete"))}</button>` : `<button class="ghost danger delete-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("delete"))}</button>`}`}
          </div>
          ${capabilities.has("adapter_access") ? `<button class="ghost key-library key-library-fab" data-id="${escapeHtml(library.id)}" title="${escapeHtml(t("databaseAccessKey"))}" aria-label="${escapeHtml(t("databaseAccessKey"))}">${libraryAccessKeyIconMarkup()}</button>` : ""}
          <button class="library-expand-toggle ${isExpanded ? "expanded" : ""}" data-id="${escapeHtml(library.id)}" title="${escapeHtml(expandTitle)}" aria-label="${escapeHtml(expandTitle)}" aria-expanded="${isExpanded ? "true" : "false"}">
            ${LIBRARY_EXPAND_ICON}
          </button>
        </article>`;
      })
      .join("") || `<div class="database-empty-state"><strong>${escapeHtml(
        state.databaseCategory === "knowledge" ? t("noKnowledgeTypes") : t("noMemoryLibraries")
      )}</strong><span>${escapeHtml(
        state.databaseCategory === "knowledge" ? t("knowledgeTypesComing") : t("createMemoryLibraryHint")
      )}</span></div>`;
  bindLibraryCardActions();
  bindUsedListDetails($("library-cards"));
  refreshDatabaseContext();
}

async function loadDatabases(render = true, options = {}) {
  try {
    await fetchCatalog(options);
    await ensureDatabaseSelection();
    updateDatabaseCategoryTabs();
    if (!render) {
      refreshDatabaseContext();
      return;
    }
    renderDatabaseCards();
  } catch (error) {
    if (error?.name === "AbortError") return;
    toast(error.message, true);
  }
}

async function refreshCatalogAfterMutationConflict(error) {
  if (Number(error?.status || 0) !== 409) return false;
  await loadDatabases(state.page === "libraries", { force: true });
  return true;
}

function bindLibraryCardActions() {
  const buttonRef = (button) => ({
    id: button.dataset.id || button.closest(".library-card")?.dataset.id || "",
    databaseType: button.closest(".library-card")?.dataset.databaseType || DEFAULT_DATABASE_TYPE,
  });
  const libraryCards = $("library-cards");
  libraryCards.onclick = (event) => {
    if (event.target.closest("button,a,input,select,textarea,label")) return;
    const card = event.target.closest(".library-card");
    if (!card) return;
    selectDatabase(card.dataset.id, { databaseType: card.dataset.databaseType });
  };
  libraryCards.onkeydown = (event) => {
    if (event.target.closest("button,a,input,select,textarea,label")) return;
    const card = event.target.closest(".library-card");
    if (!card) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    selectDatabase(card.dataset.id, { databaseType: card.dataset.databaseType });
  };
  document.querySelectorAll(".enter-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      selectDatabase(ref.id, { databaseType: ref.databaseType });
      const driver = databaseUiRegistry.get(ref.databaseType);
      if (driver) navigate(databasePageRoute(ref.databaseType, driver.defaultPage));
    };
  });
  document.querySelectorAll(".manage-text-media").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      selectDatabase(ref.id, { databaseType: ref.databaseType });
      await activatePage(databasePageRoute(ref.databaseType, "content"));
    };
  });
  document.querySelectorAll(".edit-text-media").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      selectDatabase(ref.id, { databaseType: ref.databaseType });
      databaseUiRegistry.get(ref.databaseType)?.actions?.openEdit(
        databaseForRef(ref.databaseType, ref.id),
      );
    };
  });
  document.querySelectorAll(".edit-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      await loadProviders(false);
      const ref = buttonRef(button);
      const driver = databaseUiRegistry.get(ref.databaseType);
      if (driver?.settingsPage) {
        selectDatabase(ref.id, { databaseType: ref.databaseType });
        await activatePage(databasePageRoute(ref.databaseType, driver.settingsPage));
        return;
      }
      openLibraryModal(databaseForRef(ref.databaseType, ref.id));
    };
  });
  document.querySelectorAll(".copy-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      await asyncGuard.run(`database:${databaseRefKey(ref.databaseType, ref.id)}:copy`, async () => {
        const tempJobId = addOptimisticTask({
          kind: "library_copy",
          databaseId: ref.id,
          databaseType: ref.databaseType,
        });
        try {
          const result = await api(databaseApiPath(ref.databaseType, ref.id, "/copy"), { method: "POST" });
          toast(t("libraryCopyQueued", { id: button.dataset.id }));
          trackQueuedJob(result, {
            kind: "library_copy",
            databaseId: ref.id,
            databaseType: ref.databaseType,
            tempId: tempJobId,
          });
        } catch (error) {
          removeOptimisticTask(tempJobId);
          toast(error.message, true);
        }
      }, {
        button,
        busyText: t("taskSubmitting"),
        toast,
        startMessage: t("taskSubmitting"),
      });
    };
  });
  document.querySelectorAll(".import-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      openImportModal(ref.id, ref.databaseType);
    };
  });
  document.querySelectorAll(".rebuild-library-index").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      if (!(await confirmDialog({
        title: t("confirmRebuildTitle"),
        message: t("confirmRebuildLibraryIndex", { library: button.dataset.id }),
        confirmText: t("rebuildIndex"),
      }))) return;
      try {
        const ref = buttonRef(button);
        await startDatabaseIndexRebuild(
          ref.id,
          button.dataset.provider || "",
          "library_card_manual_rebuild",
          {
            button,
            startMessage: t("taskSubmitting"),
            databaseType: ref.databaseType,
          },
        );
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".backup-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      await asyncGuard.run(`database:${databaseRefKey(ref.databaseType, ref.id)}:backup`, async () => {
      try {
        const tempJobId = addOptimisticTask({
          kind: "library_backup",
          databaseId: ref.id,
          databaseType: ref.databaseType,
        });
        try {
          const result = await api(databaseApiPath(ref.databaseType, ref.id, "/backup"), { method: "POST" });
          trackQueuedJob(result, {
            kind: "library_backup",
            databaseId: ref.id,
            databaseType: ref.databaseType,
            tempId: tempJobId,
          });
          toast(t("backupQueued", { library: ref.id }));
        } catch (error) {
          removeOptimisticTask(tempJobId);
          throw error;
        }
      } catch (error) {
        toast(error.message, true);
      }
      }, {
        button,
        busyText: t("loading"),
      });
    };
  });
  document.querySelectorAll(".key-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      const ref = buttonRef(button);
      openLibraryPskModal(ref.id, ref.databaseType);
    };
  });
  document.querySelectorAll(".library-expand-toggle").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      const { id: databaseId, databaseType } = buttonRef(button);
      if (!databaseId) return;
      const refKey = databaseRefKey(databaseType, databaseId);
      const expanded = !state.expandedDatabaseRefs.has(refKey);
      if (expanded) {
        state.expandedDatabaseRefs.add(refKey);
      } else {
        state.expandedDatabaseRefs.delete(refKey);
      }
      applyLibraryCardExpandedState(
        button.closest(".library-card"),
        expanded,
      );
    };
  });
  document.querySelectorAll(".default-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      try {
        const ref = buttonRef(button);
        await api(databaseApiPath(ref.databaseType, ref.id, "/set-default"), { method: "POST" });
        await loadDatabases();
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".delete-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      if (!(await confirmDialog({
        title: t("confirmTitle"),
        message: t("confirmDeleteLibrary", { id: button.dataset.id }),
        confirmText: t("delete"),
        danger: true,
      }))) return;
      try {
        const ref = buttonRef(button);
        await api(databaseApiPath(ref.databaseType, ref.id), { method: "DELETE" });
        await loadDatabases();
      } catch (error) {
        await refreshCatalogAfterMutationConflict(error);
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".adapter-delete-blocked").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      toast(t("adapterDeleteBlocked"), true);
    };
  });
  document.querySelectorAll(".disconnect-adapter").forEach((button) => {
    button.onclick = async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const databaseId = button.dataset.databaseId || "";
      const databaseType = button.closest(".library-card")?.dataset.databaseType || DEFAULT_DATABASE_TYPE;
      const adapterId = button.dataset.adapterId || "";
      const instanceId = button.dataset.instanceId || "";
      if (!databaseId || !adapterId || !instanceId) return;
      if (!(await confirmDialog({
        title: t("forceDisconnectAdapterTitle"),
        message: t("confirmForceDisconnectAdapter", { id: adapterId }),
        confirmText: t("forceDisconnectAdapterAction"),
        danger: true,
      }))) return;
      try {
        await api(
          databaseApiPath(databaseType, databaseId, `/adapters/${encodeURIComponent(adapterId)}/disconnect`),
          {
            method: "POST",
            body: JSON.stringify({ instance_id: instanceId }),
          },
        );
        toast(t("adapterDisconnected", { id: adapterId }));
        await loadDatabases();
      } catch (error) {
        toast(error.message, true);
        await loadDatabases();
      }
    };
  });
}

function bindUsedListDetails(root = document) {
  root.querySelectorAll(".used-lib-dd").forEach((details) => {
    const summary = details.querySelector("summary");
    if (!summary) return;
    summary.setAttribute("aria-expanded", details.open ? "true" : "false");
    summary.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (details.classList.contains("used-lib-dd-animating")) return;
      const animationMs = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 320;
      details.classList.add("used-lib-dd-animating");
      if (details.open) {
        summary.setAttribute("aria-expanded", "false");
        details.classList.add("used-lib-dd-closing");
        window.setTimeout(() => {
          details.open = false;
          details.classList.remove("used-lib-dd-closing", "used-lib-dd-animating");
        }, animationMs);
        return;
      }
      details.open = true;
      summary.setAttribute("aria-expanded", "true");
      details.classList.add("used-lib-dd-opening");
      requestAnimationFrame(() => {
        details.classList.remove("used-lib-dd-opening");
        window.setTimeout(() => {
          details.classList.remove("used-lib-dd-animating");
        }, animationMs);
      });
    };
  });
}

function applyLibraryCardExpandedState(card, expanded) {
  if (!card) return;
  card.classList.toggle("library-card-expanded", expanded);
  card.classList.toggle("library-card-collapsed", !expanded);
  const toggle = card.querySelector(".library-expand-toggle");
  if (!toggle) return;
  toggle.classList.toggle("expanded", expanded);
  const title = expanded ? t("collapseLibraryCard") : t("expandLibraryCard");
  toggle.title = title;
  toggle.setAttribute("aria-label", title);
  toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function openLibraryPskModal(databaseId, databaseType = DEFAULT_DATABASE_TYPE) {
  const library = databaseForRef(databaseType, databaseId);
  const keyPrefix = library?.database_category === "knowledge" ? "pkb-" : "psk-";
  $("library-psk-id").value = databaseId || "";
  $("library-psk-form").dataset.databaseType = databaseType;
  $("library-psk-title").textContent = t("libraryPskTitle", { library: library?.name || databaseId });
  $("library-psk-hint").textContent = t("databaseKeyHint", { prefix: keyPrefix });
  $("library-psk-password").value = "";
  $("library-psk-url").textContent = state.settings?.recommended_adapter_url || "";
  $("library-psk-libid").textContent = databaseId || "";
  $("library-psk-value").textContent = "";
  $("library-psk-result").classList.add("hidden");
  $("library-psk-hint").classList.remove("hidden");
  $("library-psk-password-row").classList.remove("hidden");
  $("library-psk-submit").classList.remove("hidden");
  $("library-psk-cancel").textContent = t("cancel");
  $("library-psk-modal").classList.remove("hidden");
  setTimeout(() => $("library-psk-password")?.focus(), 0);
}

$("library-psk-form").onsubmit = async (event) => {
  event.preventDefault();
  await asyncGuard.run("library:psk", async () => {
    const databaseId = $("library-psk-id").value;
    const databaseType = event.currentTarget.dataset.databaseType || DEFAULT_DATABASE_TYPE;
    const password = $("library-psk-password").value;
    try {
      const result = await api(databaseApiPath(databaseType, databaseId, "/access-key"), {
        method: "POST",
        body: JSON.stringify({ password }),
      });
      $("library-psk-url").textContent =
        result.adapter_url || state.settings?.recommended_adapter_url || "";
      $("library-psk-value").textContent = result.access_key || "";
      $("library-psk-result").classList.remove("hidden");
      $("library-psk-hint").classList.add("hidden");
      $("library-psk-password-row").classList.add("hidden");
      $("library-psk-submit").classList.add("hidden");
      $("library-psk-cancel").textContent = t("close");
      toast(t("libraryPskReady"));
    } catch (error) {
      toast(error.message, true);
    }
  }, {
    form: event.currentTarget,
    button: event.submitter,
    busyText: t("loading"),
  });
};

$("library-psk-url-copy").onclick = async () => {
  const value = $("library-psk-url").textContent.trim();
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("library-psk-url"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
  toast(t("adapterUrlCopied"));
};

$("library-psk-copy").onclick = async () => {
  const value = $("library-psk-value").textContent.trim();
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("library-psk-value"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
  toast(t("libraryPskCopied"));
};

$("library-psk-libid-copy").onclick = async () => {
  const value = $("library-psk-libid").textContent.trim();
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("library-psk-libid"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
  toast(t("libraryIdCopied"));
};

document.querySelectorAll(".database-category-tab").forEach((button) => {
  const iconMap = {
    memory: "/static/icons/memory-library.svg",
    knowledge: "/static/icons/knowledge-library.svg",
  };
  let label = button.querySelector(".database-category-tab-label");
  if (!label) {
    const text = button.textContent.trim();
    const i18nKey = button.getAttribute("data-i18n") || "";
    button.textContent = "";
    if (i18nKey) {
      button.removeAttribute("data-i18n");
    }
    const icon = document.createElement("span");
    icon.className = "database-category-tab-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.style.setProperty(
      "--database-category-icon-mask",
      `url("${iconMap[button.dataset.databaseCategory] || iconMap.memory}")`,
    );
    label = document.createElement("span");
    label.className = "database-category-tab-label";
    if (i18nKey) {
      label.dataset.i18n = i18nKey;
    }
    label.textContent = text;
    const current = document.createElement("span");
    current.className = "database-category-tab-current hidden";
    button.append(icon, label, current);
  }
  if (!button.querySelector(".database-category-tab-current")) {
    const current = document.createElement("span");
    current.className = "database-category-tab-current hidden";
    button.append(current);
  }
  const active = button.dataset.databaseCategory === state.databaseCategory;
  button.classList.toggle("active", active);
  button.setAttribute("aria-selected", active ? "true" : "false");
  button.onclick = () => {
    setDatabaseCategory(button.dataset.databaseCategory || "memory");
    renderDatabaseCards();
    refreshDatabaseContext();
  };
});
window.addEventListener("prag-database-selection-changed", updateDatabaseCategoryTabs);
updateDatabaseCategoryTabs();

$("library-create").onclick = async () => {
  const types = state.databaseTypes.filter(
    (type) => type.category === state.databaseCategory,
  );
  $("database-type-cards").innerHTML = types.map((type) => `
    <button type="button" class="database-type-card" data-type="${escapeHtml(type.id)}">
      <span class="database-type-icon" data-icon="${escapeHtml(type.icon || "/static/icons/livingmemory-v8.svg")}" aria-hidden="true"></span>
      <span><b>${escapeHtml(type.display_name)}</b><small>${escapeHtml(type.description)}</small></span>
    </button>`).join("") || `<div class="database-empty-state"><strong>${escapeHtml(t("noDatabaseTypes"))}</strong></div>`;
  document.querySelectorAll(".database-type-icon").forEach((icon) => {
    const iconPath = String(icon.dataset.icon || "/static/icons/livingmemory-v8.svg").replace(/"/g, '\\"');
    icon.style.setProperty("--database-type-icon-mask", `url("${iconPath}")`);
  });
  document.querySelectorAll(".database-type-card").forEach((button) => {
    button.onclick = async () => {
      await loadProviders(false);
      closeOverlay("database-type-modal");
      const driver = databaseUiRegistry.get(button.dataset.type);
      state.selectedDatabaseType = button.dataset.type;
      if (driver?.actions?.openCreate) {
        await driver.actions.openCreate();
        return;
      }
      openLibraryModal(null, button.dataset.type);
    };
  });
  $("database-type-modal").classList.remove("hidden");
};

function boundedNumberField(id, fallback, min = null, max = null) {
  const raw = $(id)?.value;
  let value = Number(raw);
  if (!Number.isFinite(value)) {
    value = fallback;
  }
  if (min !== null) {
    value = Math.max(min, value);
  }
  if (max !== null) {
    value = Math.min(max, value);
  }
  return value;
}

function fillNumberField(id, value, fallback) {
  const element = $(id);
  if (element) {
    element.value = value ?? fallback;
  }
}

function fillBooleanField(id, value, fallback = false) {
  const element = $(id);
  if (element) {
    element.checked = Boolean(value ?? fallback);
  }
}

function fillLibrarySettingsFields(library = null) {
  const conversation = {
    ...DEFAULT_LIBRARY_CONVERSATION_SETTINGS,
    ...(library?.conversation_settings || {}),
  };
  const recall = {
    ...DEFAULT_LIBRARY_RECALL_SETTINGS,
    ...(library?.recall_settings || {}),
  };
  const maintenance = {
    ...DEFAULT_LIBRARY_MAINTENANCE_SETTINGS,
    ...(library?.maintenance_settings || {}),
  };
  fillNumberField("library-conversation-max-sessions", conversation.max_sessions, DEFAULT_LIBRARY_CONVERSATION_SETTINGS.max_sessions);
  fillNumberField("library-conversation-session-ttl", conversation.session_ttl, DEFAULT_LIBRARY_CONVERSATION_SETTINGS.session_ttl);
  fillNumberField("library-conversation-context-window-size", conversation.context_window_size, DEFAULT_LIBRARY_CONVERSATION_SETTINGS.context_window_size);
  fillNumberField("library-conversation-max-messages", conversation.max_messages_per_session, DEFAULT_LIBRARY_CONVERSATION_SETTINGS.max_messages_per_session);
  fillNumberField("library-conversation-cleanup-batch-size", conversation.cleanup_batch_size, DEFAULT_LIBRARY_CONVERSATION_SETTINGS.cleanup_batch_size);
  fillNumberField("library-importance-weight", recall.importance_weight, DEFAULT_LIBRARY_RECALL_SETTINGS.importance_weight);
  fillNumberField("library-min-importance-retrieval", recall.min_importance_for_retrieval, DEFAULT_LIBRARY_RECALL_SETTINGS.min_importance_for_retrieval);
  fillNumberField("library-min-similarity-retrieval", recall.min_similarity_for_retrieval, DEFAULT_LIBRARY_RECALL_SETTINGS.min_similarity_for_retrieval);
  fillNumberField("library-recent-memory-count", recall.recent_memory_count, DEFAULT_LIBRARY_RECALL_SETTINGS.recent_memory_count);
  fillNumberField("library-recent-memory-max-age", recall.recent_memory_max_age_hours, DEFAULT_LIBRARY_RECALL_SETTINGS.recent_memory_max_age_hours);
  const memoryTypeFilter = $("library-memory-type-filter");
  if (memoryTypeFilter) memoryTypeFilter.value = recall.memory_type_filter || DEFAULT_LIBRARY_RECALL_SETTINGS.memory_type_filter;
  fillBooleanField("library-search-cache-enabled", recall.search_cache_enabled, DEFAULT_LIBRARY_RECALL_SETTINGS.search_cache_enabled);
  fillNumberField("library-search-cache-ttl", recall.search_cache_ttl_seconds, DEFAULT_LIBRARY_RECALL_SETTINGS.search_cache_ttl_seconds);
  fillNumberField("library-search-cache-max-size", recall.search_cache_max_size, DEFAULT_LIBRARY_RECALL_SETTINGS.search_cache_max_size);
  fillBooleanField("library-persona-filtering", recall.use_persona_filtering, DEFAULT_LIBRARY_RECALL_SETTINGS.use_persona_filtering);
  fillBooleanField("library-session-filtering", recall.use_session_filtering, DEFAULT_LIBRARY_RECALL_SETTINGS.use_session_filtering);
  fillNumberField("library-rrf-k", recall.rrf_k, DEFAULT_LIBRARY_RECALL_SETTINGS.rrf_k);
  fillBooleanField("library-graph-enabled", recall.graph_memory_enabled, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_memory_enabled);
  fillNumberField("library-document-route-weight", recall.document_route_weight, DEFAULT_LIBRARY_RECALL_SETTINGS.document_route_weight);
  fillNumberField("library-graph-route-weight", recall.graph_route_weight, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_route_weight);
  fillNumberField("library-cross-route-bonus", recall.cross_route_bonus, DEFAULT_LIBRARY_RECALL_SETTINGS.cross_route_bonus);
  fillNumberField("library-graph-expansion-limit", recall.graph_expansion_limit, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_expansion_limit);
  fillNumberField("library-graph-expansion-hops", recall.graph_expansion_hops, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_expansion_hops);
  fillNumberField("library-graph-second-hop-weight", recall.graph_second_hop_weight, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_second_hop_weight);
  fillBooleanField("library-dynamic-route-weighting", recall.dynamic_route_weighting, DEFAULT_LIBRARY_RECALL_SETTINGS.dynamic_route_weighting);
  fillNumberField("library-graph-max-topics", recall.graph_max_topics, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_topics);
  fillNumberField("library-graph-max-participants", recall.graph_max_participants, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_participants);
  fillNumberField("library-graph-max-facts", recall.graph_max_facts, DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_facts);
  fillBooleanField("library-atom-enabled", maintenance.atom_enabled, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_enabled);
  fillNumberField("library-atom-maintenance-interval", maintenance.atom_maintenance_interval_hours, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_maintenance_interval_hours);
  fillNumberField("library-atom-forget-delay", maintenance.atom_forget_delay_days, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_forget_delay_days);
  fillNumberField("library-atom-purge-delay", maintenance.atom_purge_delay_days, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_purge_delay_days);
  fillNumberField("library-decay-rate", recall.decay_rate, DEFAULT_LIBRARY_RECALL_SETTINGS.decay_rate);
  fillNumberField("library-access-decay-window-days", recall.access_decay_window_days, DEFAULT_LIBRARY_RECALL_SETTINGS.access_decay_window_days);
  fillNumberField("library-access-decay-max-count", recall.access_decay_max_count, DEFAULT_LIBRARY_RECALL_SETTINGS.access_decay_max_count);
  fillNumberField("library-access-count-decay-multiplier", recall.access_count_decay_multiplier, DEFAULT_LIBRARY_RECALL_SETTINGS.access_count_decay_multiplier);
  fillBooleanField("library-backup-enabled", maintenance.backup_enabled, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.backup_enabled);
  fillNumberField("library-backup-keep-days", maintenance.backup_keep_days, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.backup_keep_days);
  fillBooleanField("library-auto-cleanup-enabled", maintenance.auto_cleanup_enabled, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.auto_cleanup_enabled);
  fillBooleanField("library-auto-archived-enabled", maintenance.auto_archived_enabled, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.auto_archived_enabled);
  fillNumberField("library-cleanup-days-threshold", maintenance.cleanup_days_threshold, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.cleanup_days_threshold);
  fillNumberField("library-cleanup-importance-threshold", maintenance.cleanup_importance_threshold, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.cleanup_importance_threshold);
  fillNumberField("library-protected-importance-threshold", maintenance.protected_importance_threshold, DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.protected_importance_threshold);
}

function libraryConversationFromForm() {
  return {
    max_sessions: Math.max(1, Math.round(boundedNumberField("library-conversation-max-sessions", DEFAULT_LIBRARY_CONVERSATION_SETTINGS.max_sessions, 1))),
    session_ttl: Math.max(60, Math.round(boundedNumberField("library-conversation-session-ttl", DEFAULT_LIBRARY_CONVERSATION_SETTINGS.session_ttl, 60))),
    context_window_size: Math.max(1, Math.round(boundedNumberField("library-conversation-context-window-size", DEFAULT_LIBRARY_CONVERSATION_SETTINGS.context_window_size, 1, 1000))),
    max_messages_per_session: Math.max(1, Math.round(boundedNumberField("library-conversation-max-messages", DEFAULT_LIBRARY_CONVERSATION_SETTINGS.max_messages_per_session, 1))),
    cleanup_batch_size: Math.max(1, Math.round(boundedNumberField("library-conversation-cleanup-batch-size", DEFAULT_LIBRARY_CONVERSATION_SETTINGS.cleanup_batch_size, 1))),
  };
}

function libraryRecallFromForm() {
  return {
    importance_weight: boundedNumberField("library-importance-weight", DEFAULT_LIBRARY_RECALL_SETTINGS.importance_weight, 0, 10),
    min_importance_for_retrieval: boundedNumberField("library-min-importance-retrieval", DEFAULT_LIBRARY_RECALL_SETTINGS.min_importance_for_retrieval, 0, 1),
    min_similarity_for_retrieval: boundedNumberField("library-min-similarity-retrieval", DEFAULT_LIBRARY_RECALL_SETTINGS.min_similarity_for_retrieval, 0, 1),
    recent_memory_count: Math.max(0, Math.round(boundedNumberField("library-recent-memory-count", DEFAULT_LIBRARY_RECALL_SETTINGS.recent_memory_count, 0, 100))),
    recent_memory_max_age_hours: Math.max(0, Math.round(boundedNumberField("library-recent-memory-max-age", DEFAULT_LIBRARY_RECALL_SETTINGS.recent_memory_max_age_hours, 0, 8760))),
    memory_type_filter: ["all", "event_only"].includes($("library-memory-type-filter")?.value)
      ? $("library-memory-type-filter").value
      : DEFAULT_LIBRARY_RECALL_SETTINGS.memory_type_filter,
    search_cache_enabled: Boolean($("library-search-cache-enabled")?.checked),
    search_cache_ttl_seconds: boundedNumberField("library-search-cache-ttl", DEFAULT_LIBRARY_RECALL_SETTINGS.search_cache_ttl_seconds, 0, 600),
    search_cache_max_size: Math.max(0, Math.round(boundedNumberField("library-search-cache-max-size", DEFAULT_LIBRARY_RECALL_SETTINGS.search_cache_max_size, 0, 10000))),
    rrf_k: Math.max(1, Math.round(boundedNumberField("library-rrf-k", DEFAULT_LIBRARY_RECALL_SETTINGS.rrf_k, 1, 1000))),
    graph_memory_enabled: Boolean($("library-graph-enabled")?.checked),
    document_route_weight: boundedNumberField("library-document-route-weight", DEFAULT_LIBRARY_RECALL_SETTINGS.document_route_weight, 0, 1),
    graph_route_weight: boundedNumberField("library-graph-route-weight", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_route_weight, 0, 1),
    cross_route_bonus: boundedNumberField("library-cross-route-bonus", DEFAULT_LIBRARY_RECALL_SETTINGS.cross_route_bonus, 0, 0.5),
    graph_expansion_limit: Math.max(1, Math.round(boundedNumberField("library-graph-expansion-limit", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_expansion_limit, 1, 200))),
    graph_expansion_hops: Math.max(1, Math.round(boundedNumberField("library-graph-expansion-hops", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_expansion_hops, 1, 2))),
    graph_second_hop_weight: boundedNumberField("library-graph-second-hop-weight", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_second_hop_weight, 0, 1),
    dynamic_route_weighting: Boolean($("library-dynamic-route-weighting")?.checked),
    graph_max_topics: Math.max(1, Math.round(boundedNumberField("library-graph-max-topics", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_topics, 1, 20))),
    graph_max_participants: Math.max(1, Math.round(boundedNumberField("library-graph-max-participants", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_participants, 1, 30))),
    graph_max_facts: Math.max(1, Math.round(boundedNumberField("library-graph-max-facts", DEFAULT_LIBRARY_RECALL_SETTINGS.graph_max_facts, 1, 30))),
    use_persona_filtering: Boolean($("library-persona-filtering")?.checked),
    use_session_filtering: Boolean($("library-session-filtering")?.checked),
    decay_rate: boundedNumberField("library-decay-rate", DEFAULT_LIBRARY_RECALL_SETTINGS.decay_rate, 0, 1),
    access_decay_window_days: boundedNumberField("library-access-decay-window-days", DEFAULT_LIBRARY_RECALL_SETTINGS.access_decay_window_days, 1),
    access_decay_max_count: Math.max(1, Math.round(boundedNumberField("library-access-decay-max-count", DEFAULT_LIBRARY_RECALL_SETTINGS.access_decay_max_count, 1))),
    access_count_decay_multiplier: boundedNumberField("library-access-count-decay-multiplier", DEFAULT_LIBRARY_RECALL_SETTINGS.access_count_decay_multiplier, 0, 1),
  };
}

function libraryMaintenanceFromForm() {
  return {
    atom_enabled: Boolean($("library-atom-enabled")?.checked),
    atom_maintenance_interval_hours: boundedNumberField("library-atom-maintenance-interval", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_maintenance_interval_hours, 1, 168),
    atom_forget_delay_days: boundedNumberField("library-atom-forget-delay", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_forget_delay_days, 1, 90),
    atom_purge_delay_days: boundedNumberField("library-atom-purge-delay", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.atom_purge_delay_days, 1, 365),
    backup_enabled: Boolean($("library-backup-enabled")?.checked),
    backup_keep_days: Math.max(1, Math.round(boundedNumberField("library-backup-keep-days", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.backup_keep_days, 1))),
    auto_cleanup_enabled: Boolean($("library-auto-cleanup-enabled")?.checked),
    auto_archived_enabled: Boolean($("library-auto-archived-enabled")?.checked),
    cleanup_days_threshold: Math.max(1, Math.round(boundedNumberField("library-cleanup-days-threshold", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.cleanup_days_threshold, 1))),
    cleanup_importance_threshold: boundedNumberField("library-cleanup-importance-threshold", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.cleanup_importance_threshold, 0, 1),
    protected_importance_threshold: boundedNumberField("library-protected-importance-threshold", DEFAULT_LIBRARY_MAINTENANCE_SETTINGS.protected_importance_threshold, 0, 1),
  };
}

function openLibraryModal(library = null, databaseType = DEFAULT_DATABASE_TYPE) {
  const hasAdapters = Boolean((library?.adapter_connections || []).length);
  const idLocked = hasAdapters;
  $("library-original-id").value = library?.id || "";
  $("library-database-type").value = library?.database_type || databaseType;
  $("library-id").value = library?.id || "";
  $("library-id").readOnly = idLocked;
  $("library-id").classList.toggle("readonly-lock", idLocked);
  $("library-id").setAttribute("aria-readonly", idLocked ? "true" : "false");
  $("library-id-readonly-note").textContent = t("adapterProtectedLibrary");
  $("library-id-readonly-note").classList.toggle("hidden", !idLocked);
  $("library-name").value = library?.name || "";
  $("library-description").value = library?.description || "";
  $("library-persona").value = library?.default_persona_id || "";
  fillLibrarySettingsFields(library);
  $("library-provider-field").classList.remove("hidden");
  fillProviderSelect($("library-provider"), library?.provider_id, "embedding");
  fillProviderSelect(
    $("library-rerank-provider"),
    library?.rerank_provider_id || library?.rerank_provider?.id || "",
    "rerank",
    { optional: true, emptyLabel: t("noRerankProvider") },
  );
  $("library-modal-title").textContent = library ? `编辑记忆库：${library.name}` : "新增记忆库";
  $("library-modal").classList.remove("hidden");
}

async function confirmSensitiveLibraryEdit(originalLibrary, payload) {
  if (!originalLibrary) {
    return true;
  }
  const changedFields = [];
  const notes = [];
  if (payload.id && payload.id !== originalLibrary.id) {
    changedFields.push(t("databaseId"));
    notes.push(t("libraryEditRenameNote"));
  }
  if (
    payload.provider_id
    && payload.provider_id !== originalLibrary.provider_id
  ) {
    changedFields.push(t("providerManagement"));
    notes.push(t("libraryEditProviderNote"));
    const currentProvider = latestProvider(originalLibrary.provider_id)
      || originalLibrary.provider
      || null;
    const nextProvider = latestProvider(payload.provider_id);
    const currentMaxContext = providerMaxContextTokens(currentProvider);
    const nextMaxContext = providerMaxContextTokens(nextProvider);
    if (
      currentMaxContext
      && nextMaxContext
      && nextMaxContext < currentMaxContext
    ) {
      notes.push(t("libraryEditProviderLowContextWarning"));
    }
  }
  if (!changedFields.length) {
    return true;
  }
  return confirmDialog({
    title: t("confirmLibraryEditSensitiveTitle"),
    message: [
      t("confirmLibraryEditSensitiveMessage", {
        library: originalLibrary.name || originalLibrary.id,
        fields: joinLocalizedList(changedFields),
      }),
      ...notes,
    ].join(" "),
    confirmText: t("savePlain"),
  });
}

async function confirmSensitiveProviderEdit(originalProvider, payload) {
  if (!originalProvider || !payload.id || payload.id === originalProvider.id) {
    return true;
  }
  return confirmDialog({
    title: t("confirmProviderEditSensitiveTitle"),
    message: [
      t("confirmProviderEditSensitiveMessage", {
        provider: originalProvider.display_name || originalProvider.id,
        currentId: originalProvider.id,
        nextId: payload.id,
      }),
      t("providerEditRenameNote"),
    ].join(" "),
    confirmText: t("savePlain"),
  });
}

$("library-form").onsubmit = async (event) => {
  event.preventDefault();
  const originalId = $("library-original-id").value;
  const databaseType = $("library-database-type").value || DEFAULT_DATABASE_TYPE;
  const requestedId = validateIdentifierInput($("library-id"));
  if (requestedId === null) return;
  const payload = {
    id: requestedId,
    name: $("library-name").value.trim(),
    description: $("library-description").value.trim(),
    default_persona_id: $("library-persona").value.trim(),
    provider_id: $("library-provider").value,
    rerank_provider_id: $("library-rerank-provider").value,
    conversation_settings: libraryConversationFromForm(),
    recall_settings: libraryRecallFromForm(),
    maintenance_settings: libraryMaintenanceFromForm(),
  };
  await asyncGuard.run(`library-form:${originalId || payload.id || "new"}`, async () => {
  try {
    if (originalId) {
      const originalLibrary = databaseForRef(databaseType, originalId);
      if ((originalLibrary?.adapter_connections || []).length) {
        payload.id = originalId;
      }
      const wasSelected = state.selectedDatabaseId === originalId;
      const providerChanged = Boolean(
        originalLibrary
        && payload.provider_id
        && payload.provider_id !== originalLibrary.provider_id,
      );
      if (!(await confirmSensitiveLibraryEdit(originalLibrary, payload))) {
        return;
      }
      const updatedLibrary = await api(databaseApiPath(databaseType, originalId), {
        method: "PATCH",
        body: JSON.stringify({
          id: payload.id,
          name: payload.name,
          description: payload.description,
          default_persona_id: payload.default_persona_id,
          rerank_provider_id: payload.rerank_provider_id || "",
          conversation_settings: payload.conversation_settings,
          recall_settings: payload.recall_settings,
          maintenance_settings: payload.maintenance_settings,
        }),
      });
      const savedLibraryId = updatedLibrary.id || payload.id || originalId;
      if (wasSelected && savedLibraryId !== originalId) {
        selectDatabase(savedLibraryId, { resetMemoryPage: false, databaseType });
      }
      await loadProviders(false);
      const selectedProvider = latestProvider(payload.provider_id);
      const rebuildReason = providerChanged
        ? "library_edit_provider_switch"
        : libraryEditNeedsRebuild(updatedLibrary, selectedProvider);
      if (rebuildReason) {
        closeOverlay("library-modal");
        markDatabaseIndexConflict(savedLibraryId, databaseType);
        await startDatabaseIndexRebuild(savedLibraryId, payload.provider_id, rebuildReason, {
          markLibraryConflict: true,
          databaseType,
        });
        toast(t("providerSwitchQueued", { library: savedLibraryId }));
      }
    } else {
      await api(databaseCollectionApiPath(databaseType), {
        method: "POST",
        body: JSON.stringify(payload),
      });
    }
    closeOverlay("library-modal");
    toast("记忆库已保存");
    await loadDatabases();
  } catch (error) {
    await refreshCatalogAfterMutationConflict(error);
    toast(error.message, true);
  }
  }, {
    form: event.currentTarget,
    button: event.submitter,
    busyText: t("loading"),
  });
};

function openImportModal(databaseId, databaseType = DEFAULT_DATABASE_TYPE) {
  $("import-library-id").value = databaseId;
  $("import-form").dataset.databaseType = databaseType;
  $("import-file").value = "";
  $("import-file-name").textContent = "";
  $("import-conversations-file").value = "";
  $("import-conversations-file-name").textContent = "";
  $("import-modal-title").textContent = `${t("importMemory")}：${databaseId}`;
  $("import-modal").classList.remove("hidden");
}

function selectedImportFile() {
  return $("import-file").files?.[0] || null;
}

function selectedImportConversationsFile() {
  return $("import-conversations-file").files?.[0] || null;
}

$("import-file")?.addEventListener("change", () => {
  const file = selectedImportFile();
  $("import-file-name").textContent = file ? file.name : "";
});

function setupImportDropZone(zoneId, inputId, nameId) {
  const dropZone = $(zoneId);
  if (!dropZone) return;
  ["dragenter", "dragover"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $(inputId).files = transfer.files;
    $(nameId).textContent = file.name;
  });
}

$("import-conversations-file")?.addEventListener("change", () => {
  const file = selectedImportConversationsFile();
  $("import-conversations-file-name").textContent = file ? file.name : "";
});

setupImportDropZone("import-drop-zone", "import-file", "import-file-name");
setupImportDropZone(
  "import-conversations-drop-zone",
  "import-conversations-file",
  "import-conversations-file-name",
);

$("import-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const databaseId = $("import-library-id").value;
  const databaseType = event.currentTarget.dataset.databaseType || DEFAULT_DATABASE_TYPE;
  const file = selectedImportFile();
  const conversationsFile = selectedImportConversationsFile();
  if (!file) {
    toast(t("chooseImportFile"), true);
    return;
  }
  const form = new FormData();
  form.append("file", file, file.name);
  if (conversationsFile) {
    form.append("conversations_file", conversationsFile, conversationsFile.name);
  }
  await asyncGuard.run(`database:${databaseRefKey(databaseType, databaseId)}:livingmemory-import`, async () => {
  const tempJobId = addOptimisticTask({
    kind: "livingmemory_import",
    databaseId,
    databaseType,
    markLibraryConflict: true,
  });
  try {
    const result = await api(databaseApiPath(databaseType, databaseId, "/imports/livingmemory-db"), {
      method: "POST",
      body: form,
    });
    closeOverlay("import-modal");
    toast(t("importQueued", { library: databaseId }));
    trackQueuedJob(result, {
      kind: "livingmemory_import",
      databaseId,
      databaseType,
      tempId: tempJobId,
    });
  } catch (error) {
    removeOptimisticTask(tempJobId);
    toast(error.message, true);
  }
  }, {
    form: event.currentTarget,
    button: event.submitter,
    busyText: t("taskSubmitting"),
    toast,
    startMessage: t("taskSubmitting"),
  });
});


  return {
    loadDatabases,
    ensureDatabaseSelection,
    navigate,
    bindUsedListDetails,
    confirmSensitiveProviderEdit,
    markDatabaseIndexConflict,
    clearDatabaseIndexConflict,
  };
}
