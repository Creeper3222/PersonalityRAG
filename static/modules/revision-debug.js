const GROUPS = Object.freeze([
  ["connection", ["type", "api_base", "model_endpoint"]],
  ["model", ["model", "dimensions", "truncate"]],
  ["context", ["context_length_mode", "max_context_tokens", "max_context_tokens_source"]],
  ["execution", ["timeout_seconds", "batch_size", "concurrency", "max_retries", "launch_model_if_not_running", "enabled"]],
  ["rebuild", ["index_rebuild_settings"]],
]);

export function createRevisionDebugController({
  $, state, t, api, toast, escapeHtml, navigate, confirmDialog, asyncGuard,
}) {
  let session = null;
  let overview = null;
  let activeView = "providers";
  let countdownTimer = null;
  let pendingAction = null;

  const encode = (value) => encodeURIComponent(String(value ?? ""));
  const decode = (value) => decodeURIComponent(String(value ?? ""));
  const formatTime = (value) => {
    const timestamp = Number(value || 0);
    return timestamp ? new Date(timestamp * 1000).toLocaleString() : "—";
  };
  const humanKey = (value) => String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
  const label = (key) => {
    const translated = t(`revisionDebugField_${key}`);
    return translated === `revisionDebugField_${key}` ? humanKey(key) : translated;
  };
  const boolPill = (value) => `<span class="status-pill ${value ? "success" : "warning"}">${escapeHtml(value ? t("yes") : t("no"))}</span>`;
  const valueMarkup = (value) => {
    if (typeof value === "boolean") return boolPill(value);
    if (value === null || value === undefined || value === "") return `<span class="muted">—</span>`;
    if (Array.isArray(value)) return value.length
      ? `<span>${value.map((item) => escapeHtml(item)).join(" · ")}</span>`
      : `<span class="muted">—</span>`;
    return `<code>${escapeHtml(value)}</code>`;
  };

  function clearTimer() {
    if (countdownTimer) clearInterval(countdownTimer);
    countdownTimer = null;
  }

  function remainingSeconds() {
    return Math.max(0, Math.ceil(Number(session?.expires_at || 0) - Date.now() / 1000));
  }

  function updateCountdown() {
    const seconds = remainingSeconds();
    const target = $("revision-debug-countdown");
    if (target) {
      const minutes = Math.floor(seconds / 60);
      target.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }
    if (session?.unlocked && seconds <= 0) {
      clearTimer();
      handleLocked({ redirect: true });
    }
  }

  function startTimer() {
    clearTimer();
    updateCountdown();
    if (session?.unlocked) countdownTimer = setInterval(updateCountdown, 1000);
  }

  function updateSettingsCard() {
    const configured = Boolean(session?.password_configured);
    const unlocked = Boolean(session?.unlocked && remainingSeconds() > 0);
    $("revision-debug-password-state").innerHTML = configured
      ? `<span class="status-pill success">${escapeHtml(t("revisionDebugPasswordConfigured"))}</span>`
      : `<span class="status-pill danger">${escapeHtml(t("revisionDebugPasswordMissing"))}</span>`;
    $("revision-debug-unlock-state").innerHTML = unlocked
      ? `<span class="status-pill success">${escapeHtml(t("revisionDebugUnlocked"))}</span>`
      : `<span class="status-pill warning">${escapeHtml(t("revisionDebugLocked"))}</span>`;
    const enter = $("revision-debug-enter");
    enter.disabled = !configured;
    enter.textContent = t(unlocked ? "revisionDebugEnter" : "revisionDebugVerifyEnter");
    $("revision-debug-settings-note").textContent = configured
      ? t("revisionDebugRiskNotice")
      : t("revisionDebugConfigurePasswordFirst");
    $("revision-debug-nav")?.classList.toggle("hidden", !unlocked);
  }

  function handleLocked({ redirect = false } = {}) {
    session = {
      ...(session || {}),
      unlocked: false,
      expires_at: null,
      remaining_seconds: 0,
    };
    overview = null;
    clearTimer();
    updateSettingsCard();
    $("revision-debug-content").replaceChildren();
    if (redirect && state.page === "revision-debug") {
      toast(t("revisionDebugExpired"), true, { log: false });
      navigate("settings").catch(() => {});
    }
  }

  async function refreshSessionStatus({ redirectIfLocked = false } = {}) {
    try {
      session = await api("/debug/session", { pageScoped: false });
    } catch (error) {
      handleLocked({ redirect: redirectIfLocked });
      throw error;
    }
    if (!session.unlocked || remainingSeconds() <= 0) {
      handleLocked({ redirect: redirectIfLocked });
      return session;
    }
    updateSettingsCard();
    startTimer();
    return session;
  }

  function closeModal(id) {
    $(id)?.classList.add("hidden");
  }

  function openUnlockModal() {
    const overlay = $("revision-debug-unlock-modal");
    const input = $("revision-debug-unlock-password");
    input.value = "";
    overlay.classList.remove("hidden");
    setTimeout(() => input.focus(), 0);
  }

  async function unlock(password) {
    session = await api("/debug/session", {
      method: "POST",
      body: JSON.stringify({ password }),
      pageScoped: false,
    });
    closeModal("revision-debug-unlock-modal");
    updateSettingsCard();
    startTimer();
    toast(t("revisionDebugUnlockSuccess"), false, { log: false });
    await navigate("revision-debug");
  }

  async function lock() {
    try {
      await api("/debug/session", { method: "DELETE", pageScoped: false });
    } finally {
      handleLocked();
      await navigate("settings");
    }
  }

  function groupConfig(config = {}) {
    const used = new Set(["id", "display_name", "provider_kind", "has_api_key", "api_key"]);
    const result = [];
    for (const [name, fields] of GROUPS) {
      const entries = [];
      for (const field of fields) {
        if (!(field in config)) continue;
        used.add(field);
        if (field === "index_rebuild_settings" && config[field] && typeof config[field] === "object") {
          Object.entries(config[field]).forEach(([key, value]) => entries.push([`index_rebuild_settings.${key}`, value]));
        } else {
          entries.push([field, config[field]]);
        }
      }
      if (entries.length) result.push([name, entries]);
    }
    const specific = Object.entries(config).filter(([key]) => !used.has(key));
    if (specific.length) result.push(["provider", specific]);
    result.push(["credential", [["has_api_key", Boolean(config.has_api_key)]]]);
    return result;
  }

  function configMarkup(config) {
    return `<div class="revision-debug-config-groups">${groupConfig(config).map(([name, entries]) => `
      <section class="revision-debug-config-group">
        <h5>${escapeHtml(t(`revisionDebugGroup_${name}`))}</h5>
        <dl class="revision-debug-config-list">${entries.map(([key, value]) => `
          <dt>${escapeHtml(label(key))}</dt><dd>${valueMarkup(value)}</dd>
        `).join("")}</dl>
      </section>`).join("")}</div>`;
  }

  function referenceSummary(revision) {
    const refs = revision.references || [];
    if (!refs.length) return `<span class="status-pill">${escapeHtml(t("revisionDebugUnreferenced"))}</span>`;
    const generationCount = refs.filter((item) => item.kind === "index_generation").length;
    const bindingCount = refs.filter((item) => item.kind === "database_binding").length;
    const taskCount = refs.filter((item) => item.kind === "task_checkpoint").length;
    return [
      [bindingCount, "revisionDebugReferenceBindings"],
      [generationCount, "revisionDebugReferenceGenerations"],
      [taskCount, "revisionDebugReferenceTasks"],
    ].filter(([count]) => count).map(([count, key]) => `<span class="status-pill">${count} ${escapeHtml(t(key))}</span>`).join("");
  }

  function revisionMarkup(provider, revision) {
    const latest = revision.is_latest;
    const equivalent = revision.functionally_equal_to_latest;
    const providerKey = encode(provider.provider_id);
    const changed = revision.changed_fields || [];
    return `<article class="revision-debug-revision">
      <header>
        <div><strong>revision ${revision.revision}</strong><div class="revision-debug-card-badges">${latest ? `<span class="status-pill success">${escapeHtml(t("revisionDebugLatest"))}</span>` : ""}${!latest && equivalent ? `<span class="status-pill success">${escapeHtml(t("revisionDebugEquivalent"))}</span>` : ""}${!equivalent ? `<span class="status-pill warning">${escapeHtml(t("revisionDebugDifferent"))}</span>` : ""}</div></div>
        <div class="revision-debug-revision-actions">
          ${!latest ? `<button class="ghost" type="button" data-debug-action="set-latest" data-provider="${providerKey}" data-revision="${revision.revision}">${escapeHtml(t("revisionDebugSetLatest"))}</button>` : ""}
          <button class="ghost danger" type="button" data-debug-action="edit-revision" data-provider="${providerKey}" data-revision="${revision.revision}">${escapeHtml(t("revisionDebugEditRevision"))}</button>
          ${revision.config?.has_api_key ? `<button class="ghost danger" type="button" data-debug-action="replace-key" data-provider="${providerKey}" data-revision="${revision.revision}">${escapeHtml(t("revisionDebugReplaceKey"))}</button><button class="ghost danger" type="button" data-debug-action="clear-key" data-provider="${providerKey}" data-revision="${revision.revision}">${escapeHtml(t("revisionDebugClearKey"))}</button>` : `<button class="ghost danger" type="button" data-debug-action="replace-key" data-provider="${providerKey}" data-revision="${revision.revision}">${escapeHtml(t("revisionDebugSetKey"))}</button>`}
        </div>
      </header>
      <dl class="revision-debug-meta-grid"><dt>${escapeHtml(t("revisionDebugCreatedAt"))}</dt><dd>${escapeHtml(formatTime(revision.created_at))}</dd><dt>${escapeHtml(t("revisionDebugConfigHash"))}</dt><dd><code class="revision-debug-hash">${escapeHtml(revision.config_sha256)}</code></dd><dt>${escapeHtml(t("revisionDebugFunctionalHash"))}</dt><dd><code class="revision-debug-hash">${escapeHtml(revision.functional_sha256)}</code></dd></dl>
      <div class="revision-debug-field-diffs">${changed.length ? changed.map((field) => `<code>${escapeHtml(field)}</code>`).join("") : `<span class="muted">${escapeHtml(t("revisionDebugNoFieldDifferences"))}</span>`}</div>
      ${referenceSummary(revision)}
      <details class="revision-debug-detail"><summary>${escapeHtml(t("revisionDebugStructuredConfig"))}</summary><div class="revision-debug-detail-body">${configMarkup(revision.config || {})}</div></details>
    </article>`;
  }

  function providerMarkup(provider) {
    const revisions = [...(provider.revisions || [])].sort((a, b) => b.revision - a.revision);
    const providerKey = encode(provider.provider_id);
    return `<article class="panel revision-debug-provider-card" data-filter-text="${escapeHtml([provider.provider_id, provider.display_name, provider.provider_type, provider.provider_kind].join(" ").toLowerCase())}">
      <header class="revision-debug-card-header"><div class="revision-debug-card-title"><span class="revision-debug-settings-icon" aria-hidden="true"></span><div><h3>${escapeHtml(provider.display_name || provider.provider_id)}</h3><p>${escapeHtml(provider.provider_id)} · ${escapeHtml(provider.provider_type)} · ${escapeHtml(provider.provider_kind)}</p></div></div><div class="revision-debug-card-badges"><span class="status-pill success">latest ${provider.latest_revision}</span><span class="status-pill">${revisions.length} ${escapeHtml(t("revisionDebugRevisions"))}</span></div></header>
      <details class="revision-debug-revisions"><summary><span>${escapeHtml(t("revisionDebugRevisionHistory"))}</span><span>${revisions.length}</span></summary><div class="revision-debug-revision-list">${revisions.map((revision) => revisionMarkup(provider, revision)).join("")}</div></details>
      <section class="revision-debug-danger-zone"><h4>${escapeHtml(t("revisionDebugDangerZone"))}</h4><p>${escapeHtml(t("revisionDebugDeleteNewerHint"))}</p><div class="revision-debug-danger-actions"><button class="ghost danger" type="button" data-debug-action="delete-newer" data-provider="${providerKey}">${escapeHtml(t("revisionDebugDeleteNewer"))}</button></div></section>
    </article>`;
  }

  function bindingStatus(binding) {
    if (!binding.provider_exists || !binding.revision_exists || binding.fingerprint_matches_revision === false) return "danger";
    if (binding.needs_rebuild || binding.needs_recalibration) return "warning";
    return "success";
  }

  function bindingMarkup(database, binding) {
    const provider = overview.providers.find((item) => item.provider_id === binding.provider_id);
    const revisions = (provider?.revisions || []).slice().sort((a, b) => b.revision - a.revision);
    const current = Number(binding.provider_revision || 0);
    const pinned = binding.binding_mode === "pinned";
    const typeKey = encode(database.database_type);
    const dbKey = encode(database.database_id);
    const providerKey = encode(binding.provider_id);
    return `<section class="revision-debug-binding">
      <header><h4>${escapeHtml(t(binding.usage_kind === "rerank" ? "revisionDebugRerankBinding" : "revisionDebugEmbeddingBinding"))}</h4><span class="status-pill ${bindingStatus(binding)}">${escapeHtml(t(`revisionDebugBinding_${binding.binding_mode}`))}</span></header>
      <dl><dt>Provider ID</dt><dd><code>${escapeHtml(binding.provider_id || "—")}</code></dd><dt>revision</dt><dd>${valueMarkup(binding.provider_revision)}</dd><dt>latest revision</dt><dd>${valueMarkup(binding.latest_revision)}</dd><dt>${escapeHtml(t("revisionDebugFingerprint"))}</dt><dd><code class="revision-debug-hash">${escapeHtml(binding.provider_fingerprint || "—")}</code></dd><dt>${escapeHtml(t("revisionDebugFingerprintMatch"))}</dt><dd>${binding.fingerprint_matches_revision === null ? "—" : boolPill(binding.fingerprint_matches_revision)}</dd><dt>${escapeHtml(t("revisionDebugNeedsRebuild"))}</dt><dd>${boolPill(Boolean(binding.needs_rebuild))}</dd>${binding.usage_kind === "rerank" ? `<dt>${escapeHtml(t("revisionDebugNeedsRecalibration"))}</dt><dd>${boolPill(Boolean(binding.needs_recalibration))}</dd>` : ""}</dl>
      ${pinned && revisions.length ? `<div class="revision-debug-binding-controls"><label>${escapeHtml(t("revisionDebugTargetRevision"))}<select data-debug-binding-revision>${revisions.map((item) => `<option value="${item.revision}" ${item.revision === current ? "selected" : ""}>revision ${item.revision}${item.is_latest ? ` · ${escapeHtml(t("revisionDebugLatest"))}` : ""}${item.functionally_equal_to_latest ? ` · ${escapeHtml(t("revisionDebugEquivalent"))}` : ""}</option>`).join("")}</select></label><button class="ghost" type="button" data-debug-action="repair-binding" data-database-type="${typeKey}" data-database-id="${dbKey}" data-usage-kind="${escapeHtml(binding.usage_kind)}" data-provider="${providerKey}">${escapeHtml(t("revisionDebugRepairBinding"))}</button></div>` : `<p class="muted">${escapeHtml(t(pinned ? "revisionDebugRevisionUnavailable" : "revisionDebugFollowsLatestHint"))}</p>`}
    </section>`;
  }

  function rowsMarkup(items = []) {
    if (!items.length) return `<div class="revision-debug-empty">${escapeHtml(t("revisionDebugNoData"))}</div>`;
    return items.map((item) => `<div class="revision-debug-detail-row">${Object.entries(item).map(([key, value]) => `<span>${escapeHtml(label(key))}</span><div>${valueMarkup(value)}</div>`).join("")}</div>`).join("");
  }

  function databaseMarkup(database) {
    const issues = database.issues || [];
    const blockers = [...(database.active_tasks || []).map((item) => `${t("revisionDebugActiveTask")}: ${item.id}`), ...(database.active_adapters || []).map((item) => `${t("revisionDebugOnlineAdapter")}: ${item.adapter_id || item.instance_id}`)];
    const details = [
      ["revisionDebugIndexGenerations", database.embedding_generations || []],
      ["revisionDebugMediaEmbeddings", database.media_embeddings || []],
      ["revisionDebugCalibration", database.relation_calibrations || []],
      ["revisionDebugStrengthCalibration", database.strength_calibrations || []],
    ].filter(([, items]) => items.length);
    const filterText = [database.database_type, database.database_id, database.database_name, database.database_category, ...issues.map((item) => item.code)].join(" ").toLowerCase();
    return `<article class="panel revision-debug-database-card" data-filter-text="${escapeHtml(filterText)}">
      <header class="revision-debug-card-header"><div class="revision-debug-card-title"><div><h3>${escapeHtml(database.database_name || database.database_id)}</h3><p>${escapeHtml(database.database_type)}:${escapeHtml(database.database_id)} · ${escapeHtml(database.database_category)}</p></div></div><div class="revision-debug-card-badges">${issues.length ? `<span class="status-pill danger">${issues.length} ${escapeHtml(t("revisionDebugIssues"))}</span>` : `<span class="status-pill success">${escapeHtml(t("revisionDebugHealthy"))}</span>`}${database.needs_rebuild ? `<span class="status-pill warning">${escapeHtml(t("revisionDebugNeedsRebuild"))}</span>` : ""}${database.needs_recalibration ? `<span class="status-pill warning">${escapeHtml(t("revisionDebugNeedsRecalibration"))}</span>` : ""}${database.missing_storage ? `<span class="status-pill danger">${escapeHtml(t("revisionDebugMissingStorage"))}</span>` : ""}</div></header>
      ${blockers.length ? `<div class="revision-debug-blockers">${blockers.map((item) => `<span class="status-pill danger">${escapeHtml(item)}</span>`).join("")}</div>` : ""}
      <div class="revision-debug-binding-grid">${(database.bindings || []).map((binding) => bindingMarkup(database, binding)).join("")}</div>
      ${details.map(([key, items]) => `<details class="revision-debug-detail"><summary>${escapeHtml(t(key))}<span>${items.length}</span></summary><div class="revision-debug-detail-body">${rowsMarkup(items)}</div></details>`).join("")}
    </article>`;
  }

  function issueText(issue) {
    const translated = t(`revisionDebugIssue_${issue.code}`);
    return translated === `revisionDebugIssue_${issue.code}` ? humanKey(issue.code) : translated;
  }

  function renderSummary() {
    const summary = overview?.summary || {};
    const items = [
      ["provider_count", "revisionDebugProviderCount"],
      ["revision_count", "revisionDebugRevisionCount"],
      ["database_count", "revisionDebugDatabaseCount"],
      ["issue_count", "revisionDebugIssueCount"],
      ["active_task_count", "revisionDebugActiveTaskCount"],
      ["active_adapter_count", "revisionDebugActiveAdapterCount"],
    ];
    $("revision-debug-summary").innerHTML = items.map(([key, textKey]) => `<div class="revision-debug-stat ${key === "issue_count" && Number(summary[key]) ? "alert" : ""}"><strong>${escapeHtml(summary[key] ?? 0)}</strong><span>${escapeHtml(t(textKey))}</span></div>`).join("");
    const issues = overview?.issues || [];
    $("revision-debug-issues-panel").classList.toggle("hidden", !issues.length);
    $("revision-debug-issue-count").textContent = String(issues.length);
    $("revision-debug-issues").innerHTML = issues.map((issue) => `<div class="revision-debug-issue"><span class="status-pill danger">!</span><div><strong>${escapeHtml(issueText(issue))}</strong><br><code>${escapeHtml([issue.provider_id, issue.database_type && `${issue.database_type}:${issue.database_id}`, issue.usage_kind, issue.revision && `revision ${issue.revision}`, issue.generation].filter(Boolean).join(" · "))}</code></div></div>`).join("");
    $("revision-debug-captured-at").textContent = `${t("revisionDebugCapturedAt")}: ${formatTime(overview?.captured_at)}`;
  }

  function render() {
    if (!overview) return;
    renderSummary();
    document.querySelectorAll("[data-revision-debug-view]").forEach((button) => button.classList.toggle("active", button.dataset.revisionDebugView === activeView));
    const items = activeView === "providers" ? overview.providers : overview.databases;
    $("revision-debug-content").innerHTML = items.length
      ? items.map((item) => activeView === "providers" ? providerMarkup(item) : databaseMarkup(item)).join("")
      : `<section class="panel revision-debug-empty">${escapeHtml(t("revisionDebugNoData"))}</section>`;
    applyFilter();
  }

  function applyFilter() {
    const query = String($("revision-debug-filter")?.value || "").trim().toLowerCase();
    $("revision-debug-content")?.querySelectorAll("[data-filter-text]").forEach((card) => {
      card.classList.toggle("hidden", Boolean(query) && !card.dataset.filterText.includes(query));
    });
  }

  async function loadOverview() {
    overview = await api("/debug/revisions/overview", { pageScoped: false });
    render();
  }

  async function load() {
    const status = await refreshSessionStatus({ redirectIfLocked: true });
    if (!status?.unlocked) return;
    await loadOverview();
  }

  function findProvider(providerId) {
    return overview?.providers.find((item) => item.provider_id === providerId);
  }

  function findRevision(providerId, revision) {
    return findProvider(providerId)?.revisions.find((item) => item.revision === Number(revision));
  }

  function inputMarkup(key, value) {
    const fieldLabel = escapeHtml(label(key));
    const path = escapeHtml(key);
    if (typeof value === "boolean") return `<label><span>${fieldLabel}</span><select data-debug-config-path="${path}"><option value="true" ${value ? "selected" : ""}>${escapeHtml(t("yes"))}</option><option value="false" ${!value ? "selected" : ""}>${escapeHtml(t("no"))}</option></select></label>`;
    const numeric = typeof value === "number";
    return `<label><span>${fieldLabel}</span><input data-debug-config-path="${path}" type="${numeric ? "number" : "text"}" value="${escapeHtml(value ?? "")}" ${numeric ? "step=\"any\"" : ""}></label>`;
  }

  function editableFields(config = {}) {
    return groupConfig(config).filter(([name]) => name !== "credential").map(([name, entries]) => `<section class="revision-debug-config-group"><h5>${escapeHtml(t(`revisionDebugGroup_${name}`))}</h5>${entries.filter(([key]) => !["id", "type", "provider_kind", "has_api_key", "api_key"].includes(key)).map(([key, value]) => inputMarkup(key, value)).join("")}</section>`).join("");
  }

  function openAction({ title, message, fields = "", dangerous = true, submitText, run }) {
    pendingAction = { run, dangerous };
    $("revision-debug-action-title").textContent = title;
    $("revision-debug-action-message").textContent = message;
    $("revision-debug-action-fields").innerHTML = fields;
    $("revision-debug-action-password-row").classList.toggle("hidden", !dangerous);
    $("revision-debug-action-risk-row").classList.toggle("hidden", !dangerous);
    $("revision-debug-action-password").value = "";
    $("revision-debug-action-risk").checked = false;
    $("revision-debug-action-submit").textContent = submitText || t("confirmAction");
    $("revision-debug-action-modal").classList.remove("hidden");
    setTimeout(() => {
      const target = dangerous
        ? $("revision-debug-action-password")
        : $("revision-debug-action-fields")?.querySelector("input,select");
      target?.focus();
    }, 0);
  }

  function readConfigPatch(original) {
    const patch = {};
    const nested = {};
    $("revision-debug-action-fields").querySelectorAll("[data-debug-config-path]").forEach((input) => {
      const path = input.dataset.debugConfigPath;
      const originalValue = path.startsWith("index_rebuild_settings.")
        ? original.index_rebuild_settings?.[path.split(".")[1]] : original[path];
      let value = input.value;
      if (typeof originalValue === "boolean") value = value === "true";
      if (typeof originalValue === "number") value = Number(value);
      if (value === originalValue) return;
      if (path.startsWith("index_rebuild_settings.")) nested[path.split(".")[1]] = value;
      else patch[path] = value;
    });
    if (Object.keys(nested).length) patch.index_rebuild_settings = { ...(original.index_rebuild_settings || {}), ...nested };
    return patch;
  }

  async function performMutation(work) {
    try {
      await work();
      closeModal("revision-debug-action-modal");
      toast(t("revisionDebugOperationComplete"), false, { log: false });
      await loadOverview();
    } catch (error) {
      if ([401, 404].includes(Number(error.status)) && state.page === "revision-debug") {
        await refreshSessionStatus({ redirectIfLocked: true }).catch(() => {});
      }
      throw error;
    }
  }

  async function setLatest(providerId, revision) {
    const item = findRevision(providerId, revision);
    if (!item) return;
    if (item.functionally_equal_to_latest) {
      const confirmed = await confirmDialog({ title: t("revisionDebugSetLatest"), message: t("revisionDebugSetLatestSafeHint"), confirmText: t("confirmAction") });
      if (!confirmed) return;
      await performMutation(() => api(`/debug/providers/${encode(providerId)}/revisions/reset`, { method: "POST", body: JSON.stringify({ latest_revision: Number(revision) }), pageScoped: false }));
      return;
    }
    openAction({ title: t("revisionDebugSetLatest"), message: t("revisionDebugSetLatestDangerHint"), run: ({ password }) => performMutation(() => api(`/debug/providers/${encode(providerId)}/revisions/reset`, { method: "POST", body: JSON.stringify({ latest_revision: Number(revision), force_non_equivalent: true, password, risk_confirmed: true }), pageScoped: false })) });
  }

  function editRevision(providerId, revision) {
    const item = findRevision(providerId, revision);
    if (!item) return;
    openAction({ title: `${t("revisionDebugEditRevision")} · ${providerId}@${revision}`, message: t("revisionDebugEditRevisionHint"), fields: editableFields(item.config || {}), run: ({ password }) => {
      const patch = readConfigPatch(item.config || {});
      if (!Object.keys(patch).length) throw new Error(t("revisionDebugNoChanges"));
      return performMutation(() => api(`/debug/providers/${encode(providerId)}/revisions/${revision}`, { method: "PATCH", body: JSON.stringify({ patch, password, risk_confirmed: true }), pageScoped: false }));
    } });
  }

  function keyAction(providerId, revision, clear = false) {
    openAction({ title: t(clear ? "revisionDebugClearKey" : "revisionDebugReplaceKey"), message: t(clear ? "revisionDebugClearKeyHint" : "revisionDebugReplaceKeyHint"), fields: clear ? "" : `<label><span>${escapeHtml(t("revisionDebugNewApiKey"))}</span><input id="revision-debug-new-api-key" type="password" autocomplete="new-password" required></label>`, run: ({ password }) => {
      const key = $("revision-debug-new-api-key")?.value || "";
      if (!clear && !key) throw new Error(t("revisionDebugApiKeyRequired"));
      return performMutation(() => api(`/debug/providers/${encode(providerId)}/revisions/${revision}`, { method: "PATCH", body: JSON.stringify({ patch: clear ? { clear_api_key: true } : { api_key: key }, password, risk_confirmed: true }), pageScoped: false }));
    } });
  }

  function deleteNewer(providerId) {
    const provider = findProvider(providerId);
    if (!provider) return;
    const options = [...provider.revisions].sort((a, b) => b.revision - a.revision).map((item) => `<option value="${item.revision}" ${item.revision === provider.latest_revision ? "selected" : ""}>revision ${item.revision}${item.references?.length ? ` · ${item.references.length} ${escapeHtml(t("revisionDebugReferences"))}` : ""}</option>`).join("");
    openAction({ title: t("revisionDebugDeleteNewer"), message: t("revisionDebugDeleteNewerDangerHint"), fields: `<label><span>${escapeHtml(t("revisionDebugKeepThroughRevision"))}</span><select id="revision-debug-delete-target">${options}</select><small>${escapeHtml(t("revisionDebugDeleteReferencedBlocked"))}</small></label>`, run: ({ password }) => performMutation(() => api(`/debug/providers/${encode(providerId)}/revisions/reset`, { method: "POST", body: JSON.stringify({ latest_revision: Number($("revision-debug-delete-target").value), delete_revisions_after_latest: true, force_non_equivalent: true, password, risk_confirmed: true }), pageScoped: false })) });
  }

  async function repairBinding(button) {
    const databaseType = decode(button.dataset.databaseType);
    const databaseId = decode(button.dataset.databaseId);
    const usageKind = button.dataset.usageKind;
    const providerId = decode(button.dataset.provider);
    const revision = Number(button.closest(".revision-debug-binding").querySelector("[data-debug-binding-revision]").value);
    const database = overview.databases.find((item) => item.database_type === databaseType && item.database_id === databaseId);
    const binding = database?.bindings.find((item) => item.usage_kind === usageKind);
    const currentRevision = findRevision(providerId, binding?.provider_revision);
    const targetRevision = findRevision(providerId, revision);
    const equivalent = Boolean(currentRevision && targetRevision && currentRevision.functional_sha256 === targetRevision.functional_sha256);
    const endpoint = `/debug/databases/${encode(databaseType)}/${encode(databaseId)}/bindings/${encode(usageKind)}`;
    const base = { provider_id: providerId, revision };
    if (equivalent) {
      const confirmed = await confirmDialog({ title: t("revisionDebugRepairBinding"), message: t("revisionDebugRepairSafeHint"), confirmText: t("confirmAction") });
      if (!confirmed) return;
      await performMutation(() => api(endpoint, { method: "PATCH", body: JSON.stringify(base), pageScoped: false }));
      return;
    }
    openAction({ title: t("revisionDebugRepairBinding"), message: t("revisionDebugRepairDangerHint"), run: ({ password }) => performMutation(() => api(endpoint, { method: "PATCH", body: JSON.stringify({ ...base, assert_functional_compatibility: true, password, risk_confirmed: true }), pageScoped: false })) });
  }

  function bind() {
    $("revision-debug-enter")?.addEventListener("click", async () => {
      if (session?.unlocked && remainingSeconds() > 0) await navigate("revision-debug");
      else openUnlockModal();
    });
    $("revision-debug-unlock-form")?.addEventListener("submit", async (event) => {
      event.preventDefault();
      await asyncGuard.run("revision-debug:unlock", () => unlock($("revision-debug-unlock-password").value), { form: event.currentTarget, button: event.submitter, busyText: t("loading") }).catch((error) => toast(error.message, true, { log: false }));
    });
    $("revision-debug-lock")?.addEventListener("click", () => lock().catch((error) => toast(error.message, true, { log: false })));
    $("revision-debug-back")?.addEventListener("click", () => navigate("settings"));
    $("revision-debug-refresh")?.addEventListener("click", (event) => asyncGuard.run("revision-debug:refresh", loadOverview, { button: event.currentTarget, busyText: t("loading") }).catch((error) => toast(error.message, true, { log: false })));
    $("revision-debug-filter")?.addEventListener("input", applyFilter);
    document.querySelectorAll("[data-revision-debug-view]").forEach((button) => button.addEventListener("click", () => { activeView = button.dataset.revisionDebugView; render(); }));
    $("revision-debug-content")?.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-debug-action]");
      if (!button) return;
      const action = button.dataset.debugAction;
      const providerId = decode(button.dataset.provider || "");
      const revision = Number(button.dataset.revision || 0);
      try {
        if (action === "set-latest") await setLatest(providerId, revision);
        if (action === "edit-revision") editRevision(providerId, revision);
        if (action === "replace-key") keyAction(providerId, revision, false);
        if (action === "clear-key") keyAction(providerId, revision, true);
        if (action === "delete-newer") deleteNewer(providerId);
        if (action === "repair-binding") await repairBinding(button);
      } catch (error) { toast(error.message, true, { log: false }); }
    });
    $("revision-debug-action-form")?.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!pendingAction) return;
      const password = $("revision-debug-action-password").value;
      if (pendingAction.dangerous && (!password || !$("revision-debug-action-risk").checked)) {
        toast(t("revisionDebugPasswordRiskRequired"), true, { log: false });
        return;
      }
      await asyncGuard.run("revision-debug:action", () => pendingAction.run({ password }), { form: event.currentTarget, button: event.submitter, busyText: t("loading") }).catch((error) => toast(error.message, true, { log: false }));
    });
    ["revision-debug-unlock-modal", "revision-debug-action-modal"].forEach((id) => {
      const overlay = $(id);
      overlay?.querySelectorAll(".modal-dismiss").forEach((button) => button.addEventListener("click", () => closeModal(id)));
      overlay?.addEventListener("click", (event) => { if (event.target === overlay) closeModal(id); });
    });
  }

  function onLanguageChange() {
    updateSettingsCard();
    if (overview) render();
  }

  function clear() {
    handleLocked();
  }

  bind();
  return { load, refreshSessionStatus, onLanguageChange, clear, lock };
}
