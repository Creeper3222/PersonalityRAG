export function createSettingsController({
  $,
  state,
  t,
  api,
  toast,
  stopLogPolling,
  stopTaskPolling,
  resetTaskState = () => {},
  isValidPage,
  restartReturnPage,
  escapeHtml,
  confirmDialog,
}) {
function normalizePragFilename(name) {
  const value = String(name || "personalityrag.prag");
  return value.endsWith(".prag") ? value : `${value}.prag`;
}

function downloadPragBlob(blob, suggestedName) {
  const filename = normalizePragFilename(suggestedName);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function createPragSaveTarget(suggestedName) {
  const filename = normalizePragFilename(suggestedName);
  if (window.showSaveFilePicker) {
    const handle = await window.showSaveFilePicker({
      suggestedName: filename,
      types: [
        {
          description: "PersonalityRAG config package",
          accept: { "application/octet-stream": [".prag"] },
        },
      ],
    });
    return {
      save: async (blob) => {
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      },
    };
  }
  return {
    save: async (blob, fallbackName = filename) => {
      downloadPragBlob(blob, fallbackName);
    },
  };
}

function requestBackupPassword() {
  return new Promise((resolve) => {
    const overlay = $("backup-password-modal");
    const form = $("backup-password-form");
    const input = $("backup-password-input");
    input.value = "";
    overlay.classList.remove("hidden");
    const cleanup = (value) => {
      overlay.classList.add("hidden");
      form.onsubmit = null;
      overlay.onclick = null;
      overlay.querySelectorAll(".modal-dismiss").forEach((button) => {
        button.onclick = null;
      });
      resolve(value);
    };
    form.onsubmit = (event) => {
      event.preventDefault();
      const value = input.value || "";
      if (!value.trim()) {
        toast(t("configPasswordRequired"), true);
        return;
      }
      cleanup(value);
    };
    overlay.querySelectorAll(".modal-dismiss").forEach((button) => {
      button.onclick = () => cleanup(null);
    });
    overlay.onclick = (event) => {
      if (event.target === overlay) cleanup(null);
    };
    setTimeout(() => input.focus(), 0);
  });
}

function requestBackupExportScope({ suggestedName = "" } = {}) {
  return new Promise((resolve, reject) => {
    const overlay = $("backup-export-scope-modal");
    const form = $("backup-export-scope-form");
    const libraries = $("backup-export-libraries");
    const providers = $("backup-export-providers");
    const busy = $("backup-export-busy");
    let resolved = false;
    libraries.checked = true;
    providers.checked = true;
    form.classList.remove("backup-export-modal-exporting");
    busy?.classList.add("hidden");
    overlay.classList.remove("hidden");
    const finish = () => {
      overlay.classList.add("hidden");
      form.classList.remove("backup-export-modal-exporting");
      busy?.classList.add("hidden");
      form.onsubmit = null;
      overlay.onclick = null;
      overlay.querySelectorAll(".modal-dismiss").forEach((button) => {
        button.onclick = null;
      });
    };
    const cleanup = (value) => {
      finish();
      if (!resolved) {
        resolved = true;
        resolve(value);
      }
    };
    form.onsubmit = async (event) => {
      event.preventDefault();
      if (form.classList.contains("backup-export-modal-exporting")) return;
      form.classList.add("backup-export-modal-exporting");
      busy?.classList.remove("hidden");
      overlay.onclick = null;
      try {
        const saveTarget = await createPragSaveTarget(suggestedName);
        resolved = true;
        resolve({
          include_libraries: Boolean(libraries.checked),
          include_providers: Boolean(providers.checked),
          save: saveTarget.save,
          close: finish,
        });
      } catch (error) {
        if (error?.name === "AbortError") {
          cleanup(null);
          return;
        }
        finish();
        if (!resolved) {
          resolved = true;
          reject(error);
        }
      }
    };
    overlay.querySelectorAll(".modal-dismiss").forEach((button) => {
      button.onclick = () => cleanup(null);
    });
    overlay.onclick = (event) => {
      if (event.target === overlay) cleanup(null);
    };
  });
}

function settingsDraft() {
  return {
    access_base_url: $("settings-access-base-url")?.value.trim() || "",
    port: Number($("settings-port")?.value || 0),
    access_port: Number($("settings-access-port")?.value || 0),
    new_password: $("settings-password")?.value || "",
    clear_password: Boolean($("settings-clear-password")?.checked),
    runtime_idle_minutes: Number($("settings-runtime-idle-minutes")?.value || 0),
    max_non_default_runtimes: Number($("settings-runtime-max-non-default")?.value || 0),
  };
}

function hasUnsavedSettingsChanges() {
  if (!state.settings) return false;
  const draft = settingsDraft();
  return (
    draft.access_base_url !== (state.settings.access_base_url || "http://127.0.0.1") ||
    draft.port !== Number(state.settings.configured_port || 8765) ||
    draft.access_port !== Number(state.settings.configured_access_port || 8766) ||
    draft.runtime_idle_minutes !== Number(state.settings.runtime_residency?.idle_minutes || 30) ||
    draft.max_non_default_runtimes !== Number(state.settings.runtime_residency?.max_non_default_runtimes || 4) ||
    Boolean(draft.new_password) ||
    draft.clear_password
  );
}

function clearRestartTimer() {
  if (state.restart.timer) {
    clearTimeout(state.restart.timer);
    state.restart.timer = null;
  }
}

function normalizeRestartProbeUrls(urls = [], fallbackUrl = "") {
  const ordered = [...urls, fallbackUrl].filter(Boolean);
  const unique = [];
  const seen = new Set();
  ordered.forEach((url) => {
    if (seen.has(url)) return;
    seen.add(url);
    unique.push(url);
  });
  return unique;
}

function prioritizedRestartProbeUrls() {
  return normalizeRestartProbeUrls(
    [state.restart.targetUrl, ...(state.restart.probeUrls || [])],
    "",
  );
}

function nextRestartProbeBatch(batchSize = 6) {
  const urls = prioritizedRestartProbeUrls();
  if (!urls.length) return [];
  const limit = Math.min(batchSize, urls.length);
  const priorityCount = Math.min(2, limit);
  const batch = urls.slice(0, priorityCount);
  const remainder = urls.slice(priorityCount);
  if (!remainder.length || batch.length >= limit) {
    state.restart.probeCursor = 0;
    return batch;
  }
  const extraCount = limit - batch.length;
  for (let index = 0; index < extraCount; index += 1) {
    const cursor = (state.restart.probeCursor + index) % remainder.length;
    batch.push(remainder[cursor]);
  }
  state.restart.probeCursor =
    (state.restart.probeCursor + extraCount) % remainder.length;
  return batch;
}

function buildAppPageUrl(url, page = "") {
  const target = new URL(url);
  if (isValidPage(page)) {
    target.searchParams.set("page", page);
  } else {
    target.searchParams.delete("page");
  }
  return target.toString();
}

function restartRedirectUrl(url) {
  const target = new URL(buildAppPageUrl(url, restartReturnPage));
  target.searchParams.set("restart", String(Date.now()));
  return target.toString();
}

function updateRestartScreen(phaseKey, detailText = "") {
  const phaseText = t(phaseKey);
  $("restart-phase").textContent = phaseText;
  $("restart-title").textContent = state.restart.updateTransaction ? t("updateRestartTitle") : t("restartTitle");
  $("restart-message").textContent = state.restart.updateTransaction ? t("updateRestartMessage") : t("restartMessage");
  $("restart-status-detail").textContent = detailText || phaseText;
}

function updateRestartElapsed() {
  if (!state.restarting) return;
  const seconds = Math.max(
    0,
    Math.round((Date.now() - (state.restart.startedAt || Date.now())) / 1000),
  );
  $("restart-elapsed").textContent = t("restartElapsed", { seconds });
  if (seconds >= 8) {
    $("restart-open-link").classList.remove("hidden");
  }
}

async function probeRestartUrl(url, timeoutMs = 1400) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    await fetch(`${new URL("/api/v1/health", url)}?restart_probe=${Date.now()}`, {
      mode: "no-cors",
      cache: "no-store",
      signal: controller.signal,
    });
    return url;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

async function firstReachableRestartUrl(candidates) {
  if (!candidates.length) return null;
  return new Promise((resolve) => {
    let pending = candidates.length;
    let settled = false;
    candidates.forEach((url) => {
      probeRestartUrl(url).then((result) => {
        if (settled) return;
        if (result) {
          settled = true;
          resolve(result);
          return;
        }
        pending -= 1;
        if (pending <= 0) {
          resolve(null);
        }
      });
    });
  });
}

async function pollRestartStatus() {
  if (!state.restarting) return;
  updateRestartScreen("restartPhaseWaiting", t("restartStatusPolling"));
  updateRestartElapsed();
  const candidates = nextRestartProbeBatch();
  const reachableUrl = await firstReachableRestartUrl(candidates);
  if (reachableUrl) {
    updateRestartScreen("restartPhaseConnecting", reachableUrl);
    window.location.replace(restartRedirectUrl(reachableUrl));
    return;
  }
  state.restart.timer = setTimeout(pollRestartStatus, 1200);
}

function showRestartScreen(payload) {
  clearRestartTimer();
  stopLogPolling();
  stopTaskPolling();
  resetTaskState({ render: false });
  state.restarting = true;
  state.restart.startedAt = Date.now();
  state.restart.updateTransaction = payload.transaction_id || "";
  state.restart.targetUrl =
    payload.configured_webui_url || payload.webui_url || window.location.origin + "/";
  state.restart.probeUrls = normalizeRestartProbeUrls(
    payload.restart_probe_urls,
    state.restart.targetUrl,
  );
  state.restart.probeCursor = 0;
  document.body.classList.add("restarting");
  $("restart-target-url").textContent = state.restart.targetUrl;
  $("restart-open-link").href = buildAppPageUrl(
    state.restart.targetUrl,
    restartReturnPage,
  );
  $("restart-open-link").classList.add("hidden");
  $("restart-screen").classList.remove("hidden");
  updateRestartScreen(
    payload.transaction_id ? "updateRestartPhase" : "restartPhaseStopping",
    payload.transaction_id ? t("updateRestartPreparing") : t("restartStatusPreparing"),
  );
  updateRestartElapsed();
  state.restart.timer = setTimeout(pollRestartStatus, 700);
}

async function loadSettings() {
  try {
    if (!$("settings-access-base-url") || !$("settings-access-port")) {
      window.location.reload();
      return;
    }
    const data = await api("/settings");
    state.settings = data;
    if ($("sidebar-version")) $("sidebar-version").textContent = data.version ? `v${data.version}` : "v—";
    $("settings-access-url").textContent = formatRuntimePendingValue(
      data.webui_url || data.access_url || "—",
      data.configured_webui_url || data.webui_url || data.access_url || "—",
    );
    $("settings-access-base-url").value = data.access_base_url || "http://127.0.0.1";
    $("settings-port").value = data.configured_port || 8765;
    $("settings-actual-port").textContent = formatRuntimePendingValue(
      data.actual_port,
      data.configured_port,
    );
    $("settings-api-access-url").textContent = formatRuntimePendingValue(
      data.api_access_url || "—",
      data.configured_api_access_url || data.api_access_url || "—",
    );
    $("settings-access-port").value = data.configured_access_port || 8766;
    $("settings-actual-access-port").textContent = formatRuntimePendingValue(
      data.actual_access_port,
      data.configured_access_port,
    );
    $("settings-login-mode").textContent =
      data.login_password_enabled ? t("loginModePassword") : t("loginModeApiKey");
    $("settings-password").value = "";
    $("settings-clear-password").checked = false;
    $("settings-runtime-idle-minutes").value = data.runtime_residency?.idle_minutes || 30;
    $("settings-runtime-max-non-default").value = data.runtime_residency?.max_non_default_runtimes || 4;
    $("settings-note").textContent = t("settingsPortRestartHint");
    await loadUpdateStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

function updateRelation(tag, currentTag) {
  const parse = (value) => String(value || "").replace(/^v/, "").split(/[.-]/).map((part) => /^\d+$/.test(part) ? Number(part) : part);
  const left = parse(tag);
  const right = parse(currentTag);
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    if (a === b) continue;
    if (typeof a === "number" && typeof b === "number") return a > b ? 1 : -1;
    return String(a).localeCompare(String(b));
  }
  return 0;
}

function updateAction(tag, currentTag) {
  const relation = updateRelation(tag, currentTag);
  return relation > 0 ? "update" : relation < 0 ? "rollback" : "reinstall";
}

function renderUpdateStatus(payload) {
  state.updates.status = payload;
  const currentTag = payload.current_tag || (payload.current_version ? `v${payload.current_version}` : "v—");
  if ($("sidebar-version")) $("sidebar-version").textContent = currentTag;
  if ($("update-current-version")) $("update-current-version").textContent = currentTag;
  if ($("update-latest-version")) $("update-latest-version").textContent = payload.latest_tag || currentTag;
  if ($("update-checked-at")) {
    $("update-checked-at").textContent = payload.checked_at
      ? new Date(payload.checked_at * 1000).toLocaleString()
      : "—";
  }
  if ($("update-check-state")) {
    $("update-check-state").textContent = payload.error
      ? t("updateCheckUnavailable")
      : payload.update_available
        ? t("updateAvailable")
        : t("upToDate");
  }
  $("update-available-badge")?.classList.toggle("hidden", !payload.update_available);
}

async function loadUpdateStatus({ refresh = false } = {}) {
  const payload = await api(`/updates/status${refresh ? "?refresh=true" : ""}`);
  renderUpdateStatus(payload);
  return payload;
}

function renderReleaseList() {
  const currentTag = state.updates.status?.current_tag || "v—";
  const target = $("updates-release-list");
  if (!state.updates.releases.length) {
    target.innerHTML = `<div class="empty">${escapeHtml(t("noCompatibleReleases"))}</div>`;
    return;
  }
  target.innerHTML = state.updates.releases.map((release) => {
    const action = updateAction(release.tag_name, currentTag);
    const actionKey = action === "update" ? "updateAction" : action === "rollback" ? "rollbackAction" : "reinstallAction";
    return `<article class="update-release-card">
      <div class="update-release-main">
        <div class="update-release-title"><strong>${escapeHtml(release.tag_name)}</strong>${release.prerelease ? `<span class="pill warning">${escapeHtml(t("prerelease"))}</span>` : ""}</div>
        <time>${escapeHtml(release.published_at ? new Date(release.published_at).toLocaleString() : "—")}</time>
        <p>${escapeHtml(release.notes || t("noReleaseNotes"))}</p>
      </div>
      <button type="button" class="${action === "rollback" ? "ghost" : "primary"}" data-update-tag="${escapeHtml(release.tag_name)}" data-update-action="${action}">${escapeHtml(t(actionKey))}</button>
    </article>`;
  }).join("");
  target.querySelectorAll("[data-update-tag]").forEach((button) => {
    button.onclick = () => switchVersion(button.dataset.updateTag, button.dataset.updateAction);
  });
}

async function refreshUpdateReleases({ refresh = true } = {}) {
  const query = refresh ? "?refresh=true" : "";
  const payload = await api(`/updates/releases${query}`);
  state.updates.releases = payload.releases || [];
  await loadUpdateStatus({ refresh: false });
  renderReleaseList();
  return payload;
}

async function openVersionSelector() {
  const overlay = $("updates-modal");
  overlay.classList.remove("hidden");
  $("updates-release-list").innerHTML = `<div class="empty">${escapeHtml(t("updateLoading"))}</div>`;
  try {
    await refreshUpdateReleases({ refresh: false });
  } catch (error) {
    overlay.classList.add("hidden");
    throw error;
  }
}

async function switchVersion(tag, action) {
  if (state.updates.switching) return;
  const actionKey = action === "update" ? "updateAction" : action === "rollback" ? "rollbackAction" : "reinstallAction";
  const confirmed = await confirmDialog({
    title: t("switchVersionTitle"),
    message: t("switchVersionConfirm", { action: t(actionKey), tag }),
    confirmText: t(actionKey),
    danger: action === "rollback",
  });
  if (!confirmed) return;
  state.updates.switching = true;
  $("updates-modal").classList.add("update-switching");
  $("updates-modal-busy").classList.remove("hidden");
  $("updates-release-list").querySelectorAll("button").forEach((button) => { button.disabled = true; });
  try {
    const payload = await api("/updates/switch", {
      method: "POST",
      body: JSON.stringify({ tag_name: tag }),
    });
    localStorage.setItem("personalityrag_update_transaction", payload.transaction_id || "");
    showRestartScreen(payload);
  } catch (error) {
    state.updates.switching = false;
    $("updates-modal").classList.remove("update-switching");
    $("updates-modal-busy").classList.add("hidden");
    renderReleaseList();
    throw error;
  }
}

async function checkLastUpdateTransaction() {
  const transactionId = localStorage.getItem("personalityrag_update_transaction") || "";
  if (!transactionId) return null;
  try {
    const payload = await api(`/updates/transactions/${encodeURIComponent(transactionId)}`);
    if (payload.status === "completed") {
      localStorage.removeItem("personalityrag_update_transaction");
      toast(t("versionSwitchCompleted", { tag: payload.target_tag || payload.target_version }));
    } else if (payload.status === "rolled_back" || payload.status === "failed") {
      localStorage.removeItem("personalityrag_update_transaction");
      toast(t("versionSwitchRolledBack"), true);
    } else if (payload.status === "recovery_required") {
      toast(t("versionSwitchRecoveryRequired"), true);
    }
    return payload;
  } catch (error) {
    if (error?.status === 404) localStorage.removeItem("personalityrag_update_transaction");
    return null;
  }
}


function formatRuntimePendingValue(currentValue, nextValue) {
  const currentText = String(currentValue ?? "").trim();
  const nextText = String(nextValue ?? "").trim();
  if (!currentText && !nextText) return "—";
  if (!currentText) return nextText || "—";
  if (!nextText || currentText === nextText) return currentText;
  return t("settingsRuntimePendingValue", {
    current: currentText,
    next: nextText,
  });
}


  return {
    requestBackupPassword,
    requestBackupExportScope,
    settingsDraft,
    hasUnsavedSettingsChanges,
    showRestartScreen,
    loadSettings,
    loadUpdateStatus,
    openVersionSelector,
    refreshUpdateReleases,
    checkLastUpdateTransaction,
  };
}
