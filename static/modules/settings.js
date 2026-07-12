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
  $("restart-title").textContent = t("restartTitle");
  $("restart-message").textContent = t("restartMessage");
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
    if (
      state.restart.requireOfflineTransition
      && !state.restart.sawOfflineTransition
    ) {
      state.restart.timer = setTimeout(pollRestartStatus, 350);
      return;
    }
    updateRestartScreen("restartPhaseConnecting", reachableUrl);
    window.location.replace(restartRedirectUrl(reachableUrl));
    return;
  }
  state.restart.sawOfflineTransition = true;
  state.restart.timer = setTimeout(pollRestartStatus, 1200);
}

function showRestartScreen(payload) {
  clearRestartTimer();
  stopLogPolling();
  stopTaskPolling();
  resetTaskState({ render: false });
  state.restarting = true;
  state.restart.startedAt = Date.now();
  const containerRestart = payload.restart_strategy === "container";
  state.restart.targetUrl = containerRestart
    ? `${window.location.origin}/`
    : payload.configured_webui_url || payload.webui_url || `${window.location.origin}/`;
  state.restart.probeUrls = containerRestart
    ? [state.restart.targetUrl]
    : normalizeRestartProbeUrls(payload.restart_probe_urls, state.restart.targetUrl);
  state.restart.probeCursor = 0;
  state.restart.requireOfflineTransition = containerRestart;
  state.restart.sawOfflineTransition = false;
  document.body.classList.add("restarting");
  $("restart-target-url").textContent = state.restart.targetUrl;
  $("restart-open-link").href = buildAppPageUrl(
    state.restart.targetUrl,
    restartReturnPage,
  );
  $("restart-open-link").classList.add("hidden");
  $("restart-screen").classList.remove("hidden");
  updateRestartScreen("restartPhaseStopping", t("restartStatusPreparing"));
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
    $("settings-access-url").textContent = formatRuntimePendingValue(
      data.webui_url || data.access_url || "—",
      data.configured_webui_url || data.webui_url || data.access_url || "—",
    );
    $("settings-access-base-url").value = data.access_base_url || "http://127.0.0.1";
    $("settings-port").value = data.configured_port || 8765;
    const managedSettings = new Set(data.managed_settings || []);
    $("settings-port").disabled = managedSettings.has("port");
    $("settings-actual-port").textContent = formatRuntimePendingValue(
      data.actual_port,
      data.configured_port,
    );
    $("settings-api-access-url").textContent = formatRuntimePendingValue(
      data.api_access_url || "—",
      data.configured_api_access_url || data.api_access_url || "—",
    );
    $("settings-access-port").value = data.configured_access_port || 8766;
    $("settings-access-port").disabled = managedSettings.has("access_port");
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
    $("settings-note").textContent = data.deployment_mode === "docker"
      ? t("settingsDockerManagedPorts")
      : t("settingsPortRestartHint");
  } catch (error) {
    toast(error.message, true);
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
  };
}
