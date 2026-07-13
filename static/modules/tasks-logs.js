export function createTaskLogController({ $, state, t, api, toast, escapeHtml, formatBytes, selectedLibrary, loadLibraries, loadGraph, loadMemories, loadSystem, confirmDialog, markLibraryIndexConflict, clearLibraryIndexConflict, refreshLibraryContext }) {
function taskKindLabel(kind) {
  return {
    index_rebuild: t("taskKindIndexRebuild"),
    graph_rebuild: t("taskKindGraphRebuild"),
    library_copy: t("taskKindLibraryCopy"),
    livingmemory_import: t("taskKindImport"),
    livingmemory_migration: t("taskKindMigration"),
    memory_create: t("taskKindMemoryCreate"),
    memory_update: t("taskKindMemoryUpdate"),
    memory_delete: t("taskKindMemoryDelete"),
  }[kind] || kind || "Task";
}

function taskStatusLabel(status) {
  return {
    queued: t("taskQueued"),
    running: t("taskRunning"),
    pausing: t("taskPausing"),
    paused: t("taskPaused"),
    interrupted: t("taskInterrupted"),
    stopping: t("taskStopping"),
    completed: t("taskCompleted"),
    failed: t("taskFailed"),
    stopped: t("taskStopped"),
    cancelled: t("taskCancelled"),
  }[status] || status || "";
}

function taskStatusMark(status) {
  if (status === "completed") return "✅";
  if (status === "failed" || status === "cancelled" || status === "stopped") return "⏹";
  if (status === "running") return "▶";
  if (status === "paused" || status === "interrupted") return "⏸";
  return "⏳";
}

function taskTimestamp() {
  return Date.now() / 1000;
}

function isActiveTask(job) {
  return ["queued", "running", "pausing", "paused", "interrupted", "stopping"].includes(String(job?.status || ""));
}

function taskReasonLabel(reason) {
  return {
    manual: t("taskReasonManual"),
    shutdown: t("taskReasonShutdown"),
    process_interrupted: t("taskReasonProcessInterrupted"),
    provider_unavailable: t("taskReasonProviderUnavailable"),
    rollback_failed: t("taskReasonRollbackFailed"),
    checkpoint_conflict: t("taskReasonCheckpointConflict"),
    checkpoint_corrupt: t("taskReasonCheckpointCorrupt"),
    source_changed: t("taskReasonSourceChanged"),
  }[String(reason || "")] || "";
}

function taskActionButtons(job) {
  const capabilities = job?.capabilities || {};
  const buttons = [];
  if (capabilities.pause) buttons.push(["pause", t("taskPause"), "ghost"]);
  if (capabilities.resume) buttons.push(["resume", t("taskResume"), "primary"]);
  if (capabilities.stop) buttons.push(["stop", t("taskStop"), "danger"]);
  if (capabilities.cancel) buttons.push(["cancel", t("taskCancel"), "ghost"]);
  if (!buttons.length) return "";
  return `<div class="task-actions">${buttons.map(([action, label, style]) =>
    `<button class="${style}" type="button" data-job-action="${action}" data-job-id="${escapeHtml(String(job.id || ""))}">${escapeHtml(label)}</button>`,
  ).join("")}</div>`;
}

function mergeTaskList(fetched = [], scope = "active") {
  const fetchedIds = new Set(fetched.map((job) => job.id).filter(Boolean));
  const now = taskTimestamp();
  state.tasks.optimistic = state.tasks.optimistic.filter((job) => {
    if (fetchedIds.has(job.id)) return false;
    if (!String(job.id || "").startsWith("local-")) return false;
    return now - Number(job.created_at || now) < 600;
  });
  const optimistic = state.tasks.optimistic.filter((job) =>
    scope === "finished" ? !isActiveTask(job) : isActiveTask(job),
  );
  return [...optimistic, ...fetched];
}

function syncVisibleTasks() {
  state.tasks.active = mergeTaskList(state.tasks.active.filter((job) => !job.optimistic), "active");
  state.tasks.finished = mergeTaskList(state.tasks.finished.filter((job) => !job.optimistic), "finished");
}

function addOptimisticTask({
  id = "",
  kind = "index_rebuild",
  libraryId = "",
  message = "",
  markLibraryConflict = false,
} = {}) {
  const effectiveLibraryId = libraryId || state.selectedLibraryId || "";
  const job = {
    id: id || `local-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    kind,
    library_id: effectiveLibraryId,
    status: "queued",
    progress: 0,
    message: message || t("taskSubmitting"),
    created_at: taskTimestamp(),
    updated_at: taskTimestamp(),
    optimistic: true,
  };
  if (markLibraryConflict && effectiveLibraryId) {
    markLibraryIndexConflict(effectiveLibraryId);
  }
  state.tasks.optimistic = [
    job,
    ...state.tasks.optimistic.filter((item) => item.id !== job.id),
  ];
  syncVisibleTasks();
  if (state.page === "logs") renderTasks();
  return job.id;
}

function updateOptimisticTask(id, patch = {}) {
  if (!id) return;
  state.tasks.optimistic = state.tasks.optimistic.map((job) =>
    job.id === id
      ? { ...job, ...patch, updated_at: taskTimestamp(), optimistic: true }
      : job,
  );
  syncVisibleTasks();
  if (state.page === "logs") renderTasks();
}

function removeOptimisticTask(id) {
  if (!id) return;
  state.tasks.optimistic = state.tasks.optimistic.filter((job) => job.id !== id);
  syncVisibleTasks();
  if (state.page === "logs") renderTasks();
}

function upsertJobSnapshot(job) {
  if (!job?.id) return;
  state.tasks.optimistic = state.tasks.optimistic.filter((item) => item.id !== job.id);
  const active = isActiveTask(job);
  const target = active ? "active" : "finished";
  const other = active ? "finished" : "active";
  state.tasks[target] = [
    job,
    ...state.tasks[target].filter((item) => item.id !== job.id),
  ];
  state.tasks[other] = state.tasks[other].filter((item) => item.id !== job.id);
  if (state.page === "logs") renderTasks();
}

function taskHistoryCollapsedHeight(list) {
  const items = Array.from(list.querySelectorAll(".task-item"));
  if (items.length < 3) return 0;
  const styles = window.getComputedStyle(list);
  const gap = Number.parseFloat(styles.rowGap || styles.gap || "0") || 0;
  const heights = items.slice(0, 3).map((item) => item.offsetHeight || 0);
  if (heights.some((value) => value <= 0)) return 0;
  return Math.round(heights[0] + gap + heights[1] + gap + heights[2] * 0.5);
}

function applyTaskHistoryCollapseState() {
  const panel = $("task-history-panel");
  const content = $("task-history-primary");
  const list = $("task-list");
  const toggle = $("task-history-toggle");
  if (!panel || !content || !list || !toggle) return;
  const items = Array.from(list.querySelectorAll(".task-item"));
  const finishedScope = state.tasks.scope === "finished";

  if (!finishedScope || items.length < 3) {
    panel.classList.remove("task-history-panel-collapsed", "task-history-panel-expanded");
    panel.style.removeProperty("--task-history-collapsed-height");
    toggle.classList.add("hidden");
    toggle.classList.remove("expanded");
    toggle.setAttribute("aria-expanded", "false");
    return;
  }

  const collapsedHeight = taskHistoryCollapsedHeight(list);
  const overflowing = collapsedHeight > 0 && content.scrollHeight > collapsedHeight + 24;
  if (!overflowing) {
    panel.classList.remove("task-history-panel-collapsed", "task-history-panel-expanded");
    panel.style.removeProperty("--task-history-collapsed-height");
    toggle.classList.add("hidden");
    toggle.classList.remove("expanded");
    toggle.setAttribute("aria-expanded", "false");
    return;
  }

  panel.style.setProperty("--task-history-collapsed-height", `${collapsedHeight}px`);
  panel.classList.toggle("task-history-panel-collapsed", !state.tasks.finishedExpanded);
  panel.classList.toggle("task-history-panel-expanded", state.tasks.finishedExpanded);
  toggle.classList.toggle("expanded", state.tasks.finishedExpanded);
  toggle.classList.remove("hidden");
  const title = state.tasks.finishedExpanded ? t("collapseLibraryCard") : t("expandLibraryCard");
  toggle.title = title;
  toggle.setAttribute("aria-label", title);
  toggle.setAttribute("aria-expanded", state.tasks.finishedExpanded ? "true" : "false");
}

function renderFinishedTaskClearButton() {
  const button = $("tasks-finished-clear");
  if (!button) return;
  const finishedScope = state.tasks.scope === "finished";
  const hasItems = state.tasks.finished.length > 0;
  button.classList.toggle("hidden", !finishedScope);
  button.disabled = !finishedScope || !hasItems;
}

function hideTrackedJobProgress() {
  const box = $("job-progress");
  if (!box) return;
  box.classList.add("hidden");
  const bar = box.querySelector("div");
  const label = box.querySelector("span");
  if (bar) {
    bar.style.width = "0%";
  }
  if (label) {
    label.textContent = "";
  }
}

function resetTaskState({ render = true } = {}) {
  stopTaskPolling();
  state.tasks.watchers.forEach((watcher) => {
    watcher.done = true;
    watcher.source?.close();
  });
  state.tasks.watchers.clear();
  state.tasks.active = [];
  state.tasks.finished = [];
  state.tasks.optimistic = [];
  state.tasks.finishedExpanded = false;
  hideTrackedJobProgress();
  if (render) {
    renderTasks();
  } else {
    applyTaskHistoryCollapseState();
  }
}

function renderTasks() {
  const list = $("task-list");
  if (!list) return;
  syncVisibleTasks();
  const items = state.tasks.scope === "finished" ? state.tasks.finished : state.tasks.active;
  document.querySelectorAll("[data-task-scope]").forEach((button) => {
    button.classList.toggle("active", button.dataset.taskScope === state.tasks.scope);
  });
  renderFinishedTaskClearButton();
  if (!items.length) {
    list.innerHTML = `<div class="task-empty">${escapeHtml(t(state.tasks.scope === "finished" ? "noFinishedTasks" : "noActiveTasks"))}</div>`;
    applyTaskHistoryCollapseState();
    return;
  }
  list.innerHTML = items
    .map((job) => {
      const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
      const status = String(job.status || "");
      const message = job.error || job.message || "";
      const library = job.library_id || "—";
      const reason = taskReasonLabel(job.status_reason);
      const checkpoint = job.checkpoint || {};
      const checkpointText = checkpoint.phase
        ? t("taskCheckpoint", {
            phase: checkpoint.phase,
            completed: checkpoint.completed_documents ?? checkpoint.completed_graph_entries ?? 0,
          })
        : "";
      return `<article class="task-item task-status-${escapeHtml(status)}">
        <div class="task-head">
          <div>
            <div class="task-title"><span>${taskStatusMark(status)}</span><span>${escapeHtml(taskKindLabel(job.kind))}</span></div>
            <div class="task-meta">
              <span>job ${escapeHtml(String(job.id || "").slice(0, 8))}</span>
              <span>${escapeHtml(library)}</span>
              <span>${escapeHtml(taskStatusLabel(status))}</span>
              ${reason ? `<span>${escapeHtml(reason)}</span>` : ""}
              ${checkpointText ? `<span>${escapeHtml(checkpointText)}</span>` : ""}
            </div>
          </div>
          <b>${Math.round(progress * 100)}%</b>
        </div>
        <div class="task-progress"><div style="width:${progress * 100}%"></div><span>${escapeHtml(message)}</span></div>
        ${taskActionButtons(job)}
      </article>`;
    })
    .join("");
  applyTaskHistoryCollapseState();
}

async function controlTask(jobId, action, button) {
  if (!jobId || !action) return;
  if (action === "stop") {
    if (!(await confirmDialog({
      title: t("confirmStopTaskTitle"),
      message: t("confirmStopTask"),
      confirmText: t("taskStop"),
      danger: true,
    }))) return;
  }
  if (action === "cancel") {
    if (!(await confirmDialog({
      title: t("confirmCancelTaskTitle"),
      message: t("confirmCancelTask"),
      confirmText: t("taskCancel"),
    }))) return;
  }
  if (button) button.disabled = true;
  try {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}/${action}`, { method: "POST" });
    upsertJobSnapshot(job);
    toast(t("taskControlAccepted"));
    watchJob(jobId);
    await loadTasks(state.tasks.scope);
  } catch (error) {
    toast(error.message || String(error), true);
  } finally {
    if (button) button.disabled = false;
  }
}

$("task-list")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-job-action]");
  if (!button) return;
  controlTask(button.dataset.jobId, button.dataset.jobAction, button);
});

async function clearFinishedTasks() {
  const button = $("tasks-finished-clear");
  if (!(await confirmDialog({
    title: t("confirmClearFinishedTasksTitle"),
    message: t("confirmClearFinishedTasks"),
    confirmText: t("clearFinishedTasks"),
    danger: true,
  }))) return;
  if (button) button.disabled = true;
  try {
    const payload = await api("/jobs/finished/clear", { method: "POST" });
    state.tasks.finished = [];
    state.tasks.finishedExpanded = false;
    renderTasks();
    toast(t("finishedTasksCleared", { count: payload.cleared || 0 }));
    if (state.page === "logs") {
      await loadTasks("finished");
    }
  } catch (error) {
    toast(error.message || String(error), true);
  } finally {
    renderFinishedTaskClearButton();
  }
}

$("tasks-finished-clear")?.addEventListener("click", () => {
  clearFinishedTasks().catch((error) => {
    toast(error.message || String(error), true);
  });
});

async function loadTasks(scope = state.tasks.scope) {
  const data = await api(`/jobs?scope=${encodeURIComponent(scope)}`, {
    suppressOperationalError: true,
  });
  if (scope === "active") {
    state.tasks.active = mergeTaskList(data.items || [], "active");
  } else if (scope === "finished") {
    state.tasks.finished = mergeTaskList(data.items || [], "finished");
  } else {
    const items = data.items || [];
    state.tasks.active = mergeTaskList(items.filter((job) => isActiveTask(job)), "active");
    state.tasks.finished = mergeTaskList(items.filter((job) => !isActiveTask(job)), "finished");
  }
  renderTasks();
}

function startTaskPolling() {
  if (state.tasks.polling) return;
  state.tasks.polling = true;
  const poll = async () => {
    if (!state.tasks.polling) return;
    try {
      await loadTasks("active");
      if (state.tasks.scope === "finished") {
        await loadTasks("finished");
      } else {
        renderTasks();
      }
    } catch (error) {
      console.error("task polling failed", error);
    } finally {
      if (state.tasks.polling) {
        state.tasks.pollTimer = setTimeout(poll, 1200);
      }
    }
  };
  poll();
}

function stopTaskPolling() {
  state.tasks.polling = false;
  if (state.tasks.pollTimer) {
    clearTimeout(state.tasks.pollTimer);
    state.tasks.pollTimer = null;
  }
}

$("task-history-toggle")?.addEventListener("click", () => {
  state.tasks.finishedExpanded = !state.tasks.finishedExpanded;
  applyTaskHistoryCollapseState();
});

function renderLogAutoScrollState() {
  const toggle = $("log-auto-scroll");
  const label = $("log-auto-scroll-label");
  if (toggle) toggle.checked = state.logs.autoScroll;
  if (label) label.textContent = state.logs.autoScroll ? t("logAutoScrollOn") : t("logAutoScrollOff");
}

function renderLogs({ scrollToBottom = false } = {}) {
  const consoleElement = $("log-console");
  if (!consoleElement) return;
  const activeLevels = state.logs.activeLevels;
  const visible = state.logs.items.filter((item) => activeLevels.has(item.level));
  if (!visible.length) {
    consoleElement.innerHTML = `<div class="log-empty">${escapeHtml(t("logsEmpty"))}</div>`;
  } else {
    consoleElement.innerHTML = visible
      .map((item) => `
        <div class="log-entry log-${String(item.level || "INFO").toLowerCase()}">
          <span class="log-entry-level">${escapeHtml(item.level || "")}</span>
          <span class="log-entry-line">${escapeHtml(item.line || item.message || "")}</span>
        </div>
      `)
      .join("");
  }
  const summary = state.logs.buffer || {};
  $("log-meta").textContent = t("logsMeta", {
    count: state.logs.items.length,
    max: state.logs.maxEntries,
    bytes: formatBytes(summary.bytes || 0),
    maxBytes: formatBytes(summary.max_bytes || 0),
  });
  document.querySelectorAll(".log-filter[data-log-level]").forEach((button) => {
    button.classList.toggle("active", state.logs.activeLevels.has(button.dataset.logLevel));
  });
  renderLogAutoScrollState();
  if (scrollToBottom && state.logs.autoScroll) {
    consoleElement.scrollTop = consoleElement.scrollHeight;
  }
}

async function loadLogs({ reset = false, waitSeconds = 0, signal = null } = {}) {
  const afterId = reset ? 0 : state.logs.lastId;
  const generation = state.logs.generation;
  const query = new URLSearchParams({
    after_id: String(afterId),
    wait: String(Math.max(0, Number(waitSeconds) || 0)),
  });
  const payload = await api(`/logs?${query.toString()}`, {
    signal,
    suppressOperationalError: true,
  });
  if (generation !== state.logs.generation) return;
  if (reset || payload.reset) {
    state.logs.items = [];
    state.logs.lastId = 0;
  }
  const incoming = Array.isArray(payload.items) ? payload.items : [];
  if (incoming.length) {
    state.logs.items.push(...incoming);
    state.logs.lastId = Number(incoming[incoming.length - 1].id) || state.logs.lastId;
  }
  state.logs.maxEntries = Number(payload.max_entries) || state.logs.maxEntries;
  state.logs.buffer = payload.buffer || state.logs.buffer || {};
  if (state.logs.items.length > state.logs.maxEntries) {
    state.logs.items = state.logs.items.slice(-state.logs.maxEntries);
  }
  renderLogs({ scrollToBottom: incoming.length > 0 });
}

function startLogPolling() {
  if (state.logs.polling) return;
  state.logs.polling = true;
  loadLogs({ reset: true }).catch((error) => console.error("log reset failed", error));
  const poll = async () => {
    if (!state.logs.polling) return;
    const controller = new AbortController();
    state.logs.abortController = controller;
    try {
      await loadLogs({ waitSeconds: 25, signal: controller.signal });
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("log polling failed", error);
      }
    } finally {
      if (state.logs.abortController === controller) {
        state.logs.abortController = null;
      }
      if (state.logs.polling) {
        state.logs.pollTimer = setTimeout(poll, 250);
      }
    }
  };
  poll();
}

function stopLogPolling() {
  state.logs.polling = false;
  if (state.logs.pollTimer) {
    clearTimeout(state.logs.pollTimer);
    state.logs.pollTimer = null;
  }
  if (state.logs.abortController) {
    state.logs.abortController.abort();
    state.logs.abortController = null;
  }
}

async function clearLogs() {
  if (!(await confirmDialog({
    title: t("confirmClearLogsTitle"),
    message: t("confirmClearLogs"),
    confirmText: t("clearLogs"),
    danger: true,
  }))) return;
  const payload = await api("/logs/clear", { method: "POST" });
  state.logs.generation += 1;
  state.logs.items = [];
  state.logs.lastId = 0;
  state.logs.maxEntries = Number(payload.max_entries) || state.logs.maxEntries;
  state.logs.buffer = { bytes: 0, max_bytes: state.logs.buffer?.max_bytes || 0 };
  renderLogs();
  toast(t("logsCleared", { count: payload.cleared || 0 }));
  if (state.page === "logs") {
    await loadLogs({ reset: true });
  }
}

function trackQueuedJob(payload, options = {}) {
  const jobId = payload?.job_id;
  const tempId = options.tempId || "";
  const kind = options.kind || "index_rebuild";
  const libraryId = options.libraryId || state.selectedLibraryId || "";
  if (tempId && jobId) {
    updateOptimisticTask(tempId, {
      id: jobId,
      kind,
      library_id: libraryId,
      status: "queued",
      progress: 0,
      message: t("taskQueued"),
    });
  } else if (tempId && !jobId) {
    removeOptimisticTask(tempId);
  }
  if (!jobId) return;
  addOptimisticTask({
    id: jobId,
    kind,
    libraryId,
    message: t("taskQueued"),
  });
  watchJob(jobId);
  if (state.page === "logs") {
    loadTasks("active").catch((error) => console.error("task refresh failed", error));
  }
}

function renderTrackedJobProgress(job) {
  const box = $("job-progress");
  if (!box) return;
  const bar = box.querySelector("div");
  const label = box.querySelector("span");
  box.classList.remove("hidden");
  if (bar) {
    bar.style.width = `${job.progress * 100}%`;
  }
  if (label) {
    label.textContent = `${t("progressPrefix")} ${Math.round(job.progress * 100)}% · ${job.message}`;
  }
}

async function finishTrackedJob(job) {
  const terminalMessage =
    job.status === "completed"
      ? t("jobCompleted")
      : job.status === "failed"
        ? t("jobFailed")
        : job.status === "stopped"
          ? t("jobStopped")
          : t("jobCancelled");
  toast(terminalMessage, job.status !== "completed");
  await loadLibraries(state.page === "libraries");
  await loadSystem();
  if (state.page === "graph") await loadGraph();
  if (state.page === "memory") await loadMemories();
  if (job.status === "completed") {
    clearLibraryIndexConflict(job.library_id);
  }
  refreshLibraryContext();
  if (state.page === "logs") {
    loadTasks(state.tasks.scope).catch((error) => console.error("task refresh failed", error));
  }
}

function watchJob(id) {
  if (!id || state.tasks.watchers.has(id)) return;
  const watcher = { source: null, polling: false, done: false };
  state.tasks.watchers.set(id, watcher);

  const accept = async (job) => {
    if (!job || watcher.done) return;
    upsertJobSnapshot(job);
    renderTrackedJobProgress(job);
    if (!["completed", "failed", "stopped", "cancelled"].includes(job.status)) return;
    watcher.done = true;
    watcher.source?.close();
    state.tasks.watchers.delete(id);
    try {
      await finishTrackedJob(job);
    } catch (error) {
      console.error("task completion refresh failed", error);
    }
  };

  const pollFallback = async () => {
    if (watcher.polling || watcher.done) return;
    watcher.polling = true;
    while (!watcher.done) {
      try {
        await accept(await api("/jobs/" + encodeURIComponent(id)));
      } catch (error) {
        console.error("task polling fallback failed", error);
      }
      if (!watcher.done) {
        await new Promise((resolve) => setTimeout(resolve, 800));
      }
    }
  };

  if (!("EventSource" in window)) {
    pollFallback();
    return;
  }

  const source = new EventSource(`/api/v1/jobs/${encodeURIComponent(id)}/events`);
  watcher.source = source;
  source.onmessage = (event) => {
    try {
      accept(JSON.parse(event.data)).catch((error) => {
        console.error("task SSE update failed", error);
      });
    } catch (error) {
      console.error("task SSE payload failed", error);
    }
  };
  source.onerror = () => {
    source.close();
    watcher.source = null;
    pollFallback();
  };
}

  return {
    loadTasks,
    startTaskPolling,
    stopTaskPolling,
    resetTaskState,
    applyTaskHistoryCollapseState,
    renderTasks,
    loadLogs,
    startLogPolling,
    stopLogPolling,
    clearLogs,
    trackQueuedJob,
    addOptimisticTask,
    updateOptimisticTask,
    removeOptimisticTask,
    renderLogs,
    renderLogAutoScrollState,
  };
}
