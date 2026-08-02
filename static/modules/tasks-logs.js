export function createTaskLogController({ $, state, t, api, toast, escapeHtml, formatBytes, selectedDatabase, onTaskFinished, confirmDialog, markDatabaseIndexConflict, clearDatabaseIndexConflict, refreshDatabaseContext }) {
let taskPoll = null;
let logPoll = null;
let logReset = null;
let taskDetailRequest = 0;
let taskDetailCollapseFrame = 0;
let taskDetailCollapseResizeHandler = null;

function taskKindLabel(kind) {
  return {
    index_rebuild: t("taskKindIndexRebuild"),
    graph_rebuild: t("taskKindGraphRebuild"),
    library_copy: t("taskKindLibraryCopy"),
    library_backup: t("taskKindLibraryBackup"),
    livingmemory_import: t("taskKindImport"),
    livingmemory_migration: t("taskKindMigration"),
    memory_create: t("taskKindMemoryCreate"),
    memory_update: t("taskKindMemoryUpdate"),
    memory_delete: t("taskKindMemoryDelete"),
    text_media_index_rebuild: t("taskKindTextMediaIndexRebuild"),
    text_media_document_ingest: t("taskKindTextMediaDocumentIngest"),
    text_media_ingest_batch: t("taskKindTextMediaIngestBatch"),
    text_media_entry_create: t("taskKindTextMediaEntryCreate"),
    text_media_entry_update: t("taskKindTextMediaEntryUpdate"),
    text_media_document_delete: t("taskKindTextMediaDocumentDelete"),
    text_media_entry_delete: t("taskKindTextMediaEntryDelete"),
    text_media_media_calibration: t("taskKindTextMediaCalibration"),
    text_media_media_descriptions_update: t("taskKindTextMediaDescriptionsUpdate"),
    text_media_image_upload: t("taskKindTextMediaImageUpload"),
    text_media_image_delete: t("taskKindTextMediaImageDelete"),
    text_media_relation_update: t("taskKindTextMediaRelationUpdate"),
    text_media_relation_delete: t("taskKindTextMediaRelationDelete"),
    text_media_visual_intent_policy_update: t("taskKindTextMediaVisualIntentPolicyUpdate"),
    tmkb_export: t("taskKindTmkbExport"),
    tmkb_import: t("taskKindTmkbImport"),
    tmkbs_export: t("taskKindTmkbsExport"),
    tmkbs_import: t("taskKindTmkbsImport"),
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
  }[String(reason || "")] || String(reason || "");
}

function taskActionButtons(job) {
  const capabilities = job?.capabilities || {};
  const buttons = [];
  if (capabilities.pause) buttons.push(["pause", t("taskPause"), "ghost"]);
  if (capabilities.resume) buttons.push(["resume", t("taskResume"), "primary"]);
  if (capabilities.stop) buttons.push(["stop", t("taskStop"), "danger"]);
  if (capabilities.cancel) buttons.push(["cancel", t("taskCancel"), "ghost"]);
  const controls = buttons.map(([action, label, style]) =>
    `<button class="${style}" type="button" data-job-action="${action}" data-job-id="${escapeHtml(String(job.id || ""))}">${escapeHtml(label)}</button>`,
  ).join("");
  if (!controls) return "";
  return `<div class="task-actions">${controls}</div>`;
}

function formatTaskTime(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return "—";
  return new Date(numeric * 1000).toLocaleString();
}

function formatTaskDuration(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) return "—";
  if (numeric < 1) return t("taskDurationMilliseconds", { value: Math.round(numeric * 1000) });
  if (numeric < 60) return t("taskDurationSeconds", { value: numeric.toFixed(numeric < 10 ? 2 : 1) });
  const minutes = Math.floor(numeric / 60);
  const seconds = Math.round(numeric % 60);
  return t("taskDurationMinutes", { minutes, seconds });
}

const taskDataLabelKeys = {
  database: "taskDataGroupDatabase",
  provider_binding: "taskDataGroupProviderBinding",
  provider: "taskDataGroupProvider",
  statistics: "taskDataGroupStatistics",
  indexes: "taskDataGroupIndexes",
  manifest: "taskDataGroupManifest",
  validation: "taskDataGroupValidation",
  embedding_context: "taskDataGroupEmbeddingContext",
  embedding_capability: "taskDataGroupEmbeddingCapability",
  graph_recovery: "taskDataGroupGraphRecovery",
  fts: "taskDataGroupFullTextIndex",
  before: "taskStateBefore",
  after: "taskStateAfter",
  schema_version: "taskFieldSchemaVersion",
  captured_at: "taskFieldCapturedAt",
  exists: "taskFieldExists",
  database_type: "taskDetailDatabaseType",
  database_id: "taskDetailDatabaseId",
  library_id: "taskFieldLibraryId",
  name: "taskFieldName",
  status: "taskDetailStatus",
  is_default: "taskFieldDefaultDatabase",
  created_at: "taskFieldCreatedAt",
  updated_at: "taskFieldUpdatedAt",
  saved_at: "taskFieldSavedAt",
  detected_at: "taskFieldDetectedAt",
  embedding_provider_id: "taskFieldEmbeddingProvider",
  embedding_provider_revision: "taskFieldProviderRevision",
  rerank_provider_id: "taskFieldRerankProvider",
  provider_id: "taskFieldProviderId",
  provider_revision: "taskFieldProviderRevision",
  provider_fingerprint: "taskFieldProviderFingerprint",
  provider_config_sha256: "taskFieldProviderFingerprint",
  provider_type: "taskFieldProviderType",
  configured_model: "taskFieldConfiguredModel",
  resolved_model: "taskFieldResolvedModel",
  generation: "taskFieldGeneration",
  generation_id: "taskFieldGeneration",
  loaded: "taskFieldLoaded",
  vector_count: "taskFieldVectorCount",
  document_vectors: "taskFieldDocumentVectors",
  graph_vectors: "taskFieldGraphVectors",
  media_vector_count: "taskFieldMediaVectors",
  dimension: "taskFieldDimensions",
  dimensions: "taskFieldDimensions",
  media_dimensions: "taskFieldMediaDimensions",
  total_memories: "taskFieldTotalMemories",
  document_count: "taskFieldDocumentCount",
  graph_entry_count: "taskFieldGraphEntries",
  graph_entries: "taskFieldGraphEntries",
  graph_nodes: "taskFieldGraphNodes",
  graph_edges: "taskFieldGraphEdges",
  atom_count: "taskFieldAtomCount",
  chunk_count: "taskFieldChunkCount",
  total_embedding_chunks: "taskFieldEmbeddingChunks",
  calibration_count: "taskFieldCalibrationCount",
  filename: "taskFieldFilename",
  path: "taskFieldPath",
  size_bytes: "taskFieldSize",
  sha256: "taskFieldSha256",
  reason: "taskFieldReason",
  rebuilt: "taskFieldRebuilt",
  phase: "taskFieldPhase",
  metric: "taskFieldMetric",
  rows: "taskFieldRows",
  id: "taskFieldId",
  revision: "taskFieldRevision",
  fingerprint: "taskFieldProviderFingerprint",
  format: "taskFieldFormat",
  version: "taskFieldVersion",
  description: "taskFieldDescription",
  download_token: "taskFieldDownloadToken",
};

function taskDataLabel(key) {
  const raw = String(key ?? "");
  const translationKey = taskDataLabelKeys[raw];
  if (translationKey) return t(translationKey);
  return raw.replace(/[_\-.]+/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function taskDataPathLabel(path) {
  return String(path || "").split(".").filter(Boolean).map(taskDataLabel).join(" / ");
}

function taskDataValue(value, key = "") {
  const field = String(key || "").toLowerCase();
  if (value === null || value === undefined || value === "") {
    return `<span class="task-detail-value task-detail-value-empty">${escapeHtml(t("taskDataUnset"))}</span>`;
  }
  if (typeof value === "boolean") {
    return `<span class="task-detail-value-badge ${value ? "positive" : "neutral"}">${escapeHtml(value ? t("taskDataYes") : t("taskDataNo"))}</span>`;
  }
  if (typeof value === "number") {
    let text = Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    if (/(^|_)(created_at|updated_at|captured_at|saved_at|detected_at|timestamp|finished_at|started_at)$/.test(field) && value > 1000000000) {
      text = formatTaskTime(value);
    } else if (field.endsWith("_bytes")) {
      text = formatBytes(value);
    } else if (field.endsWith("_duration_seconds")) {
      text = formatTaskDuration(value);
    }
    return `<span class="task-detail-value task-detail-value-number">${escapeHtml(text)}</span>`;
  }
  let text = String(value);
  if (field === "status") text = taskStatusLabel(text) || text;
  if (field === "reason" && text === "not_needed") text = t("taskReasonNotNeeded");
  const codeLike = /(^|_)(id|sha256|fingerprint|generation|provider|model|metric|path|filename|phase)(_|$)/.test(field) || text.startsWith("gen-");
  const longToken = /(?:sha256|fingerprint)/.test(field) && text.length > 24;
  const display = longToken ? `${text.slice(0, 12)}…${text.slice(-8)}` : text;
  return `<span class="task-detail-value${codeLike ? " task-detail-value-code" : ""}"${longToken ? ` title="${escapeHtml(text)}"` : ""}>${escapeHtml(display)}</span>`;
}

function renderTaskData(value, { emptyMessage = t("taskMetadataUnavailable"), level = 0 } = {}) {
  if (value === null || value === undefined) return `<div class="task-detail-data-empty">${escapeHtml(emptyMessage)}</div>`;
  if (Array.isArray(value)) {
    if (!value.length) return `<div class="task-detail-data-empty">${escapeHtml(t("taskDataEmptyCollection"))}</div>`;
    if (value.every((item) => item === null || typeof item !== "object")) {
      return `<div class="task-detail-chip-list">${value.map((item) => `<span>${taskDataValue(item)}</span>`).join("")}</div>`;
    }
    return `<div class="task-detail-data-collection">${value.map((item, index) => `<article class="task-detail-data-group"><h5>${escapeHtml(t("taskDataItem", { index: index + 1 }))}</h5>${renderTaskData(item, { emptyMessage, level: level + 1 })}</article>`).join("")}</div>`;
  }
  if (typeof value !== "object") return taskDataValue(value);
  const entries = Object.entries(value);
  if (!entries.length) return `<div class="task-detail-data-empty">${escapeHtml(t("taskDataEmptyCollection"))}</div>`;
  const simple = entries.filter(([, item]) => item === null || typeof item !== "object");
  const nested = entries.filter(([, item]) => item !== null && typeof item === "object");
  return `<div class="task-detail-data${level ? " nested" : ""}">
    ${simple.length ? `<dl class="task-detail-data-list">${simple.map(([key, item]) => `<div class="task-detail-data-row"><dt>${escapeHtml(taskDataLabel(key))}</dt><dd>${taskDataValue(item, key)}</dd></div>`).join("")}</dl>` : ""}
    ${nested.length ? `<div class="task-detail-data-grid${nested.length === 1 ? " single" : ""}">${nested.map(([key, item]) => `<article class="task-detail-data-group"><h4>${escapeHtml(taskDataLabel(key))}</h4>${renderTaskData(item, { emptyMessage, level: level + 1 })}</article>`).join("")}</div>` : ""}
  </div>`;
}

function taskMetaItem(label, value, { code = false } = {}) {
  const tag = code ? "code" : "strong";
  return `<div class="task-detail-meta-item"><span>${escapeHtml(label)}</span><${tag}>${escapeHtml(value ?? "—")}</${tag}></div>`;
}

function taskPolicyChip(label, enabled = false) {
  return `<span class="task-detail-policy${enabled ? " enabled" : ""}">${escapeHtml(label)}</span>`;
}

function renderTaskTimeline(history = []) {
  if (!history.length) return `<div class="task-detail-empty">${escapeHtml(t("taskTimelineUnavailable"))}</div>`;
  return `<ol class="task-detail-timeline">${history.map((item) => {
    const status = String(item.status || "");
    const reason = taskReasonLabel(item.reason);
    const message = String(item.message || "");
    return `<li>
      <span class="task-detail-timeline-dot"></span>
      <div class="task-detail-timeline-main">
        <strong>${escapeHtml(taskStatusLabel(status))}</strong>
        ${message ? `<span>${escapeHtml(message)}</span>` : ""}
        ${reason ? `<span>${escapeHtml(t("taskDetailReason", { reason }))}</span>` : ""}
      </div>
      <time>${escapeHtml(formatTaskTime(item.timestamp))}</time>
    </li>`;
  }).join("")}</ol>`;
}

function taskSnapshotMetrics(snapshot = {}) {
  const indexes = snapshot.indexes || {};
  const statistics = snapshot.statistics || {};
  const candidates = [
    ["generation", indexes.generation || indexes.generation_id],
    ["vector_count", indexes.vector_count],
    ["document_vectors", indexes.document_vectors],
    ["graph_vectors", indexes.graph_vectors],
    ["media_vector_count", indexes.media_vector_count],
    ["dimensions", indexes.dimensions || indexes.dimension],
    ["total_memories", statistics.total_memories],
    ["document_count", statistics.document_count],
  ].filter(([, value]) => value !== null && value !== undefined);
  return candidates.slice(0, 6);
}

function renderTaskSnapshot(snapshot, phase) {
  if (!snapshot) return `<div class="task-detail-data-empty">${escapeHtml(t("taskStateUnavailable"))}</div>`;
  const metrics = taskSnapshotMetrics(snapshot);
  const groups = ["database", "provider_binding", "statistics", "indexes"]
    .filter((key) => snapshot[key] !== null && snapshot[key] !== undefined)
    .map((key) => `<article class="task-detail-data-group"><h4>${escapeHtml(taskDataLabel(key))}</h4>${renderTaskData(snapshot[key], { level: 1 })}</article>`)
    .join("");
  return `<article class="task-detail-snapshot task-detail-snapshot-${phase}">
    <header><div><span class="task-detail-snapshot-mark"></span><h4>${escapeHtml(t(phase === "before" ? "taskStateBefore" : "taskStateAfter"))}</h4></div><time>${escapeHtml(formatTaskTime(snapshot.captured_at))}</time></header>
    ${metrics.length ? `<div class="task-detail-snapshot-metrics">${metrics.map(([key, value]) => `<div><span>${escapeHtml(taskDataLabel(key))}</span>${taskDataValue(value, key)}</div>`).join("")}</div>` : ""}
    <div class="task-detail-snapshot-groups">${groups}</div>
  </article>`;
}

function renderTaskStateComparison(comparison) {
  if (!comparison) return "";
  const errors = comparison.capture_errors || {};
  if (!comparison.available) {
    return `<section class="task-detail-section">
      <header><div><h3>${escapeHtml(t("taskStateComparison"))}</h3><p>${escapeHtml(t("taskStateComparisonHint"))}</p></div></header>
      <div class="task-detail-empty">${escapeHtml(t("taskStateUnavailable"))}</div>
      ${Object.keys(errors).length ? `<div class="task-detail-error">${renderTaskData(errors)}</div>` : ""}
    </section>`;
  }
  const changes = Array.isArray(comparison.changes) ? comparison.changes : [];
  const table = changes.length
    ? `<div class="task-detail-table-wrap"><table class="task-detail-diff-table">
        <thead><tr><th>${escapeHtml(t("taskStateField"))}</th><th>${escapeHtml(t("taskStateChange"))}</th><th>${escapeHtml(t("taskStateBefore"))}</th><th>${escapeHtml(t("taskStateAfter"))}</th></tr></thead>
        <tbody>${changes.map((item) => `<tr>
          <td data-label="${escapeHtml(t("taskStateField"))}"><strong>${escapeHtml(taskDataPathLabel(item.path))}</strong></td>
          <td data-label="${escapeHtml(t("taskStateChange"))}" class="task-detail-change-${escapeHtml(item.change || "changed")}">${escapeHtml(t(`taskStateChange${String(item.change || "changed").replace(/^./, (value) => value.toUpperCase())}`))}</td>
          <td data-label="${escapeHtml(t("taskStateBefore"))}">${taskDataValue(item.before, String(item.path || "").split(".").pop())}</td>
          <td data-label="${escapeHtml(t("taskStateAfter"))}">${taskDataValue(item.after, String(item.path || "").split(".").pop())}</td>
        </tr>`).join("")}</tbody>
      </table></div>`
    : `<div class="task-detail-empty">${escapeHtml(t("taskStateUnchanged"))}</div>`;
  return `<section class="task-detail-section">
    <header><div><h3>${escapeHtml(t("taskStateComparison"))}</h3><p>${escapeHtml(t("taskStateComparisonHint"))}</p></div></header>
    <div class="task-detail-comparison-summary">
      <strong>${escapeHtml(comparison.changed ? t("taskStateChanged") : t("taskStateUnchanged"))}</strong>
      <span>${escapeHtml(t("taskStateChangeCount", { count: comparison.change_count || 0 }))}</span>
    </div>
    ${table}
    ${Object.keys(errors).length ? `<div class="task-detail-error">${renderTaskData(errors)}</div>` : ""}
    <div class="task-detail-state-grid">
      ${renderTaskSnapshot(comparison.before, "before")}
      ${renderTaskSnapshot(comparison.after, "after")}
    </div>
  </section>`;
}

function disconnectTaskDetailCollapsibles() {
  if (taskDetailCollapseFrame) cancelAnimationFrame(taskDetailCollapseFrame);
  taskDetailCollapseFrame = 0;
  if (taskDetailCollapseResizeHandler) {
    window.removeEventListener("resize", taskDetailCollapseResizeHandler);
    taskDetailCollapseResizeHandler = null;
  }
}

function setupTaskDetailCollapsibles() {
  disconnectTaskDetailCollapsibles();
  const body = $("task-detail-body");
  if (!body) return;
  const topLevelSections = Array.from(body.querySelectorAll(":scope > .task-detail-section"))
    .filter((section) =>
      !section.classList.contains("task-detail-execution-section")
      && !section.querySelector(".task-detail-state-grid")
    );
  const candidates = [
    ...topLevelSections,
    ...body.querySelectorAll(".task-detail-snapshot"),
    ...body.querySelectorAll(".task-detail-table-wrap"),
  ];
  const items = candidates.map((container, index) => {
    const header = container.querySelector(":scope > header");
    const content = document.createElement("div");
    content.id = `task-detail-collapse-content-${index}`;
    content.className = "task-detail-collapse-content";
    if (container.classList.contains("task-detail-table-wrap")) {
      content.classList.add("task-detail-collapse-content-scroll");
    }
    Array.from(container.children)
      .filter((child) => child !== header)
      .forEach((child) => content.appendChild(child));
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "library-expand-toggle task-detail-collapse-toggle hidden";
    toggle.setAttribute("aria-controls", content.id);
    toggle.setAttribute("aria-expanded", "false");
    toggle.innerHTML = `<svg viewBox="0 0 240 24" aria-hidden="true" focusable="false">
      <circle cx="92" cy="12" r="3"></circle>
      <circle cx="120" cy="12" r="3"></circle>
      <circle cx="148" cy="12" r="3"></circle>
    </svg>`;
    const topToggle = document.createElement("button");
    topToggle.type = "button";
    topToggle.className = "collapse-top-toggle task-detail-collapse-top hidden";
    topToggle.setAttribute("aria-controls", content.id);
    topToggle.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m6 15 6-6 6 6"></path></svg>`;
    container.classList.add("task-detail-collapsible");
    container.append(topToggle, content, toggle);
    return { container, content, toggle, topToggle, header, expanded: false };
  });

  const applyState = () => {
    taskDetailCollapseFrame = 0;
    const collapsedHeight = Math.max(1, Math.floor(window.innerHeight * 0.5));
    items.forEach((item) => {
      const containerStyle = window.getComputedStyle(item.container);
      const padding = (Number.parseFloat(containerStyle.paddingTop) || 0)
        + (Number.parseFloat(containerStyle.paddingBottom) || 0);
      const headerHeight = item.header?.getBoundingClientRect().height || 0;
      const naturalHeight = padding + headerHeight + item.content.scrollHeight + 54;
      const overflowing = naturalHeight > collapsedHeight + 8;
      item.container.style.setProperty("--task-detail-collapsed-height", `${collapsedHeight}px`);
      item.container.classList.toggle("task-detail-collapse-active", overflowing);
      item.toggle.classList.toggle("hidden", !overflowing);
      if (!overflowing) item.expanded = false;
      item.container.classList.toggle("task-detail-collapsed", overflowing && !item.expanded);
      item.container.classList.toggle("task-detail-expanded", overflowing && item.expanded);
      item.toggle.classList.toggle("expanded", overflowing && item.expanded);
      item.toggle.setAttribute("aria-expanded", String(overflowing && item.expanded));
      item.topToggle.classList.toggle("hidden", !overflowing || !item.expanded);
      if (overflowing) {
        const label = t(item.expanded ? "collapseLibraryCard" : "expandLibraryCard");
        item.toggle.title = label;
        item.toggle.setAttribute("aria-label", label);
        const collapseLabel = t("collapseLibraryCard");
        item.topToggle.title = collapseLabel;
        item.topToggle.setAttribute("aria-label", collapseLabel);
      } else {
        item.toggle.removeAttribute("title");
        item.toggle.removeAttribute("aria-label");
        item.topToggle.removeAttribute("title");
        item.topToggle.removeAttribute("aria-label");
      }
    });
  };
  const scheduleApply = () => {
    if (taskDetailCollapseFrame) cancelAnimationFrame(taskDetailCollapseFrame);
    taskDetailCollapseFrame = requestAnimationFrame(applyState);
  };
  items.forEach((item) => {
    item.toggle.addEventListener("click", () => {
      item.expanded = !item.expanded;
      applyState();
    });
    item.topToggle.addEventListener("click", () => {
      item.expanded = false;
      applyState();
    });
  });
  taskDetailCollapseResizeHandler = scheduleApply;
  window.addEventListener("resize", taskDetailCollapseResizeHandler);
  scheduleApply();
}

function renderTaskDetail(job) {
  const body = $("task-detail-body");
  const title = $("task-detail-title");
  const subtitle = $("task-detail-subtitle");
  if (!body || !title || !subtitle) return;
  const status = String(job.status || "");
  const databaseId = job.memory_store_id || job.knowledge_base_id || job.database_id || "—";
  const taskType = job.task_type || {};
  const timing = job.timing || {};
  title.textContent = taskKindLabel(job.kind);
  subtitle.innerHTML = `<span class="task-detail-status task-detail-status-${escapeHtml(status)}">${escapeHtml(taskStatusLabel(status))}</span><span>${escapeHtml(String(job.id || ""))}</span>`;
  const metadata = [
    taskMetaItem(t("taskDetailJobId"), String(job.id || "—"), { code: true }),
    taskMetaItem(t("taskDetailKind"), `${taskKindLabel(job.kind)} · ${job.kind || "—"}`),
    taskMetaItem(t("taskDetailStatus"), taskStatusLabel(status)),
    taskMetaItem(t("taskDetailDatabaseType"), String(job.database_type || "—"), { code: true }),
    taskMetaItem(t("taskDetailDatabaseId"), String(databaseId), { code: true }),
    taskMetaItem(t("taskDetailProgress"), `${Math.round(Number(job.progress || 0) * 100)}%`),
    taskMetaItem(t("taskDetailCreatedAt"), formatTaskTime(timing.created_at || job.created_at)),
    taskMetaItem(t("taskDetailStartedAt"), formatTaskTime(timing.started_at || job.started_at)),
    taskMetaItem(t("taskDetailFinishedAt"), formatTaskTime(timing.finished_at || job.finished_at)),
    taskMetaItem(t("taskDetailQueueDuration"), formatTaskDuration(timing.queue_duration_seconds)),
    taskMetaItem(t("taskDetailExecutionDuration"), formatTaskDuration(timing.execution_duration_seconds)),
    taskMetaItem(t("taskDetailTotalDuration"), formatTaskDuration(timing.total_duration_seconds)),
    taskMetaItem(t("taskDetailAttempts"), String(job.attempt_count ?? 0)),
    taskMetaItem(t("taskDetailStatusReason"), taskReasonLabel(job.status_reason) || "—"),
    taskMetaItem(t("taskDetailMessage"), String(job.message || "—")),
  ].join("");
  const policies = [
    taskPolicyChip(t(`taskLane${taskType.lane === "long" ? "Long" : "Short"}`), true),
    taskPolicyChip(t("taskPolicyResumable"), Boolean(taskType.resumable)),
    taskPolicyChip(t("taskPolicyAdapterBlocking"), Boolean(taskType.adapter_blocking)),
    taskPolicyChip(t("taskPolicyRuntimePause"), Boolean(taskType.runtime_pause)),
    taskPolicyChip(t("taskPolicyReadOnly"), Boolean(taskType.read_only)),
    taskPolicyChip(`${t("taskPolicyEmbedding")}: ${taskType.embedding_context_policy || "none"}`, taskType.embedding_context_policy !== "none"),
  ].join("");
  const requestSection = `<section class="task-detail-section"><header><div><h3>${escapeHtml(t("taskRequestMetadata"))}</h3><p>${escapeHtml(t("taskRequestMetadataHint"))}</p></div></header>${renderTaskData(job.request_metadata, { emptyMessage: t("taskMetadataUnavailable") })}</section>`;
  const resultSection = `<section class="task-detail-section"><header><div><h3>${escapeHtml(t("taskResultMetadata"))}</h3><p>${escapeHtml(t("taskResultMetadataHint"))}</p></div></header>${job.error ? `<div class="task-detail-error">${escapeHtml(String(job.error))}</div>` : ""}${renderTaskData(job.result, { emptyMessage: t("taskResultUnavailable") })}${job.checkpoint == null ? "" : `<div class="task-detail-checkpoint"><h4>${escapeHtml(t("taskCheckpointMetadata"))}</h4>${renderTaskData(job.checkpoint)}</div>`}</section>`;
  body.innerHTML = `
    <section class="task-detail-section task-detail-execution-section"><header><div><h3>${escapeHtml(t("taskExecutionMetadata"))}</h3><p>${escapeHtml(t("taskExecutionMetadataHint"))}</p></div></header><div class="task-detail-meta-grid">${metadata}</div><div class="task-detail-policy-list">${policies}</div></section>
    <section class="task-detail-section"><header><div><h3>${escapeHtml(t("taskTimeline"))}</h3><p>${escapeHtml(t("taskTimelineHint"))}</p></div></header>${renderTaskTimeline(job.status_history || [])}</section>
    ${renderTaskStateComparison(job.database_state_comparison)}
    ${requestSection}
    ${resultSection}
  `;
  setupTaskDetailCollapsibles();
}

function closeTaskDetail() {
  disconnectTaskDetailCollapsibles();
  taskDetailRequest += 1;
  $("task-detail-overlay")?.classList.add("hidden");
  $("task-detail-overlay")?.setAttribute("aria-hidden", "true");
  $("task-detail-panel")?.classList.remove("visible");
  $("task-detail-panel")?.setAttribute("aria-hidden", "true");
}

async function openTaskDetail(jobId) {
  if (!jobId) return;
  disconnectTaskDetailCollapsibles();
  const request = ++taskDetailRequest;
  const body = $("task-detail-body");
  const title = $("task-detail-title");
  const subtitle = $("task-detail-subtitle");
  if (title) title.textContent = t("taskDetailTitle");
  if (subtitle) subtitle.textContent = jobId;
  if (body) body.innerHTML = `<div class="task-detail-loading">${escapeHtml(t("taskDetailLoading"))}</div>`;
  $("task-detail-overlay")?.classList.remove("hidden");
  $("task-detail-overlay")?.setAttribute("aria-hidden", "false");
  $("task-detail-panel")?.classList.add("visible");
  $("task-detail-panel")?.setAttribute("aria-hidden", "false");
  $("task-detail-close")?.focus();
  try {
    const job = await api(`/jobs/${encodeURIComponent(jobId)}/details`, { pageScoped: false });
    if (request !== taskDetailRequest) return;
    renderTaskDetail(job);
  } catch (error) {
    if (request !== taskDetailRequest || !body) return;
    body.innerHTML = `<div class="task-detail-error">${escapeHtml(error.message || String(error))}</div>`;
  }
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
  databaseId = "",
  databaseType = "",
  message = "",
  markLibraryConflict = false,
} = {}) {
  const effectiveDatabaseId = databaseId || state.selectedDatabaseId || "";
  const effectiveDatabaseType = databaseType || state.selectedDatabaseType || "livingmemory_v8";
  const job = {
    id: id || `local-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    kind,
    database_type: effectiveDatabaseType,
    database_id: effectiveDatabaseId,
    status: "queued",
    progress: 0,
    message: message || t("taskSubmitting"),
    created_at: taskTimestamp(),
    updated_at: taskTimestamp(),
    optimistic: true,
  };
  if (markLibraryConflict && effectiveDatabaseId) {
    markDatabaseIndexConflict(effectiveDatabaseId, effectiveDatabaseType);
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
  const topToggle = $("task-history-collapse-top");
  if (!panel || !content || !list || !toggle || !topToggle) return;
  const items = Array.from(list.querySelectorAll(".task-item"));
  const finishedScope = state.tasks.scope === "finished";

  if (!finishedScope || items.length < 3) {
    panel.classList.remove("task-history-panel-collapsed", "task-history-panel-expanded");
    panel.style.removeProperty("--task-history-collapsed-height");
    toggle.classList.add("hidden");
    toggle.classList.remove("expanded");
    topToggle.classList.add("hidden");
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
    topToggle.classList.add("hidden");
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
  topToggle.classList.toggle("hidden", !state.tasks.finishedExpanded);
  const collapseTitle = t("collapseLibraryCard");
  topToggle.title = collapseTitle;
  topToggle.setAttribute("aria-label", collapseTitle);
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
  closeTaskDetail();
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
      const databaseId = job.memory_store_id
        || job.knowledge_base_id
        || job.database_id
        || job.library_id
        || "—";
      const reason = taskReasonLabel(job.status_reason);
      const checkpoint = job.checkpoint || {};
      const checkpointText = checkpoint.phase
        ? t("taskCheckpoint", {
            phase: checkpoint.phase,
            completed: checkpoint.completed_documents ?? checkpoint.completed_graph_entries ?? 0,
          })
        : "";
      const opensDetail = !isActiveTask(job);
      const detailClass = opensDetail ? " task-item-detail" : "";
      const detailAttrs = opensDetail
        ? ` data-job-detail="${escapeHtml(String(job.id || ""))}" tabindex="0" role="button" aria-label="${escapeHtml(`${t("viewTaskDetails")}: ${taskKindLabel(job.kind)}`)}"`
        : "";
      return `<article class="task-item task-status-${escapeHtml(status)}${detailClass}"${detailAttrs}>
        <div class="task-head">
          <div>
            <div class="task-title"><span>${taskStatusMark(status)}</span><span>${escapeHtml(taskKindLabel(job.kind))}</span></div>
            <div class="task-meta">
              <span>job ${escapeHtml(String(job.id || "").slice(0, 8))}</span>
              <span>${escapeHtml(databaseId)}</span>
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
  if (button) {
    controlTask(button.dataset.jobId, button.dataset.jobAction, button);
    return;
  }
  const detailItem = event.target.closest("[data-job-detail]");
  if (!detailItem) return;
  openTaskDetail(detailItem.dataset.jobDetail).catch((error) => {
    toast(error.message || String(error), true);
  });
});

$("task-list")?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const detailItem = event.target.closest("[data-job-detail]");
  if (!detailItem || event.target.closest("[data-job-action]")) return;
  event.preventDefault();
  openTaskDetail(detailItem.dataset.jobDetail).catch((error) => {
    toast(error.message || String(error), true);
  });
});

$("task-detail-close")?.addEventListener("click", closeTaskDetail);
$("task-detail-overlay")?.addEventListener("click", closeTaskDetail);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && $("task-detail-panel")?.classList.contains("visible")) {
    closeTaskDetail();
  }
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
    closeTaskDetail();
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
    if (document.hidden) return;
    try {
      await loadTasks("active");
      if (state.tasks.scope === "finished") {
        await loadTasks("finished");
      } else {
        renderTasks();
      }
    } catch (error) {
      if (error?.name !== "AbortError") console.error("task polling failed", error);
    } finally {
      if (state.tasks.polling && !document.hidden) {
        const hasActiveTasks = state.tasks.active.some(isActiveTask);
        state.tasks.pollTimer = setTimeout(poll, hasActiveTasks ? 1200 : 5000);
      }
    }
  };
  taskPoll = poll;
  taskPoll();
}

function stopTaskPolling() {
  state.tasks.polling = false;
  if (state.tasks.pollTimer) {
    clearTimeout(state.tasks.pollTimer);
    state.tasks.pollTimer = null;
  }
  taskPoll = null;
}

document.addEventListener("visibilitychange", () => {
  if (state.tasks.polling) {
    if (state.tasks.pollTimer) {
      clearTimeout(state.tasks.pollTimer);
      state.tasks.pollTimer = null;
    }
    if (!document.hidden && taskPoll) {
      state.tasks.pollTimer = setTimeout(taskPoll, 0);
    }
  }
  if (state.logs.polling) {
    if (state.logs.pollTimer) {
      clearTimeout(state.logs.pollTimer);
      state.logs.pollTimer = null;
    }
    if (document.hidden) {
      state.logs.abortController?.abort();
    } else if (logPoll && !logReset) {
      state.logs.pollTimer = setTimeout(logPoll, 0);
    }
  }
});

$("task-history-toggle")?.addEventListener("click", () => {
  state.tasks.finishedExpanded = !state.tasks.finishedExpanded;
  applyTaskHistoryCollapseState();
});

$("task-history-collapse-top")?.addEventListener("click", () => {
  state.tasks.finishedExpanded = false;
  applyTaskHistoryCollapseState();
});

function renderLogAutoScrollState() {
  const toggle = $("log-auto-scroll");
  const label = $("log-auto-scroll-label");
  if (toggle) toggle.checked = state.logs.autoScroll;
  if (label) label.textContent = state.logs.autoScroll ? t("logAutoScrollOn") : t("logAutoScrollOff");
}

function logEntryHtml(item) {
  return `<div class="log-entry log-${String(item.level || "INFO").toLowerCase()}">
    <span class="log-entry-level">${escapeHtml(item.level || "")}</span>
    <span class="log-entry-line">${escapeHtml(item.line || item.message || "")}</span>
  </div>`;
}

function updateLogChrome() {
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
}

function renderLogs({ scrollToBottom = false } = {}) {
  const consoleElement = $("log-console");
  if (!consoleElement) return;
  const activeLevels = state.logs.activeLevels;
  const visible = state.logs.items.filter((item) => activeLevels.has(item.level));
  if (!visible.length) {
    consoleElement.innerHTML = `<div class="log-empty">${escapeHtml(t("logsEmpty"))}</div>`;
  } else {
    consoleElement.innerHTML = visible.map(logEntryHtml).join("");
  }
  updateLogChrome();
  if (scrollToBottom && state.logs.autoScroll) {
    consoleElement.scrollTop = consoleElement.scrollHeight;
  }
}

function appendLogEntries(items) {
  const consoleElement = $("log-console");
  if (!consoleElement) return;
  const visible = items.filter((item) => state.logs.activeLevels.has(item.level));
  if (visible.length) {
    consoleElement.querySelector(".log-empty")?.remove();
    consoleElement.insertAdjacentHTML("beforeend", visible.map(logEntryHtml).join(""));
  }
  updateLogChrome();
  if (visible.length && state.logs.autoScroll) {
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
  const replaced = reset || payload.reset;
  if (replaced) {
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
  const trimmed = state.logs.items.length > state.logs.maxEntries;
  if (trimmed) {
    state.logs.items = state.logs.items.slice(-state.logs.maxEntries);
  }
  if (replaced || trimmed) {
    renderLogs({ scrollToBottom: incoming.length > 0 });
  } else if (incoming.length) {
    appendLogEntries(incoming);
  } else {
    updateLogChrome();
  }
}

function startLogPolling() {
  if (state.logs.polling) return;
  state.logs.polling = true;
  const poll = async () => {
    if (!state.logs.polling) return;
    if (document.hidden) return;
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
      if (state.logs.polling && !document.hidden) {
        state.logs.pollTimer = setTimeout(poll, 250);
      }
    }
  };
  logPoll = poll;
  if (!document.hidden) {
    logReset = loadLogs({ reset: true })
      .catch((error) => console.error("log reset failed", error))
      .finally(() => {
        logReset = null;
        if (state.logs.polling && !document.hidden && logPoll) logPoll();
      });
  }
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
  logPoll = null;
  logReset = null;
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
  const databaseId = options.databaseId || state.selectedDatabaseId || "";
  const databaseType = options.databaseType || state.selectedDatabaseType || "livingmemory_v8";
  if (tempId && jobId) {
    updateOptimisticTask(tempId, {
      id: jobId,
      kind,
      database_type: databaseType,
      database_id: databaseId,
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
    databaseId,
    databaseType,
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
  await onTaskFinished(job);
  if (job.status === "completed") {
    clearDatabaseIndexConflict(
      job.memory_store_id || job.knowledge_base_id || job.database_id || job.library_id,
      job.database_type || "livingmemory_v8",
    );
  }
  refreshDatabaseContext();
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
        await accept(await api("/jobs/" + encodeURIComponent(id), { pageScoped: false }));
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
