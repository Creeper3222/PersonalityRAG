export function createMemoriesController({ $, state, t, toast, api, selectedDatabaseApi, escapeHtml, formatMemoryTime, displayStatus, memoryStatusPill, memoryImportanceBar, normalizeMemoryDetail, memoryMetaItem, memoryListSection, memoryTagsSection, renderMemoryMiniGraph, loadDatabases, debounce, confirmDialog, asyncGuard }) {
async function loadMemories() {
  if (state.memoryTransferDatabaseId && state.memoryTransferDatabaseId !== state.selectedDatabaseId) {
    resetTransferPreview();
  }
  state.memoryTransferDatabaseId = state.selectedDatabaseId;
  const params = new URLSearchParams({
    page: state.memoryPage,
    page_size: state.memoryPageSize,
    keyword: $("memory-keyword").value,
    session_id: $("memory-session")?.value || "",
    persona_id: $("memory-persona").value,
    status: $("memory-status")?.value || "",
    sort: $("memory-sort").value,
  });
  try {
    const data = await selectedDatabaseApi("/memories?" + params);
    state.memoryHasMore = data.has_more;
    state.memoryItems = Array.isArray(data.items) ? data.items : [];
    $("memory-rows").innerHTML =
      state.memoryItems
        .map(
          (item) => {
            const metadata = item.metadata || {};
            const personaSummary = String(metadata.persona_summary || item.text || "");
            const updated = formatMemoryTime(metadata.updated_at ?? item.updated_at ?? metadata.create_time);
            const created = formatMemoryTime(metadata.create_time ?? item.created_at);
            return `<tr class="memory-row" data-id="${escapeHtml(item.id)}" tabindex="0">
            <td class="memory-id">${escapeHtml(item.id)}</td>
            <td class="memory-summary-cell" title="${escapeHtml(personaSummary)}"><div class="memory-summary-text">${escapeHtml(personaSummary)}</div><div class="memory-summary-meta">${escapeHtml(t("updatedAt"))} ${escapeHtml(updated)}</div></td>
            <td>${memoryImportanceBar(metadata.importance ?? 0.5)}</td>
            <td>${memoryStatusPill(metadata.status || "active")}</td>
            <td>${escapeHtml(created)}</td>
          </tr>`;
          },
        )
        .join("") || `<tr><td colspan="5">${escapeHtml(t("tableEmpty"))}</td></tr>`;
    $("memory-page-info").textContent = t("pageOfTotal", { page: state.memoryPage, total: data.total });
    $("memory-prev").disabled = state.memoryPage <= 1;
    $("memory-next").disabled = !data.has_more;
    document.querySelectorAll(".memory-row").forEach((row) => {
      const open = () => openMemoryDetail(Number(row.dataset.id));
      row.onclick = open;
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open();
        }
      };
    });
  } catch (error) {
    if (error?.name === "AbortError") return;
    toast(error.message, true);
  }
}

$("memory-refresh").onclick = async () => {
  state.memoryPage = 1;
  await loadMemories();
  toast(t("pageRefreshed"));
};
$("memory-keyword").oninput = debounce(() => {
  state.memoryPage = 1;
  loadMemories();
}, 350);
$("memory-persona").oninput = debounce(() => {
  state.memoryPage = 1;
  loadMemories();
}, 350);
$("memory-session").oninput = debounce(() => {
  state.memoryPage = 1;
  loadMemories();
}, 350);
$("memory-status").onchange = () => {
  state.memoryPage = 1;
  loadMemories();
};
$("memory-sort").onchange = () => loadMemories();
$("memory-page-size").onchange = () => {
  state.memoryPageSize = Number($("memory-page-size").value) || 20;
  state.memoryPage = 1;
  loadMemories();
};
$("memory-prev").onclick = () => {
  if (state.memoryPage > 1) {
    state.memoryPage -= 1;
    loadMemories();
  }
};
$("memory-next").onclick = () => {
  if (state.memoryHasMore) {
    state.memoryPage += 1;
    loadMemories();
  }
};

function transferApiPath(suffix = "") {
  if (!state.selectedDatabaseId) throw new Error(t("selectLibraryFirst"));
  return `/api/v1/memory-libraries/livingmemory_v8/${encodeURIComponent(state.selectedDatabaseId)}/transfers${suffix}`;
}

function transferPreviewCard(item) {
  const summary = String(item?.canonical_summary || item?.content || "");
  const flags = [
    item?.duplicate ? t("transferDuplicate") : "",
    item?.needs_summary ? t("transferNeedsSummary") : "",
  ].filter(Boolean);
  return `<article class="memory-transfer-item ${item?.duplicate ? "is-duplicate" : ""} ${item?.needs_summary ? "needs-summary" : ""}">
    <header><strong>${escapeHtml(t("transferRow", { row: item?.row_number ?? "—" }))}</strong><span>${escapeHtml(flags.join(" · ") || t("transferReady"))}</span></header>
    <p>${escapeHtml(summary || t("transferSourceOnly"))}</p>
    <footer><code>${escapeHtml(item?.session_id || "—")}</code><code>${escapeHtml(item?.persona_id || "—")}</code></footer>
  </article>`;
}

function renderTransferPreview(data) {
  const counts = data?.counts || {};
  const items = Array.isArray(data?.items) ? data.items : [];
  const invalid = Array.isArray(data?.invalid_items) ? data.invalid_items : [];
  const needsSummary = Number(counts.needs_summary || 0);
  state.memoryTransferPreview = data;
  $("memory-transfer-preview-result").classList.remove("hidden");
  $("memory-transfer-preview-result").innerHTML = `
    <div class="memory-transfer-stat-grid">
      ${memoryMetaItem(t("transferInput"), escapeHtml(String(counts.input || 0)))}
      ${memoryMetaItem(t("transferPlannedImport"), escapeHtml(String(counts.planned_import || 0)))}
      ${memoryMetaItem(t("transferDuplicates"), escapeHtml(String(counts.duplicates || 0)))}
      ${memoryMetaItem(t("transferInvalid"), escapeHtml(String(counts.invalid || 0)))}
      ${memoryMetaItem(t("transferNeedsSummary"), escapeHtml(String(needsSummary)))}
    </div>
    ${needsSummary ? `<div class="info-banner warning">${escapeHtml(t("transferNeedsAdapterSummary"))}</div>` : ""}
    ${invalid.length ? `<details><summary>${escapeHtml(t("transferInvalidDetails", { count: invalid.length }))}</summary><div class="memory-transfer-errors">${invalid.slice(0, 50).map((entry) => `<div><strong>${escapeHtml(t("transferRow", { row: entry.row_number }))}</strong><span>${escapeHtml(entry.error || t("unknownError"))}</span></div>`).join("")}</div></details>` : ""}
    <div class="memory-transfer-list">${items.slice(0, 50).map(transferPreviewCard).join("")}</div>
    ${items.length > 50 ? `<p class="panel-hint">${escapeHtml(t("transferPreviewLimited", { count: items.length }))}</p>` : ""}
    <div class="memory-detail-actions">
      <button id="memory-transfer-reset" type="button" class="ghost">${escapeHtml(t("clearPreview"))}</button>
      <button id="memory-transfer-commit" type="button" class="primary" ${needsSummary || !Number(counts.planned_import || 0) ? "disabled" : ""}>${escapeHtml(t("commitImport"))}</button>
    </div>`;
  $("memory-transfer-reset").onclick = resetTransferPreview;
  $("memory-transfer-commit").onclick = commitTransferImport;
}

function resetTransferPreview() {
  state.memoryTransferPreview = null;
  const result = $("memory-transfer-preview-result");
  result?.classList.add("hidden");
  if (result) result.innerHTML = "";
}

$("memory-transfer-file")?.addEventListener("change", resetTransferPreview);
$("memory-transfer-preview")?.addEventListener("click", async (event) => {
  const file = $("memory-transfer-file")?.files?.[0];
  if (!file) {
    toast(t("chooseTransferFile"), true);
    return;
  }
  await asyncGuard.run("memory-transfer:preview", async () => {
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const data = await selectedDatabaseApi("/transfers/imports/preview", { method: "POST", body: form });
      renderTransferPreview(data);
    } catch (error) {
      toast(error.message, true);
    }
  }, { button: event.currentTarget, busyText: t("loading") });
});

async function commitTransferImport() {
  const preview = state.memoryTransferPreview;
  if (!preview?.preview_id) return;
  await asyncGuard.run("memory-transfer:commit", async () => {
    try {
      const result = await selectedDatabaseApi(`/transfers/imports/${encodeURIComponent(preview.preview_id)}/commit`, {
        method: "POST",
        body: JSON.stringify({
          duplicate_mode: $("memory-transfer-duplicate-mode")?.value || "skip",
          summaries: [],
        }),
      });
      resetTransferPreview();
      toast(t("transferTaskQueued"));
      if (result?.job_id) toast(`${t("taskId")}: ${result.job_id}`);
    } catch (error) {
      toast(error.message, true);
    }
  }, { button: $("memory-transfer-commit"), busyText: t("loading") });
}

$("memory-transfer-export")?.addEventListener("click", async (event) => {
  await asyncGuard.run("memory-transfer:export", async () => {
    try {
      const format = $("memory-transfer-export-format")?.value || "json";
      const response = await fetch(transferApiPath(`/export?format=${encodeURIComponent(format)}`), {
        credentials: "same-origin",
        redirect: "error",
      });
      if (!response.ok) throw new Error((await response.text()) || response.statusText);
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const match = disposition.match(/filename="?([^";]+)"?/i);
      const filename = match?.[1] || `${state.selectedDatabaseId}-memories.${format}`;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      toast(t("transferExported"));
    } catch (error) {
      toast(error.message, true);
    }
  }, { button: event.currentTarget, busyText: t("loading") });
});

async function openMemoryDetail(id, fallback = null) {
  state.selectedMemoryId = id;
  $("memory-detail-title").textContent = t("memoryDetails", { id });
  $("memory-detail-body").innerHTML = `<div class="memory-detail-empty">${escapeHtml(t("loading"))}</div>`;
  $("memory-detail-overlay").classList.remove("hidden");
  $("memory-detail-panel").classList.add("visible");
  try {
    const item = await selectedDatabaseApi("/memories/" + id);
    state.selectedMemoryDetail = item;
    renderMemoryDetailView(item);
  } catch (error) {
    if (fallback) {
      state.selectedMemoryDetail = fallback;
      renderMemoryDetailView(fallback);
      toast(error.message, true);
      return;
    }
    toast(error.message, true);
    closeMemoryDetail();
  }
}

function closeMemoryDetail() {
  $("memory-detail-overlay")?.classList.add("hidden");
  $("memory-detail-panel")?.classList.remove("visible");
  state.selectedMemoryId = null;
  state.selectedMemoryDetail = null;
}

function renderMemoryDetailView(raw) {
  const detail = normalizeMemoryDetail(raw);
  state.selectedMemoryDetail = raw;
  $("memory-detail-title").textContent = t("memoryDetails", { id: detail.id });
  const historyItems = detail.updateHistory.map((item) => {
    const time = formatMemoryTime(item.timestamp || item.time);
    const text = item.description || `${item.field || ""}: ${item.old_value ?? ""} -> ${item.new_value ?? ""}`;
    return `${time} ${text}`.trim();
  });
  $("memory-detail-body").innerHTML = `
    <div class="memory-detail-top">
      <div class="memory-detail-header">
        ${memoryStatusPill(detail.status)}
        <span class="memory-detail-tag">${escapeHtml(t("importanceField"))}: ${detail.importance.toFixed(1)}/10</span>
      </div>
      <div class="memory-detail-actions">
        <button type="button" class="ghost" id="memory-detail-edit">${escapeHtml(t("editMemory"))}</button>
        ${detail.hasSource ? `<button type="button" class="ghost" id="memory-detail-source">${escapeHtml(t("viewSourceMessages"))}</button>` : ""}
        ${detail.status === "archived"
          ? `<button type="button" class="primary" id="memory-detail-restore">${escapeHtml(t("restoreMemory"))}</button>`
          : `<button type="button" class="ghost" id="memory-detail-archive">${escapeHtml(t("archiveMemory"))}</button>`}
        <button type="button" class="ghost danger" id="memory-detail-delete">${escapeHtml(t("deleteMemory"))}</button>
      </div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("personaSummary"))}</div>
      <div class="memory-detail-content">${escapeHtml(detail.personaSummary || detail.canonicalSummary || detail.text)}</div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("canonicalSummary"))}</div>
      <div class="memory-detail-content">${escapeHtml(detail.canonicalSummary || detail.text)}</div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("graphContext"))}</div>
      <div class="memory-detail-graph">${renderMemoryMiniGraph(detail.graph)}</div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("metadata"))}</div>
      <div class="memory-detail-meta-grid">
        ${memoryMetaItem(t("statusField"), memoryStatusPill(detail.status))}
        ${memoryMetaItem(t("importanceField"), `${detail.importance.toFixed(1)} / 10`)}
        ${memoryMetaItem(t("sessionField"), `<code>${escapeHtml(detail.sessionId)}</code>`)}
        ${memoryMetaItem(t("personaField"), `<div class="memory-persona-value"><code>${escapeHtml(detail.personaId)}</code><button type="button" class="ghost" id="memory-detail-edit-persona">${escapeHtml(t("editPersona"))}</button></div>`)}
        ${memoryMetaItem(t("createdAt"), escapeHtml(detail.createdAt))}
        ${memoryMetaItem(t("updatedAt"), escapeHtml(detail.updatedAt))}
        ${memoryMetaItem("last_access_time", escapeHtml(detail.lastAccess))}
        ${memoryMetaItem(t("sourceRetention"), detail.hasSource ? escapeHtml(t("sourceAvailableOnDemand")) : escapeHtml(t("sourceUnavailable")))}
        ${memoryMetaItem(t("sourceTimeStrategy"), escapeHtml(t(`sourceTimeStrategy_${detail.sourceTimeStrategy}`)))}
      </div>
    </div>
    ${renderSourceTimeTags(detail.sourceTimeTags)}
    ${memoryListSection(t("factsField"), detail.keyFacts)}
    ${memoryTagsSection(t("topicsField"), detail.topics)}
    ${memoryTagsSection(t("participantsField"), detail.participants)}
    ${memoryListSection(t("editHistory"), historyItems)}
    <details class="memory-detail-section">
      <summary class="memory-detail-section-title">${escapeHtml(t("rawMetadata"))}</summary>
      <pre class="memory-detail-json">${escapeHtml(JSON.stringify(detail.metadata, null, 2))}</pre>
    </details>
  `;
  $("memory-detail-edit").onclick = () => renderMemoryEditView(raw);
  $("memory-detail-edit-persona").onclick = () => renderMemoryPersonaEditView(raw);
  if (detail.hasSource) $("memory-detail-source").onclick = () => loadMemorySource(detail, raw);
  if (detail.status === "archived") $("memory-detail-restore").onclick = () => restoreMemory(detail.id);
  else $("memory-detail-archive").onclick = () => archiveMemory(detail.id);
  $("memory-detail-delete").onclick = () => deleteMemory(detail.id);
}

function renderSourceTimeTags(tags = {}) {
  const items = Object.entries(tags).filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!items.length) return "";
  return `<div class="memory-detail-section">
    <div class="memory-detail-section-title">${escapeHtml(t("sourceTimeTags"))}</div>
    <div class="memory-detail-meta-grid">${items.map(([key, value]) => memoryMetaItem(
      t(`sourceTimeTag_${key}`),
      escapeHtml(typeof value === "number" ? formatMemoryTime(value) : String(value)),
    )).join("")}</div>
  </div>`;
}

function renderSourceMessage(message, index) {
  const role = String(message?.role || message?.sender_name || t("unknownRole"));
  const sender = String(message?.sender_name || message?.sender_id || "");
  const timestamp = message?.timestamp == null ? "" : formatMemoryTime(message.timestamp);
  const content = String(message?.content || message?.text || "");
  return `<article class="memory-source-message">
    <header><strong>${escapeHtml(role)}</strong><span>${escapeHtml(sender)}</span><time>${escapeHtml(timestamp)}</time></header>
    <div>${escapeHtml(content || t("emptySourceMessage"))}</div>
    <small>${escapeHtml(t("sourceMessageNumber", { number: index + 1 }))}</small>
  </article>`;
}

async function loadMemorySource(detail, raw) {
  const button = $("memory-detail-source");
  if (button) button.disabled = true;
  try {
    const data = await selectedDatabaseApi(`/memories/${detail.id}/source`);
    const messages = Array.isArray(data?.source_messages) ? data.source_messages : [];
    $("memory-detail-body").innerHTML = `
      <div class="memory-detail-top">
        <div class="memory-detail-header"><span class="memory-detail-tag">${escapeHtml(t("sourceMessageCount", { count: messages.length }))}</span></div>
        <div class="memory-detail-actions"><button type="button" class="ghost" id="memory-source-back">${escapeHtml(t("backToMemoryDetail"))}</button></div>
      </div>
      <p class="panel-hint">${escapeHtml(t("sourceMessagesPrivacyHint"))}</p>
      <div class="memory-source-list">${messages.map(renderSourceMessage).join("") || `<div class="memory-detail-empty">${escapeHtml(t("sourceUnavailable"))}</div>`}</div>`;
    $("memory-source-back").onclick = () => renderMemoryDetailView(raw);
  } catch (error) {
    toast(error.message, true);
  } finally {
    if (button) button.disabled = false;
  }
}

async function archiveMemory(id) {
  return asyncGuard.run(`memory:${id}:archive`, async () => {
    if (!(await confirmDialog({
      title: t("archiveMemory"),
      message: t("archiveMemoryConfirm"),
      confirmText: t("archiveMemory"),
    }))) return;
    await selectedDatabaseApi(`/memories/${id}/archive`, { method: "POST", body: JSON.stringify({}) });
    toast(t("memoryArchived"));
    await loadMemories();
    await loadDatabases(false);
    await openMemoryDetail(id);
  }, { button: $("memory-detail-archive"), busyText: t("loading") });
}

async function restoreMemory(id) {
  return asyncGuard.run(`memory:${id}:restore`, async () => {
    if (!(await confirmDialog({
      title: t("restoreMemory"),
      message: t("restoreMemoryConfirm"),
      confirmText: t("restoreMemory"),
    }))) return;
    await selectedDatabaseApi(`/memories/${id}/restore`, { method: "POST", body: JSON.stringify({}) });
    toast(t("memoryRestored"));
    await loadMemories();
    await loadDatabases(false);
    await openMemoryDetail(id);
  }, { button: $("memory-detail-restore"), busyText: t("loading") });
}

function renderMemoryPersonaEditView(raw) {
  const detail = normalizeMemoryDetail(raw);
  $("memory-detail-title").textContent = t("editingMemoryPersona", { id: detail.id });
  $("memory-detail-body").innerHTML = `
    <div class="memory-detail-actions">
      <button type="button" class="primary" id="memory-persona-save">${escapeHtml(t("savePersona"))}</button>
      <button type="button" class="ghost" id="memory-persona-cancel">${escapeHtml(t("cancel"))}</button>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("personaField"))}</div>
      <input id="memory-persona-edit-value" value="${escapeHtml(detail.personaId === "—" ? "" : detail.personaId)}" placeholder="${escapeHtml(t("personaOptional"))}">
      <p class="panel-hint">${escapeHtml(t("personaEditIsolationHint"))}</p>
    </div>
  `;
  $("memory-persona-save").onclick = () => saveMemoryPersonaEdit(detail);
  $("memory-persona-cancel").onclick = () => renderMemoryDetailView(raw);
  setTimeout(() => $("memory-persona-edit-value")?.focus(), 0);
}

async function saveMemoryPersonaEdit(detail) {
  const personaId = $("memory-persona-edit-value").value.trim();
  const currentPersonaId = detail.personaId === "—" ? "" : detail.personaId;
  if (personaId === currentPersonaId) {
    toast(t("noChanges"));
    return;
  }
  const saveButton = $("memory-persona-save");
  saveButton.disabled = true;
  try {
    const updated = await selectedDatabaseApi(`/memories/${detail.id}/persona`, {
      method: "PATCH",
      body: JSON.stringify({ persona_id: personaId }),
    });
    state.selectedMemoryDetail = updated;
    toast(t("personaSavedNoReindex"));
    await loadMemories();
    await openMemoryDetail(detail.id, updated);
  } catch (error) {
    toast(error.message, true);
  } finally {
    saveButton.disabled = false;
  }
}

function renderMemoryEditView(raw) {
  const detail = normalizeMemoryDetail(raw);
  $("memory-detail-title").textContent = t("editingMemory", { id: detail.id });
  $("memory-detail-body").innerHTML = `
    <div class="memory-detail-actions">
      <button type="button" class="primary" id="memory-detail-save">${escapeHtml(t("saveMemory"))}</button>
      <button type="button" class="ghost" id="memory-detail-cancel">${escapeHtml(t("cancel"))}</button>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("content"))}</div>
      <textarea id="memory-edit-content" class="memory-detail-edit-area" required>${escapeHtml(detail.text)}</textarea>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("metadata"))}</div>
      <div class="memory-detail-edit-grid">
        <label><span>${escapeHtml(t("statusField"))}</span><div class="readonly-output">${memoryStatusPill(detail.status)}</div><small>${escapeHtml(t("memoryStatusManagedHint"))}</small></label>
        <label class="wide"><span>${escapeHtml(t("importanceField"))}</span><div class="memory-detail-slider"><input id="memory-edit-importance" type="range" min="0" max="10" step="0.1" value="${detail.importance.toFixed(1)}"><strong id="memory-edit-importance-value">${detail.importance.toFixed(1)}</strong></div></label>
        <label class="wide"><span>${escapeHtml(t("updateReason"))}</span><input id="memory-edit-reason" placeholder="${escapeHtml(t("reasonPlaceholder"))}"></label>
      </div>
    </div>
  `;
  $("memory-edit-importance").oninput = (event) => {
    $("memory-edit-importance-value").textContent = Number(event.target.value).toFixed(1);
  };
  $("memory-detail-save").onclick = () => saveMemoryDetailEdit(detail);
  $("memory-detail-cancel").onclick = () => renderMemoryDetailView(raw);
}

async function saveMemoryDetailEdit(detail) {
  const content = $("memory-edit-content").value.trim();
  const status = detail.status;
  const importance = Number($("memory-edit-importance").value);
  const reason = $("memory-edit-reason").value.trim();
  if (!content) {
    toast("content is required", true);
    return;
  }
  const metadata = {};
  if (reason) {
    metadata.update_history = [
      ...(Array.isArray(detail.metadata.update_history) ? detail.metadata.update_history : []),
      {
        timestamp: Math.floor(Date.now() / 1000),
        description: reason,
      },
    ];
  }
  const payload = {
    status,
    importance,
    value_scale: "display",
    metadata,
  };
  if (content !== detail.text) payload.content = content;
  if (
    content === detail.text
    && status === detail.status
    && Math.abs(importance - detail.importance) < 0.01
    && !reason
  ) {
    toast(t("noChanges"));
    return;
  }
  const saveButton = $("memory-detail-save");
  saveButton.disabled = true;
  try {
    const updated = await selectedDatabaseApi("/memories/" + detail.id, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    state.selectedMemoryDetail = updated;
    toast(t("saved"));
    await loadMemories();
    await loadDatabases(false);
    await openMemoryDetail(updated.new_memory_id || updated.id || detail.id, updated);
  } catch (error) {
    toast(error.message, true);
  } finally {
    saveButton.disabled = false;
  }
}

$("memory-detail-close")?.addEventListener("click", closeMemoryDetail);
$("memory-detail-overlay")?.addEventListener("click", closeMemoryDetail);

function splitList(value) {
  return value
    .split(/[,，\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

$("memory-create").onclick = () => openModal();

function openModal(item = null) {
  $("form-id").value = item?.id || "";
  $("form-content").value = item?.text || "";
  $("form-persona").value = item?.metadata?.persona_id || "";
  $("form-session").value = item?.metadata?.session_id || "";
  $("form-importance").value = item?.metadata?.importance ?? 0.5;
  $("form-status").value = item?.metadata?.status || "active";
  $("form-topics").value = (item?.metadata?.topics || []).join(", ");
  $("form-participants").value = (item?.metadata?.participants || []).join(", ");
  $("form-facts").value = (item?.metadata?.key_facts || []).join(", ");
  $("modal-title").textContent = item ? `${t("memoryModalEdit")}${item.id}` : t("memoryModalNew");
  $("modal").classList.remove("hidden");
}

function closeModal() {
  $("modal").classList.add("hidden");
}

$("modal-close").onclick = closeModal;
$("modal-cancel").onclick = closeModal;

$("memory-form").onsubmit = async (event) => {
  event.preventDefault();
  const id = Number($("form-id").value) || null;
  const topics = splitList($("form-topics").value);
  const participants = splitList($("form-participants").value);
  const facts = splitList($("form-facts").value);
  const payload = {
    content: $("form-content").value,
    persona_id: $("form-persona").value || null,
    session_id: $("form-session").value || null,
    importance: Number($("form-importance").value),
    status: $("form-status").value,
    topics,
    participants,
    key_facts: facts,
    metadata: {
      topics,
      participants,
      key_facts: facts,
    },
  };
  if (id) {
    payload.value_scale = "stored";
  }
  await asyncGuard.run(`memory-form:${id || "new"}`, async () => {
  try {
    await selectedDatabaseApi(id ? "/memories/" + id : "/memories", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify(payload),
    });
    closeModal();
    toast(t("saved"));
    await loadMemories();
    await loadDatabases(false);
  } catch (error) {
    toast(error.message, true);
  }
  }, {
    form: event.currentTarget,
    button: event.submitter,
    busyText: t("loading"),
  });
};

async function deleteMemory(id) {
  return asyncGuard.run(`memory:${id}:delete`, async () => {
  if (!(await confirmDialog({
    title: t("confirmTitle"),
    message: `${t("confirmDelete")}${id}？`,
    confirmText: t("tableDelete"),
    danger: true,
  }))) {
    return;
  }
  try {
    await selectedDatabaseApi("/memories/" + id, { method: "DELETE" });
    toast(t("deleted"));
    closeMemoryDetail();
    await loadMemories();
    await loadDatabases(false);
  } catch (error) {
    toast(error.message, true);
  }
  }, {
    button: $("memory-detail-delete"),
    busyText: t("loading"),
  });
}

  return { loadMemories };
}
