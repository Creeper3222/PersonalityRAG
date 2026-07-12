export function createMemoriesController({ $, state, t, toast, api, libraryApi, escapeHtml, formatMemoryTime, displayMemoryType, displayStatus, memoryStatusPill, memoryTypeTag, memoryImportanceBar, normalizeMemoryDetail, memoryMetaItem, memoryListSection, memoryTagsSection, renderMemoryMiniGraph, loadLibraries, debounce, confirmDialog }) {
async function loadMemories() {
  const params = new URLSearchParams({
    page: state.memoryPage,
    page_size: state.memoryPageSize,
    keyword: $("memory-keyword").value,
    session_id: $("memory-session")?.value || "",
    persona_id: $("memory-persona").value,
    status: $("memory-status")?.value || "",
    memory_type: $("memory-type")?.value || "",
    sort: $("memory-sort").value,
  });
  try {
    const data = await libraryApi("/memories?" + params);
    state.memoryHasMore = data.has_more;
    state.memoryItems = Array.isArray(data.items) ? data.items : [];
    $("memory-rows").innerHTML =
      state.memoryItems
        .map(
          (item) => {
            const metadata = item.metadata || {};
            const updated = formatMemoryTime(metadata.updated_at ?? item.updated_at ?? metadata.create_time);
            const created = formatMemoryTime(metadata.create_time ?? item.created_at);
            return `<tr class="memory-row" data-id="${escapeHtml(item.id)}" tabindex="0">
            <td class="memory-id">${escapeHtml(item.id)}</td>
            <td class="memory-summary-cell" title="${escapeHtml(item.text)}"><div class="memory-summary-text">${escapeHtml(item.text)}</div><div class="memory-summary-meta">${escapeHtml(t("updatedAt"))} ${escapeHtml(updated)}</div></td>
            <td>${memoryTypeTag(metadata.memory_type || "GENERAL")}</td>
            <td>${memoryImportanceBar(metadata.importance ?? 0.5)}</td>
            <td>${memoryStatusPill(metadata.status || "active")}</td>
            <td>${escapeHtml(created)}</td>
          </tr>`;
          },
        )
        .join("") || `<tr><td colspan="6">${escapeHtml(t("tableEmpty"))}</td></tr>`;
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
$("memory-type").onchange = () => {
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

async function openMemoryDetail(id, fallback = null) {
  state.selectedMemoryId = id;
  $("memory-detail-title").textContent = t("memoryDetails", { id });
  $("memory-detail-badge").textContent = "memory";
  $("memory-detail-body").innerHTML = `<div class="memory-detail-empty">${escapeHtml(t("loading"))}</div>`;
  $("memory-detail-overlay").classList.remove("hidden");
  $("memory-detail-panel").classList.add("visible");
  try {
    const item = await libraryApi("/memories/" + id);
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
  $("memory-detail-badge").textContent = displayMemoryType(detail.type);
  const historyItems = detail.updateHistory.map((item) => {
    const time = formatMemoryTime(item.timestamp || item.time);
    const text = item.description || `${item.field || ""}: ${item.old_value ?? ""} -> ${item.new_value ?? ""}`;
    return `${time} ${text}`.trim();
  });
  $("memory-detail-body").innerHTML = `
    <div class="memory-detail-top">
      <div class="memory-detail-header">
        ${memoryStatusPill(detail.status)}
        ${memoryTypeTag(detail.type)}
        <span class="memory-type-tag">${escapeHtml(t("importanceField"))}: ${detail.importance.toFixed(1)}/10</span>
      </div>
      <div class="memory-detail-actions">
        <button type="button" class="ghost" id="memory-detail-edit">${escapeHtml(t("editMemory"))}</button>
        <button type="button" class="ghost danger" id="memory-detail-delete">${escapeHtml(t("deleteMemory"))}</button>
      </div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("content"))}</div>
      <div class="memory-detail-content">${escapeHtml(detail.text)}</div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("graphContext"))}</div>
      <div class="memory-detail-graph">${renderMemoryMiniGraph(detail.graph)}</div>
    </div>
    <div class="memory-detail-section">
      <div class="memory-detail-section-title">${escapeHtml(t("metadata"))}</div>
      <div class="memory-detail-meta-grid">
        ${memoryMetaItem(t("statusField"), memoryStatusPill(detail.status))}
        ${memoryMetaItem(t("typeLabel"), memoryTypeTag(detail.type))}
        ${memoryMetaItem(t("importanceField"), `${detail.importance.toFixed(1)} / 10`)}
        ${memoryMetaItem(t("sessionField"), `<code>${escapeHtml(detail.sessionId)}</code>`)}
        ${memoryMetaItem(t("personaField"), `<div class="memory-persona-value"><code>${escapeHtml(detail.personaId)}</code><button type="button" class="ghost" id="memory-detail-edit-persona">${escapeHtml(t("editPersona"))}</button></div>`)}
        ${memoryMetaItem(t("createdAt"), escapeHtml(detail.createdAt))}
        ${memoryMetaItem(t("updatedAt"), escapeHtml(detail.updatedAt))}
        ${memoryMetaItem("last_access_time", escapeHtml(detail.lastAccess))}
      </div>
    </div>
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
  $("memory-detail-delete").onclick = () => deleteMemory(detail.id);
}

function renderMemoryPersonaEditView(raw) {
  const detail = normalizeMemoryDetail(raw);
  $("memory-detail-title").textContent = t("editingMemoryPersona", { id: detail.id });
  $("memory-detail-badge").textContent = displayMemoryType(detail.type);
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
    const updated = await libraryApi(`/memories/${detail.id}/persona`, {
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
  $("memory-detail-badge").textContent = displayMemoryType(detail.type);
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
        <label><span>${escapeHtml(t("statusField"))}</span><select id="memory-edit-status">
          <option value="active" ${detail.status === "active" ? "selected" : ""}>${escapeHtml(displayStatus("active"))}</option>
          <option value="archived" ${detail.status === "archived" ? "selected" : ""}>${escapeHtml(displayStatus("archived"))}</option>
          <option value="deleted" ${detail.status === "deleted" ? "selected" : ""}>${escapeHtml(displayStatus("deleted"))}</option>
        </select></label>
        <label><span>${escapeHtml(t("typeLabel"))}</span><input id="memory-edit-type" value="${escapeHtml(detail.type)}"></label>
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
  const status = $("memory-edit-status").value;
  const memoryType = $("memory-edit-type").value.trim() || "GENERAL";
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
    memory_type: memoryType,
    importance,
    value_scale: "display",
    metadata,
  };
  if (content !== detail.text) payload.content = content;
  if (
    content === detail.text
    && status === detail.status
    && memoryType === detail.type
    && Math.abs(importance - detail.importance) < 0.01
    && !reason
  ) {
    toast(t("noChanges"));
    return;
  }
  const saveButton = $("memory-detail-save");
  saveButton.disabled = true;
  try {
    const updated = await libraryApi("/memories/" + detail.id, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    state.selectedMemoryDetail = updated;
    toast(t("saved"));
    await loadMemories();
    await loadLibraries(false);
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
  $("form-type").value = item?.metadata?.memory_type || "GENERAL";
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
    memory_type: $("form-type").value,
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
  try {
    await libraryApi(id ? "/memories/" + id : "/memories", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify(payload),
    });
    closeModal();
    toast(t("saved"));
    await loadMemories();
    await loadLibraries(false);
  } catch (error) {
    toast(error.message, true);
  }
};

async function deleteMemory(id) {
  if (!(await confirmDialog({
    title: t("confirmTitle"),
    message: `${t("confirmDelete")}${id}？`,
    confirmText: t("tableDelete"),
    danger: true,
  }))) {
    return;
  }
  try {
    await libraryApi("/memories/" + id, { method: "DELETE" });
    toast(t("deleted"));
    closeMemoryDetail();
    await loadMemories();
    await loadLibraries(false);
  } catch (error) {
    toast(error.message, true);
  }
}

  return { loadMemories };
}
