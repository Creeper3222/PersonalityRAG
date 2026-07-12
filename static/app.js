import enUS from "./locales/en.js";
import ruRU from "./locales/ru.js";
import zhCN from "./locales/zh.js";
import { createGraphController } from "./modules/graph.js";
import { createTaskLogController } from "./modules/tasks-logs.js";
import { createApiClient, parseDownloadFilename, responseError } from "./modules/api.js";
import { createSettingsController } from "./modules/settings.js";
import { createLibrariesController } from "./modules/libraries.js";
import { createProvidersController } from "./modules/providers.js";
import { createMemoriesController } from "./modules/memories.js";
import { createRecallController } from "./modules/recall.js";
import { createSystemController } from "./modules/system.js";
import { createFileManagerController } from "./modules/files.js";
const $ = (id) => document.getElementById(id);

document.querySelector('.nav[data-page="system"]')?.after(
  document.querySelector('.nav[data-page="files"]'),
);

const APP_PAGES = new Set([
  "libraries",
  "providers",
  "graph",
  "memory",
  "recall",
  "system",
  "files",
  "settings",
  "logs",
]);
const RESTART_RETURN_PAGE = "settings";

function isValidPage(page) {
  return APP_PAGES.has(page);
}

function consumeInitialRouteState() {
  const url = new URL(window.location.href);
  const requestedPage = url.searchParams.get("page");
  const page = isValidPage(requestedPage) ? requestedPage : "libraries";
  const shouldCleanup =
    url.searchParams.has("page") || url.searchParams.has("restart");
  if (shouldCleanup) {
    url.searchParams.delete("page");
    url.searchParams.delete("restart");
    window.history.replaceState({}, document.title, url);
  }
  return { page };
}

const initialRouteState = consumeInitialRouteState();

const state = {
  page: initialRouteState.page,
  memoryPage: 1,
  memoryPageSize: 20,
  memoryHasMore: false,
  memoryItems: [],
  selectedMemoryId: null,
  selectedMemoryDetail: null,
  stats: null,
  libraries: [],
  providers: [],
  providerTypes: [],
  providerStatuses: {},
  providerKind: localStorage.getItem("prag_provider_kind") || "embedding",
  recallView: "embedding",
  recallCache: {
    embedding: [],
    rerank: [],
    rerankMeta: null,
    summary: null,
  },
  settings: null,
  loginMode: "api_key",
  authGeneration: 0,
  systemProviderExpanded: false,
  systemIndexExpanded: false,
  systemPanelRefreshFrame: 0,
  restarting: false,
  restart: {
    timer: null,
    startedAt: 0,
    targetUrl: "",
    probeUrls: [],
    probeCursor: 0,
    requireOfflineTransition: false,
    sawOfflineTransition: false,
    lastPhase: "",
  },
  selectedLibraryId: localStorage.getItem("prag_library_id") || "",
  expandedLibraryIds: new Set(),
  optimisticIndexConflicts: new Set(),
  logs: {
    items: [],
    lastId: 0,
    maxEntries: 2000,
    activeLevels: new Set(["INFO", "WARN", "ERROR"]),
    autoScroll: true,
    polling: false,
    pollTimer: null,
    abortController: null,
    generation: 0,
  },
  tasks: {
    active: [],
    finished: [],
    optimistic: [],
    finishedExpanded: false,
    scope: "active",
    polling: false,
    pollTimer: null,
    watchers: new Map(),
  },
};

const DEFAULT_LIBRARY_CONVERSATION_SETTINGS = {
  max_sessions: 100,
  session_ttl: 3600,
  context_window_size: 300,
  max_messages_per_session: 1000,
  cleanup_batch_size: 50,
};

const DEFAULT_LIBRARY_RECALL_SETTINGS = {
  rrf_k: 60,
  decay_rate: 0,
  access_decay_window_days: 30,
  access_decay_max_count: 10,
  access_count_decay_multiplier: 0.5,
  importance_weight: 1,
  graph_memory_enabled: true,
  document_route_weight: 0.65,
  graph_route_weight: 0.35,
  cross_route_bonus: 0.08,
  graph_expansion_limit: 24,
  graph_expansion_hops: 1,
  graph_second_hop_weight: 0.4,
  dynamic_route_weighting: true,
  graph_max_topics: 6,
  graph_max_participants: 8,
  graph_max_facts: 8,
  use_persona_filtering: true,
  use_session_filtering: false,
  search_cache_enabled: true,
  search_cache_ttl_seconds: 45,
  search_cache_max_size: 256,
};

const DEFAULT_LIBRARY_MAINTENANCE_SETTINGS = {
  atom_enabled: true,
  atom_maintenance_interval_hours: 24,
  atom_forget_delay_days: 7,
  atom_purge_delay_days: 30,
  backup_enabled: true,
  backup_keep_days: 7,
  auto_cleanup_enabled: false,
  cleanup_days_threshold: 7,
  cleanup_importance_threshold: 0.3,
};

const DEFAULT_RECALL_K = 5;
const DEFAULT_RERANK_K = 5;
const IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]+$/;
const LIBRARY_EXPAND_ICON = `<svg viewBox="0 0 240 24" aria-hidden="true"><circle cx="24" cy="12" r="4.5"/><circle cx="120" cy="12" r="4.5"/><circle cx="216" cy="12" r="4.5"/></svg>`;

function validateIdentifierInput(input) {
  const value = input?.value ?? "";
  if (IDENTIFIER_PATTERN.test(value)) return value;
  toast(t("invalidIdentifier"), true);
  input?.focus();
  return null;
}

function clampRecallK(value, max = 50) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_RECALL_K;
  return Math.max(1, Math.min(max, Math.round(numeric)));
}

function setRecallK(value = DEFAULT_RECALL_K) {
  const next = clampRecallK(value);
  const input = $("recall-k");
  const output = $("recall-k-value");
  if (input) input.value = String(next);
  if (output) output.value = String(next);
  if (output) output.textContent = String(next);
  const rerankInput = $("recall-rerank-k");
  if (rerankInput) {
    rerankInput.max = String(next);
    setRecallRerankK(Math.min(Number(rerankInput.value || DEFAULT_RERANK_K), next));
  }
  return next;
}

function setRecallRerankK(value = DEFAULT_RERANK_K) {
  const embeddingK = Number($("recall-k")?.value || DEFAULT_RECALL_K);
  const next = clampRecallK(value, Math.max(1, embeddingK));
  const input = $("recall-rerank-k");
  const output = $("recall-rerank-k-value");
  if (input) {
    input.max = String(Math.max(1, embeddingK));
    input.value = String(next);
  }
  if (output) output.value = String(next);
  if (output) output.textContent = String(next);
  return next;
}

const strings = { zh: zhCN, en: enUS, ru: ruRU };
let lang = localStorage.getItem("prag_lang") || "zh";

function t(key, replacements = {}) {
  const template = strings[lang]?.[key] ?? strings.zh[key] ?? key;
  return Object.entries(replacements).reduce(
    (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
    template,
  );
}

function joinLocalizedList(items = []) {
  const values = items.filter(Boolean);
  if (values.length <= 1) {
    return values[0] || "";
  }
  return values.join(lang === "zh" ? "、" : ", ");
}

function applyLanguage() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (key) {
      element.textContent = t(key);
    }
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => {
    const key = element.dataset.i18nPlaceholder;
    if (key) {
      element.placeholder = t(key);
    }
  });
  $("page-title").textContent = t(state.page);
  applyLoginMode({ login_mode: state.loginMode });
  $("memory-sort").querySelector('[value="created_desc"]').textContent = t("sortNewest");
  $("memory-sort").querySelector('[value="created_asc"]').textContent = t("sortOldest");
  $("memory-sort").querySelector('[value="importance_desc"]').textContent = t("sortImportanceDesc");
  $("memory-sort").querySelector('[value="importance_asc"]').textContent = t("sortImportanceAsc");
  $("language").value = lang;
}

function logUiFeedback(message, level = "INFO") {
  fetch("/api/v1/logs/ui", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      level,
      message,
      context: { page: state.page },
    }),
  }).catch(() => {});
}

function toast(message, error = false, options = {}) {
  const element = $("toast");
  element.textContent = message;
  element.classList.remove("success", "error", "show");
  element.classList.add(error ? "error" : "success");
  element.classList.add("show");
  if (options.log !== false) {
    logUiFeedback(message, options.level || (error ? "WARN" : "INFO"));
  }
  clearTimeout(element._timer);
  element._timer = setTimeout(() => element.classList.remove("show"), 2600);
}

function logDirectRequestError(error) {
  if (error?.name === "AbortError") return;
  if (!Number(error?.status) || Number(error.status) >= 500) {
    logUiFeedback(error?.message || "Network request failed", "ERROR");
  }
}

function confirmDialog({ title = t("confirmTitle"), message = "", confirmText = t("confirmAction"), danger = false } = {}) {
  return new Promise((resolve) => {
    const overlay = $("confirm-modal");
    const ok = $("confirm-ok");
    const cancel = $("confirm-cancel");
    const close = $("confirm-close");
    $("confirm-title").textContent = title;
    $("confirm-message").textContent = message;
    ok.textContent = confirmText;
    overlay.querySelector(".confirm-modal").classList.toggle("danger", Boolean(danger));
    overlay.classList.remove("hidden");
    const cleanup = (value) => {
      overlay.classList.add("hidden");
      ok.onclick = null;
      cancel.onclick = null;
      close.onclick = null;
      overlay.onclick = null;
      resolve(value);
    };
    ok.onclick = () => cleanup(true);
    cancel.onclick = () => cleanup(false);
    close.onclick = () => cleanup(false);
    overlay.onclick = (event) => {
      if (event.target === overlay) cleanup(false);
    };
  });
}

const api = createApiClient({
  onUnauthorized: () => showLogin(),
  unauthorizedMessage: () => t("unauthorized"),
  onOperationalError: (error) => logUiFeedback(error.message, "ERROR"),
  getUnauthorizedGeneration: () => state.authGeneration,
});

const {
  requestBackupPassword, requestBackupExportScope,
  settingsDraft, hasUnsavedSettingsChanges, showRestartScreen, loadSettings,
} = createSettingsController({
  $, state, t, api, toast,
  stopLogPolling: (...args) => stopLogPolling(...args),
  stopTaskPolling: (...args) => stopTaskPolling(...args),
  resetTaskState: (...args) => resetTaskState(...args),
  isValidPage,
  restartReturnPage: RESTART_RETURN_PAGE,
});
function libraryApi(path, options = {}) {
  if (!state.selectedLibraryId) {
    throw new Error("请先选择记忆库");
  }
  return api(`/libraries/${encodeURIComponent(state.selectedLibraryId)}${path}`, options);
}

function selectedLibrary() {
  return state.libraries.find((item) => item.id === state.selectedLibraryId) || null;
}

function selectedLibraryHasRerank() {
  const library = selectedLibrary();
  return Boolean(library?.rerank_provider_id || library?.rerank_provider?.id);
}

function updateRecallRerankControls() {
  const hasRerank = selectedLibraryHasRerank();
  $("recall-k-control")?.classList.remove("hidden");
  $("recall-rerank-k-control")?.classList.toggle("hidden", !hasRerank);
  $("recall-view-rerank")?.classList.toggle("hidden", !hasRerank);
  if ($("recall-rerank-k")) {
    $("recall-rerank-k").disabled = !hasRerank;
  }
  if ($("recall-view-rerank")) {
    $("recall-view-rerank").disabled = !hasRerank;
    $("recall-view-rerank").setAttribute("aria-hidden", hasRerank ? "false" : "true");
  }
  if (!hasRerank && state.recallView === "rerank") {
    state.recallView = "embedding";
  }
  document.querySelectorAll("[data-recall-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.recallView === state.recallView);
  });
}

function refreshSidebarLibrary() {
  const box = $("sidebar-current-library");
  if (!box) return;
  const library = selectedLibrary();
  if (!library) {
    box.classList.add("hidden");
    return;
  }
  const provider = library.provider || {};
  $("sidebar-library-name").textContent = library.name || library.id;
  $("sidebar-library-id").textContent = library.id;
  $("sidebar-library-provider").textContent = `${provider.display_name || provider.id || "未绑定"}`;
  box.title = `${library.name || library.id}\n${library.id}`;
  box.classList.remove("hidden");
}

function refreshLibraryContext() {
  refreshSidebarLibrary();
  updateRecallRerankControls();
  const button = $("library-context");
  const library = selectedLibrary();
  const libraryPages = ["graph", "memory", "recall", "system"];
  if (!library || !libraryPages.includes(state.page)) {
    button.classList.add("hidden");
    return;
  }
  const provider = library.provider || {};
  const generation = library.indexes?.generation || "尚未构建";
  button.innerHTML = `<strong>${escapeHtml(library.name)}</strong><small>${escapeHtml(provider.display_name || provider.id || "未绑定")} · ${escapeHtml(generation)}</small>`;
  button.classList.remove("hidden");
}

function updateLibraryCardSelection() {
  document.querySelectorAll(".library-card").forEach((card) => {
    const selected = card.dataset.id === state.selectedLibraryId;
    card.classList.toggle("active-card", selected);
    card.setAttribute("aria-pressed", selected ? "true" : "false");
    card.querySelector(".selected-library-badge")?.classList.toggle("hidden", !selected);
  });
}

function selectLibrary(libraryId, options = {}) {
  if (!libraryId) return;
  const changed = state.selectedLibraryId !== libraryId;
  state.selectedLibraryId = libraryId;
  localStorage.setItem("prag_library_id", state.selectedLibraryId);
  if (options.resetMemoryPage !== false) {
    state.memoryPage = 1;
  }
  if (changed) {
    state.recallCache = {
      embedding: [],
      rerank: [],
      rerankMeta: null,
      summary: null,
    };
    if ($("recall-results")) {
      renderRecallResults();
    }
  }
  refreshLibraryContext();
  updateLibraryCardSelection();
}

function showLogin() {
  stopLogPolling();
  stopTaskPolling();
  resetTaskState({ render: false });
  if (!$("app").classList.contains("hidden")) {
    state.authGeneration += 1;
  }
  $("login").classList.remove("hidden");
  $("app").classList.add("hidden");
}

function showApp() {
  $("login").classList.add("hidden");
  $("app").classList.remove("hidden");
}

function displayStatus(status) {
  const mapping = {
    active: t("statusActive"),
    archived: t("statusArchived"),
    deleted: t("statusDeleted"),
  };
  return mapping[status] || status;
}

function applyLoginMode(status = {}) {
  state.loginMode = status.login_mode || (status.login_password_enabled ? "password" : "api_key");
  const hint = $("login")?.querySelector("[data-i18n='loginHint']");
  const input = $("api-key");
  if (!hint || !input) return;
  if (state.loginMode === "password") {
    hint.textContent = t("loginHintPassword");
    input.placeholder = "WebUI 登录密码";
  } else {
    hint.textContent = t("loginHint");
    input.placeholder = "prag_...";
  }
}

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  try {
    await api("/auth/login", {
      method: "POST",
      body: JSON.stringify({ credential: $("api-key").value }),
      suppressUnauthorizedHandler: true,
    });
    state.authGeneration += 1;
    showApp();
    await activatePage(state.page);
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});

$("logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
  state.authGeneration += 1;
  showLogin();
});

$("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("prag_theme", next);
  graphController.render();
});

$("language").addEventListener("change", (event) => {
  lang = event.target.value;
  localStorage.setItem("prag_lang", lang);
  applyLanguage();
  if (state.page === "graph" && state.stats) {
    loadGraph();
  }
  if (state.page === "memory") {
    loadMemories();
  }
  if (state.page === "system" && state.stats) {
    loadSystem();
  }
  if (state.page === "files") fileController.renderFiles();
  if (state.page === "libraries") loadLibraries();
  if (state.page === "providers") loadProviders();
});

document.documentElement.dataset.theme = localStorage.getItem("prag_theme") || "light";

function setActivePage(page) {
  const nextPage = isValidPage(page) ? page : "libraries";
  document.querySelectorAll(".nav[data-page]").forEach((item) => {
    item.classList.toggle("active", item.dataset.page === nextPage);
  });
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("active", section.id === `page-${nextPage}`);
  });
  state.page = nextPage;
  applyLanguage();
  refreshLibraryContext();
}

async function activatePage(page) {
  setActivePage(page);
  await loadPage(state.page);
}

document.querySelectorAll(".nav[data-page]").forEach((button) =>
  button.addEventListener("click", async () => {
    await activatePage(button.dataset.page);
  }),
);

async function loadPage(page) {
  if (page !== "logs") {
    stopLogPolling();
    stopTaskPolling();
  }
  if (page === "libraries") await loadLibraries();
  if (page === "providers") await loadProviders();
  if (["graph", "memory", "recall", "system"].includes(page)) {
    await ensureLibrarySelection();
  }
  if (page === "graph") await loadGraph();
  if (page === "memory") await loadMemories();
  if (page === "system") await loadSystem();
  if (page === "files") await fileController.loadFiles();
  if (page === "settings") await loadSettings();
  if (page === "logs") {
    state.tasks.scope = state.tasks.scope || "active";
    renderTasks();
    renderLogs();
    startLogPolling();
    startTaskPolling();
  }
  refreshLibraryContext();
}

function statCards(target, items) {
  target.innerHTML = items
    .map(
      ([label, value]) =>
        `<div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`,
    )
    .join("");
}

function formatVersionTag(value) {
  const text = String(value ?? "").trim();
  if (!text) return "—";
  return text.startsWith("v") ? text : `v${text}`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function formatMemoryTime(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    return new Date(value * 1000).toLocaleString();
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric > 1000000000) {
    return new Date(numeric * 1000).toLocaleString();
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function normalizeMemoryImportance(value) {
  const numeric = Number(value ?? 0.5);
  if (!Number.isFinite(numeric)) return 5;
  const displayValue = numeric <= 1 ? numeric * 10 : numeric;
  return Math.max(0, Math.min(10, displayValue));
}

function memoryImportanceClass(value) {
  const score = normalizeMemoryImportance(value);
  return score >= 7 ? "high" : score >= 4 ? "medium" : "low";
}

function displayMemoryType(type) {
  const normalized = String(type || "GENERAL").toUpperCase();
  const mapping = {
    GENERAL: t("typeGeneral"),
    FACT: t("typeFact"),
    EVENT: t("typeEvent"),
    PREFERENCE: t("typePreference"),
    OPINION: t("typeOpinion"),
    FACTUAL: t("typeFactual"),
    EPISODIC: t("typeEpisodic"),
    RELATIONAL: t("typeRelational"),
    PLANNED: t("typePlanned"),
    GROUP_CHAT: t("typeGroupChat"),
    PRIVATE_CHAT: t("typePrivateChat"),
    MANUAL: t("typeManual"),
  };
  return mapping[normalized] || normalized;
}

function memoryStatusPill(status) {
  const value = String(status || "active").toLowerCase();
  return `<span class="memory-status-pill ${escapeHtml(value)}">${escapeHtml(displayStatus(value))}</span>`;
}

function memoryTypeTag(type) {
  return `<span class="memory-type-tag">${escapeHtml(displayMemoryType(type))}</span>`;
}

function memoryImportanceBar(value) {
  const score = normalizeMemoryImportance(value);
  return `<div class="memory-importance">
    <div class="memory-importance-track"><div class="memory-importance-fill ${memoryImportanceClass(score)}" style="width:${score * 10}%"></div></div>
    <span>${score.toFixed(1)}</span>
  </div>`;
}

function normalizeMemoryDetail(raw = {}) {
  const metadata = raw.metadata || {};
  return {
    id: Number(raw.id ?? raw.memory_id),
    text: raw.text || raw.content || raw.summary || "",
    metadata,
    type: metadata.memory_type || raw.memory_type || "GENERAL",
    importance: normalizeMemoryImportance(metadata.importance ?? raw.importance),
    status: metadata.status || raw.status || "active",
    sessionId: metadata.session_id ?? raw.session_id ?? "—",
    personaId: metadata.persona_id ?? raw.persona_id ?? "—",
    createdAt: formatMemoryTime(metadata.create_time ?? raw.created_at),
    updatedAt: formatMemoryTime(metadata.updated_at ?? raw.updated_at ?? metadata.create_time),
    lastAccess: formatMemoryTime(metadata.last_access_time),
    topics: Array.isArray(metadata.topics) ? metadata.topics : [],
    participants: Array.isArray(metadata.participants) ? metadata.participants : [],
    keyFacts: Array.isArray(metadata.key_facts) ? metadata.key_facts : [],
    updateHistory: Array.isArray(metadata.update_history) ? metadata.update_history : [],
    graph: raw.graph_context || null,
    raw,
  };
}

function memoryMetaItem(label, value) {
  return `<div class="memory-detail-meta-item"><span>${escapeHtml(label)}</span><strong>${value}</strong></div>`;
}

function memoryListSection(title, items) {
  if (!items?.length) return "";
  return `<div class="memory-detail-section">
    <div class="memory-detail-section-title">${escapeHtml(title)}</div>
    <div class="memory-detail-list">${items.map((item) => `<div class="memory-detail-list-item">${escapeHtml(item)}</div>`).join("")}</div>
  </div>`;
}

function memoryTagsSection(title, items) {
  if (!items?.length) return "";
  return `<div class="memory-detail-section">
    <div class="memory-detail-section-title">${escapeHtml(title)}</div>
    <div class="memory-detail-tags">${items.map((item) => `<span class="memory-detail-tag">${escapeHtml(item)}</span>`).join("")}</div>
  </div>`;
}

function renderMemoryMiniGraph(graph) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes.slice(0, 24) : [];
  if (!nodes.length) {
    return `<div class="memory-detail-empty">${escapeHtml(t("noGraphContext"))}</div>`;
  }
  const edges = Array.isArray(graph?.edges) ? graph.edges.slice(0, 36) : [];
  const width = 520;
  const height = 180;
  const cx = width / 2;
  const cy = height / 2;
  const rx = 150;
  const ry = 58;
  const positions = new Map();
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
    positions.set(String(node.id), {
      x: cx + Math.cos(angle) * rx,
      y: cy + Math.sin(angle) * ry,
    });
  });
  const lines = edges
    .map((edge) => {
      const source = positions.get(String(edge.source));
      const target = positions.get(String(edge.target));
      if (!source || !target) return "";
      return `<line x1="${source.x.toFixed(1)}" y1="${source.y.toFixed(1)}" x2="${target.x.toFixed(1)}" y2="${target.y.toFixed(1)}" />`;
    })
    .join("");
  const fallbackLines = nodes
    .map((node) => {
      const point = positions.get(String(node.id));
      return point ? `<line x1="${cx}" y1="${cy}" x2="${point.x.toFixed(1)}" y2="${point.y.toFixed(1)}" />` : "";
    })
    .join("");
  const dots = nodes
    .map((node) => {
      const point = positions.get(String(node.id));
      return point ? `<circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="5.2"><title>${escapeHtml(node.label || node.canonical_value || node.id)}</title></circle>` : "";
    })
    .join("");
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(t("graphContext"))}">
    <g stroke="rgba(148,163,184,.36)" stroke-width="1.2">${lines || fallbackLines}</g>
    <circle cx="${cx}" cy="${cy}" r="7" fill="var(--accent)" opacity=".85"></circle>
    <g fill="#3b82f6">${dots}</g>
  </svg>`;
}

const graphController = createGraphController({ $, state, t, libraryApi, statCards, toast, escapeHtml });
async function loadGraph(payload = null) {
  return graphController.loadGraph(payload);
}

const { loadMemories } = createMemoriesController({
  $, state, t, toast, api, libraryApi, escapeHtml, formatMemoryTime,
  displayMemoryType, memoryStatusPill, memoryTypeTag, memoryImportanceBar,
  displayStatus,
  normalizeMemoryDetail, memoryMetaItem, memoryListSection, memoryTagsSection,
  renderMemoryMiniGraph, loadLibraries: (...args) => loadLibraries(...args),
  debounce, confirmDialog,
});
const { setRecallView, renderRecallResults } = createRecallController({
  $, state, t, toast, libraryApi, escapeHtml, selectedLibrary, selectedLibraryHasRerank,
  setRecallK, setRecallRerankK, updateRecallRerankControls,
});
const { loadSystem, scheduleSystemPanelsRefresh, formatBytes } = createSystemController({
  $, state, t, toast, libraryApi, statCards, formatVersionTag, escapeHtml,
});
const fileController = createFileManagerController({
  $, state, t, api, toast, responseError, parseDownloadFilename, confirmDialog, escapeHtml,
});
const {
  loadTasks, startTaskPolling, stopTaskPolling, renderTasks,
  resetTaskState, applyTaskHistoryCollapseState: refreshTaskHistoryCollapseState,
  loadLogs, startLogPolling, stopLogPolling, clearLogs,
  trackQueuedJob, addOptimisticTask, updateOptimisticTask, removeOptimisticTask,
  renderLogs, renderLogAutoScrollState,
} = createTaskLogController({
  $, state, t, api, toast, escapeHtml, formatBytes, selectedLibrary,
  loadLibraries: (...args) => loadLibraries(...args),
  loadGraph, loadMemories, loadSystem,
  confirmDialog,
  clearLibraryIndexConflict: (...args) => clearLibraryIndexConflict(...args),
  refreshLibraryContext,
});
const { loadLibraries, ensureLibrarySelection, navigate, bindUsedListDetails, confirmSensitiveProviderEdit, clearLibraryIndexConflict } = createLibrariesController({
  $, state, t, toast, api, libraryApi, selectedLibrary, selectLibrary, refreshLibraryContext,
  escapeHtml, confirmDialog, addOptimisticTask, removeOptimisticTask, trackQueuedJob,
  validateIdentifierInput, LIBRARY_EXPAND_ICON,
  activatePage, closeOverlay, joinLocalizedList,
  DEFAULT_LIBRARY_CONVERSATION_SETTINGS, DEFAULT_LIBRARY_RECALL_SETTINGS,
  DEFAULT_LIBRARY_MAINTENANCE_SETTINGS,
  loadProviders: (...args) => loadProviders(...args),
  fillProviderSelect: (...args) => fillProviderSelect(...args),
});
const { loadProviders, fillProviderSelect } = createProvidersController({
  $, state, t, toast, api, escapeHtml, validateIdentifierInput,
  confirmSensitiveProviderEdit, navigate, loadLibraries, confirmDialog, closeOverlay, selectLibrary,
});
$("settings-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const draft = settingsDraft();
  const payload = {
    access_base_url: draft.access_base_url,
    port: draft.port,
    access_port: draft.access_port,
    new_password: draft.new_password || null,
    clear_password: draft.clear_password,
    runtime_idle_minutes: draft.runtime_idle_minutes,
    max_non_default_runtimes: draft.max_non_default_runtimes,
  };
  try {
    const data = await api("/settings", {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    state.settings = data;
    state.loginMode = data.login_mode || state.loginMode;
    applyLoginMode(data);
    await loadSettings();
    toast(t("settingsSaved"));
  } catch (error) {
    toast(error.message, true);
  }
});

$("backup-migration-export")?.addEventListener("click", async () => {
  const password = await requestBackupPassword();
  if (!password) return;
  const suggestedName = `personalityrag-${new Date().toISOString().slice(0, 10)}.prag`;
  let exportSession = null;
  try {
    exportSession = await requestBackupExportScope({ suggestedName });
  } catch (error) {
    logDirectRequestError(error);
    toast(error.message, true);
    return;
  }
  if (!exportSession) return;
  toast(t("configPackageExportStarted"));
  try {
    const response = await fetch("/api/v1/settings/backup-migration/export", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        password,
        include_libraries: Boolean(exportSession.include_libraries),
        include_providers: Boolean(exportSession.include_providers),
      }),
    });
    if (response.status === 401) {
      showLogin();
      throw new Error(t("unauthorized"));
    }
    if (!response.ok) {
      throw await responseError(response);
    }
    const blob = await response.blob();
    const filename = parseDownloadFilename(
      response.headers.get("Content-Disposition"),
      suggestedName,
    );
    await exportSession.save(blob, filename.endsWith(".prag") ? filename : `${filename}.prag`);
    toast(t("configPackageExported"));
  } catch (error) {
    if (error?.name === "AbortError") {
      return;
    }
    logDirectRequestError(error);
    toast(error.message, true);
  } finally {
    exportSession.close?.();
  }
});

$("backup-migration-import")?.addEventListener("click", () => {
  $("backup-migration-import-file").value = "";
  $("backup-migration-import-file").click();
});

$("backup-migration-import-file")?.addEventListener("change", async (event) => {
  const file = event.target.files?.[0] || null;
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".prag")) {
    toast(t("choosePragPackage"), true);
    event.target.value = "";
    return;
  }
  const password = await requestBackupPassword();
  if (!password) {
    event.target.value = "";
    return;
  }
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("password", password);
  try {
    const result = await api("/settings/backup-migration/import", {
      method: "POST",
      body: form,
    });
    toast(
      result?.indexes_pending
        ? t("configPackageImportedIndexPending")
        : t("configPackageImported"),
    );
    await loadSettings();
    await loadProviders();
    await loadLibraries();
  } catch (error) {
    toast(error.message, true);
  } finally {
    event.target.value = "";
  }
});

$("settings-restart")?.addEventListener("click", async () => {
  if (hasUnsavedSettingsChanges()) {
    const confirmed = await confirmDialog({
      message: t("restartUnsavedConfirm"),
      confirmText: t("restartService"),
    });
    if (!confirmed) return;
  }
  try {
    const payload = await api("/settings/restart", { method: "POST" });
    showRestartScreen(payload);
  } catch (error) {
    toast(error.message, true);
  }
});

function closeOverlay(id) {
  $(id).classList.add("hidden");
}

document.querySelectorAll(".modal-dismiss").forEach((button) => {
  button.onclick = () => button.closest(".modal-overlay").classList.add("hidden");
});

document.querySelectorAll(".log-filter[data-log-level]").forEach((button) => {
  button.onclick = () => {
    const level = button.dataset.logLevel;
    if (state.logs.activeLevels.has(level)) {
      state.logs.activeLevels.delete(level);
    } else {
      state.logs.activeLevels.add(level);
    }
    renderLogs();
  };
});

document.querySelectorAll("[data-task-scope]").forEach((button) => {
  button.onclick = () => {
    state.tasks.scope = button.dataset.taskScope || "active";
    renderTasks();
    loadTasks(state.tasks.scope).catch((error) => toast(error.message, true));
  };
});

document.querySelectorAll("[data-provider-kind]").forEach((button) => {
  button.onclick = () => {
    state.providerKind = button.dataset.providerKind || "embedding";
    localStorage.setItem("prag_provider_kind", state.providerKind);
    loadProviders().catch((error) => toast(error.message, true));
  };
});

document.querySelectorAll("[data-recall-view]").forEach((button) => {
  button.onclick = () => setRecallView(button.dataset.recallView || "embedding");
});

$("log-auto-scroll")?.addEventListener("change", (event) => {
  state.logs.autoScroll = Boolean(event.target.checked);
  renderLogAutoScrollState();
  if (state.logs.autoScroll) {
    const consoleElement = $("log-console");
    consoleElement.scrollTop = consoleElement.scrollHeight;
  }
});

$("logs-clear")?.addEventListener("click", async () => {
  try {
    await clearLogs();
  } catch (error) {
    toast(error.message, true);
  }
});

$("libraries-refresh")?.addEventListener("click", async () => { await loadLibraries(); toast(t("pageRefreshed")); });
$("providers-refresh")?.addEventListener("click", async () => { await loadProviders(); toast(t("pageRefreshed")); });
$("graph-refresh")?.addEventListener("click", async () => { await loadGraph(); toast(t("pageRefreshed")); });
$("recall-refresh")?.addEventListener("click", () => {
  setRecallK(DEFAULT_RECALL_K);
  setRecallRerankK(DEFAULT_RERANK_K);
  if (!$("recall-query").value.trim()) {
    toast(t("recallRefreshEmpty"), true);
    return;
  }
  $("run-recall").click();
});
$("system-refresh")?.addEventListener("click", async () => { await loadSystem(); toast(t("pageRefreshed")); });
$("settings-refresh")?.addEventListener("click", async () => { await loadSettings(); toast(t("pageRefreshed")); });
$("logs-refresh")?.addEventListener("click", async () => {
  try {
    await loadTasks(state.tasks.scope);
    await loadLogs({ reset: true });
    toast(t("pageRefreshed"));
  } catch (error) {
    toast(error.message, true);
  }
});

function debounce(callback, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), delay);
  };
}

window.addEventListener("resize", debounce(() => {
  scheduleSystemPanelsRefresh();
  refreshTaskHistoryCollapseState();
}, 120));

setRecallK(DEFAULT_RECALL_K);
setRecallRerankK(DEFAULT_RERANK_K);
setRecallView("embedding");
applyLanguage();

fetch("/api/v1/auth/status", { credentials: "same-origin" })
  .then((response) => response.json())
  .then((status) => {
    applyLoginMode(status);
    if (status.authenticated) {
      showApp();
      activatePage(state.page);
    } else {
      showLogin();
    }
  })
  .catch((error) => {
    logDirectRequestError(error);
    showLogin();
  });
