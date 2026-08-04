import enUS from "./locales/en.js";
import ruRU from "./locales/ru.js";
import zhCN from "./locales/zh.js";
import { createTaskLogController } from "./modules/tasks-logs.js";
import { createApiClient, parseDownloadFilename, responseError } from "./modules/api.js";
import { createSettingsController } from "./modules/settings.js";
import { createLibrariesController } from "./modules/libraries.js";
import { createProvidersController } from "./modules/providers.js";
import { createFileManagerController } from "./modules/files.js";
import { createGlobalSystemController } from "./modules/global-system.js";
import { createRevisionDebugController } from "./modules/revision-debug.js";
import { createAsyncActionGuard } from "./modules/ui-guard.js";
import {
  createDatabaseUiRegistry,
  databasePageRoute, databaseTypePageRoute, globalPageRoute, parsePageRoute,
} from "./modules/database-ui-registry.js";
const DEFAULT_DATABASE_TYPE = "livingmemory_v8";
const $ = (id) => document.getElementById(id);
document.querySelector('.nav[data-page="system"]')?.after(
  document.querySelector('.nav[data-page="files"]'),
);

const APP_PAGES = new Set([
  "libraries",
  "providers",
  "database",
  "graph",
  "memory",
  "recall",
  "system",
  "files",
  "settings",
  "revision-debug",
  "logs",
]);
const GLOBAL_PAGES = new Set(["libraries", "providers", "system", "files", "settings", "revision-debug", "logs"]);
const LEGACY_DATABASE_PAGES = Object.freeze({
  graph: "graph",
  memory: "memories",
  recall: "recall",
});
const RESTART_RETURN_PAGE = "libraries";

function isValidPage(page) {
  return APP_PAGES.has(page);
}

function consumeInitialRouteState() {
  const url = new URL(window.location.href);
  const requestedPage = url.searchParams.get("page");
  const page = url.searchParams.has("restart") ? "libraries" : requestedPage || "libraries";
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
  pageRequests: {
    generation: 0,
    controller: new AbortController(),
  },
  memoryPage: 1,
  memoryPageSize: 20,
  memoryHasMore: false,
  memoryItems: [], selectedMemoryIds: new Set(),
  selectedMemoryId: null,
  selectedMemoryDetail: null,
  stats: null,
  databases: [],
  databaseTypes: [],
  databaseCategory: "memory",
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
  updates: {
    status: null,
    releases: [],
    timer: null,
    switching: false,
  },
  loginMode: "api_key",
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
    lastPhase: "",
  },
  selectedDatabaseId: "",
  selectedDatabaseType: DEFAULT_DATABASE_TYPE,
  selectedDatabaseRefByCategory: { memory: null, knowledge: null },
  expandedDatabaseRefs: new Set(),
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
    finishedExpanded: false, finishedCursor: null, scope: "active",
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
  min_importance_for_retrieval: 0,
  min_similarity_for_retrieval: 0,
  recent_memory_count: 2,
  recent_memory_max_age_hours: 72,
  memory_type_filter: "all",
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
  auto_archived_enabled: false,
  cleanup_days_threshold: 7,
  cleanup_importance_threshold: 0.3,
  protected_importance_threshold: 1,
};

const DEFAULT_RECALL_K = 5;
const DEFAULT_RERANK_K = 5;
const IDENTIFIER_PATTERN = /^[A-Za-z0-9_-]+$/;
const LIBRARY_EXPAND_ICON = `<svg viewBox="0 0 240 24" aria-hidden="true"><circle cx="24" cy="12" r="4.5"/><circle cx="120" cy="12" r="4.5"/><circle cx="216" cy="12" r="4.5"/></svg>`;
const asyncGuard = createAsyncActionGuard();

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
  $("page-title").textContent = t(state.pageTitleKey || state.page);
  applyLoginMode({ login_mode: state.loginMode });
  $("memory-sort")?.querySelector('[value="created_desc"]')?.replaceChildren(t("sortNewest"));
  $("memory-sort")?.querySelector('[value="created_asc"]')?.replaceChildren(t("sortOldest"));
  $("memory-sort")?.querySelector('[value="importance_desc"]')?.replaceChildren(t("sortImportanceDesc"));
  $("memory-sort")?.querySelector('[value="importance_asc"]')?.replaceChildren(t("sortImportanceAsc"));
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

const rawApi = createApiClient({
  onUnauthorized: () => showLogin(),
  unauthorizedMessage: () => t("unauthorized"),
  onOperationalError: (error) => logUiFeedback(error.message, "ERROR"),
});

function pageRequestAbortError() {
  const error = new Error("Page request superseded");
  error.name = "AbortError";
  return error;
}

function beginPageRequestGeneration() {
  state.pageRequests.controller.abort();
  state.pageRequests = {
    generation: state.pageRequests.generation + 1,
    controller: new AbortController(),
  };
  return state.pageRequests.generation;
}

async function api(path, options = {}) {
  const { pageScoped = true, ...requestOptions } = options;
  const method = String(requestOptions.method || "GET").toUpperCase();
  const scoped = pageScoped && method === "GET" && !requestOptions.signal;
  const generation = state.pageRequests.generation;
  if (scoped) requestOptions.signal = state.pageRequests.controller.signal;
  const payload = await rawApi(path, requestOptions);
  if (scoped && generation !== state.pageRequests.generation) {
    throw pageRequestAbortError();
  }
  return payload;
}

const {
  requestBackupPassword, requestBackupExportScope,
  settingsDraft, hasUnsavedSettingsChanges, showRestartScreen, loadSettings,
  loadUpdateStatus, openVersionSelector, refreshUpdateReleases,
  checkLastUpdateTransaction,
} = createSettingsController({
  $, state, t, api, toast,
  stopLogPolling: (...args) => stopLogPolling(...args),
  stopTaskPolling: (...args) => stopTaskPolling(...args),
  resetTaskState: (...args) => resetTaskState(...args),
  isValidPage,
  restartReturnPage: RESTART_RETURN_PAGE,
  escapeHtml,
  confirmDialog,
});
function databaseRefKey(databaseType, databaseId) {
  return `${databaseType || DEFAULT_DATABASE_TYPE}:${databaseId || ""}`;
}

const DATABASE_TYPE_API_CLIENTS = Object.freeze({
  livingmemory_v8: Object.freeze({
    collection: "/memory-libraries/livingmemory_v8",
  }),
  text_media_v1: Object.freeze({
    collection: "/knowledge-libraries/text_media_v1",
  }),
});

function databaseTypeApiClient(databaseType) {
  const normalizedType = databaseType || DEFAULT_DATABASE_TYPE;
  const client = DATABASE_TYPE_API_CLIENTS[normalizedType];
  if (!client) {
    throw new Error(`Unsupported database type: ${normalizedType}`);
  }
  return client;
}

function databaseCollectionApiPath(databaseType) {
  return databaseTypeApiClient(databaseType).collection;
}

function databaseApiPath(databaseType, databaseId, path = "") {
  return `${databaseCollectionApiPath(databaseType)}/${encodeURIComponent(databaseId)}${path}`;
}

function selectedDatabaseApi(path, options = {}) {
  if (!state.selectedDatabaseId) {
    throw new Error("请先选择记忆库");
  }
  return api(databaseApiPath(
    state.selectedDatabaseType,
    state.selectedDatabaseId,
    path,
  ), options);
}

function selectedDatabase() {
  return state.databases.find((item) => (
    item.id === state.selectedDatabaseId
    && (item.database_type || DEFAULT_DATABASE_TYPE) === state.selectedDatabaseType
  )) || null;
}

function selectedDatabaseHasRerank() {
  const library = selectedDatabase();
  return Boolean(library?.rerank_provider_id || library?.rerank_provider?.id);
}

function updateRecallRerankControls() {
  const hasRerank = selectedDatabaseHasRerank();
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

function refreshSidebarDatabase() {
  const box = $("sidebar-current-library");
  if (!box) return;
  const library = selectedDatabase();
  if (!library || parsePageRoute(state.route)?.scope === "database_type") {
    box.classList.add("hidden");
    return;
  }
  const provider = library.provider || {};
  const categoryLabel = box.querySelector("span");
  if (categoryLabel) {
    const key = library.database_category === "knowledge"
      ? "currentKnowledgeLibrary"
      : "currentLibrary";
    categoryLabel.dataset.i18n = key;
    categoryLabel.textContent = t(key);
  }
  $("sidebar-library-name").textContent = library.name || library.id;
  $("sidebar-library-id").textContent = library.id;
  $("sidebar-library-provider").textContent = `${provider.display_name || provider.id || "未绑定"}`;
  box.title = `${library.name || library.id}\n${library.id}`;
  box.classList.remove("hidden");
}

const DATABASE_NAV_ICONS = Object.freeze({
  overview: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4h18v13H3V4Zm2 2v9h14V6H5Zm3 13h8v2H8v-2Z"/></svg>',
  graph: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3a3 3 0 1 1-1 5.83v6.34A3 3 0 1 1 7.83 18h6.34A3 3 0 1 1 17 20.83V17a3 3 0 0 1-2.83-2H7.83A3 3 0 0 1 7 16.83V8.83A3 3 0 0 1 6 9V3Zm11 3a3 3 0 1 1 0 6 3 3 0 0 1 0-6ZM8 8.83v4.34A3 3 0 0 1 7.83 13h6.34A3 3 0 0 1 16 11.83 3 3 0 0 1 14.17 8H7.83L8 8.83Z"/></svg>',
  memories: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2 9 4.5-9 4.5-9-4.5L12 2Zm-7.2 8.4L12 14l7.2-3.6L21 12l-9 4.5L3 12l1.8-1.6Zm0 5.5 7.2 3.6 7.2-3.6L21 17.5 12 22l-9-4.5 1.8-1.6Z"/></svg>',
  recall: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.5 3a7.5 7.5 0 1 1-4.71 13.34L2.6 19.53l-1.42-1.42 3.2-3.2A7.5 7.5 0 0 1 10.5 3Zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z"/></svg>',
  content: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 2h10l4 4v16H5V2Zm2 2v16h10V7h-3V4H7Zm2 7h6v2H9v-2Zm0 4h6v2H9v-2Z"/></svg>',
  media: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4h18v16H3V4Zm2 2v10.17L9.17 12 12 14.83 14.83 12 19 16.17V6H5Zm10 1.5A2.5 2.5 0 1 1 15 12a2.5 2.5 0 0 1 0-5Z"/></svg>',
  search: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.5 3a7.5 7.5 0 1 1-4.71 13.34L2.6 19.53l-1.42-1.42 3.2-3.2A7.5 7.5 0 0 1 10.5 3Zm0 2a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11Z"/></svg>',
  settings: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m10.8 2 2.4.01.55 2.05c.55.2 1.07.5 1.54.89l2.05-.57 1.21 2.08-1.5 1.5c.1.57.1 1.15 0 1.72l1.5 1.5-1.2 2.09-2.06-.57c-.47.39-.99.69-1.54.89l-.55 2.05h-2.4l-.55-2.05a6.5 6.5 0 0 1-1.54-.89l-2.05.57-1.21-2.08 1.5-1.51a5.2 5.2 0 0 1 0-1.72l-1.5-1.5 1.2-2.08 2.06.57c.47-.39.99-.69 1.54-.89L10.8 2ZM12 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z"/></svg>',
});

function renderDatabaseNavigation() {
  const container = $("database-type-nav");
  if (!container) return;
  const database = selectedDatabase();
  const databaseType = database?.database_type || state.selectedDatabaseType;
  const pages = [...(database ? databaseUiRegistry.availablePages(database) : []),
    ...databaseUiRegistry.availableTypePages(databaseType)];
  container.innerHTML = pages.map((page) => {
    const route = page.scope === "database_type" ? databaseTypePageRoute(databaseType, page.id)
      : databasePageRoute(database.database_type, page.id);
    const icon = page.iconMarkup || `<span class="nav-ico">${DATABASE_NAV_ICONS[page.id] || DATABASE_NAV_ICONS.content}</span>`;
    return `<button class="nav" type="button" data-route="${escapeHtml(route)}">${icon}<b data-i18n="${escapeHtml(page.labelKey)}">${escapeHtml(t(page.labelKey))}</b></button>`;
  }).join("");
  container.querySelectorAll(".nav[data-route]").forEach((button) => {
    button.onclick = () => activatePage(button.dataset.route);
  });
  container.querySelector(`[data-route="${CSS.escape(state.route || "")}"]`)?.classList.add("active");
}

function refreshDatabaseContext() {
  refreshSidebarDatabase();
  updateRecallRerankControls();
  renderDatabaseNavigation();
  const button = $("library-context");
  const library = selectedDatabase();
  if (!library || parsePageRoute(state.route)?.scope !== "database") {
    button.classList.add("hidden");
    return;
  }
  const provider = library.provider || {};
  const generation = library.indexes?.generation || "尚未构建";
  button.innerHTML = `<strong>${escapeHtml(library.name)}</strong><small>${escapeHtml(provider.display_name || provider.id || "未绑定")} · ${escapeHtml(generation)}</small>`;
  button.classList.remove("hidden");
}

function updateDatabaseCardSelection() {
  document.querySelectorAll(".library-card").forEach((card) => {
    const selected = card.dataset.id === state.selectedDatabaseId
      && card.dataset.databaseType === state.selectedDatabaseType;
    card.classList.toggle("active-card", selected);
    card.setAttribute("aria-pressed", selected ? "true" : "false");
    card.querySelector(".selected-library-badge")?.classList.toggle("hidden", !selected);
  });
}

function selectDatabase(databaseId, options = {}) {
  if (!databaseId) return;
  const databaseType = options.databaseType || DEFAULT_DATABASE_TYPE;
  const database = state.databases.find((item) => (
    item.id === databaseId
    && (item.database_type || DEFAULT_DATABASE_TYPE) === databaseType
  ));
  const databaseCategory = options.databaseCategory || database?.database_category || "";
  const previousDatabaseType = state.selectedDatabaseType;
  const changed = state.selectedDatabaseId !== databaseId
    || state.selectedDatabaseType !== databaseType;
  state.selectedDatabaseId = databaseId;
  state.selectedDatabaseType = databaseType;
  if (databaseCategory && state.selectedDatabaseRefByCategory) {
    state.selectedDatabaseRefByCategory[databaseCategory] = { id: databaseId, databaseType };
  }
  window.dispatchEvent(new CustomEvent("prag-database-selection-changed"));
  if (options.resetMemoryPage !== false) {
    state.memoryPage = 1;
  }
  if (changed) {
    beginPageRequestGeneration();
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
  refreshDatabaseContext();
  updateDatabaseCardSelection();
  const activeRoute = parsePageRoute(state.route);
  if (changed && ["database", "database_type"].includes(activeRoute?.scope)) {
    const driver = databaseUiRegistry.get(databaseType);
    const keepTypePage = activeRoute.scope === "database_type" && previousDatabaseType === databaseType
      && databaseUiRegistry.resolveTypePage(databaseType, activeRoute.pageId);
    const pageId = previousDatabaseType === databaseType && activeRoute.scope === "database"
      ? activeRoute.pageId : driver?.defaultPage;
    if (!keepTypePage && pageId) {
      queueMicrotask(() => activatePage(
        databasePageRoute(databaseType, pageId),
      ).catch((error) => toast(error.message, true)));
    }
  }
}

function showLogin() {
  state.pageRequests.controller.abort();
  stopLogPolling();
  if (state.updates.timer) clearInterval(state.updates.timer);
  state.updates.timer = null;
  revisionDebugController?.clear();
  $("login").classList.remove("hidden");
  $("app").classList.add("hidden");
}

function showApp() {
  $("login").classList.add("hidden");
  $("app").classList.remove("hidden");
  loadUpdateStatus().catch(() => {});
  checkLastUpdateTransaction().catch(() => {});
  if (!state.updates.timer) {
    state.updates.timer = setInterval(() => loadUpdateStatus().catch(() => {}), 15 * 60 * 1000);
  }
  revisionDebugController?.refreshSessionStatus().catch(() => {});
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
    });
    showApp();
    await activatePage(state.page);
  } catch (error) {
    $("login-error").textContent = error.message;
  }
});

$("logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
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
  const activeRoute = parsePageRoute(state.route);
  if (["database", "database_type"].includes(activeRoute?.scope)) {
    databaseUiRegistry.get(activeRoute.databaseType)
      ?.onLanguageChange(activeRoute.pageId, activeRoute.scope)
      .catch((error) => toast(error.message, true));
  }
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
  if (state.page === "libraries") loadDatabases();
  if (state.page === "providers") loadProviders();
  if (state.page === "revision-debug") revisionDebugController.onLanguageChange();
});

document.documentElement.dataset.theme = localStorage.getItem("prag_theme") || "light";

function normalizePageRoute(value) {
  const parsed = parsePageRoute(value);
  if (parsed) return value;
  if (GLOBAL_PAGES.has(value)) return globalPageRoute(value);
  if (LEGACY_DATABASE_PAGES[value]) {
    const typeId = state.selectedDatabaseType || DEFAULT_DATABASE_TYPE;
    const driver = databaseUiRegistry.get(typeId);
    const pageId = typeId === DEFAULT_DATABASE_TYPE
      ? LEGACY_DATABASE_PAGES[value]
      : driver?.defaultPage;
    return pageId ? databasePageRoute(typeId, pageId) : globalPageRoute("libraries");
  }
  return globalPageRoute("libraries");
}

function resolvePageRoute(value) {
  const route = normalizePageRoute(value);
  const parsed = parsePageRoute(route);
  if (parsed?.scope === "database_type") {
    if (parsed.databaseType !== state.selectedDatabaseType) return globalPageRoute("libraries");
    const page = databaseUiRegistry.resolveTypePage(parsed.databaseType, parsed.pageId);
    return page ? databaseTypePageRoute(parsed.databaseType, page.id) : globalPageRoute("libraries");
  }
  if (parsed?.scope !== "database") return route;
  const database = selectedDatabase();
  if (!database) return globalPageRoute("libraries");
  const requestedPage = parsed.databaseType === database.database_type
    ? parsed.pageId
    : "";
  const page = databaseUiRegistry.resolvePage(database, requestedPage);
  return page
    ? databasePageRoute(database.database_type, page.id)
    : globalPageRoute("libraries");
}

function setActivePage(page) {
  const route = resolvePageRoute(page);
  const parsed = parsePageRoute(route);
  const database = parsed?.scope === "database" ? selectedDatabase() : null;
  const descriptor = parsed?.scope === "database_type" ? databaseUiRegistry.resolveTypePage(parsed.databaseType, parsed.pageId)
    : database ? databaseUiRegistry.resolvePage(database, parsed.pageId) : null;
  const nextPage = parsed?.scope === "global"
    ? parsed.pageId
    : descriptor?.viewId || "database";
  const nextType = ["database", "database_type"].includes(parsed?.scope) ? parsed.databaseType : null;
  if (state.activeDatabaseUiType && state.activeDatabaseUiType !== nextType) {
    databaseUiRegistry.get(state.activeDatabaseUiType)?.unmount();
  }
  if (nextType && state.activeDatabaseUiType !== nextType) {
    databaseUiRegistry.get(nextType)?.mount({ host: $("database-page-host") });
  }
  state.activeDatabaseUiType = nextType;
  document.querySelectorAll(".nav[data-page]").forEach((item) => {
    item.classList.toggle(
      "active",
      parsed?.scope === "global" && item.dataset.page === nextPage,
    );
  });
  document.querySelectorAll(".nav[data-route]").forEach((item) => {
    item.classList.toggle("active", item.dataset.route === route);
  });
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("active", section.id === `page-${nextPage}`);
  });
  state.page = nextPage;
  state.route = route;
  state.pageTitleKey = descriptor?.titleKey || nextPage;
  applyLanguage();
  refreshDatabaseContext();
}

async function activatePage(page) {
  beginPageRequestGeneration();
  const requested = normalizePageRoute(page);
  const requestedRoute = parsePageRoute(requested);
  if (
    requestedRoute?.scope === "database"
    || (requestedRoute?.scope === "global" && requestedRoute.pageId === "libraries")
  ) {
    await ensureDatabaseSelection();
  }
  setActivePage(page);
  try {
    await loadPage(state.route);
  } catch (error) {
    if (error?.name !== "AbortError") throw error;
  }
}

document.querySelectorAll(".nav[data-page]").forEach((button) =>
  button.addEventListener("click", async () => {
    await activatePage(button.dataset.page);
  }),
);

async function loadPage(page) {
  const parsed = parsePageRoute(page) || parsePageRoute(normalizePageRoute(page));
  if (!(parsed?.scope === "global" && parsed.pageId === "logs")) {
    stopLogPolling();
    stopTaskPolling();
  }
  if (["database", "database_type"].includes(parsed?.scope)) {
    const database = selectedDatabase();
    const driver = databaseUiRegistry.get(parsed.databaseType);
    if (driver) await driver.load({ scope: parsed.scope, pageId: parsed.pageId,
      database: parsed.scope === "database" ? database : null, databaseType: parsed.databaseType });
  }
  if (parsed?.scope === "global" && parsed.pageId === "libraries") await loadDatabases();
  if (parsed?.scope === "global" && parsed.pageId === "providers") await loadProviders();
  if (parsed?.scope === "global" && parsed.pageId === "system") await globalSystemController.loadGlobalSystem();
  if (parsed?.scope === "global" && parsed.pageId === "files") await fileController.loadFiles();
  if (parsed?.scope === "global" && parsed.pageId === "settings") {
    await loadSettings();
    await revisionDebugController.refreshSessionStatus();
  }
  if (parsed?.scope === "global" && parsed.pageId === "revision-debug") await revisionDebugController.load();
  if (parsed?.scope === "global" && parsed.pageId === "logs") {
    state.tasks.scope = state.tasks.scope || "active";
    renderTasks();
    renderLogs();
    startLogPolling();
    startTaskPolling();
  }
  refreshDatabaseContext();
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

function memoryStatusPill(status) {
  const value = String(status || "active").toLowerCase();
  return `<span class="memory-status-pill ${escapeHtml(value)}">${escapeHtml(displayStatus(value))}</span>`;
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
  const canonicalSummary = String(metadata.canonical_summary || raw.text || raw.content || raw.summary || "");
  const personaSummary = String(metadata.persona_summary || "");
  return {
    id: Number(raw.id ?? raw.memory_id),
    text: raw.text || raw.content || raw.summary || "",
    metadata,
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
    canonicalSummary,
    personaSummary,
    hasSource: Boolean(raw.has_source ?? metadata.has_source),
    sourceTimeStrategy: String(metadata.source_time_strategy || "preserve"),
    sourceTimeTags: metadata.source_time_tags && typeof metadata.source_time_tags === "object"
      ? metadata.source_time_tags
      : {},
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

const databaseUiRegistry = createDatabaseUiRegistry({
  $, state, t, toast, api, responseError, selectedDatabaseApi, escapeHtml, formatMemoryTime,
  statCards, formatVersionTag,
  memoryStatusPill, memoryImportanceBar,
  displayStatus,
  normalizeMemoryDetail, memoryMetaItem, memoryListSection, memoryTagsSection,
  renderMemoryMiniGraph, loadDatabases: (...args) => loadDatabases(...args),
  debounce, confirmDialog, asyncGuard,
  selectedDatabase, selectedDatabaseHasRerank, selectDatabase,
  setRecallK, setRecallRerankK, updateRecallRerankControls,
  closeOverlay,
  navigate: (...args) => activatePage(...args),
  loadProviders: (...args) => loadProviders(...args),
  fillProviderSelect: (...args) => fillProviderSelect(...args),
  addOptimisticTask: (...args) => addOptimisticTask(...args),
  removeOptimisticTask: (...args) => removeOptimisticTask(...args),
  trackQueuedJob: (...args) => trackQueuedJob(...args),
});
await databaseUiRegistry.preload();
const livingMemoryUi = databaseUiRegistry.get(DEFAULT_DATABASE_TYPE);
await livingMemoryUi.mount({ root: document.querySelector("main") });
const textMediaUi = databaseUiRegistry.get("text_media_v1");
await textMediaUi.mount({ host: $("database-page-host") });
const typeNav = $("database-type-nav");
document.querySelector('.nav[data-page="providers"]')?.after(typeNav);
for (const page of ["graph", "memory", "recall"]) {
  document.querySelector(`.nav[data-page="${page}"]`)?.remove();
}
const {
  loadGraph,
  loadMemories,
  setRecallView,
  renderRecallResults,
  resetRecallTest,
  loadOverview: loadSystem,
  scheduleOverviewRefresh: scheduleSystemPanelsRefresh,
  formatBytes,
} = livingMemoryUi.actions;
const graphController = { render: livingMemoryUi.actions.renderGraph };
const fileController = createFileManagerController({
  $, state, t, api, toast, responseError, parseDownloadFilename, confirmDialog, escapeHtml, asyncGuard,
});
const globalSystemController = createGlobalSystemController({
  $, api, t, escapeHtml, statCards,
});
const {
  loadTasks, startTaskPolling, stopTaskPolling, renderTasks,
  resetTaskState, applyTaskHistoryCollapseState: refreshTaskHistoryCollapseState,
  loadLogs, startLogPolling, stopLogPolling, clearLogs,
  trackQueuedJob, addOptimisticTask, updateOptimisticTask, removeOptimisticTask,
  renderLogs, renderLogAutoScrollState,
} = createTaskLogController({
  $, state, t, api, toast, escapeHtml, formatBytes, selectedDatabase,
  onTaskFinished: async (job) => {
    await loadDatabases(state.page === "libraries");
    const route = parsePageRoute(state.route);
    if (route?.scope === "global" && route.pageId === "system") {
      await globalSystemController.loadGlobalSystem();
      return;
    }
    if (route?.scope === "database_type") {
      if ((job.database_type || DEFAULT_DATABASE_TYPE) !== route.databaseType) return;
      await databaseUiRegistry.get(route.databaseType)?.load({ scope: route.scope,
        pageId: route.pageId, databaseType: route.databaseType });
      return;
    }
    if (route?.scope !== "database") return;
    const database = selectedDatabase();
    const jobType = job.database_type || DEFAULT_DATABASE_TYPE;
    const jobDatabaseId = job.memory_store_id || job.knowledge_base_id
      || job.database_id
      || job.library_id;
    if (!database || database.id !== jobDatabaseId || database.database_type !== jobType) return;
    await databaseUiRegistry.get(database.database_type)?.load({
      pageId: route.pageId,
      database,
    });
  },
  confirmDialog,
  markDatabaseIndexConflict: (...args) => markDatabaseIndexConflict(...args),
  clearDatabaseIndexConflict: (...args) => clearDatabaseIndexConflict(...args),
  refreshDatabaseContext,
});
const { loadDatabases, ensureDatabaseSelection, navigate, bindUsedListDetails, confirmSensitiveProviderEdit, markDatabaseIndexConflict, clearDatabaseIndexConflict } = createLibrariesController({
  $, state, t, toast, api, selectedDatabaseApi, selectedDatabase, selectDatabase, refreshDatabaseContext,
  databaseApiPath, databaseCollectionApiPath, databaseRefKey, DEFAULT_DATABASE_TYPE,
  escapeHtml, confirmDialog, addOptimisticTask, removeOptimisticTask, trackQueuedJob,
  validateIdentifierInput, LIBRARY_EXPAND_ICON,
  activatePage, closeOverlay, joinLocalizedList,
  DEFAULT_LIBRARY_CONVERSATION_SETTINGS, DEFAULT_LIBRARY_RECALL_SETTINGS,
  DEFAULT_LIBRARY_MAINTENANCE_SETTINGS, asyncGuard,
  loadProviders: (...args) => loadProviders(...args),
  fillProviderSelect: (...args) => fillProviderSelect(...args),
  databaseUiRegistry, databasePageRoute, databaseTypePageRoute,
});
const { loadProviders, fillProviderSelect } = createProvidersController({
  $, state, t, toast, api, escapeHtml, validateIdentifierInput,
  confirmSensitiveProviderEdit, navigate, loadDatabases, confirmDialog, closeOverlay, selectDatabase, asyncGuard,
});
const revisionDebugController = createRevisionDebugController({
  $, state, t, api, toast, escapeHtml, navigate: (...args) => activatePage(...args),
  confirmDialog, asyncGuard,
});
$("settings-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  await asyncGuard.run("settings:save", async () => {
  const draft = settingsDraft();
  const payload = {
    access_base_url: draft.access_base_url, public_adapter_url: draft.public_adapter_url,
    port: draft.port,
    access_port: draft.access_port,
    new_password: draft.new_password || null,
    clear_password: draft.clear_password,
    performance_profile: draft.performance_profile,
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
    await revisionDebugController.refreshSessionStatus();
    toast(t("settingsSaved"));
  } catch (error) {
    toast(error.message, true);
  }
  }, {
    form: event.currentTarget,
    button: event.submitter,
    busyText: t("loading"),
  });
});

$("backup-migration-export")?.addEventListener("click", async (event) => {
  await asyncGuard.run("settings:backup-export", async () => {
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
  }, {
    button: event.currentTarget,
    busyText: t("loading"),
  });
});

$("backup-migration-import")?.addEventListener("click", () => {
  $("backup-migration-import-file").value = "";
  $("backup-migration-import-file").click();
});

$("backup-migration-import-file")?.addEventListener("change", async (event) => {
  await asyncGuard.run("settings:backup-import", async () => {
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
    await loadDatabases();
  } catch (error) {
    toast(error.message, true);
  } finally {
    event.target.value = "";
  }
  }, {
    button: $("backup-migration-import"),
    busyText: t("loading"),
  });
});

$("settings-restart")?.addEventListener("click", async (event) => {
  await asyncGuard.run("settings:restart", async () => {
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
  }, {
    button: event.currentTarget,
    busyText: t("loading"),
  });
});

$("updates-refresh")?.addEventListener("click", async (event) => {
  await asyncGuard.run("updates:refresh", async () => {
  try {
    await refreshUpdateReleases();
    toast(t("updateCheckComplete"));
  } catch (error) {
    toast(error.message, true);
  }
  }, {
    button: event.currentTarget,
    busyText: t("loading"),
  });
});

$("updates-select")?.addEventListener("click", async (event) => {
  await asyncGuard.run("updates:select", async () => {
  try {
    await openVersionSelector();
  } catch (error) {
    toast(error.message, true);
  }
  }, {
    button: event.currentTarget,
    busyText: t("loading"),
  });
});

$("update-available-badge")?.addEventListener("click", async () => {
  await activatePage("settings");
  await openVersionSelector();
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

$("libraries-refresh")?.addEventListener("click", async () => { await loadDatabases(true, { force: true }); toast(t("pageRefreshed")); });
$("providers-refresh")?.addEventListener("click", async () => { await loadProviders(); toast(t("pageRefreshed")); });
$("graph-refresh")?.addEventListener("click", async () => { await loadGraph(); toast(t("pageRefreshed")); });
$("recall-refresh")?.addEventListener("click", () => {
  resetRecallTest();
  toast(t("pageRefreshed"));
});
$("system-refresh")?.addEventListener("click", async () => { await loadSystem(); toast(t("pageRefreshed")); });
$("settings-refresh")?.addEventListener("click", async () => { await loadSettings(); await revisionDebugController.refreshSessionStatus(); toast(t("pageRefreshed")); });
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
    if ($("sidebar-version")) $("sidebar-version").textContent = status.version ? `v${status.version}` : "v-";
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
