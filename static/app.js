const $ = (id) => document.getElementById(id);

const APP_PAGES = new Set([
  "libraries",
  "providers",
  "graph",
  "memory",
  "recall",
  "system",
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
  selectedLibraryId: localStorage.getItem("prag_library_id") || "",
  expandedLibraryIds: new Set(),
  optimisticIndexConflicts: new Set(),
  logs: {
    items: [],
    lastId: 0,
    maxEntries: 2000,
    activeLevels: new Set(["DEBUG", "INFO", "WARN", "ERROR"]),
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
    scope: "active",
    polling: false,
    pollTimer: null,
  },
};

const DEFAULT_RECALL_K = 5;
const DEFAULT_RERANK_K = 5;
const LIBRARY_EXPAND_ICON = `<svg viewBox="0 0 240 24" aria-hidden="true"><circle cx="24" cy="12" r="4.5"/><circle cx="120" cy="12" r="4.5"/><circle cx="216" cy="12" r="4.5"/></svg>`;

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

const strings = {
  zh: {
    libraries: "记忆库",
    providers: "模型提供商",
    graph: "知识图谱",
    memory: "记忆管理",
    recall: "召回测试",
    system: "系统概览",
    theme: "切换主题",
    logout: "退出",
    login: "登录",
    loginHint: "输入 launcher 输出或 config/config.json 中的 API 密钥。",
    graphQuery: "输入人物、主题、事实或整句",
    memoryId: "记忆 ID",
    sessionOptional: "session_id（可选）",
    sessionRecallOptional: "session_id（当前配置不启用过滤）",
    personaOptional: "persona_id（可选）",
    searchGraph: "检索图谱",
    overview: "最近概览",
    memorySearch: "搜索正文、元数据或 ID",
    refresh: "刷新",
    createMemory: "新增记忆",
    content: "内容",
    persona: "人格",
    importance: "重要性",
    status: "状态",
    actions: "操作",
    recallQuery: "输入要回忆的问题、人物、关系或事件",
    runRecall: "执行召回",
    test: "测试",
    indexGeneration: "索引代次",
    importanceDistribution: "重要性分布",
    atomTypes: "原子类型",
    backups: "备份与迁移归档",
    integrity: "完整性检查",
    cancel: "取消",
    save: "保存并重建索引",
    serviceEyebrow: "人格记忆库",
    online: "在线",
    loading: "加载中…",
    sortNewest: "最新优先",
    sortOldest: "最早优先",
    sortImportanceDesc: "重要性从高到低",
    sortImportanceAsc: "重要性从低到高",
    previousPage: "上一页",
    nextPage: "下一页",
    topKLabel: "嵌入召回条数",
    rerankKLabel: "重排输出条数",
    embeddingProvider: "嵌入模型提供商",
    libraryManagement: "记忆库管理",
    libraryManagementHint: "每个记忆库拥有独立数据库、索引、缓存和模型绑定。",
    createLibrary: "新增记忆库",
    providerManagement: "模型提供商",
    providerManagementHint: "统一管理可供不同记忆库绑定的 Embedding Provider。",
    createProvider: "新增模型提供商",
    currentProviderBinding: "当前模型绑定",
    manageProviders: "管理 Provider",
    personaField: "人格 ID",
    sessionField: "会话 ID",
    importanceField: "重要性",
    statusField: "状态",
    statusActive: "正常",
    statusArchived: "归档",
    statusDeleted: "删除",
    memoryTypeField: "记忆类型",
    topicsField: "主题",
    participantsField: "参与者",
    factsField: "关键事实",
    commaSeparated: "逗号分隔",
    statsMemories: "总记忆",
    statsNodes: "节点",
    statsRelations: "关系",
    statsSessions: "会话",
    statsActiveSessions: "活跃会话",
    statsGraphEntries: "图条目",
    statsAtoms: "原子",
    statsMessages: "消息",
    selectedLibraryStatsTitle: "当前选中记忆库相关信息",
    selectedLibraryStatsHint: "以下七项统计仅对应当前选中的记忆库，不代表全部记忆库总量。",
    serviceVersion: "PersonalityRAG 版本",
    livingMemoryDbVersion: "主 LivingMemory DB 版本",
    graphNoData: "暂无图谱数据",
    legendPerson: "人物",
    legendTopic: "主题",
    legendFact: "事实",
    legendSummary: "摘要",
    legendOther: "其它",
    unnamedNode: "未命名节点",
    nodeMemories: "关联记忆",
    nodeDegree: "连接度",
    nodeEntries: "条目",
    nodeWeight: "权重",
    tableEdit: "编辑",
    tableDelete: "删除",
    tableEmpty: "暂无数据",
    memoryModalEdit: "记忆 #",
    memoryModalNew: "新增记忆",
    saved: "已保存",
    deleted: "已删除",
    confirmDelete: "确定删除记忆 #",
    recallSummary: "共 {total} 条结果，用时 {elapsed} ms",
    recallNoResult: "暂无结果",
    recallScoreBreakdown: "评分明细",
    filesUnit: "个文件",
    noBackups: "暂无备份",
    progressPrefix: "进度",
    jobCompleted: "重建完成",
    jobFailed: "重建失败",
    jobCancelled: "任务已取消",
    integrityReady: "完整性检查已完成",
    pageOfTotal: "第 {page} 页 · 共 {total} 条",
    unauthorized: "登录状态已失效，请重新登录。",
    defaultBadge: "默认",
    selectedBadge: "当前选择",
    currentLibrary: "当前记忆库",
    noDescription: "暂无描述",
    providerLabel: "模型提供商",
    modelDimension: "模型 / 维度",
    generationLabel: "索引代次",
    indexStatus: "索引状态",
    indexHealthy: "健康",
    indexMismatch: "数量异常",
    indexPending: "待构建",
    defaultPersona: "默认人格",
    noLimit: "不限制",
    enterLibrary: "进入记忆库",
    edit: "编辑",
    backupNow: "立即备份",
    setDefault: "设为默认",
    delete: "删除",
    noLibraries: "暂无记忆库",
    typeLabel: "类型",
    modelLabel: "模型",
    dimensionLabel: "维度",
    autoDetect: "自动检测",
    endpointLabel: "API 地址",
    usedLibraries: "使用中的库",
    none: "无",
    copy: "复制",
    notTested: "尚未测试",
    availableStatus: "可用 · {elapsed} ms",
    unavailableStatus: "不可用 · {error}",
    pendingRebuild: "待重建 · {libraries}",
    stillUsesRevision: "{library} 仍使用 r{revision}",
    noProviders: "暂无模型提供商",
    libraryId: "记忆库 ID",
    displayName: "显示名称",
    defaultPersonaOptional: "默认人格 ID（可选）",
    description: "描述",
    savePlain: "保存",
    selectProviderType: "选择模型提供商类型",
    enabled: "启用",
    apiKey: "API Key",
    keepSecretPlaceholder: "留空保持原值",
    clearApiKey: "清除已有 API Key",
    apiBaseUrl: "API Base URL",
    embeddingModel: "嵌入模型",
    embeddingDimension: "嵌入维度",
    dimensionHelp: "0 表示自动检测。vLLM 不会把此值作为 dimensions 参数发送。",
    timeoutSeconds: "超时时间（秒）",
    proxyAddress: "代理地址",
    proxyPlaceholder: "例如 http://127.0.0.1:7890",
    batchSize: "批量大小",
    concurrency: "并发数",
    maxRetries: "最大重试次数",
    testConfig: "测试配置",
  },
  en: {
    libraries: "Libraries",
    providers: "Providers",
    graph: "Knowledge Graph",
    memory: "Memories",
    recall: "Recall Test",
    system: "System",
    theme: "Theme",
    logout: "Logout",
    login: "Login",
    loginHint: "Enter the API key shown by launcher or stored in config/config.json.",
    graphQuery: "Person, topic, fact, or sentence",
    memoryId: "Memory ID",
    sessionOptional: "session_id (optional)",
    sessionRecallOptional: "session_id (filter currently disabled)",
    personaOptional: "persona_id (optional)",
    searchGraph: "Search Graph",
    overview: "Overview",
    memorySearch: "Search content, metadata, or ID",
    refresh: "Refresh",
    createMemory: "New Memory",
    content: "Content",
    persona: "Persona",
    importance: "Importance",
    status: "Status",
    actions: "Actions",
    recallQuery: "What should this persona remember?",
    runRecall: "Execute Recall",
    test: "Test",
    indexGeneration: "Index Generation",
    importanceDistribution: "Importance Distribution",
    atomTypes: "Atom Types",
    backups: "Backups & Imports",
    integrity: "Integrity Check",
    cancel: "Cancel",
    save: "Save & Rebuild",
    serviceEyebrow: "Personality Memory Service",
    online: "Online",
    loading: "Loading…",
    sortNewest: "Newest first",
    sortOldest: "Oldest first",
    sortImportanceDesc: "Importance high to low",
    sortImportanceAsc: "Importance low to high",
    previousPage: "Previous",
    nextPage: "Next",
    topKLabel: "Embedding recall count",
    rerankKLabel: "Rerank output count",
    embeddingProvider: "Embedding Model Provider",
    libraryManagement: "Memory Libraries",
    libraryManagementHint: "Each library owns an isolated database, index, cache, and provider binding.",
    createLibrary: "New Library",
    providerManagement: "Model Providers",
    providerManagementHint: "Manage embedding providers shared by memory libraries.",
    createProvider: "New Provider",
    currentProviderBinding: "Current Provider Binding",
    manageProviders: "Manage Providers",
    personaField: "persona_id",
    sessionField: "session_id",
    importanceField: "Importance",
    statusField: "Status",
    statusActive: "Active",
    statusArchived: "Archived",
    statusDeleted: "Deleted",
    memoryTypeField: "memory_type",
    topicsField: "topics",
    participantsField: "participants",
    factsField: "key_facts",
    statsMemories: "Memories",
    statsNodes: "Nodes",
    statsRelations: "Relations",
    statsSessions: "Sessions",
    statsActiveSessions: "Active Sessions",
    statsGraphEntries: "Graph entries",
    statsAtoms: "Atoms",
    statsMessages: "Messages",
    selectedLibraryStatsTitle: "Selected Library Stats",
    selectedLibraryStatsHint: "These seven metrics only describe the currently selected memory library, not the total across all libraries.",
    serviceVersion: "PersonalityRAG Version",
    livingMemoryDbVersion: "LivingMemory DB Version",
    graphNoData: "No graph data",
    legendPerson: "Person",
    legendTopic: "Topic",
    legendFact: "Fact",
    legendSummary: "Summary",
    legendOther: "Other",
    unnamedNode: "Unnamed node",
    nodeMemories: "Memories",
    nodeDegree: "Degree",
    nodeEntries: "Entries",
    nodeWeight: "Weight",
    tableEdit: "Edit",
    tableDelete: "Delete",
    tableEmpty: "No data",
    memoryModalEdit: "Memory #",
    memoryModalNew: "New Memory",
    saved: "Saved",
    deleted: "Deleted",
    confirmDelete: "Delete memory #",
    recallSummary: "{total} results · {elapsed} ms",
    recallNoResult: "No result",
    recallScoreBreakdown: "Score breakdown",
    filesUnit: "files",
    noBackups: "No backups",
    progressPrefix: "Progress",
    jobCompleted: "Rebuild completed",
    jobFailed: "Rebuild failed",
    jobCancelled: "Job cancelled",
    integrityReady: "Integrity check completed",
    pageOfTotal: "Page {page} · {total} total",
    unauthorized: "Session expired. Please log in again.",
    defaultBadge: "Default",
    selectedBadge: "Selected",
    currentLibrary: "Current library",
    noDescription: "No description",
    providerLabel: "Provider",
    modelDimension: "Model / dimension",
    generationLabel: "Generation",
    indexStatus: "Index status",
    indexHealthy: "Healthy",
    indexMismatch: "Count mismatch",
    indexPending: "Pending build",
    defaultPersona: "Default persona",
    noLimit: "Unrestricted",
    enterLibrary: "Open library",
    edit: "Edit",
    backupNow: "Back up now",
    setDefault: "Set default",
    delete: "Delete",
    noLibraries: "No memory libraries",
    typeLabel: "Type",
    modelLabel: "Model",
    dimensionLabel: "Dimension",
    autoDetect: "Auto detect",
    endpointLabel: "API endpoint",
    usedLibraries: "Used by libraries",
    none: "None",
    copy: "Copy",
    notTested: "Not tested",
    availableStatus: "Available · {elapsed} ms",
    unavailableStatus: "Unavailable · {error}",
    pendingRebuild: "Pending rebuild · {libraries}",
    stillUsesRevision: "{library} still uses r{revision}",
    noProviders: "No model providers",
    libraryId: "Library ID",
    displayName: "Display name",
    defaultPersonaOptional: "Default persona ID (optional)",
    description: "Description",
    savePlain: "Save",
    selectProviderType: "Select provider type",
    enabled: "Enabled",
    apiKey: "API Key",
    keepSecretPlaceholder: "Leave blank to keep the current value",
    clearApiKey: "Clear saved API Key",
    apiBaseUrl: "API Base URL",
    embeddingModel: "Embedding model",
    embeddingDimension: "Embedding dimension",
    dimensionHelp: "0 means auto-detect. vLLM never receives this value as a dimensions parameter.",
    timeoutSeconds: "Timeout (seconds)",
    proxyAddress: "Proxy URL",
    proxyPlaceholder: "Example: http://127.0.0.1:7890",
    batchSize: "Batch size",
    concurrency: "Concurrency",
    maxRetries: "Maximum retries",
    testConfig: "Test configuration",
  },
  ru: {
    libraries: "Библиотеки",
    providers: "Провайдеры",
    graph: "Граф знаний",
    memory: "Память",
    recall: "Проверка поиска",
    system: "Система",
    theme: "Тема",
    logout: "Выйти",
    login: "Войти",
    loginHint: "Введите API-ключ из launcher или config/config.json.",
    graphQuery: "Человек, тема, факт или фраза",
    memoryId: "ID памяти",
    sessionOptional: "session_id (необязательно)",
    sessionRecallOptional: "session_id (фильтр сейчас отключён)",
    personaOptional: "persona_id (необязательно)",
    searchGraph: "Поиск в графе",
    overview: "Обзор",
    memorySearch: "Поиск по памяти",
    refresh: "Обновить",
    createMemory: "Новая запись",
    content: "Содержание",
    persona: "Персона",
    importance: "Важность",
    status: "Статус",
    actions: "Действия",
    recallQuery: "Что должна вспомнить персона?",
    runRecall: "Выполнить поиск",
    test: "Проверить",
    indexGeneration: "Поколение индекса",
    importanceDistribution: "Распределение важности",
    atomTypes: "Типы атомов",
    backups: "Резервные копии",
    integrity: "Проверка целостности",
    cancel: "Отмена",
    save: "Сохранить и перестроить",
    serviceEyebrow: "Сервис памяти личности",
    online: "Онлайн",
    loading: "Загрузка…",
    sortNewest: "Сначала новые",
    sortOldest: "Сначала старые",
    sortImportanceDesc: "Важность по убыванию",
    sortImportanceAsc: "Важность по возрастанию",
    previousPage: "Назад",
    nextPage: "Далее",
    topKLabel: "Число результатов эмбеддинга",
    rerankKLabel: "Число результатов rerank",
    embeddingProvider: "Провайдер модели эмбеддингов",
    libraryManagement: "Библиотеки памяти",
    libraryManagementHint: "У каждой библиотеки отдельные база, индекс, кэш и провайдер.",
    createLibrary: "Новая библиотека",
    providerManagement: "Провайдеры моделей",
    providerManagementHint: "Управление провайдерами эмбеддингов для библиотек.",
    createProvider: "Новый провайдер",
    currentProviderBinding: "Текущая привязка",
    manageProviders: "Провайдеры",
    personaField: "persona_id",
    sessionField: "session_id",
    importanceField: "Важность",
    statusField: "Статус",
    statusActive: "Активна",
    statusArchived: "Архив",
    statusDeleted: "Удалена",
    memoryTypeField: "memory_type",
    topicsField: "topics",
    participantsField: "participants",
    factsField: "key_facts",
    statsMemories: "Память",
    statsNodes: "Узлы",
    statsRelations: "Связи",
    statsSessions: "Сессии",
    statsActiveSessions: "Активные сессии",
    statsGraphEntries: "Записи графа",
    statsAtoms: "Атомы",
    statsMessages: "Сообщения",
    selectedLibraryStatsTitle: "Статистика выбранной библиотеки",
    selectedLibraryStatsHint: "Эти семь показателей относятся только к текущей выбранной библиотеке памяти, а не ко всем библиотекам сразу.",
    serviceVersion: "Версия PersonalityRAG",
    livingMemoryDbVersion: "Версия LivingMemory DB",
    graphNoData: "Нет данных графа",
    legendPerson: "Персона",
    legendTopic: "Тема",
    legendFact: "Факт",
    legendSummary: "Сводка",
    legendOther: "Другое",
    unnamedNode: "Без имени",
    nodeMemories: "Память",
    nodeDegree: "Связность",
    nodeEntries: "Записи",
    nodeWeight: "Вес",
    tableEdit: "Изменить",
    tableDelete: "Удалить",
    tableEmpty: "Нет данных",
    memoryModalEdit: "Память #",
    memoryModalNew: "Новая запись",
    saved: "Сохранено",
    deleted: "Удалено",
    confirmDelete: "Удалить запись #",
    recallSummary: "Результатов: {total}, время: {elapsed} мс",
    recallNoResult: "Нет результатов",
    recallScoreBreakdown: "Разбор оценки",
    filesUnit: "файлов",
    noBackups: "Нет резервных копий",
    progressPrefix: "Прогресс",
    jobCompleted: "Перестройка завершена",
    jobFailed: "Перестройка не удалась",
    jobCancelled: "Задача отменена",
    integrityReady: "Проверка завершена",
    pageOfTotal: "Страница {page} · всего {total}",
    unauthorized: "Сеанс истёк. Войдите снова.",
    defaultBadge: "По умолчанию",
    selectedBadge: "Выбрана",
    currentLibrary: "Текущая библиотека",
    noDescription: "Нет описания",
    providerLabel: "Провайдер",
    modelDimension: "Модель / размерность",
    generationLabel: "Поколение",
    indexStatus: "Состояние индекса",
    indexHealthy: "Исправен",
    indexMismatch: "Несовпадение количества",
    indexPending: "Ожидает построения",
    defaultPersona: "Персона по умолчанию",
    noLimit: "Без ограничений",
    enterLibrary: "Открыть библиотеку",
    edit: "Изменить",
    backupNow: "Создать резервную копию",
    setDefault: "Сделать основной",
    delete: "Удалить",
    noLibraries: "Нет библиотек памяти",
    typeLabel: "Тип",
    modelLabel: "Модель",
    dimensionLabel: "Размерность",
    autoDetect: "Определить автоматически",
    endpointLabel: "Адрес API",
    usedLibraries: "Используется библиотеками",
    none: "Нет",
    copy: "Копировать",
    notTested: "Не проверен",
    availableStatus: "Доступен · {elapsed} мс",
    unavailableStatus: "Недоступен · {error}",
    pendingRebuild: "Ожидает перестройки · {libraries}",
    stillUsesRevision: "{library} использует r{revision}",
    noProviders: "Нет провайдеров моделей",
    libraryId: "ID библиотеки",
    displayName: "Отображаемое имя",
    defaultPersonaOptional: "ID персоны по умолчанию (необязательно)",
    description: "Описание",
    savePlain: "Сохранить",
    selectProviderType: "Выберите тип провайдера",
    enabled: "Включён",
    apiKey: "API Key",
    keepSecretPlaceholder: "Оставьте пустым, чтобы сохранить текущее значение",
    clearApiKey: "Удалить сохранённый API Key",
    apiBaseUrl: "API Base URL",
    embeddingModel: "Модель эмбеддингов",
    embeddingDimension: "Размерность эмбеддингов",
    dimensionHelp: "0 означает автоопределение. vLLM не получает это значение в параметре dimensions.",
    timeoutSeconds: "Тайм-аут (секунды)",
    proxyAddress: "Адрес прокси",
    proxyPlaceholder: "Например: http://127.0.0.1:7890",
    batchSize: "Размер пакета",
    concurrency: "Параллелизм",
    maxRetries: "Максимум повторов",
    testConfig: "Проверить конфигурацию",
  },
};

Object.assign(strings.zh, {
  settings: "基础设置",
  settingsTitle: "基础设置",
  settingsHint: "管理 WebUI 登录/鉴权方式、WebUI 端口和记忆库接入端口。端口修改会在下次启动生效；API Token 仍可作为脚本 Bearer 凭据。",
  currentAccessUrl: "当前 WebUI 地址",
  accessBaseUrl: "服务接入端点 URL",
  accessBaseUrlPlaceholder: "例如 http://127.0.0.1",
  configuredPort: "WebUI 配置端口",
  actualPort: "WebUI 本次实际端口",
  currentApiAccessUrl: "记忆库接入地址",
  configuredAccessPort: "记忆库接入配置端口",
  actualAccessPort: "记忆库接入本次实际端口",
  loginMode: "登录方式",
  loginModePassword: "登录密码",
  loginModeApiKey: "API Token",
  newLoginPassword: "新登录/鉴权密码",
  passwordKeepPlaceholder: "留空保持不变",
  clearLoginPassword: "清除登录密码，恢复 API Token 登录",
  restartService: "重启",
  saveSettings: "保存基础设置",
  settingsSaved: "基础设置已保存",
  settingsPortRestartHint: "服务接入端点 URL 保存为不带端口和尾斜杠的基址；WebUI 端口和记忆库接入端口修改都会在下次启动生效。两个配置端口不能相同。",
  settingsRuntimePendingValue: "{current}（当前运行，重启后 {next}）",
  restartTitle: "正在重启 PersonalityRAG",
  restartMessage: "正在关闭旧实例并等待新的 WebUI 恢复连接。若端口已改动，界面会自动跳转到新地址。",
  restartPhaseStopping: "正在停止旧实例",
  restartPhaseWaiting: "等待新实例启动",
  restartPhaseConnecting: "正在恢复连接",
  restartTargetLabel: "目标 WebUI",
  restartStatusLabel: "重启状态",
  restartStatusPreparing: "正在准备重启请求…",
  restartStatusPolling: "正在探测新实例…",
  restartElapsed: "已等待 {seconds}s",
  restartOpenNow: "立即打开",
  restartUnsavedConfirm: "检测到未保存的基础设置变更；未保存内容不会在本次重启中生效。是否仍然继续重启？",
  loginHintPassword: "请输入你设置的 WebUI 登录密码。",
  confirmAction: "确认",
  confirmTitle: "请确认",
  confirmDeleteLibrary: "确定删除记忆库 {id} 吗？系统会在回收目录保留核心 livingmemory.db 与 conversations.db。",
  confirmLibraryEditSensitiveTitle: "确认关键变更",
  confirmLibraryEditSensitiveMessage: "你正在修改记忆库 {library} 的 {fields}。是否继续保存？",
  libraryEditRenameNote: "记忆库 ID 变更会立即生效，并影响该库的接入标识。",
  libraryEditProviderNote: "模型提供商变更会在保存后触发全量索引重建。",
  libraryEditProviderLowContextWarning: "新embedding模型上下文长度较低，重建是可能会导致分块被截断导致信息丢失",
  defaultLibraryIdLocked: "默认记忆库 ID 已锁定，不可修改。",
  confirmDeleteProvider: "确定删除 Provider {id} 吗？",
  confirmClearLogsTitle: "清空实时日志",
  confirmRebuildTitle: "重建索引",
  copyLibrary: "复制",
  expandLibraryCard: "展开更多信息",
  collapseLibraryCard: "收起更多信息",
  libraryCopied: "记忆库已复制：{id}",
  providerSwitchQueued: "已保存并开始为 {library} 全量重建索引",
  providerSwitchEditHint: "编辑时可选择新的模型提供商；若发生变化，保存后会对该记忆库启动全量索引重建。",
  logs: "日志",
  logsTitle: "日志",
  logsHint: "实时记录检索、写入、迁移、索引重建和模型连接等关键行为；本地文件会自动轮转限制大小。",
  logsRealtime: "实时日志",
  clearLogs: "清空实时日志",
  logAutoScrollOn: "自动滚动已开启",
  logAutoScrollOff: "自动滚动已关闭",
  logsEmpty: "暂时还没有 PersonalityRAG 实时日志。",
  logsMeta: "实时日志 · 缓存 {count}/{max} 条 · {bytes}/{maxBytes}",
  confirmClearLogs: "确认清空当前实时日志吗？这只会清空 WebUI 实时缓存，不会删除本地日志文件。",
  logsCleared: "已清空 {count} 条日志",
  conversationBuffer: "短期缓冲",
  shortSessionsUnit: "短期会话",
  messagesUnit: "消息",
  pendingMessagesUnit: "待总结",
});

Object.assign(strings.en, {
  settings: "Settings",
  settingsTitle: "Basic Settings",
  settingsHint: "Manage WebUI login / auth, the WebUI port, and the memory API access port. Port changes take effect next launch; API token remains available for Bearer automation.",
  currentAccessUrl: "Current WebUI URL",
  accessBaseUrl: "Service endpoint base URL",
  accessBaseUrlPlaceholder: "Example: http://127.0.0.1",
  configuredPort: "Configured WebUI port",
  actualPort: "Actual WebUI port",
  currentApiAccessUrl: "Memory API access URL",
  configuredAccessPort: "Configured memory API port",
  actualAccessPort: "Actual memory API port",
  loginMode: "Login mode",
  loginModePassword: "Password",
  loginModeApiKey: "API Token",
  newLoginPassword: "New login / auth password",
  passwordKeepPlaceholder: "Leave blank to keep unchanged",
  clearLoginPassword: "Clear password and restore API token login",
  restartService: "Restart",
  saveSettings: "Save settings",
  settingsSaved: "Settings saved",
  settingsPortRestartHint: "The endpoint base URL is saved without a port or trailing slash. WebUI and memory API port changes take effect on next launch. The two configured ports must differ.",
  settingsRuntimePendingValue: "{current} (running now, {next} after restart)",
  restartTitle: "Restarting PersonalityRAG",
  restartMessage: "Shutting down the current instance and waiting for the WebUI to come back. If the port changed, this page will jump to the new address automatically.",
  restartPhaseStopping: "Stopping current instance",
  restartPhaseWaiting: "Waiting for new instance",
  restartPhaseConnecting: "Restoring connection",
  restartTargetLabel: "Target WebUI",
  restartStatusLabel: "Restart status",
  restartStatusPreparing: "Preparing restart request...",
  restartStatusPolling: "Probing new instance...",
  restartElapsed: "Waited {seconds}s",
  restartOpenNow: "Open now",
  restartUnsavedConfirm: "There are unsaved basic-setting changes. They will not take effect in this restart. Continue anyway?",
  loginHintPassword: "Enter your WebUI login password.",
  confirmAction: "Confirm",
  confirmTitle: "Confirm",
  confirmDeleteLibrary: "Delete library {id}? The trash will retain livingmemory.db and conversations.db.",
  confirmLibraryEditSensitiveTitle: "Confirm critical changes",
  confirmLibraryEditSensitiveMessage: "You are changing {fields} for library {library}. Continue saving?",
  libraryEditRenameNote: "Changing the library ID takes effect immediately and changes this library's access identity.",
  libraryEditProviderNote: "Changing the provider will trigger a full index rebuild after saving.",
  libraryEditProviderLowContextWarning: "The new embedding model has a lower context length; rebuilding may truncate embedding inputs and reduce retrieval quality.",
  defaultLibraryIdLocked: "The default library ID is locked and cannot be changed.",
  confirmDeleteProvider: "Delete Provider {id}?",
  confirmClearLogsTitle: "Clear live logs",
  confirmRebuildTitle: "Rebuild indexes",
  copyLibrary: "Copy",
  expandLibraryCard: "Expand details",
  collapseLibraryCard: "Collapse details",
  libraryCopied: "Library copied: {id}",
  providerSwitchQueued: "Saved and started full index rebuild for {library}",
  providerSwitchEditHint: "When editing, selecting a different provider starts a full index rebuild for this library after saving.",
  logs: "Logs",
  logsTitle: "Logs",
  logsHint: "Live records for recall, writes, migration, index rebuilds, and provider connectivity. Local log files rotate automatically.",
  logsRealtime: "Live logs",
  clearLogs: "Clear live logs",
  logAutoScrollOn: "Auto scroll on",
  logAutoScrollOff: "Auto scroll off",
  logsEmpty: "No PersonalityRAG live logs yet.",
  logsMeta: "Live logs · {count}/{max} cached · {bytes}/{maxBytes}",
  confirmClearLogs: "Clear the current live log cache? Local log files will not be deleted.",
  logsCleared: "Cleared {count} log entries",
  conversationBuffer: "Short-term buffer",
  shortSessionsUnit: "sessions",
  messagesUnit: "messages",
  pendingMessagesUnit: "pending",
});

Object.assign(strings.ru, {
  settings: "Настройки",
  settingsTitle: "Основные настройки",
  settingsHint: "Настройка входа / аутентификации WebUI, порта WebUI и отдельного порта API памяти. Порты применяются при следующем запуске; API token остаётся для Bearer-доступа.",
  currentAccessUrl: "Текущий WebUI URL",
  accessBaseUrl: "Базовый URL сервиса",
  accessBaseUrlPlaceholder: "Например: http://127.0.0.1",
  configuredPort: "Настроенный порт WebUI",
  actualPort: "Фактический порт WebUI",
  currentApiAccessUrl: "URL API памяти",
  configuredAccessPort: "Настроенный порт API памяти",
  actualAccessPort: "Фактический порт API памяти",
  loginMode: "Способ входа",
  loginModePassword: "Пароль",
  loginModeApiKey: "API Token",
  newLoginPassword: "Новый пароль входа / аутентификации",
  passwordKeepPlaceholder: "Оставьте пустым без изменений",
  clearLoginPassword: "Удалить пароль и вернуть вход по API Token",
  restartService: "Перезапуск",
  saveSettings: "Сохранить настройки",
  settingsSaved: "Настройки сохранены",
  settingsPortRestartHint: "Базовый URL сохраняется без порта и завершающего слеша. Порты WebUI и API памяти применяются при следующем запуске. Эти два порта должны различаться.",
  settingsRuntimePendingValue: "{current} (сейчас работает, после перезапуска {next})",
  restartTitle: "Перезапуск PersonalityRAG",
  restartMessage: "Текущий экземпляр завершается, а страница ждёт возврата WebUI. Если порт изменился, переход на новый адрес произойдёт автоматически.",
  restartPhaseStopping: "Остановка текущего экземпляра",
  restartPhaseWaiting: "Ожидание нового экземпляра",
  restartPhaseConnecting: "Восстановление соединения",
  restartTargetLabel: "Целевой WebUI",
  restartStatusLabel: "Состояние перезапуска",
  restartStatusPreparing: "Подготовка запроса на перезапуск...",
  restartStatusPolling: "Проверка нового экземпляра...",
  restartElapsed: "Ожидание {seconds} с",
  restartOpenNow: "Открыть сейчас",
  restartUnsavedConfirm: "Есть несохранённые изменения основных настроек. Они не вступят в силу после этого перезапуска. Всё равно продолжить?",
  loginHintPassword: "Введите пароль WebUI.",
  confirmAction: "Подтвердить",
  confirmTitle: "Подтвердите",
  confirmDeleteLibrary: "Удалить библиотеку {id}? В корзине сохранятся livingmemory.db и conversations.db.",
  confirmLibraryEditSensitiveTitle: "Подтвердите важные изменения",
  confirmLibraryEditSensitiveMessage: "Вы изменяете поля {fields} у библиотеки {library}. Продолжить сохранение?",
  libraryEditRenameNote: "Изменение ID библиотеки вступит в силу сразу и изменит её идентификатор доступа.",
  libraryEditProviderNote: "Смена провайдера после сохранения запустит полную перестройку индекса.",
  libraryEditProviderLowContextWarning: "У новой embedding-модели меньше контекст; при перестройке входы embedding могут быть обрезаны, что снизит качество поиска.",
  defaultLibraryIdLocked: "ID библиотеки по умолчанию заблокирован и не может быть изменён.",
  confirmDeleteProvider: "Удалить Provider {id}?",
  confirmClearLogsTitle: "Очистить журнал",
  confirmRebuildTitle: "Перестроить индексы",
  copyLibrary: "Копировать",
  expandLibraryCard: "Показать детали",
  collapseLibraryCard: "Скрыть детали",
  libraryCopied: "Библиотека скопирована: {id}",
  providerSwitchQueued: "Сохранено, запущена полная перестройка индекса для {library}",
  providerSwitchEditHint: "При редактировании выбор другого провайдера запустит полную перестройку индекса этой библиотеки после сохранения.",
  logs: "Журнал",
  logsTitle: "Журнал",
  logsHint: "Живой журнал поиска, записи, миграции, перестройки индексов и подключения провайдеров. Локальные файлы автоматически ротируются.",
  logsRealtime: "Живой журнал",
  clearLogs: "Очистить живой журнал",
  logAutoScrollOn: "Автопрокрутка включена",
  logAutoScrollOff: "Автопрокрутка выключена",
  logsEmpty: "Пока нет живых логов PersonalityRAG.",
  logsMeta: "Живой журнал · {count}/{max} записей · {bytes}/{maxBytes}",
  confirmClearLogs: "Очистить текущий живой журнал? Локальные файлы логов не будут удалены.",
  logsCleared: "Очищено записей: {count}",
  conversationBuffer: "Краткий буфер",
  shortSessionsUnit: "сессий",
  messagesUnit: "сообщ.",
  pendingMessagesUnit: "ожидают",
});

Object.assign(strings.zh, {
  rebuildIndex: "\u91cd\u5efa\u7d22\u5f15",
  indexRebuildQueued: "\u5df2\u4e3a {library} \u63d0\u4ea4\u7d22\u5f15\u91cd\u5efa\u4efb\u52a1",
  confirmRebuildLibraryIndex: "\u786e\u5b9a\u8981\u4e3a {library} \u91cd\u5efa\u7d22\u5f15\u5417\uff1f",
  providerSwitchQueued: "\u5df2\u4fdd\u5b58\u5e76\u5f00\u59cb\u4e3a {library} \u5168\u91cf\u91cd\u5efa\u7d22\u5f15",
  logs: "日志与任务列表",
  logsTitle: "日志与任务列表",
  graphPageHint: "查看当前记忆库的关系网络，可手动刷新最新导入或重建后的图谱。",
  recallPageHint: "用当前记忆库执行混合召回；刷新会把结果数重置为 5 并重新运行上一次查询。",
  systemPageHint: "查看当前记忆库统计、索引代次、备份与完整性状态。",
  taskListTitle: "任务列表",
  taskListHint: "索引重建、记忆库复制和记忆导入会排队执行，同一时间只运行一个任务。",
  activeTasks: "进行中",
  finishedTasks: "已结束任务",
  noActiveTasks: "当前没有等待或运行中的任务。",
  noFinishedTasks: "本次启动后还没有已结束任务。",
  taskQueued: "等待执行",
  taskRunning: "正在执行",
  taskCompleted: "已完成",
  taskFailed: "失败",
  taskCancelled: "已取消",
  taskSubmitting: "正在提交任务",
  taskKindIndexRebuild: "索引重建",
  taskKindLibraryCopy: "记忆库复制",
  taskKindImport: "记忆导入",
  taskKindMigration: "LivingMemory 迁移",
  libraryCopyQueued: "已提交记忆库复制任务：{id}",
  importMemory: "导入记忆",
  importMemoryHint: "仅空记忆库可导入 LivingMemory 的 livingmemory.db；导入后会自动重建索引。",
  dropLivingMemoryDb: "拖入 livingmemory.db，或点击选择文件",
  startImport: "开始导入",
  importQueued: "已提交记忆导入任务：{library}",
  chooseImportFile: "请选择 livingmemory.db 文件",
  recallRefreshEmpty: "还没有可刷新的召回查询",
  dbVersionLabel: "数据库版本",
  dbVersionTarget: "目标",
  dbVersionUnknown: "未知",
  dbVersionMismatch: "数据库版本不一致",
});

Object.assign(strings.en, {
  rebuildIndex: "Rebuild index",
  indexRebuildQueued: "Index rebuild submitted for {library}",
  confirmRebuildLibraryIndex: "Rebuild indexes for {library}?",
  logs: "Logs & Tasks",
  logsTitle: "Logs & Tasks",
  graphPageHint: "View the current library graph and refresh after imports or rebuilds.",
  recallPageHint: "Run hybrid recall for the current library. Refresh resets k to 5 and reruns the last query.",
  systemPageHint: "View current library stats, index generation, backups, and integrity.",
  taskListTitle: "Tasks",
  taskListHint: "Index rebuilds, library copies, and imports run one at a time.",
  activeTasks: "Active",
  finishedTasks: "Finished",
  noActiveTasks: "No queued or running tasks.",
  noFinishedTasks: "No finished tasks in this process.",
  taskQueued: "Queued",
  taskRunning: "Running",
  taskCompleted: "Completed",
  taskFailed: "Failed",
  taskCancelled: "Cancelled",
  taskSubmitting: "Submitting task",
  taskKindIndexRebuild: "Index rebuild",
  taskKindLibraryCopy: "Library copy",
  taskKindImport: "Memory import",
  taskKindMigration: "LivingMemory migration",
  libraryCopyQueued: "Library copy queued: {id}",
  importMemory: "Import memory",
  importMemoryHint: "Only empty libraries can import LivingMemory livingmemory.db; indexes rebuild automatically.",
  dropLivingMemoryDb: "Drop livingmemory.db here, or click to choose",
  startImport: "Start import",
  importQueued: "Memory import queued: {library}",
  chooseImportFile: "Choose livingmemory.db first",
  recallRefreshEmpty: "No recall query to refresh yet",
  dbVersionLabel: "DB version",
  dbVersionTarget: "target",
  dbVersionUnknown: "unknown",
  dbVersionMismatch: "database version mismatch",
});

Object.assign(strings.ru, {
  rebuildIndex: "袩械褉械褋褌褉芯懈褌褜 懈薪写械泻褋",
  indexRebuildQueued: "袟邪写邪薪懈械 锌械褉械褋褌褉芯泄泫懈 懈薪写械泫褋邪 写谢褟 {library} 蟹邪锌褍褖械薪芯",
  confirmRebuildLibraryIndex: "袩械褉械褋褌褉芯懈褌褜 懈薪写械泫褋褘 写谢褟 {library}?",
  logs: "Logs & Tasks",
  logsTitle: "Logs & Tasks",
  graphPageHint: "Refresh the graph after imports or rebuilds.",
  recallPageHint: "Refresh resets k to 5 and reruns the last recall query.",
  systemPageHint: "Stats, index generation, backups, and integrity.",
  taskListTitle: "Tasks",
  taskListHint: "Long tasks run one at a time.",
  activeTasks: "Active",
  finishedTasks: "Finished",
  noActiveTasks: "No active tasks.",
  noFinishedTasks: "No finished tasks.",
  taskQueued: "Queued",
  taskRunning: "Running",
  taskCompleted: "Completed",
  taskFailed: "Failed",
  taskCancelled: "Cancelled",
  taskSubmitting: "Submitting task",
  taskKindIndexRebuild: "Index rebuild",
  taskKindLibraryCopy: "Library copy",
  taskKindImport: "Memory import",
  taskKindMigration: "LivingMemory migration",
  libraryCopyQueued: "Library copy queued: {id}",
  importMemory: "Import memory",
  importMemoryHint: "Only empty libraries can import livingmemory.db.",
  dropLivingMemoryDb: "Drop livingmemory.db here, or click to choose",
  startImport: "Start import",
  importQueued: "Memory import queued: {library}",
  chooseImportFile: "Choose livingmemory.db first",
  recallRefreshEmpty: "No recall query to refresh yet",
  dbVersionLabel: "DB version",
  dbVersionTarget: "target",
  dbVersionUnknown: "unknown",
  dbVersionMismatch: "database version mismatch",
});

Object.assign(strings.zh, {
  libraryPsk: "接入密钥",
  libraryPskShort: "密钥",
  libraryPskTitle: "记忆库接入密钥：{library}",
  libraryPskPassword: "请输入鉴权密码",
  libraryPskReady: "密钥已生成",
  libraryPskCopied: "密钥已复制",
  providerIdCopied: "已复制提供商 ID",
  libraryIdCopied: "已复制记忆库 ID",
  close: "关闭",
});

Object.assign(strings.en, {
  libraryPsk: "Access key",
  libraryPskShort: "Key",
  libraryPskTitle: "Library access key: {library}",
  libraryPskPassword: "Enter WebUI login password",
  libraryPskReady: "Key generated",
  libraryPskCopied: "Key copied",
  providerIdCopied: "Provider ID copied",
  libraryIdCopied: "Library ID copied",
  close: "Close",
});

Object.assign(strings.ru, {
  libraryPsk: "Access key",
  libraryPskShort: "Key",
  libraryPskTitle: "Library access key: {library}",
  libraryPskPassword: "Enter WebUI login password",
  libraryPskReady: "Key generated",
  libraryPskCopied: "Key copied",
  providerIdCopied: "Provider ID скопирован",
  libraryIdCopied: "Library ID скопирован",
  close: "Закрыть",
});

Object.assign(strings.zh, {
  confirmProviderEditSensitiveTitle: "确认 Provider ID 变更",
  confirmProviderEditSensitiveMessage: "你正在修改模型提供商 {provider} 的 Provider ID（{currentId} -> {nextId}）。是否继续保存？",
  providerEditRenameNote: "Provider ID 变更会立即生效，并影响该提供商的绑定标识。",
  providerKindEmbedding: "Embedding",
  providerKindRerank: "Rerank",
  rerankProvider: "重排模型提供商",
  rerankProviderOptional: "重排模型提供商（可选）",
  noRerankProvider: "不使用 Rerank",
  rerankModel: "重排模型",
  recallOnlyEmbedding: "仅嵌入召回",
  recallWithRerank: "重排后",
  rerankAppliedText: "Rerank 已应用：{provider}",
  rerankFallbackText: "Rerank 未应用，已回退到仅嵌入召回：{reason}",
  rerankNotConfiguredText: "当前记忆库未绑定 Rerank Provider。",
  rerankCandidateText: "候选池 {count} 条",
});

Object.assign(strings.en, {
  rerankProvider: "Rerank Model Provider",
  rerankProviderOptional: "Rerank Model Provider (optional)",
  noRerankProvider: "Do not use Rerank",
  recallOnlyEmbedding: "Embedding only",
  recallWithRerank: "Reranked",
  rerankKLabel: "Rerank output count",
  rerankAppliedText: "Rerank applied: {provider}",
  rerankFallbackText: "Rerank was not applied; showing embedding fallback: {reason}",
  rerankNotConfiguredText: "The current library has no Rerank Provider.",
  rerankCandidateText: "Candidate pool: {count}",
});

Object.assign(strings.ru, {
  rerankProvider: "Провайдер модели rerank",
  rerankProviderOptional: "Провайдер модели rerank (необязательно)",
  noRerankProvider: "Не использовать Rerank",
  recallOnlyEmbedding: "Только эмбеддинги",
  recallWithRerank: "После rerank",
  rerankKLabel: "Число результатов rerank",
  rerankAppliedText: "Rerank применён: {provider}",
  rerankFallbackText: "Rerank не применён; показан результат эмбеддинга: {reason}",
  rerankNotConfiguredText: "У текущей библиотеки нет Rerank Provider.",
  rerankCandidateText: "Пул кандидатов: {count}",
});

Object.assign(strings.zh, {
  maxContextTokens: "\u6700\u5927\u4e0a\u4e0b\u6587\u957f\u5ea6",
  maxContextHelp: "0 \u8868\u793a\u672a\u77e5\uff1b\u82e5\u4fdd\u5b58\u65f6\u80fd\u81ea\u52a8\u63a2\u6d4b\u5230\uff0c\u5c06\u9501\u5b9a\u4e3a\u63a2\u6d4b\u503c\u3002",
  maxContextAutoHelp: "\u5df2\u81ea\u52a8\u63a2\u6d4b\u5e76\u9501\u5b9a\uff1b\u4fee\u6539 API Base URL \u6216\u6a21\u578b\u540e\u4f1a\u91cd\u65b0\u63a2\u6d4b\u3002",
  maxContextManualHelp: "\u5f53\u524d\u672a\u80fd\u81ea\u52a8\u63a2\u6d4b\uff0c\u53ef\u624b\u52a8\u586b\u5199\u7528\u4e8e\u7d22\u5f15\u5b89\u5168\u8bc4\u4f30\u3002",
  detectMaxContextTokens: "\u5c1d\u8bd5\u63a2\u6d4b\u6700\u5927\u4e0a\u4e0b\u6587\u957f\u5ea6",
});

Object.assign(strings.en, {
  maxContextTokens: "Max context length",
  maxContextHelp: "0 means unknown; if detected on save, the value will be locked.",
  maxContextAutoHelp: "Auto-detected and locked; changing API Base URL or model will detect again.",
  maxContextManualHelp: "Auto-detection is unavailable; enter a manual value for index safety checks.",
  detectMaxContextTokens: "Probe max context length",
});

Object.assign(strings.ru, {
  maxContextTokens: "Max context length",
  maxContextHelp: "0 means unknown; if detected on save, the value will be locked.",
  maxContextAutoHelp: "Auto-detected and locked; changing API Base URL or model will detect again.",
  maxContextManualHelp: "Auto-detection is unavailable; enter a manual value for index safety checks.",
  detectMaxContextTokens: "Probe max context length",
});

Object.assign(strings.en, {
  confirmProviderEditSensitiveTitle: "Confirm Provider ID change",
  confirmProviderEditSensitiveMessage: "You are changing the Provider ID for {provider} ({currentId} -> {nextId}). Continue saving?",
  providerEditRenameNote: "Changing the Provider ID takes effect immediately and changes this provider's binding identity.",
});

Object.assign(strings.ru, {
  confirmProviderEditSensitiveTitle: "Подтвердите изменение Provider ID",
  confirmProviderEditSensitiveMessage: "Вы изменяете Provider ID у провайдера {provider} ({currentId} -> {nextId}). Продолжить сохранение?",
  providerEditRenameNote: "Изменение Provider ID вступит в силу сразу и изменит идентификатор привязки этого провайдера.",
});

Object.assign(strings.zh, {
  memorySummary: "摘要",
  createdAt: "创建时间",
  updatedAt: "更新时间",
  filterStatusAll: "全部状态",
  filterTypeAll: "全部类型",
  typeGeneral: "通用",
  typeFact: "事实",
  typeEvent: "事件",
  typePreference: "偏好",
  typeOpinion: "观点",
  typeFactual: "事实型",
  typeEpisodic: "情节型",
  typeRelational: "关系型",
  typePlanned: "计划型",
  sortUpdatedDesc: "最近更新",
  sortTypeAsc: "类型 A-Z",
  perPage20: "20 条/页",
  perPage50: "50 条/页",
  perPage100: "100 条/页",
  memoryDetails: "记忆 #{id}",
  graphContext: "知识图谱关联",
  metadata: "元数据",
  rawMetadata: "原始元数据",
  noGraphContext: "暂无知识图谱关联",
  editMemory: "编辑",
  deleteMemory: "删除",
  saveMemory: "保存",
  editingMemory: "正在编辑 #{id}",
  updateReason: "更新原因",
  reasonPlaceholder: "可选，用于记录这次修改的原因",
  editHistory: "编辑历史",
  noChanges: "没有检测到变更",
});

Object.assign(strings.en, {
  memorySummary: "Summary",
  createdAt: "Created",
  updatedAt: "Updated",
  filterStatusAll: "All status",
  filterTypeAll: "All types",
  typeGeneral: "General",
  typeFact: "Fact",
  typeEvent: "Event",
  typePreference: "Preference",
  typeOpinion: "Opinion",
  typeFactual: "Factual",
  typeEpisodic: "Episodic",
  typeRelational: "Relational",
  typePlanned: "Planned",
  sortUpdatedDesc: "Recently updated",
  sortTypeAsc: "Type A-Z",
  perPage20: "20 / page",
  perPage50: "50 / page",
  perPage100: "100 / page",
  memoryDetails: "Memory #{id}",
  graphContext: "Graph context",
  metadata: "Metadata",
  rawMetadata: "Raw metadata",
  noGraphContext: "No graph context yet",
  editMemory: "Edit",
  deleteMemory: "Delete",
  saveMemory: "Save",
  editingMemory: "Editing #{id}",
  updateReason: "Update reason",
  reasonPlaceholder: "Optional reason for this edit",
  editHistory: "Edit history",
  noChanges: "No changes detected",
});

Object.assign(strings.ru, {
  memorySummary: "摘要",
  createdAt: "创建时间",
  updatedAt: "更新时间",
  filterStatusAll: "全部状态",
  filterTypeAll: "全部类型",
  typeGeneral: "通用",
  typeFact: "事实",
  typeEvent: "事件",
  typePreference: "偏好",
  typeOpinion: "观点",
  typeFactual: "事实型",
  typeEpisodic: "情节型",
  typeRelational: "关系型",
  typePlanned: "计划型",
  sortUpdatedDesc: "最近更新",
  sortTypeAsc: "类型 A-Z",
  perPage20: "20 条/页",
  perPage50: "50 条/页",
  perPage100: "100 条/页",
  memoryDetails: "记忆 #{id}",
  graphContext: "知识图谱关联",
  metadata: "元数据",
  rawMetadata: "原始元数据",
  noGraphContext: "暂无知识图谱关联",
  editMemory: "编辑",
  deleteMemory: "删除",
  saveMemory: "保存",
  editingMemory: "正在编辑 #{id}",
  updateReason: "更新原因",
  reasonPlaceholder: "可选，用于记录这次修改的原因",
  editHistory: "编辑历史",
  noChanges: "没有检测到变更",
});

Object.assign(strings.zh, {
  providerManagementHint: "\u7edf\u4e00\u7ba1\u7406\u8bb0\u5fc6\u5e93\u53ef\u7ed1\u5b9a\u7684 Embedding \u4e0e Rerank Provider\uff0c\u53ef\u5728\u4e0b\u65b9\u5b50\u83dc\u5355\u95f4\u5207\u6362\u3002",
  providerKindEmbedding: "\u5d4c\u5165(Embedding)",
  providerKindRerank: "\u91cd\u6392(Rerank)",
});

Object.assign(strings.en, {
  providerManagementHint: "Manage model providers shared by memory libraries, and switch between Embedding and Rerank below.",
  providerKindEmbedding: "Embedding",
  providerKindRerank: "Rerank",
});

Object.assign(strings.ru, {
  providerManagementHint: "Manage model providers shared by memory libraries, and switch between Embedding and Rerank below.",
  providerKindEmbedding: "Embedding",
  providerKindRerank: "Rerank",
});

strings.zh.indexConflict = "\u7d22\u5f15\u51b2\u7a81";
strings.en.indexConflict = "Index conflict";
strings.ru.indexConflict = "\u041a\u043e\u043d\u0444\u043b\u0438\u043a\u0442 \u0438\u043d\u0434\u0435\u043a\u0441\u0430";

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

function logUiFeedback(message, error = false) {
  fetch("/api/v1/logs/ui", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      level: error ? "ERROR" : "INFO",
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
    logUiFeedback(message, error);
  }
  clearTimeout(element._timer);
  element._timer = setTimeout(() => element.classList.remove("show"), 2600);
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

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch("/api/v1" + path, {
    credentials: "same-origin",
    headers,
    ...options,
  });
  if (response.status === 401) {
    showLogin();
    throw new Error(t("unauthorized"));
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function settingsDraft() {
  return {
    access_base_url: $("settings-access-base-url")?.value.trim() || "",
    port: Number($("settings-port")?.value || 0),
    access_port: Number($("settings-access-port")?.value || 0),
    new_password: $("settings-password")?.value || "",
    clear_password: Boolean($("settings-clear-password")?.checked),
  };
}

function hasUnsavedSettingsChanges() {
  if (!state.settings) return false;
  const draft = settingsDraft();
  return (
    draft.access_base_url !== (state.settings.access_base_url || "http://127.0.0.1") ||
    draft.port !== Number(state.settings.configured_port || 8765) ||
    draft.access_port !== Number(state.settings.configured_access_port || 8766) ||
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
  const target = new URL(buildAppPageUrl(url, RESTART_RETURN_PAGE));
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
  state.restarting = true;
  state.restart.startedAt = Date.now();
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
    RESTART_RETURN_PAGE,
  );
  $("restart-open-link").classList.add("hidden");
  $("restart-screen").classList.remove("hidden");
  updateRestartScreen("restartPhaseStopping", t("restartStatusPreparing"));
  updateRestartElapsed();
  state.restart.timer = setTimeout(pollRestartStatus, 700);
}

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
  graphView.renderer?.render();
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

async function loadGraph(payload = null) {
  try {
    const data = payload
      ? await libraryApi("/graph/query", { method: "POST", body: JSON.stringify(payload) })
      : await libraryApi("/graph/overview?session_id=" + encodeURIComponent($("graph-session").value || ""));
    const snapshot = data.snapshot || {};
    const stats = data.stats || state.stats || {};
    statCards($("graph-stats"), [
      [t("statsMemories"), stats.total_memories ?? snapshot.memories?.length ?? 0],
      [t("statsNodes"), stats.graph_nodes ?? snapshot.nodes?.length ?? 0],
      [t("statsRelations"), stats.graph_edges ?? snapshot.edges?.length ?? 0],
      [t("statsSessions"), Object.keys(stats.sessions || {}).length],
    ]);
    drawGraph(snapshot, { focusMemoryId: payload?.memory_id || null });
  } catch (error) {
    toast(error.message, true);
  }
}

$("graph-search").addEventListener("click", () =>
  loadGraph({
    query: $("graph-query").value,
    memory_id: Number($("graph-memory-id").value) || null,
    session_id: $("graph-session").value || null,
    limit_memories: 10,
  }),
);

$("graph-overview").addEventListener("click", () => loadGraph());
$("graph-peek-close")?.addEventListener("click", () => closeGraphPeek());
$("graph-peek-overlay")?.addEventListener("click", () => closeGraphPeek());

const GRAPH_TYPE_COLORS = {
  person: "#7367d8",
  topic: "#805bd1",
  fact: "#d5a20a",
  summary: "#ef4d86",
  other: "#8492a6",
};

const graphView = {
  renderer: null,
  index: null,
  selectedNodeId: null,
  selectedMemoryId: null,
};

function graphKey(value) {
  return String(value ?? "");
}

function graphHashUnit(value, salt = 0) {
  const str = `${value}:${salt}`;
  let hash = 2166136261;
  for (let index = 0; index < str.length; index += 1) {
    hash ^= str.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return ((hash >>> 0) % 100000) / 100000;
}

function graphClamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function graphLerp(start, end, amount) {
  return start + (end - start) * amount;
}

function graphHexToRgba(color, alpha) {
  const value = String(color || "#000").replace("#", "").trim();
  const hex = value.length === 3
    ? value
        .split("")
        .map((item) => item + item)
        .join("")
    : value.padEnd(6, "0").slice(0, 6);
  const int = Number.parseInt(hex, 16);
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${graphClamp(alpha, 0, 1)})`;
}

function graphThemeColor(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

class PersonalityGraph2D {
  constructor(container, callbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "graph-surface";
    this.container.innerHTML = "";
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.nodes = [];
    this.edges = [];
    this.nodeMap = new Map();
    this.drawnNodes = [];
    this.viewport = { scale: 1, ox: 0, oy: 0 };
    this.selection = null;
    this.hoverId = null;
    this.drag = null;
    this.moved = false;
    this.rafId = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.boundPointerDown = (event) => this.onPointerDown(event);
    this.boundPointerMove = (event) => this.onPointerMove(event);
    this.boundPointerUp = (event) => this.onPointerUp(event);
    this.boundWheel = (event) => this.onWheel(event);
    this.boundDblClick = (event) => this.onDblClick(event);
    this.boundAnimate = (time) => this.animate(time);
    this.boundTheme = () => this.render();
    this.canvas.addEventListener("pointerdown", this.boundPointerDown);
    window.addEventListener("pointermove", this.boundPointerMove);
    window.addEventListener("pointerup", this.boundPointerUp);
    window.addEventListener("pointercancel", this.boundPointerUp);
    this.canvas.addEventListener("wheel", this.boundWheel, { passive: false });
    this.canvas.addEventListener("dblclick", this.boundDblClick);
    window.addEventListener("storage", this.boundTheme);
    this.resize();
    this.startAnimation();
  }

  destroy() {
    this.resizeObserver?.disconnect();
    this.canvas.removeEventListener("pointerdown", this.boundPointerDown);
    window.removeEventListener("pointermove", this.boundPointerMove);
    window.removeEventListener("pointerup", this.boundPointerUp);
    window.removeEventListener("pointercancel", this.boundPointerUp);
    this.canvas.removeEventListener("wheel", this.boundWheel);
    this.canvas.removeEventListener("dblclick", this.boundDblClick);
    window.removeEventListener("storage", this.boundTheme);
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, rect.width || 1100);
    const height = Math.max(320, rect.height || 600);
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = width;
    this.height = height;
    this.render();
  }

  loadData(snapshot = {}) {
    const nodes = snapshot.nodes || [];
    const edges = snapshot.edges || [];
    this.nodes = nodes.map((node) => {
      const id = graphKey(node.id);
      const angle = graphHashUnit(id, 13) * Math.PI * 2;
      const distance = Math.sqrt(graphHashUnit(id, 17)) * 180 + 20;
      return {
        ...node,
        id,
        radius: this.nodeRadius(node),
        x: Math.cos(angle) * distance,
        y: Math.sin(angle) * distance,
        vx: 0,
        vy: 0,
        fixed: false,
        phase: graphHashUnit(id, 29) * Math.PI * 2,
      };
    });
    this.nodeMap = new Map(this.nodes.map((node) => [node.id, node]));
    this.edges = edges
      .map((edge) => ({
        ...edge,
        source: graphKey(edge.source),
        target: graphKey(edge.target),
        weight: Number(edge.weight || 1),
        confidence: Number(edge.confidence || 0.8),
      }))
      .filter((edge) => this.nodeMap.has(edge.source) && this.nodeMap.has(edge.target));
    this.selection = null;
    this.hoverId = null;
    this.viewport = { scale: 1, ox: 0, oy: 0 };
    this.computeLayout();
    this.nodes.forEach((node) => {
      const weight = graphClamp(Number(node.weight || 0), 0, 20);
      node.homeX = node.x;
      node.homeY = node.y;
      node.floatAmp = node.type === "fact" ? 2.45 : node.type === "topic" ? 1.45 : 1.75 + Math.sqrt(weight) * 0.16;
    });
    this.fitToView();
    this.render();
  }

  nodeRadius(node) {
    const weight = graphClamp(Number(node.weight || 0), 0, 32);
    const memoryCount = graphClamp(Number(node.memory_count || 0), 0, 20);
    const base = node.type === "fact" ? 7.7 : node.type === "topic" ? 4.45 : node.type === "person" ? 5.2 : 4.9;
    return graphClamp(base + Math.sqrt(weight) * 0.45 + Math.sqrt(memoryCount) * 0.55, 4, 15);
  }

  startAnimation() {
    if (!this.rafId) this.rafId = requestAnimationFrame(this.boundAnimate);
  }

  animate(time) {
    const now = time / 1000;
    if (!this.drag || this.drag.type !== "node") {
      this.nodes.forEach((node) => {
        if (node.fixed) return;
        const primaryAmp = node.floatAmp || 1;
        const secondaryAmp = primaryAmp * 0.32;
        const targetX =
          node.homeX +
          Math.sin(now * 1.28 + node.phase) * primaryAmp +
          Math.cos(now * 2.05 + node.phase * 0.73) * secondaryAmp;
        const targetY =
          node.homeY +
          Math.cos(now * 1.11 + node.phase) * primaryAmp +
          Math.sin(now * 1.82 + node.phase * 0.69) * secondaryAmp;
        node.x = graphLerp(node.x, targetX, 0.18);
        node.y = graphLerp(node.y, targetY, 0.18);
      });
      this.render();
    }
    this.rafId = requestAnimationFrame(this.boundAnimate);
  }

  computeLayout() {
    const count = this.nodes.length;
    if (count <= 1) return;
    const iterations = count > 200 ? 260 : count > 100 ? 320 : 380;
    for (let step = 0; step < iterations; step += 1) {
      const alpha = 1 - step / iterations;
      const cooled = 0.3 + alpha * 0.7;
      for (let i = 0; i < this.nodes.length; i += 1) {
        const a = this.nodes[i];
        for (let j = i + 1; j < this.nodes.length; j += 1) {
          const b = this.nodes[j];
          let dx = a.x - b.x;
          let dy = a.y - b.y;
          let distSq = dx * dx + dy * dy;
          if (distSq < 0.01) {
            const kick = graphHashUnit(`${a.id}:${b.id}`, 43) * Math.PI * 2;
            dx = Math.cos(kick) * 0.1;
            dy = Math.sin(kick) * 0.1;
            distSq = dx * dx + dy * dy;
          }
          const dist = Math.sqrt(distSq);
          const minSep = (a.radius + b.radius) * 2.55 + 22;
          let repulse = (2350 * cooled) / Math.max(distSq, minSep * minSep * 0.22);
          if (dist < minSep) repulse += (minSep - dist) * 0.42;
          const fx = (dx / dist) * repulse;
          const fy = (dy / dist) * repulse;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
      }
      this.edges.forEach((edge) => {
        const source = this.nodeMap.get(edge.source);
        const target = this.nodeMap.get(edge.target);
        if (!source || !target) return;
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
        const targetDistance = 132 + graphClamp(edge.weight || 1, 0.4, 12) * 6.5;
        const force = (dist - targetDistance) * 0.028 * graphClamp(edge.confidence || 0.8, 0.2, 1);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        source.vx += fx;
        source.vy += fy;
        target.vx -= fx;
        target.vy -= fy;
      });
      this.nodes.forEach((node) => {
        node.vx += -node.x * 0.0062;
        node.vy += -node.y * 0.0062;
        node.vx = graphClamp(node.vx * 0.82, -15, 15);
        node.vy = graphClamp(node.vy * 0.82, -15, 15);
        node.x += node.vx;
        node.y += node.vy;
      });
    }
  }

  fitToView() {
    if (!this.nodes.length) return;
    const xs = this.nodes.map((node) => node.x);
    const ys = this.nodes.map((node) => node.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const graphWidth = Math.max(80, maxX - minX);
    const graphHeight = Math.max(80, maxY - minY);
    const fitScale = Math.min(this.width / (graphWidth + 140), this.height / (graphHeight + 140));
    this.viewport.scale = graphClamp(fitScale * 1.04, 0.34, 1.45);
    this.viewport.ox = -(minX + maxX) / 2;
    this.viewport.oy = -(minY + maxY) / 2;
  }

  screenToWorld(sx, sy) {
    return {
      x: (sx - this.width / 2) / this.viewport.scale - this.viewport.ox,
      y: (sy - this.height / 2) / this.viewport.scale - this.viewport.oy,
    };
  }

  worldToScreen(wx, wy) {
    return {
      x: (wx + this.viewport.ox) * this.viewport.scale + this.width / 2,
      y: (wy + this.viewport.oy) * this.viewport.scale + this.height / 2,
    };
  }

  pointerPosition(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  isRelatedToSelection(edge) {
    if (!this.selection) return false;
    if (this.selection.type === "node") return edge.source === this.selection.id || edge.target === this.selection.id;
    if (this.selection.type === "memory") return graphKey(edge.source_memory_id) === this.selection.id;
    return false;
  }

  render() {
    if (!this.ctx || !this.width || !this.height) return;
    const ctx = this.ctx;
    const darkTheme = document.documentElement.getAttribute("data-theme") === "dark";
    const accent = graphThemeColor("--accent", "#ff4f9a");
    const text = graphThemeColor("--text", "#241c2c");
    const muted = graphThemeColor("--muted", "#6b7280");
    const focusedNodeId = this.selection?.type === "node" ? this.selection.id : null;
    const focusedNeighbors = new Set();
    if (focusedNodeId) {
      this.edges.forEach((edge) => {
        if (edge.source === focusedNodeId) focusedNeighbors.add(edge.target);
        else if (edge.target === focusedNodeId) focusedNeighbors.add(edge.source);
      });
    }
    ctx.clearRect(0, 0, this.width, this.height);
    this.drawnNodes = [];
    ctx.save();
    ctx.translate(this.width / 2, this.height / 2);
    ctx.scale(this.viewport.scale, this.viewport.scale);
    ctx.translate(this.viewport.ox, this.viewport.oy);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    this.edges.forEach((edge) => {
      const source = this.nodeMap.get(edge.source);
      const target = this.nodeMap.get(edge.target);
      if (!source || !target) return;
      if (focusedNodeId && edge.source !== focusedNodeId && edge.target !== focusedNodeId) return;
      const hoverRelated = edge.source === this.hoverId || edge.target === this.hoverId;
      const active = this.isRelatedToSelection(edge) || hoverRelated;
      const strength = Math.max(0, Math.min(1, Math.sqrt(Number(edge.weight || 1)) / 3.6));
      let opacity = active ? (focusedNodeId ? 0.44 : 0.28) : (focusedNodeId ? 0.24 : 0.18);
      let width = active ? (focusedNodeId ? 1.1 : 0.92) : (focusedNodeId ? 0.78 : 0.72);
      width += strength * (focusedNodeId ? 0.24 : 0.45);
      opacity = Math.min(0.72, opacity + strength * (focusedNodeId ? 0.03 : 0.08));

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = hoverRelated
        ? graphHexToRgba(accent, Math.min(0.9, opacity + 0.22))
        : darkTheme
          ? `rgba(150,157,168,${opacity})`
          : `rgba(91,103,120,${opacity})`;
      ctx.lineWidth = width / this.viewport.scale;
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    this.nodes.forEach((node) => {
      const selected = this.selection?.type === "node" && this.selection.id === node.id;
      const hovered = this.hoverId === node.id;
      const relatedToMemory = this.selection?.type === "memory" && graphView.index?.memoryToNodes.get(this.selection.id)?.has(node.id);
      const color = GRAPH_TYPE_COLORS[node.type] || GRAPH_TYPE_COLORS.other;
      const isFocusedNeighbor = Boolean(focusedNodeId) && focusedNeighbors.has(node.id);
      const isMutedByFocus = Boolean(focusedNodeId) && !selected && !isFocusedNeighbor;
      const renderRadius = selected
        ? node.radius * 1.9
        : isMutedByFocus
          ? Math.max(node.radius * 0.34, 1.6)
          : node.radius;
      const halo = selected ? 11 / this.viewport.scale : hovered && !isMutedByFocus ? 5.5 / this.viewport.scale : relatedToMemory ? 4 / this.viewport.scale : 0;
      if (halo > 0) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, renderRadius + halo, 0, Math.PI * 2);
        ctx.fillStyle = graphHexToRgba(color, selected ? 0.16 : 0.08);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(node.x, node.y, renderRadius, 0, Math.PI * 2);
      ctx.fillStyle = isMutedByFocus ? graphHexToRgba("#cfd5e3", 0.2) : color;
      ctx.fill();
      if ((selected || hovered || relatedToMemory) && !isMutedByFocus) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, renderRadius, 0, Math.PI * 2);
        ctx.strokeStyle = selected ? color : graphHexToRgba(color, 0.82);
        ctx.lineWidth = selected ? 2 / this.viewport.scale : 1.35 / this.viewport.scale;
        ctx.stroke();
      }

      const label = String(node.label || node.canonical_value || node.id);
      const shortLabel = label.length > 24 ? `${label.slice(0, 24)}…` : label;
      const metaVisible = focusedNodeId ? selected : hovered || selected;
      const prominent = Number(node.degree || 0) >= 4 || Number(node.memory_count || 0) >= 3 || Number(node.weight || 0) >= 11;
      const labelVisible = focusedNodeId
        ? selected
        : metaVisible || relatedToMemory || (!this.selection && this.viewport.scale > 0.78 && prominent) || this.viewport.scale > 1.08;
      const labelX = node.x + renderRadius + 7 / this.viewport.scale;
      let labelWidth = 0;
      let metaWidth = 0;
      if (labelVisible) {
        const labelFontSize = (selected ? 13 : 11) / this.viewport.scale;
        const metaFontSize = (selected ? 9.5 : 8.5) / this.viewport.scale;
        ctx.font = `${selected ? 700 : 600} ${labelFontSize}px Inter, Microsoft YaHei, sans-serif`;
        ctx.fillStyle = text;
        ctx.textBaseline = "middle";
        ctx.fillText(shortLabel, labelX, node.y + (metaVisible ? -5 / this.viewport.scale : 1 / this.viewport.scale));
        labelWidth = ctx.measureText(shortLabel).width;
        if (metaVisible) {
          const metaLabel = `${Number(node.memory_count || 0)}M / ${Number(node.degree || 0)} links`;
          ctx.font = `${metaFontSize}px Inter, Microsoft YaHei, sans-serif`;
          ctx.fillStyle = muted;
          ctx.textBaseline = "top";
          ctx.fillText(metaLabel, labelX, node.y + 5 / this.viewport.scale);
          metaWidth = ctx.measureText(metaLabel).width;
        }
      }
      const screen = this.worldToScreen(node.x, node.y);
      this.drawnNodes.push({
        id: node.id,
        x: screen.x,
        y: screen.y,
        radius: renderRadius * this.viewport.scale + 7,
        labelLeft: screen.x + (renderRadius + 6 / this.viewport.scale) * this.viewport.scale,
        labelRight: screen.x + (renderRadius + 6 / this.viewport.scale) * this.viewport.scale + Math.max(labelWidth, metaWidth) * this.viewport.scale,
        labelTop: screen.y - 12,
        labelBottom: screen.y + (metaVisible ? 26 : 12),
      });
    });
    ctx.restore();
  }

  hitTestNode(sx, sy) {
    for (let index = this.drawnNodes.length - 1; index >= 0; index -= 1) {
      const item = this.drawnNodes[index];
      const dx = sx - item.x;
      const dy = sy - item.y;
      if (Math.sqrt(dx * dx + dy * dy) <= item.radius) return this.nodeMap.get(item.id);
      if (sx >= item.labelLeft && sx <= item.labelRight && sy >= item.labelTop && sy <= item.labelBottom) {
        return this.nodeMap.get(item.id);
      }
    }
    return null;
  }

  onPointerDown(event) {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    const pos = this.pointerPosition(event);
    const hit = this.hitTestNode(pos.x, pos.y);
    this.moved = false;
    event.preventDefault();
    if (hit) {
      const world = this.screenToWorld(pos.x, pos.y);
      this.drag = {
        type: "node",
        id: hit.id,
        startX: pos.x,
        startY: pos.y,
        offsetX: world.x - hit.x,
        offsetY: world.y - hit.y,
      };
    } else {
      this.drag = {
        type: "pan",
        startX: pos.x,
        startY: pos.y,
        baseX: this.viewport.ox,
        baseY: this.viewport.oy,
      };
    }
    this.canvas.classList.add("dragging");
  }

  onPointerMove(event) {
    const pos = this.pointerPosition(event);
    if (this.drag) {
      const distance = Math.hypot(pos.x - this.drag.startX, pos.y - this.drag.startY);
      if (distance > 3) this.moved = true;
      if (this.drag.type === "node") {
        const node = this.nodeMap.get(this.drag.id);
        if (!node) return;
        const world = this.screenToWorld(pos.x, pos.y);
        node.x = world.x - this.drag.offsetX;
        node.y = world.y - this.drag.offsetY;
        node.vx = 0;
        node.vy = 0;
        node.fixed = true;
        this.render();
      } else {
        this.viewport.ox = this.drag.baseX + (pos.x - this.drag.startX) / this.viewport.scale;
        this.viewport.oy = this.drag.baseY + (pos.y - this.drag.startY) / this.viewport.scale;
        this.render();
      }
      return;
    }
    const hit = this.hitTestNode(pos.x, pos.y);
    const hoverId = hit?.id || null;
    if (hoverId !== this.hoverId) {
      this.hoverId = hoverId;
      this.canvas.style.cursor = hit ? "move" : "grab";
      this.callbacks.onNodeHover?.(hoverId);
      this.render();
    }
  }

  onPointerUp(event) {
    if (!this.drag) return;
    const pos = this.pointerPosition(event);
    const drag = this.drag;
    if (drag.type === "node" && this.moved) {
      const node = this.nodeMap.get(drag.id);
      if (node) {
        node.homeX = node.x;
        node.homeY = node.y;
        node.fixed = false;
      }
    }
    this.drag = null;
    this.canvas.classList.remove("dragging");
    if (!this.moved) {
      const hit = this.hitTestNode(pos.x, pos.y);
      if (hit) {
        this.callbacks.onNodeClick?.(hit.id);
      } else if (drag.type === "pan") {
        this.callbacks.onBackgroundClick?.();
      }
    }
  }

  onDblClick(event) {
    const pos = this.pointerPosition(event);
    const hit = this.hitTestNode(pos.x, pos.y);
    if (hit) this.callbacks.onNodeDblClick?.(hit.id);
  }

  onWheel(event) {
    event.preventDefault();
    const pos = this.pointerPosition(event);
    const before = this.screenToWorld(pos.x, pos.y);
    const delta = -event.deltaY * 0.001;
    this.viewport.scale = graphClamp(this.viewport.scale * (1 + delta), 0.2, 3.5);
    const after = this.screenToWorld(pos.x, pos.y);
    this.viewport.ox += before.x - after.x;
    this.viewport.oy += before.y - after.y;
    this.render();
  }

  selectNode(nodeId, focus = false) {
    const id = graphKey(nodeId);
    if (!this.nodeMap.has(id)) return;
    this.selection = { type: "node", id };
    if (focus) this.focusNode(id);
    this.render();
  }

  selectMemory(memoryId) {
    const id = graphKey(memoryId);
    this.selection = { type: "memory", id };
    this.render();
  }

  clearSelection() {
    this.selection = null;
    this.render();
  }

  focusNode(nodeId) {
    const node = this.nodeMap.get(graphKey(nodeId));
    if (!node) return;
    this.viewport.ox = -node.x;
    this.viewport.oy = -node.y;
    this.render();
  }
}

function buildPersonalityGraphIndex(snapshot = {}) {
  const nodes = snapshot.nodes || [];
  const edges = snapshot.edges || [];
  const entries = snapshot.entries || [];
  const memories = snapshot.memories || [];
  const nodeMap = new Map(nodes.map((node) => [graphKey(node.id), node]));
  const memoryMap = new Map(memories.map((memory) => [graphKey(memory.memory_id ?? memory.id), memory]));
  const nodeToMemories = new Map();
  const memoryToNodes = new Map();
  const nodeToEntries = new Map();
  const neighborMap = new Map();
  const ensureSet = (map, key) => {
    if (!map.has(key)) map.set(key, new Set());
    return map.get(key);
  };
  entries.forEach((entry) => {
    const memoryId = graphKey(entry.source_memory_id ?? entry.memory_id);
    (entry.node_ids || []).forEach((nodeIdValue) => {
      const nodeId = graphKey(nodeIdValue);
      ensureSet(memoryToNodes, memoryId).add(nodeId);
      ensureSet(nodeToMemories, nodeId).add(memoryId);
      if (!nodeToEntries.has(nodeId)) nodeToEntries.set(nodeId, []);
      nodeToEntries.get(nodeId).push(entry);
    });
  });
  edges.forEach((edge) => {
    const source = graphKey(edge.source);
    const target = graphKey(edge.target);
    const memoryId = graphKey(edge.source_memory_id);
    ensureSet(memoryToNodes, memoryId).add(source);
    ensureSet(memoryToNodes, memoryId).add(target);
    ensureSet(nodeToMemories, source).add(memoryId);
    ensureSet(nodeToMemories, target).add(memoryId);
    ensureSet(neighborMap, source).add(target);
    ensureSet(neighborMap, target).add(source);
  });
  return { nodeMap, memoryMap, nodeToMemories, memoryToNodes, nodeToEntries, neighborMap };
}

function renderGraphLegend(snapshot = {}) {
  const counts = {};
  (snapshot.nodes || []).forEach((node) => {
    counts[node.type || "other"] = (counts[node.type || "other"] || 0) + 1;
  });
  const labels = {
    person: t("legendPerson"),
    topic: t("legendTopic"),
    fact: t("legendFact"),
    summary: t("legendSummary"),
    other: t("legendOther"),
  };
  $("graph-legend").innerHTML = Object.entries(GRAPH_TYPE_COLORS)
    .filter(([key]) => counts[key])
    .map(([key, color]) => `<span><i style="background:${color}"></i> ${escapeHtml(labels[key] || key)} · ${counts[key]}</span>`)
    .join("");
}

function drawGraph(snapshot, options = {}) {
  const container = $("graph-canvas");
  const nodes = snapshot.nodes || [];
  graphView.index = buildPersonalityGraphIndex(snapshot);
  graphView.selectedNodeId = null;
  graphView.selectedMemoryId = null;
  closeGraphPeek();
  if (!nodes.length) {
    graphView.renderer?.destroy();
    graphView.renderer = null;
    container.innerHTML = `<div class="empty">${escapeHtml(t("graphNoData"))}</div>`;
    $("graph-legend").innerHTML = "";
    return;
  }
  if (!graphView.renderer || graphView.renderer.container !== container) {
    graphView.renderer?.destroy();
    graphView.renderer = new PersonalityGraph2D(container, {
      onNodeClick: (nodeId) => selectGraphNode(nodeId, false),
      onNodeDblClick: (nodeId) => selectGraphNode(nodeId, true),
      onBackgroundClick: () => clearGraphSelection(),
    });
  }
  graphView.renderer.loadData(snapshot);
  renderGraphLegend(snapshot);
  if (options.focusMemoryId) {
    selectGraphMemory(options.focusMemoryId);
  }
}

function clearGraphSelection() {
  graphView.selectedNodeId = null;
  graphView.selectedMemoryId = null;
  graphView.renderer?.clearSelection();
  closeGraphPeek();
}

function selectGraphNode(nodeId, focus = false) {
  const id = graphKey(nodeId);
  const node = graphView.index?.nodeMap.get(id);
  if (!node) return;
  graphView.selectedNodeId = id;
  graphView.selectedMemoryId = null;
  graphView.renderer?.selectNode(id, focus);
  openGraphNodePeek(node);
}

function selectGraphMemory(memoryId) {
  const id = graphKey(memoryId);
  const memory = graphView.index?.memoryMap.get(id);
  graphView.selectedMemoryId = id;
  graphView.selectedNodeId = null;
  graphView.renderer?.selectMemory(id);
  if (memory) openGraphMemoryPeek(memory);
}

function openGraphNodePeek(node) {
  const panel = $("graph-peek-panel");
  if (!panel) return;
  const typeClass = graphKey(node.type || "other").replace(/[^a-z0-9_-]/gi, "") || "other";
  $("graph-peek-badge").textContent = node.type || t("legendOther");
  $("graph-peek-badge").className = `graph-peek-badge ${typeClass}`;
  $("graph-peek-title").textContent = node.label || node.canonical_value || t("unnamedNode");
  $("graph-peek-body").innerHTML = `<div class="peek-meta-grid">
    <div class="peek-meta-item"><span>${escapeHtml(t("nodeMemories"))}</span><strong>${escapeHtml(node.memory_count || 0)}</strong></div>
    <div class="peek-meta-item"><span>${escapeHtml(t("nodeDegree"))}</span><strong>${escapeHtml(node.degree || 0)}</strong></div>
    <div class="peek-meta-item"><span>${escapeHtml(t("nodeEntries"))}</span><strong>${escapeHtml(node.entry_count || 0)}</strong></div>
    <div class="peek-meta-item"><span>${escapeHtml(t("nodeWeight"))}</span><strong>${escapeHtml(Number(node.weight || 0).toFixed(2))}</strong></div>
  </div>
  <dl class="peek-detail-list">
    <dt>ID</dt><dd>${escapeHtml(node.id)}</dd>
    <dt>${escapeHtml(t("typeLabel"))}</dt><dd>${escapeHtml(node.type || "other")}</dd>
    <dt>${escapeHtml(t("content"))}</dt><dd>${escapeHtml(node.canonical_value || node.label || "")}</dd>
  </dl>`;
  openGraphPeek();
}

function openGraphMemoryPeek(memory) {
  const panel = $("graph-peek-panel");
  if (!panel) return;
  $("graph-peek-badge").textContent = memory.memory_type || t("memory");
  $("graph-peek-badge").className = "graph-peek-badge memory";
  $("graph-peek-title").textContent = `#${memory.memory_id ?? memory.id}`;
  const metadata = memory.metadata || {};
  $("graph-peek-body").innerHTML = `<p class="peek-memory-summary">${escapeHtml(memory.summary || memory.content || memory.text || "")}</p>
  <dl class="peek-detail-list">
    <dt>${escapeHtml(t("personaField"))}</dt><dd>${escapeHtml(metadata.persona_id || memory.persona_id || "—")}</dd>
    <dt>${escapeHtml(t("sessionField"))}</dt><dd>${escapeHtml(metadata.session_id || memory.session_id || "—")}</dd>
    <dt>${escapeHtml(t("importanceField"))}</dt><dd>${escapeHtml(memory.importance ?? metadata.importance ?? "—")}</dd>
  </dl>`;
  openGraphPeek();
}

function openGraphPeek() {
  $("graph-peek-overlay")?.classList.remove("hidden");
  $("graph-peek-panel")?.classList.add("visible");
}

function closeGraphPeek() {
  $("graph-peek-overlay")?.classList.add("hidden");
  $("graph-peek-panel")?.classList.remove("visible");
}

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

$("memory-refresh").onclick = () => {
  state.memoryPage = 1;
  loadMemories();
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
        ${memoryMetaItem(t("personaField"), `<code>${escapeHtml(detail.personaId)}</code>`)}
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
  $("memory-detail-delete").onclick = () => deleteMemory(detail.id);
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
    content,
    status,
    memory_type: memoryType,
    importance,
    value_scale: "display",
    metadata,
  };
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
  const tempJobId = addOptimisticTask({
    kind: "index_rebuild",
    libraryId: state.selectedLibraryId,
    markLibraryConflict: true,
  });
  try {
    const updated = await libraryApi("/memories/" + detail.id, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    state.selectedMemoryDetail = updated;
    trackQueuedJob(updated, {
      kind: "index_rebuild",
      libraryId: state.selectedLibraryId,
      tempId: tempJobId,
    });
    toast(t("saved"));
    await loadMemories();
    await openMemoryDetail(detail.id, updated);
  } catch (error) {
    removeOptimisticTask(tempJobId);
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

async function openEdit(id) {
  try {
    const item = await libraryApi("/memories/" + id);
    openModal(item);
  } catch (error) {
    toast(error.message, true);
  }
}

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
  const tempJobId = addOptimisticTask({
    kind: "index_rebuild",
    libraryId: state.selectedLibraryId,
    markLibraryConflict: true,
  });
  try {
    const result = await libraryApi(id ? "/memories/" + id : "/memories", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify(payload),
    });
    trackQueuedJob(result, {
      kind: "index_rebuild",
      libraryId: state.selectedLibraryId,
      tempId: tempJobId,
    });
    closeModal();
    toast(t("saved"));
    await loadMemories();
  } catch (error) {
    removeOptimisticTask(tempJobId);
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
  const tempJobId = addOptimisticTask({
    kind: "index_rebuild",
    libraryId: state.selectedLibraryId,
    markLibraryConflict: true,
  });
  try {
    const result = await libraryApi("/memories/" + id, { method: "DELETE" });
    trackQueuedJob(result, {
      kind: "index_rebuild",
      libraryId: state.selectedLibraryId,
      tempId: tempJobId,
    });
    toast(t("deleted"));
    closeMemoryDetail();
    await loadMemories();
  } catch (error) {
    removeOptimisticTask(tempJobId);
    toast(error.message, true);
  }
}

function setRecallView(view) {
  state.recallView = view === "rerank" && selectedLibraryHasRerank() ? "rerank" : "embedding";
  updateRecallRerankControls();
  document.querySelectorAll("[data-recall-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.recallView === state.recallView);
  });
  renderRecallResults();
}

function recallResultScore(item) {
  const breakdown = item.score_breakdown || {};
  if (state.recallView === "rerank" && breakdown.rerank_score != null) {
    return Number(breakdown.rerank_score);
  }
  return Number(item.similarity_score || 0);
}

function recallSummaryHtml(items = []) {
  const summary = state.recallCache.summary;
  if (!summary) return "";
  const modeLabel = state.recallView === "rerank"
    ? t("recallWithRerank")
    : t("recallOnlyEmbedding");
  const parts = [
    t("recallSummary", {
      total: items.length,
      elapsed: summary.elapsed_time_ms,
    }),
    `<strong>${escapeHtml(modeLabel)}</strong>`,
  ];
  const meta = state.recallCache.rerankMeta || {};
  if (state.recallView === "rerank") {
    if (meta.applied) {
      parts.push(escapeHtml(t("rerankAppliedText", {
        provider: meta.provider_id || meta.provider_type || t("rerankProvider"),
      })));
      if (meta.candidate_count != null) {
        parts.push(escapeHtml(t("rerankCandidateText", { count: meta.candidate_count })));
      }
    } else if (meta.failed || meta.requested) {
      parts.push(`<span class="danger">${escapeHtml(t("rerankFallbackText", {
        reason: meta.error || "unknown",
      }))}</span>`);
    } else {
      parts.push(escapeHtml(t("rerankNotConfiguredText")));
    }
  }
  return parts.join(" · ");
}

function renderRecallResults() {
  updateRecallRerankControls();
  document.querySelectorAll("[data-recall-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.recallView === state.recallView);
  });
  const items = state.recallView === "rerank"
    ? state.recallCache.rerank
    : state.recallCache.embedding;
  $("recall-summary").innerHTML = recallSummaryHtml(items);
  $("recall-results").innerHTML =
    items
      .map(
        (item, index) => `<article class="result">
          <header>
            <span class="rank">#${index + 1}</span>
            <b>ID ${item.memory_id}</b>
            <span class="score">${recallResultScore(item).toFixed(4)}</span>
          </header>
          <div>${escapeHtml(item.content)}</div>
          <p>${escapeHtml(item.metadata?.persona_id || "")} · ${escapeHtml(item.metadata?.session_id || "")}</p>
          <details>
            <summary>${escapeHtml(t("recallScoreBreakdown"))}</summary>
            <pre>${escapeHtml(JSON.stringify(item.score_breakdown || {}, null, 2))}</pre>
          </details>
        </article>`,
      )
      .join("") || `<div class="panel">${escapeHtml(t("recallNoResult"))}</div>`;
}

$("run-recall").onclick = async () => {
  const query = $("recall-query").value.trim();
  if (!query) {
    return;
  }
  const embeddingK = setRecallK($("recall-k").value);
  const hasRerank = selectedLibraryHasRerank();
  const rerankK = hasRerank ? setRecallRerankK($("recall-rerank-k").value) : embeddingK;
  const payload = {
    query,
    k: embeddingK,
    persona_id: $("recall-persona").value || null,
    session_id: $("recall-session").value || null,
    rerank: hasRerank,
  };
  if (hasRerank) {
    payload.rerank_k = rerankK;
    payload.include_baseline = true;
  }
  try {
    const data = await libraryApi("/recall", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.recallCache.summary = data;
    state.recallCache.embedding = data.baseline_results || data.results || [];
    state.recallCache.rerank = hasRerank ? data.results || [] : [];
    state.recallCache.rerankMeta = data.rerank || null;
    if (!hasRerank) {
      state.recallView = "embedding";
    }
    renderRecallResults();
  } catch (error) {
    toast(error.message, true);
  }
};

$("recall-k")?.addEventListener("input", (event) => {
  setRecallK(event.target.value);
});

$("recall-rerank-k")?.addEventListener("input", (event) => {
  setRecallRerankK(event.target.value);
});

async function loadSystem() {
  try {
    const data = await libraryApi("/stats");
    state.stats = data;
    statCards($("system-version-stats"), [
      [t("serviceVersion"), formatVersionTag(data.service_version)],
      [t("livingMemoryDbVersion"), formatVersionTag(data.livingmemory_database_version)],
    ]);
    statCards($("system-stats"), [
      [t("statsMemories"), data.total_memories],
      [t("statsNodes"), data.graph_nodes],
      [t("statsRelations"), data.graph_edges],
      [t("statsGraphEntries"), data.graph_entries],
      [t("statsAtoms"), data.atom_count],
      [t("statsActiveSessions"), Object.keys(data.sessions || {}).length],
      [t("statsMessages"), data.conversation_counts?.messages || 0],
    ]);
    $("provider-status").textContent = JSON.stringify(
      {
        library: data.library,
        provider: data.provider,
        status: data.provider_status,
      },
      null,
      2,
    );
    $("index-status").textContent = JSON.stringify(data.indexes, null, 2);
    scheduleSystemPanelsRefresh();
    drawBars($("importance-chart"), data.importance_distribution || {});
    drawBars($("atom-chart"), data.atom_breakdown || {});
    $("backup-list").innerHTML =
      (data.backups || [])
        .map(
          (item) =>
            `<div class="backup-item"><b>${escapeHtml(item.name)}</b><small>${item.file_count} ${escapeHtml(t("filesUnit"))}</small><small>${formatBytes(item.size_bytes)}</small></div>`,
        )
        .join("") || t("noBackups");
  } catch (error) {
    toast(error.message, true);
  }
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
    $("settings-note").textContent = t("settingsPortRestartHint");
  } catch (error) {
    toast(error.message, true);
  }
}

function systemCollapsedHeight() {
  return Math.max(280, Math.round(window.innerHeight * 0.5));
}

function applySystemPanelCollapseState({ panelId, contentId, toggleId, expanded }) {
  const panel = $(panelId);
  const content = $(contentId);
  const toggle = $(toggleId);
  if (!panel || !content || !toggle) return;
  const collapsedHeight = systemCollapsedHeight();
  panel.style.setProperty("--system-panel-collapsed-height", `${collapsedHeight}px`);
  const overflowing = content.scrollHeight > collapsedHeight + 24;
  if (!overflowing) {
    toggle.classList.add("hidden");
    toggle.classList.remove("expanded");
    panel.classList.remove("system-panel-collapsed", "system-panel-expanded");
    return;
  }
  panel.classList.toggle("system-panel-collapsed", !expanded);
  panel.classList.toggle("system-panel-expanded", expanded);
  toggle.classList.toggle("expanded", expanded);
  toggle.classList.remove("hidden");
  const title = expanded ? t("collapseLibraryCard") : t("expandLibraryCard");
  toggle.title = title;
  toggle.setAttribute("aria-label", title);
  toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
}

function applySystemProviderCollapseState() {
  applySystemPanelCollapseState({
    panelId: "system-provider-panel",
    contentId: "provider-status",
    toggleId: "system-provider-toggle",
    expanded: state.systemProviderExpanded,
  });
}

function applySystemIndexCollapseState() {
  applySystemPanelCollapseState({
    panelId: "system-index-panel",
    contentId: "index-status",
    toggleId: "system-index-toggle",
    expanded: state.systemIndexExpanded,
  });
}

function scheduleSystemPanelsRefresh() {
  if (state.systemPanelRefreshFrame) {
    cancelAnimationFrame(state.systemPanelRefreshFrame);
  }
  state.systemPanelRefreshFrame = requestAnimationFrame(() => {
    state.systemPanelRefreshFrame = 0;
    applySystemProviderCollapseState();
    applySystemIndexCollapseState();
  });
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

function drawBars(target, data) {
  const entries = Object.entries(data);
  const max = Math.max(1, ...entries.map((entry) => entry[1]));
  target.innerHTML = entries
    .map(
      ([key, value]) =>
        `<div class="bar-row"><span>${escapeHtml(key)}</span><div class="bar-track"><div class="bar-fill" style="width:${(value / max) * 100}%"></div></div><b>${value}</b></div>`,
    )
    .join("");
}

function formatBytes(size) {
  if (size < 1024) return size + " B";
  if (size < 1048576) return (size / 1024).toFixed(1) + " KB";
  if (size < 1073741824) return (size / 1048576).toFixed(1) + " MB";
  return (size / 1073741824).toFixed(2) + " GB";
}

function taskKindLabel(kind) {
  return {
    index_rebuild: t("taskKindIndexRebuild"),
    library_copy: t("taskKindLibraryCopy"),
    livingmemory_import: t("taskKindImport"),
    livingmemory_migration: t("taskKindMigration"),
  }[kind] || kind || "Task";
}

function taskStatusLabel(status) {
  return {
    queued: t("taskQueued"),
    running: t("taskRunning"),
    completed: t("taskCompleted"),
    failed: t("taskFailed"),
    cancelled: t("taskCancelled"),
  }[status] || status || "";
}

function taskStatusMark(status) {
  if (status === "completed") return "✅";
  if (status === "failed" || status === "cancelled") return "❌";
  if (status === "running") return "▶";
  return "⏳";
}

function taskTimestamp() {
  return Date.now() / 1000;
}

function isActiveTask(job) {
  return ["queued", "running"].includes(String(job?.status || ""));
}

function mergeTaskList(fetched = [], scope = "active") {
  const fetchedIds = new Set(fetched.map((job) => job.id).filter(Boolean));
  const now = taskTimestamp();
  state.tasks.optimistic = state.tasks.optimistic.filter((job) => {
    if (fetchedIds.has(job.id)) return false;
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

function renderTasks() {
  const list = $("task-list");
  if (!list) return;
  syncVisibleTasks();
  const items = state.tasks.scope === "finished" ? state.tasks.finished : state.tasks.active;
  document.querySelectorAll("[data-task-scope]").forEach((button) => {
    button.classList.toggle("active", button.dataset.taskScope === state.tasks.scope);
  });
  if (!items.length) {
    list.innerHTML = `<div class="task-empty">${escapeHtml(t(state.tasks.scope === "finished" ? "noFinishedTasks" : "noActiveTasks"))}</div>`;
    return;
  }
  list.innerHTML = items
    .map((job) => {
      const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
      const status = String(job.status || "");
      const message = job.error || job.message || "";
      const library = job.library_id || "—";
      return `<article class="task-item task-status-${escapeHtml(status)}">
        <div class="task-head">
          <div>
            <div class="task-title"><span>${taskStatusMark(status)}</span><span>${escapeHtml(taskKindLabel(job.kind))}</span></div>
            <div class="task-meta">
              <span>job ${escapeHtml(String(job.id || "").slice(0, 8))}</span>
              <span>${escapeHtml(library)}</span>
              <span>${escapeHtml(taskStatusLabel(status))}</span>
            </div>
          </div>
          <b>${Math.round(progress * 100)}%</b>
        </div>
        <div class="task-progress"><div style="width:${progress * 100}%"></div><span>${escapeHtml(message)}</span></div>
      </article>`;
    })
    .join("");
}

async function loadTasks(scope = state.tasks.scope) {
  const data = await api(`/jobs?scope=${encodeURIComponent(scope)}`);
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
  const payload = await api(`/logs?${query.toString()}`, { signal });
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

async function watchJob(id) {
  const box = $("job-progress");
  const bar = box.querySelector("div");
  const label = box.querySelector("span");
  box.classList.remove("hidden");
  while (true) {
    const job = await api("/jobs/" + id);
    upsertJobSnapshot(job);
    bar.style.width = `${job.progress * 100}%`;
    label.textContent = `${t("progressPrefix")} ${Math.round(job.progress * 100)}% · ${job.message}`;
    if (["completed", "failed", "cancelled"].includes(job.status)) {
      const terminalMessage =
        job.status === "completed"
          ? t("jobCompleted")
          : job.status === "failed"
            ? t("jobFailed")
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
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

async function startLibraryIndexRebuild(libraryId, providerId = "", reason = "manual", options = {}) {
  const payload = { reason };
  if (providerId) {
    payload.provider_id = providerId;
  }
  const tempJobId = addOptimisticTask({
    kind: "index_rebuild",
    libraryId,
    markLibraryConflict: Boolean(options.markLibraryConflict),
  });
  try {
    const result = await api(`/libraries/${encodeURIComponent(libraryId)}/indexes/rebuild`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    trackQueuedJob(result, {
      kind: "index_rebuild",
      libraryId,
      tempId: tempJobId,
    });
    if (!result.job_id && options.markLibraryConflict) {
      clearLibraryIndexConflict(libraryId);
    }
    toast(t("indexRebuildQueued", { library: libraryId }));
    return result.job_id;
  } catch (error) {
    removeOptimisticTask(tempJobId);
    if (options.markLibraryConflict) {
      clearLibraryIndexConflict(libraryId);
    }
    throw error;
  }
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
    (item) => item.usage_kind !== "rerank" && item.library_id === library.id,
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

function markLibraryIndexConflict(libraryId) {
  if (!libraryId) return;
  state.optimisticIndexConflicts.add(libraryId);
  if (state.page === "libraries") {
    renderLibraryCards();
  }
  refreshLibraryContext();
}

function clearLibraryIndexConflict(libraryId) {
  if (!libraryId || !state.optimisticIndexConflicts.has(libraryId)) return;
  state.optimisticIndexConflicts.delete(libraryId);
  if (state.page === "libraries") {
    renderLibraryCards();
  }
  refreshLibraryContext();
}

function libraryIndexState(library) {
  const stats = library?.stats || {};
  const indexes = library?.indexes || {};
  if (!indexes.generation) {
    return "pending";
  }
  if (state.optimisticIndexConflicts.has(library.id)) {
    return "conflict";
  }
  const provider = latestProvider(library?.provider_id) || library?.provider || null;
  const needsRebuild = provider ? Boolean(libraryEditNeedsRebuild(library, provider)) : false;
  const countHealthy = Number(indexes.document_vectors || 0) === Number(stats.total_memories || 0)
    && Number(indexes.graph_vectors || 0) === Number(stats.graph_entries || 0);
  if (needsRebuild || !countHealthy) {
    return "conflict";
  }
  return "healthy";
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
    const data = await libraryApi("/integrity");
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
$("system-provider-toggle")?.addEventListener("click", () => {
  state.systemProviderExpanded = !state.systemProviderExpanded;
  applySystemProviderCollapseState();
});
$("system-index-toggle")?.addEventListener("click", () => {
  state.systemIndexExpanded = !state.systemIndexExpanded;
  applySystemIndexCollapseState();
});

async function ensureLibrarySelection() {
  if (!state.libraries.length) {
    await loadLibraries(false);
  }
  let library = selectedLibrary();
  if (!library) {
    library = state.libraries.find((item) => item.is_default) || state.libraries[0];
    state.selectedLibraryId = library?.id || "";
  }
  if (state.selectedLibraryId) {
    localStorage.setItem("prag_library_id", state.selectedLibraryId);
  }
  refreshLibraryContext();
}

function renderLibraryCards() {
  $("library-cards").innerHTML =
    state.libraries
      .map((library) => {
        const stats = library.stats || {};
        const provider = library.provider || {};
        const indexes = library.indexes || {};
        const metadata = library.metadata || {};
        const compatibility = library.compatibility || {};
        const isSelected = library.id === state.selectedLibraryId;
        const isEmpty = libraryIsEmpty(library);
        const isExpanded = state.expandedLibraryIds.has(library.id);
        const targetDbVersion = compatibility.livingmemory_database_version;
        const libraryDbVersion = metadata.livingmemory_database_version;
        const activeSessionCount = Object.keys(stats.sessions || {}).length;
        const conversationCounts = stats.conversation_counts || {};
        const conversationBufferText = [
          `${Number(conversationCounts.sessions || 0)} ${t("shortSessionsUnit")}`,
          `${Number(conversationCounts.messages || 0)} ${t("messagesUnit")}`,
          `${Number(conversationCounts.pending_messages || 0)} ${t("pendingMessagesUnit")}`,
        ].join(" / ");
        const libraryDbVersionKnown = libraryDbVersion !== undefined && libraryDbVersion !== null && libraryDbVersion !== "";
        const dbVersionMismatch = String(libraryDbVersion) !== String(targetDbVersion);
        const dbVersionText = `${libraryDbVersionKnown ? `v${libraryDbVersion}` : t("dbVersionUnknown")} / ${t("dbVersionTarget")} ${targetDbVersion ? `v${targetDbVersion}` : "—"}`;
        const dbVersionClass = dbVersionMismatch ? "compat-warning" : "";
        const dbVersionWarning = dbVersionMismatch ? ` · ${escapeHtml(t("dbVersionMismatch"))}` : "";
        const indexState = libraryIndexState(library);
        const indexHealth = indexState === "healthy"
          ? `<span class="pill success">${escapeHtml(t("indexHealthy"))}</span>`
          : indexState === "pending"
            ? `<span class="pill warning">${escapeHtml(t("indexPending"))}</span>`
            : `<span class="pill danger">${escapeHtml(t("indexConflict"))}</span>`;
        const expandTitle = isExpanded ? t("collapseLibraryCard") : t("expandLibraryCard");
        return `<article class="management-card library-card ${isSelected ? "active-card" : ""} ${isExpanded ? "library-card-expanded" : "library-card-collapsed"}" data-id="${escapeHtml(library.id)}" role="button" tabindex="0" aria-pressed="${isSelected ? "true" : "false"}">
          <div class="library-card-primary">
            <header>
              <div>
                <h3>${escapeHtml(library.name)} ${library.is_default ? `<span class="pill success">${escapeHtml(t("defaultBadge"))}</span>` : ""}</h3>
                <span class="subtle">${escapeHtml(library.id)}</span>
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
              <dt>${escapeHtml(t("providerLabel"))}</dt><dd>${escapeHtml(provider.display_name || provider.id || "—")}</dd>
              <dt>${escapeHtml(t("modelDimension"))}</dt><dd>${escapeHtml(provider.model || "—")} / ${provider.dimensions || indexes.manifest?.dimension || t("autoDetect")}</dd>
            </dl>
            <div class="library-card-extra">
              <div class="library-card-extra-inner">
                <dl class="provider-meta">
                  <dt>${escapeHtml(t("dbVersionLabel"))}</dt><dd class="${dbVersionClass}">${escapeHtml(dbVersionText)}${dbVersionWarning}</dd>
                  <dt>${escapeHtml(t("generationLabel"))}</dt><dd>${escapeHtml(indexes.generation || t("indexPending"))}</dd>
                  <dt>${escapeHtml(t("indexStatus"))}</dt><dd>${indexHealth}</dd>
                  <dt>${escapeHtml(t("defaultPersona"))}</dt><dd>${escapeHtml(library.default_persona_id || t("noLimit"))}</dd>
                  <dt>${escapeHtml(t("conversationBuffer"))}</dt><dd>${escapeHtml(conversationBufferText)}</dd>
                </dl>
              </div>
            </div>
          </div>
          <div class="card-actions library-card-actions">
            <button class="primary enter-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("enterLibrary"))}</button>
            <button class="ghost edit-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("edit"))}</button>
            <button class="ghost copy-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("copyLibrary"))}</button>
            ${isEmpty ? `<button class="ghost import-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("importMemory"))}</button>` : ""}
            <button class="ghost rebuild-library-index" data-id="${escapeHtml(library.id)}" data-provider="${escapeHtml(provider.id || library.provider_id || "")}">${escapeHtml(t("rebuildIndex"))}</button>
            <button class="ghost backup-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("backupNow"))}</button>
            ${library.is_default ? "" : `<button class="ghost default-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("setDefault"))}</button><button class="ghost danger delete-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("delete"))}</button>`}
          </div>
          <button class="ghost key-library key-library-fab" data-id="${escapeHtml(library.id)}" title="${escapeHtml(t("libraryPsk"))}" aria-label="${escapeHtml(t("libraryPsk"))}">
            <svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M701.248 0A323.2 323.2 0 0 0 393.6 420.8L6.528 807.872A22.4 22.4 0 0 0 0 823.68v178.048c0 12.288 9.984 22.272 22.272 22.272h155.84c5.888 0 11.52-2.368 15.744-6.528l44.8-44.8a22.336 22.336 0 0 0 6.528-15.744v-70.848h70.848a22.272 22.272 0 0 0 22.272-22.272v-62.4h62.4c5.888 0 11.52-2.368 15.744-6.528l170.24-170.24A323.2 323.2 0 0 0 1024 322.752 323.2 323.2 0 0 0 701.248 0z m-225.28 496.96l-379.072 379.072a11.136 11.136 0 1 1-15.68-15.744L460.288 481.28a11.136 11.136 0 0 1 15.744 15.744z m385.472-35.328l-15.744 15.744L546.56 178.304l15.744-15.744a210.112 210.112 0 0 1 149.504-61.888c56.512 0 109.632 21.952 149.568 61.888a211.712 211.712 0 0 1 0 299.072z"/></svg>
          </button>
          <button class="library-expand-toggle ${isExpanded ? "expanded" : ""}" data-id="${escapeHtml(library.id)}" title="${escapeHtml(expandTitle)}" aria-label="${escapeHtml(expandTitle)}" aria-expanded="${isExpanded ? "true" : "false"}">
            ${LIBRARY_EXPAND_ICON}
          </button>
        </article>`;
      })
      .join("") || `<div class="panel">${escapeHtml(t("noLibraries"))}</div>`;
  bindLibraryCardActions();
  refreshLibraryContext();
}

async function loadLibraries(render = true) {
  try {
    const data = await api("/libraries");
    state.libraries = data.items || [];
    await ensureLibrarySelection();
    if (!render) {
      refreshLibraryContext();
      return;
    }
    renderLibraryCards();
  } catch (error) {
    toast(error.message, true);
  }
}

function bindLibraryCardActions() {
  const libraryCards = $("library-cards");
  libraryCards.onclick = (event) => {
    if (event.target.closest("button,a,input,select,textarea,label")) return;
    const card = event.target.closest(".library-card");
    if (!card) return;
    selectLibrary(card.dataset.id);
  };
  libraryCards.onkeydown = (event) => {
    if (event.target.closest("button,a,input,select,textarea,label")) return;
    const card = event.target.closest(".library-card");
    if (!card) return;
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    selectLibrary(card.dataset.id);
  };
  document.querySelectorAll(".enter-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      selectLibrary(button.dataset.id);
      navigate("graph");
    };
  });
  document.querySelectorAll(".edit-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      await loadProviders(false);
      openLibraryModal(
        state.libraries.find((item) => item.id === button.dataset.id),
      );
    };
  });
  document.querySelectorAll(".copy-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const tempJobId = addOptimisticTask({
        kind: "library_copy",
        libraryId: button.dataset.id,
      });
      try {
        const result = await api(`/libraries/${encodeURIComponent(button.dataset.id)}/copy`, { method: "POST" });
        toast(t("libraryCopyQueued", { id: button.dataset.id }));
        trackQueuedJob(result, {
          kind: "library_copy",
          libraryId: button.dataset.id,
          tempId: tempJobId,
        });
      } catch (error) {
        removeOptimisticTask(tempJobId);
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".import-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      openImportModal(button.dataset.id);
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
        await startLibraryIndexRebuild(
          button.dataset.id,
          button.dataset.provider || "",
          "library_card_manual_rebuild",
        );
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".backup-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      try {
        const result = await api(`/libraries/${encodeURIComponent(button.dataset.id)}/backup`, { method: "POST" });
        toast(`备份完成：${result.path}`);
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".key-library").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      openLibraryPskModal(button.dataset.id);
    };
  });
  document.querySelectorAll(".library-expand-toggle").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      const libraryId = button.dataset.id;
      if (!libraryId) return;
      const expanded = !state.expandedLibraryIds.has(libraryId);
      if (expanded) {
        state.expandedLibraryIds.add(libraryId);
      } else {
        state.expandedLibraryIds.delete(libraryId);
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
        await api(`/libraries/${encodeURIComponent(button.dataset.id)}/set-default`, { method: "POST" });
        await loadLibraries();
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
        await api(`/libraries/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
        await loadLibraries();
      } catch (error) {
        toast(error.message, true);
      }
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

function openLibraryPskModal(libraryId) {
  const library = state.libraries.find((item) => item.id === libraryId);
  $("library-psk-id").value = libraryId || "";
  $("library-psk-title").textContent = t("libraryPskTitle", { library: library?.name || libraryId });
  $("library-psk-password").value = "";
  $("library-psk-libid").textContent = libraryId || "";
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
  const libraryId = $("library-psk-id").value;
  const password = $("library-psk-password").value;
  try {
    const result = await api(`/libraries/${encodeURIComponent(libraryId)}/psk`, {
      method: "POST",
      body: JSON.stringify({ password }),
    });
    $("library-psk-value").textContent = result.psk || "";
    $("library-psk-result").classList.remove("hidden");
    $("library-psk-hint").classList.add("hidden");
    $("library-psk-password-row").classList.add("hidden");
    $("library-psk-submit").classList.add("hidden");
    $("library-psk-cancel").textContent = t("close");
    toast(t("libraryPskReady"));
  } catch (error) {
    toast(error.message, true);
  }
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

$("library-create").onclick = async () => {
  await loadProviders(false);
  openLibraryModal();
};

function openLibraryModal(library = null) {
  const idLocked = Boolean(library?.is_default);
  $("library-original-id").value = library?.id || "";
  $("library-id").value = library?.id || "";
  $("library-id").readOnly = idLocked;
  $("library-id").classList.toggle("readonly-lock", idLocked);
  $("library-id").setAttribute("aria-readonly", idLocked ? "true" : "false");
  $("library-id-readonly-note").classList.toggle("hidden", !idLocked);
  $("library-name").value = library?.name || "";
  $("library-description").value = library?.description || "";
  $("library-persona").value = library?.default_persona_id || "";
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
    changedFields.push(t("libraryId"));
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
  const payload = {
    id: $("library-id").value.trim(),
    name: $("library-name").value.trim(),
    description: $("library-description").value.trim(),
    default_persona_id: $("library-persona").value.trim(),
    provider_id: $("library-provider").value,
    rerank_provider_id: $("library-rerank-provider").value,
  };
  try {
    if (originalId) {
      const originalLibrary = state.libraries.find((item) => item.id === originalId);
      if (originalLibrary?.is_default) {
        payload.id = originalId;
      }
      const wasSelected = state.selectedLibraryId === originalId;
      const providerChanged = Boolean(
        originalLibrary
        && payload.provider_id
        && payload.provider_id !== originalLibrary.provider_id,
      );
      if (!(await confirmSensitiveLibraryEdit(originalLibrary, payload))) {
        return;
      }
      const updatedLibrary = await api(`/libraries/${encodeURIComponent(originalId)}`, {
        method: "PATCH",
        body: JSON.stringify({
          id: payload.id,
          name: payload.name,
          description: payload.description,
          default_persona_id: payload.default_persona_id,
          rerank_provider_id: payload.rerank_provider_id || "",
        }),
      });
      const savedLibraryId = updatedLibrary.id || payload.id || originalId;
      if (wasSelected && savedLibraryId !== originalId) {
        selectLibrary(savedLibraryId, { resetMemoryPage: false });
      }
      await loadProviders(false);
      const selectedProvider = latestProvider(payload.provider_id);
      const rebuildReason = providerChanged
        ? "library_edit_provider_switch"
        : libraryEditNeedsRebuild(updatedLibrary, selectedProvider);
      if (rebuildReason) {
        closeOverlay("library-modal");
        markLibraryIndexConflict(savedLibraryId);
        await startLibraryIndexRebuild(savedLibraryId, payload.provider_id, rebuildReason, {
          markLibraryConflict: true,
        });
        toast(t("providerSwitchQueued", { library: savedLibraryId }));
      }
    } else {
      await api("/libraries", { method: "POST", body: JSON.stringify(payload) });
    }
    closeOverlay("library-modal");
    toast("记忆库已保存");
    await loadLibraries();
  } catch (error) {
    toast(error.message, true);
  }
};

function openImportModal(libraryId) {
  $("import-library-id").value = libraryId;
  $("import-file").value = "";
  $("import-file-name").textContent = "";
  $("import-modal-title").textContent = `${t("importMemory")}：${libraryId}`;
  $("import-modal").classList.remove("hidden");
}

function selectedImportFile() {
  return $("import-file").files?.[0] || null;
}

$("import-file")?.addEventListener("change", () => {
  const file = selectedImportFile();
  $("import-file-name").textContent = file ? file.name : "";
});

const importDropZone = $("import-drop-zone");
if (importDropZone) {
  ["dragenter", "dragover"].forEach((name) => {
    importDropZone.addEventListener(name, (event) => {
      event.preventDefault();
      importDropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    importDropZone.addEventListener(name, (event) => {
      event.preventDefault();
      importDropZone.classList.remove("drag-over");
    });
  });
  importDropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    $("import-file").files = transfer.files;
    $("import-file-name").textContent = file.name;
  });
}

$("import-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const libraryId = $("import-library-id").value;
  const file = selectedImportFile();
  if (!file) {
    toast(t("chooseImportFile"), true);
    return;
  }
  const form = new FormData();
  form.append("file", file, file.name);
  const tempJobId = addOptimisticTask({
    kind: "livingmemory_import",
    libraryId,
    markLibraryConflict: true,
  });
  try {
    const result = await api(`/libraries/${encodeURIComponent(libraryId)}/imports/livingmemory-db`, {
      method: "POST",
      body: form,
    });
    closeOverlay("import-modal");
    toast(t("importQueued", { library: libraryId }));
    trackQueuedJob(result, {
      kind: "livingmemory_import",
      libraryId,
      tempId: tempJobId,
    });
  } catch (error) {
    removeOptimisticTask(tempJobId);
    toast(error.message, true);
  }
});

const PROVIDER_ICON_SVG = {
  openai_embedding: `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M565.265 954.52c-22.29 0-48.4-8.153-67.952-14.84a103.425 103.425 0 0 1-26.876-11.272c-12.737-7.77-15.411-8.343-19.806-24.774 21.972-5.158 81.581-41.905 103.871-55.342 148.897-89.16 119.029-10.445 119.029-364.982 15.03 3.566 82.79 32.416 82.79 57.317 0 133.103 20.571 273.848-52.603 354.92-22.8 25.474-91.835 58.972-138.39 58.972zM132.204 655.196c258.627 136.86 184.37 157.049 357.721 52.095 44.58-27.194 90.434-49.293 132.593-77.57V731.62C526.99 753.846 350.2 958.659 203.468 832.307L177.61 807.15c-37.32-43.943-45.344-72.41-45.344-151.89z m375.744-19.106c-19.933-13.31-79.48-51.33-101.897-57.317v-133.74a1158.312 1158.312 0 0 0 101.897-57.316c43.943 10.19 70.691 47.064 114.634 57.317v127.37c-17.832 12.101-95.847 58.719-114.634 63.75zM81.255 457.772c0-63.686-4.267-90.306 38.848-145.776 23.946-30.888 47.51-39.613 82.091-57.954v261.11c44.134 23.373 83.874 49.039 129.345 74.449l131.766 78.397c-59.546 15.921-63.686 61.33-109.603 33.753-104.7-62.73-272.383-129.345-272.383-243.915z m866.123 127.371c0 79.543-47.573 161.188-121.002 178.32V597.88c0-82.791 9.744-84.574-48.91-116.608l-212.2-118.9c15.793-23.628 22.608-19.107 48.145-34.837 41.714-25.474 39.04-16.112 117.054 28.786 94.191 54.196 216.85 100.56 216.85 228.886zM406.051 387.718V292.19c43.752-23.182 90.689-50.949 133.358-76.805 82.154-49.547 95.528-63.304 185.006-63.304 48.465 0 102.534 36.747 125.652 65.406 42.223 52.222 39.93 92.662 39.93 151.125C858.92 352.118 697.605 247.61 667.099 247.61s-229.65 123.422-261.11 140.108z m-50.948 159.214c-16.367-10.954-63.112-39.995-82.791-44.58 0-168.321-33.88-314.607 67.952-390.52 56.043-41.714 113.17-53.814 181.377-30.696 25.474 8.661 35.536 20.889 56.361 26.43-11.782 16.048-80.69 50.31-102.279 63.303-154.564 93.235-120.62 7.45-120.62 376.063zM62.15 667.934c0 169.149 115.143 280.853 274.293 273.848 59.8-2.675 26.812-7.706 69.417 25.474 97.821 76.741 228.822 73.748 319.638 1.02a251.94 251.94 0 0 0 52.604-55.662c58.209-85.275-10.954-45.599 81.963-83.62 130.237-53.24 199.4-217.358 128.645-355.428-27.448-53.56-40.249-28.85-28.276-104.699 18.723-118.582-63.176-230.032-157.622-269.772-98.903-41.587-129.09 12.737-178.892-37.574a161.889 161.889 0 0 0-43.816-32.607c-106.1-56.68-248.82-27.385-321.676 64.768-81.326 102.98 9.49 54.706-92.407 98.649C15.15 257.354-33.251 439.176 41.579 561.07c56.808 92.599 20.57 4.967 20.57 106.8z"/></svg>`,
  ollama_embedding: `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M287.922 79.6c-42.4 25.6-68.8 118-59.2 205.6l3.6 34.8-20 20c-64.8 64-81.2 163.2-40.4 243.2l10.8 21.2-10 24.8c-26 65.2-24 144 4.4 200.8l11.6 22.8-7.2 14.8c-16.4 33.6-24.4 107.6-14.8 141.2l4 15.2h60.4l-3.2-13.2c-2-6.8-3.6-30.8-3.6-52.8 0-38 0.8-41.6 15.2-71.6 8-17.2 14.8-34.4 14.8-38 0-3.6-5.6-15.6-12.8-26.8-35.6-55.6-36.4-123.6-2.8-190.4 14.8-29.6 14.4-38.8-2-58.8-22.4-26.4-33.2-69.2-26.4-104.4 9.6-50.4 40.4-92.8 82-112.4 20-9.6 29.2-11.6 54.8-11.6h31.2l8-16c18.4-36.4 50.4-61.6 93.6-74 55.6-16.4 124.4 14.8 154.8 70.4l9.6 17.6 34 2c58 4 93.2 27.2 118.4 78 25.6 52 22.4 107.2-8.4 147.6-17.2 22.4-17.2 32-2.4 61.6 33.6 66.8 32.8 134.8-2.8 190.4-7.2 11.2-12.8 23.2-12.8 26.8 0 3.6 6.8 20.8 14.8 38 14.4 30 15.2 33.6 15.2 71.6 0 22-1.6 46-3.6 52.8l-3.2 13.2h60.4l4-14.8c9.6-34 1.6-108-14.8-141.6l-7.2-14.8 11.6-22.8c28.4-56.8 30.4-135.6 4-201.6l-10-25.2 10.8-22c24-49.6 27.2-108.8 8-164.8-10.8-32-25.2-54.8-50.8-79.2l-17.2-17.2 3.6-34.8c7.2-66-8.8-148.4-35.6-183.6-21.2-28-49.6-36.4-79.2-24-27.2 11.2-50.8 51.6-62 106.4-5.2 26-8 31.6-12.8 29.6-34-15.6-60.4-21.6-94.8-21.6-35.2 0-53.2 4.4-93.2 21.6-4.8 2-7.6-3.6-12.8-29.6-11.2-54.8-34.8-95.2-62-106.4-18.4-8-41.2-6.8-55.6 2z m45.6 73.6c16 32.8 27.2 110.8 17.6 125.2-1.6 2.4-13.6 5.6-26.8 7.2-13.2 1.6-27.2 3.6-30.8 4.8-6.4 2-7.2-1.6-7.2-37.2 0-21.6 2.8-50.4 6-64.4 7.2-30 20.8-58 27.2-55.6 2.4 0.8 8.8 10 14 20z m384.8-5.2c12.8 24.8 20 62.8 20 105.6 0 30.8-1.2 38.8-5.2 37.2-3.2-1.2-17.6-3.2-32-4.8-32-3.6-33.2-5.6-29.6-54 3.6-43.2 23.2-100 35.2-100 2 0 7.2 7.2 11.6 16z"/><path d="M453.522 479.6c-44.8 18-75.2 47.6-85.6 83.6-15.6 54 8.4 101.2 64.4 127.6 22.8 10.4 27.2 11.2 80 11.2s57.2-0.8 80-11.2c41.2-19.2 64-48 68.8-86.4 5.2-46.4-24-92.4-74-117.2-24.4-12.4-30-13.2-70.8-14-35.2-0.8-47.6 0.4-62.8 6.4z m102 38.8c10 3.2 26.4 13.2 36.4 22 35.2 31.2 38 68.4 6.8 96.8-21.6 19.6-40 24.8-86.4 24.8-46.4 0-64.8-5.2-86.4-24.8-31.2-28.4-28.4-65.6 6.8-96.8 10-8.8 25.6-18.4 34.8-22 22.4-7.6 65.2-8 88 0z"/><path d="M480.722 558.8c-5.6 5.6-2 25.2 5.2 30 4.8 3.6 8.4 11.2 9.2 21.2 1.2 15.6 1.6 16 17.2 16s16-0.4 16.4-15.2c0-10 3.2-17.6 8.8-22.8 9.6-9.2 11.6-20.8 4-26.8-5.6-4.4-56.8-6-60.8-2.4z m-168.8-78.4c-18.4 9.2-28 44.8-16.4 60 7.2 9.2 20.8 15.6 34 15.6 17.2 0 36.8-23.2 36.8-43.2 0-8-2.4-17.6-5.2-21.2-11.2-14.4-32-19.2-49.2-11.2z m364.4 0.4c-12.8 7.2-17.6 16-18 32 0 20 19.6 43.2 36.8 43.2 23.2 0 38.8-14 39.2-35.2 0-32.8-32-54.8-58-40z"/></svg>`,
  vllm_embedding: `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M412.16 922.688L64 249.664h348.16v673.024z"/><path d="M667.392 922.688H412.096L586.24 226.432 899.392 64 667.328 922.688z"/></svg>`,
};
PROVIDER_ICON_SVG.vllm_rerank = PROVIDER_ICON_SVG.vllm_embedding;
PROVIDER_ICON_SVG.xinference_rerank = `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M384.128 521.984c22.08 24.416 47.072 45.984 74.496 64.224 23.52 15.744 48.608 28.96 74.88 39.52a566.304 566.304 0 0 0 116.704-156.576L822.56 128l-300.544 236.256a567.104 567.104 0 0 0-137.92 157.76zM359.52 734.432a828.416 828.416 0 0 1-61.216-45.184L194.048 896l187.392-147.328c-7.36-4.704-14.72-9.376-21.92-14.24z"/><path d="M746.112 402.4c56.48 74.88 72.96 158.08 34.688 215.36-55.904 83.648-207.488 80.416-338.56-7.168s-192-226.368-136.096-310.016c38.304-57.28 121.472-73.824 212.288-50.368-157.056-66.688-306.464-60.096-364.416 26.432-72.736 108.864 26.592 303.136 221.824 433.408 195.2 130.272 412.512 147.936 485.248 39.136 57.888-86.688 6.656-227.232-114.976-346.784z"/></svg>`;
PROVIDER_ICON_SVG.bailian_rerank = `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M122.282667 287.018667v449.962666l194.858666-112.469333V399.488z"/><path d="M317.141333 399.488v225.024L512 512z"/><path d="M512 512L317.141333 399.488l389.632-225.109333 194.944 112.64z"/><path d="M317.141333 399.488L122.282667 287.018667 512 61.994667l194.773333 112.384z"/><path d="M901.717333 736.981333L512 962.005333 317.141333 849.493333l389.717334-224.981333z"/><path d="M706.858667 624.512L901.717333 512v224.981333l-194.858666-112.469333zM317.141333 849.493333l-194.858666-112.512L512 512l194.858667 112.512-389.717334 224.981333z"/></svg>`;
PROVIDER_ICON_SVG.nvidia_rerank = `<svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M381.781333 375.381333v-61.013333a285.866667 285.866667 0 0 1 18.090667-0.768c167.338667-5.290667 277.034667 143.957333 277.034667 143.957333s-118.357333 164.309333-245.333334 164.309334a156.586667 156.586667 0 0 1-49.408-7.893334v-185.429333c65.194667 7.893333 78.378667 36.565333 117.205334 101.76l87.04-73.130667s-63.658667-83.285333-170.666667-83.285333a256.682667 256.682667 0 0 0-33.962667 1.493333m0-202.026666v91.221333l18.090667-1.152c232.533333-7.893333 384.426667 190.72 384.426667 190.72s-174.08 211.797333-355.413334 211.797333a275.626667 275.626667 0 0 1-46.72-4.138666v56.533333c12.8 1.493333 26.026667 2.645333 38.826667 2.645333 168.832 0 290.986667-86.314667 409.301333-188.074666 19.584 15.829333 99.84 53.888 116.48 70.485333-112.341333 94.208-374.272 169.984-522.794666 169.984-14.293333 0-27.861333-0.768-41.429334-2.261333v79.530666H1024V173.354667z m0 440.576v48.256c-156.032-27.904-199.381333-190.293333-199.381333-190.293334s75.008-82.944 199.381333-96.512v52.778667H381.44c-65.194667-7.936-116.48 53.12-116.48 53.12s29.013333 102.912 116.864 132.693333M104.789333 465.066667s92.330667-136.405333 277.333334-150.741334V264.576C177.194667 281.173333 0 454.528 0 454.528s100.266667 290.218667 381.781333 316.586667v-52.778667c-206.506667-25.6-276.992-253.269333-276.992-253.269333z"/></svg>`;

function providerKindOf(itemOrType = "") {
  if (typeof itemOrType === "object" && itemOrType !== null) {
    return itemOrType.provider_kind || providerKindOf(itemOrType.type || itemOrType.id || "");
  }
  const type = String(itemOrType || "");
  return type.includes("_rerank") ? "rerank" : "embedding";
}

function isRerankProvider(itemOrType = "") {
  return providerKindOf(itemOrType) === "rerank";
}

function providerIconSvg(type) {
  return PROVIDER_ICON_SVG[type] || PROVIDER_ICON_SVG.openai_embedding;
}

function providerTypeName(type) {
  return {
    openai_embedding: "OpenAI Embedding",
    ollama_embedding: "Ollama Embedding",
    vllm_embedding: "vLLM Embedding",
    vllm_rerank: "vLLM Rerank",
    xinference_rerank: "Xinference Rerank",
    bailian_rerank: "阿里云百炼重排序",
    nvidia_rerank: "NVIDIA Rerank",
  }[type] || type;
}

function providerTypeDescription(type) {
  return {
    openai_embedding: "连接 OpenAI 官方或兼容 Embedding 接口。",
    ollama_embedding: "连接 Ollama /api/embed 接口。",
    vllm_embedding: "适配 vLLM OpenAI-compatible Embedding，自动对齐 served-model-name。",
    vllm_rerank: "适配 vLLM / OpenAI-compatible Rerank 接口。",
    xinference_rerank: "适配 Xinference Rerank REST 接口。",
    bailian_rerank: "适配阿里云百炼 qwen3-rerank 与旧 DashScope 重排序接口。",
    nvidia_rerank: "适配 NVIDIA NIM Rerank 的 query/passages/rankings 格式。",
  }[type] || "自定义模型提供商。";
}

async function loadProviders(render = true) {
  try {
    const [providers, types] = await Promise.all([
      api("/providers"),
      state.providerTypes.length ? Promise.resolve({ items: state.providerTypes }) : api("/provider-types"),
    ]);
    state.providers = providers.items || [];
    state.providerTypes = types.items || [];
    if (!render) return;
    const activeKind = state.providerKind === "rerank" ? "rerank" : "embedding";
    document.querySelectorAll("[data-provider-kind]").forEach((button) => {
      const isActive = button.dataset.providerKind === activeKind;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    const visibleProviders = state.providers.filter(
      (provider) => providerKindOf(provider) === activeKind,
    );
    $("provider-cards").innerHTML =
      visibleProviders
        .map((provider) => {
          const kind = providerKindOf(provider);
          const status = state.providerStatuses[provider.id];
          const pendingLibraries = (provider.used_by || []).filter(
            (item) => item.usage_kind !== "rerank"
              && (
                item.needs_rebuild ?? (
                  Number(item.provider_revision) !== Number(provider.revision)
                )
              ),
          );
          const statusClass = pendingLibraries.length
            ? "pending"
            : status?.available
              ? "available"
              : status
                ? "unavailable"
                : "";
          const statusText = pendingLibraries.length
            ? t("pendingRebuild", {
                libraries: pendingLibraries.map((item) => t("stillUsesRevision", {
                  library: item.library_name,
                  revision: item.provider_revision,
                })).join("、"),
              })
            : status?.available
              ? t("availableStatus", { elapsed: status.elapsed_ms })
              : status
                ? t("unavailableStatus", { error: status.error || "" })
                : t("notTested");
          const libraryOrder = new Map(
            state.libraries.map((item, index) => [item.id, index]),
          );
          const usedBy = [...(provider.used_by || [])].sort((left, right) => {
            const leftIndex = libraryOrder.get(left.library_id);
            const rightIndex = libraryOrder.get(right.library_id);
            if (leftIndex != null || rightIndex != null) {
              return (leftIndex ?? Number.MAX_SAFE_INTEGER) - (rightIndex ?? Number.MAX_SAFE_INTEGER);
            }
            return String(left.library_name || left.library_id || "").localeCompare(
              String(right.library_name || right.library_id || ""),
              "zh-Hans-CN",
            );
          });
          const usedLibButton = (item, extraClass = "") => {
            const usageLabel = item.usage_kind === "rerank" ? " · Rerank" : "";
            const usageClass = item.usage_kind === "rerank" ? "used-lib-rerank" : "used-lib-embedding";
            return `<button type="button" class="${["used-lib-jump", usageClass, extraClass].filter(Boolean).join(" ")}" data-library-id="${escapeHtml(item.library_id)}">${escapeHtml(item.library_name)}${escapeHtml(usageLabel)}</button>`;
          };
          const usedLibrariesHtml = usedBy.length === 0
            ? escapeHtml(t("none"))
            : usedBy.length === 1
              ? usedLibButton(usedBy[0], "used-lib-plain")
              : `<div class="used-lib-row">${usedLibButton(usedBy[0], "used-lib-first-btn")}<details class="used-lib-dd"><summary><span class="used-lib-count">＋${usedBy.length - 1}</span><span class="used-lib-caret" aria-hidden="true">▾</span></summary><ul class="${["used-lib-list", kind === "rerank" ? "used-lib-list-rerank" : "used-lib-list-embedding"].join(" ")}">${usedBy.slice(1).map((item) => `<li>${usedLibButton(item)}</li>`).join("")}</ul></details></div>`;
          const maxContextText = provider.max_context_tokens_source?.startsWith("auto:")
            && provider.max_context_tokens
            ? `${t("autoDetect")} (${provider.max_context_tokens})`
            : (provider.max_context_tokens || t("autoDetect"));
          const providerMetaRows = [
            `<dt>${escapeHtml(t("typeLabel"))}</dt><dd>${escapeHtml(providerTypeName(provider.type))}</dd>`,
            `<dt>${escapeHtml(t("modelLabel"))}</dt><dd>${escapeHtml(provider.model)}</dd>`,
            kind === "embedding"
              ? `<dt>${escapeHtml(t("dimensionLabel"))}</dt><dd>${provider.dimensions || escapeHtml(t("autoDetect"))}</dd>`
              : "",
            kind === "embedding"
              ? `<dt>${escapeHtml(t("maxContextTokens"))}</dt><dd>${escapeHtml(String(maxContextText))}</dd>`
              : "",
            `<dt>${escapeHtml(t("endpointLabel"))}</dt><dd>${escapeHtml(provider.api_base)}</dd>`,
            kind === "rerank" && provider.api_suffix
              ? `<dt>API Suffix</dt><dd>${escapeHtml(provider.api_suffix)}</dd>`
              : "",
            kind === "rerank" && provider.model_endpoint
              ? `<dt>Model Endpoint</dt><dd>${escapeHtml(provider.model_endpoint)}</dd>`
              : "",
            `<dt>${escapeHtml(t("usedLibraries"))}</dt><dd>${usedLibrariesHtml}</dd>`,
          ].filter(Boolean).join("");
          return `<article class="management-card">
            <header>
              <div class="provider-card-head">
                <span class="provider-logo">${providerIconSvg(provider.type)}</span>
                <div><h3>${escapeHtml(provider.display_name)}</h3><span class="provider-id-line"><span class="subtle">${escapeHtml(provider.id)}</span><button type="button" class="id-copy-btn copy-provider-id" data-id="${escapeHtml(provider.id)}" title="${escapeHtml(t("copy"))} ID" aria-label="${escapeHtml(t("copy"))} ID"><svg viewBox="0 0 1024 1024" aria-hidden="true"><path d="M704 384v512H192V384h512m32-64h-576a32 32 0 0 0-32 32v576a32 32 0 0 0 32 32h576a32 32 0 0 0 32-32v-576a32 32 0 0 0-32-32z"/><path d="M320 512m32 0l192 0q32 0 32 32l0 0q0 32-32 32l-192 0q-32 0-32-32l0 0q0-32 32-32Z"/><path d="M320 704m32 0l192 0q32 0 32 32l0 0q0 32-32 32l-192 0q-32 0-32-32l0 0q0-32 32-32Z"/><path d="M928 128h-576a32 32 0 0 0-32 32V256h64V192h512v576h-64v64h96a32 32 0 0 0 32-32v-640a32 32 0 0 0-32-32z"/></svg></button></span></div>
              </div>
              <label class="switch" title="启用或停用"><input class="provider-toggle" data-id="${escapeHtml(provider.id)}" type="checkbox" ${provider.enabled ? "checked" : ""}><i></i></label>
            </header>
            <span class="status-dot ${statusClass}">${escapeHtml(statusText)}</span>
            <dl class="provider-meta">
              ${providerMetaRows}
            </dl>
            <div class="card-actions">
              <button class="ghost edit-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("edit"))}</button>
              <button class="ghost copy-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("copy"))}</button>
              <button class="ghost danger delete-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("delete"))}</button>
              <button class="primary test-provider-card" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("test"))}</button>
            </div>
          </article>`;
        })
        .join("") || `<div class="panel">${escapeHtml(t("noProviders"))}</div>`;
    bindProviderCardActions();
  } catch (error) {
    toast(error.message, true);
  }
}

function bindProviderCardActions() {
  document.querySelectorAll(".used-lib-dd").forEach((details) => {
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
  document.querySelectorAll(".provider-toggle").forEach((input) => {
    input.onchange = async () => {
      try {
        await api(`/providers/${encodeURIComponent(input.dataset.id)}`, {
          method: "PATCH",
          body: JSON.stringify({ enabled: input.checked }),
        });
      } catch (error) {
        input.checked = !input.checked;
        toast(error.message, true);
      }
      await loadProviders();
    };
  });
  document.querySelectorAll(".copy-provider-id").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      const id = button.dataset.id || "";
      if (!id) return;
      try {
        await navigator.clipboard.writeText(id);
        toast(t("providerIdCopied"));
      } catch {
        toast(id, false);
      }
    };
  });
  document.querySelectorAll(".used-lib-jump").forEach((button) => {
    button.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      const libraryId = button.dataset.libraryId || "";
      if (!libraryId) return;
      selectLibrary(libraryId);
      navigate("libraries");
    };
  });
  document.querySelectorAll(".edit-provider").forEach((button) => {
    button.onclick = () => openProviderEditor(
      state.providers.find((item) => item.id === button.dataset.id),
    );
  });
  document.querySelectorAll(".copy-provider").forEach((button) => {
    button.onclick = async () => {
      try {
        await api(`/providers/${encodeURIComponent(button.dataset.id)}/copy`, {
          method: "POST",
          body: "{}",
        });
        toast("Provider 已复制，新副本默认停用");
        await loadProviders();
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".delete-provider").forEach((button) => {
    button.onclick = async () => {
      if (!(await confirmDialog({
        title: t("confirmTitle"),
        message: t("confirmDeleteProvider", { id: button.dataset.id }),
        confirmText: t("delete"),
        danger: true,
      }))) return;
      try {
        await api(`/providers/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
        await loadProviders();
      } catch (error) {
        toast(error.message, true);
      }
    };
  });
  document.querySelectorAll(".test-provider-card").forEach((button) => {
    button.onclick = async () => {
      button.disabled = true;
      button.textContent = "测试中…";
      try {
        const result = await api(`/providers/${encodeURIComponent(button.dataset.id)}/test`, { method: "POST" });
        state.providerStatuses[button.dataset.id] = result;
        toast(result.available ? `测试成功，延迟 ${result.elapsed_ms} ms` : result.error, !result.available);
      } catch (error) {
        toast(error.message, true);
      }
      await loadProviders();
    };
  });
}

$("provider-create").onclick = async () => {
  await loadProviders(false);
  const types = state.providerTypes.filter(
    (type) => providerKindOf(type) === state.providerKind,
  );
  $("provider-type-cards").innerHTML = types
    .map((type) => `<button type="button" class="provider-type-card" data-type="${escapeHtml(type.id)}">
      <span class="provider-type-icon">${providerIconSvg(type.id)}</span><b>${escapeHtml(providerTypeName(type.id))}</b>
      <p>${escapeHtml(providerTypeDescription(type.id))}</p>
    </button>`)
    .join("") || `<div class="panel">${escapeHtml(t("noProviders"))}</div>`;
  document.querySelectorAll(".provider-type-card").forEach((button) => {
    button.onclick = () => {
      closeOverlay("provider-type-modal");
      const template = state.providerTypes.find((item) => item.id === button.dataset.type);
      openProviderEditor({ ...template, id: template.id, enabled: false }, true);
    };
  });
  $("provider-type-modal").classList.remove("hidden");
};

function setProviderRowVisible(id, visible) {
  const row = $(id);
  if (row) {
    row.classList.toggle("hidden", !visible);
  }
}

function applyProviderFormKind(provider) {
  const kind = providerKindOf(provider);
  const type = provider.type || "";
  const rerank = kind === "rerank";
  $("provider-model-label").textContent = rerank ? t("rerankModel") : t("embeddingModel");
  setProviderRowVisible("provider-dimensions-row", !rerank);
  setProviderRowVisible("provider-context-row", !rerank);
  setProviderRowVisible("provider-batch-row", !rerank);
  setProviderRowVisible("provider-concurrency-row", !rerank);
  setProviderRowVisible("provider-api-suffix-row", rerank && ["vllm_rerank", "xinference_rerank"].includes(type));
  setProviderRowVisible("provider-return-documents-row", type === "bailian_rerank");
  setProviderRowVisible("provider-instruct-row", type === "bailian_rerank");
  setProviderRowVisible("provider-model-endpoint-row", type === "nvidia_rerank");
  setProviderRowVisible("provider-truncate-row", type === "nvidia_rerank");
  setProviderRowVisible("provider-launch-model-row", type === "xinference_rerank");
}

function applyProviderContextLock(provider, options = {}) {
  const source = String(provider.max_context_tokens_source || "");
  const autoDetected = source.startsWith("auto:");
  const input = $("provider-max-context");
  const help = $("provider-context-help");
  input.value = provider.max_context_tokens ?? 0;
  input.dataset.originalValue = String(provider.max_context_tokens ?? 0);
  input.dataset.originalSource = source;
  input.dataset.manualUnlocked = "false";
  if (options.syncDraftOrigin) {
    $("provider-api-base").dataset.originalValue = $("provider-api-base").value;
    $("provider-model").dataset.originalValue = $("provider-model").value;
  }
  $("provider-max-context-source").value = source;
  input.readOnly = autoDetected;
  input.classList.toggle("readonly-lock", autoDetected);
  input.setAttribute("aria-readonly", autoDetected ? "true" : "false");
  if (help) {
    help.textContent = autoDetected
      ? `${t("maxContextAutoHelp")} ${source}`
      : t("maxContextManualHelp");
  }
}

function refreshProviderContextDraftLock() {
  const input = $("provider-max-context");
  const originalSource = String(input?.dataset.originalSource || "");
  if (!originalSource.startsWith("auto:")) return;
  const changed = $("provider-api-base").value !== ($("provider-api-base").dataset.originalValue || "")
    || $("provider-model").value !== ($("provider-model").dataset.originalValue || "");
  input.readOnly = !changed;
  input.classList.toggle("readonly-lock", !changed);
  input.setAttribute("aria-readonly", changed ? "false" : "true");
  if (changed) {
    if (input.dataset.manualUnlocked !== "true") {
      input.value = "0";
      input.dataset.manualUnlocked = "true";
    }
    $("provider-max-context-source").value = "";
    $("provider-context-help").textContent = t("maxContextManualHelp");
  } else {
    input.dataset.manualUnlocked = "false";
    input.value = input.dataset.originalValue || "0";
    $("provider-max-context-source").value = originalSource;
    $("provider-context-help").textContent = `${t("maxContextAutoHelp")} ${originalSource}`;
  }
}

function providerHintText(type) {
  return {
    vllm_embedding: "vLLM Provider 会自动忽略 dimensions，并尝试将模型名对齐到 served-model-name。",
    ollama_embedding: "Ollama Provider 使用 /api/tags 获取模型，并通过 /api/embed 生成向量。",
    openai_embedding: "OpenAI Provider 支持官方及兼容接口；维度大于 0 时会发送 dimensions。",
    vllm_rerank: "vLLM Rerank 会向 API Base + API Suffix 发送 query、documents、model 与 top_n。",
    xinference_rerank: "Xinference Rerank 使用 REST 调用，不需要额外 xinference_client 依赖。",
    bailian_rerank: "百炼 Rerank 支持 qwen3-rerank 与旧 DashScope payload 分支。",
    nvidia_rerank: "NVIDIA Rerank 会使用 query/passages/rankings 格式，并按模型 endpoint 拼接地址。",
  }[type] || providerTypeDescription(type);
}

function openProviderEditor(provider, creating = false) {
  const kind = providerKindOf(provider);
  const blankEmbeddingDraft = creating && kind !== "rerank";
  const apiBaseValue = blankEmbeddingDraft ? "" : (provider.api_base || "");
  const modelValue = blankEmbeddingDraft ? "" : (provider.model || "");
  const dimensionsValue = blankEmbeddingDraft ? "" : (provider.dimensions ?? 0);
  const idLocked = !creating && Boolean((provider.used_by || []).length);
  $("provider-original-id").value = creating ? "" : provider.id;
  $("provider-type").value = provider.type;
  $("provider-id").value = creating ? provider.id : provider.id;
  $("provider-id").readOnly = idLocked;
  $("provider-id").classList.toggle("readonly-lock", idLocked);
  $("provider-id").setAttribute("aria-readonly", idLocked ? "true" : "false");
  $("provider-id-readonly-note").classList.toggle("hidden", !idLocked);
  $("provider-name").value = provider.display_name || providerTypeName(provider.type);
  $("provider-enabled").checked = Boolean(provider.enabled);
  $("provider-api-key").value = "";
  $("provider-api-base").value = apiBaseValue;
  $("provider-api-suffix").value = provider.api_suffix || "";
  $("provider-model").value = modelValue;
  $("provider-api-base").dataset.originalValue = apiBaseValue;
  $("provider-model").dataset.originalValue = modelValue;
  $("provider-dimensions").value = dimensionsValue;
  applyProviderContextLock(provider);
  $("provider-return-documents").checked = Boolean(provider.return_documents);
  $("provider-instruct").value = provider.instruct || "";
  $("provider-model-endpoint").value = provider.model_endpoint || "";
  $("provider-truncate").value = provider.truncate || "";
  $("provider-launch-model").checked = Boolean(provider.launch_model_if_not_running);
  $("provider-timeout").value = provider.timeout_seconds || 30;
  $("provider-proxy").value = provider.proxy || "";
  $("provider-batch").value = provider.batch_size || 64;
  $("provider-concurrency").value = provider.concurrency || 2;
  $("provider-retries").value = provider.max_retries || 5;
  $("provider-clear-key").checked = false;
  $("provider-api-key-row").classList.toggle("hidden", provider.type === "ollama_embedding");
  $("provider-clear-key-row").classList.toggle("hidden", creating || !provider.has_api_key || provider.type === "ollama_embedding");
  applyProviderFormKind(provider);
  $("provider-hint").textContent = providerHintText(provider.type);
  $("provider-modal-title").textContent = creating ? `新增 ${providerTypeName(provider.type)}` : `编辑 ${provider.display_name}`;
  $("provider-test-result").classList.add("hidden");
  $("provider-modal").classList.remove("hidden");
}

function providerFormPayload() {
  const type = $("provider-type").value;
  const rerank = isRerankProvider(type);
  return {
    id: $("provider-id").value.trim(),
    display_name: $("provider-name").value.trim(),
    type: $("provider-type").value,
    enabled: $("provider-enabled").checked,
    api_base: $("provider-api-base").value.trim(),
    api_suffix: $("provider-api-suffix").value.trim(),
    api_key: $("provider-api-key").value,
    clear_api_key: $("provider-clear-key").checked,
    model: $("provider-model").value.trim(),
    dimensions: rerank ? 0 : (Number($("provider-dimensions").value) || 0),
    max_context_tokens: rerank ? 0 : (Number($("provider-max-context").value) || 0),
    max_context_tokens_source: rerank ? "" : $("provider-max-context-source").value,
    return_documents: $("provider-return-documents").checked,
    instruct: $("provider-instruct").value.trim(),
    model_endpoint: $("provider-model-endpoint").value.trim(),
    truncate: $("provider-truncate").value.trim(),
    launch_model_if_not_running: $("provider-launch-model").checked,
    timeout_seconds: Number($("provider-timeout").value),
    proxy: $("provider-proxy").value.trim(),
    batch_size: rerank ? 1 : Number($("provider-batch").value),
    concurrency: rerank ? 1 : Number($("provider-concurrency").value),
    max_retries: Number($("provider-retries").value),
  };
}

$("provider-form").onsubmit = async (event) => {
  event.preventDefault();
  const originalId = $("provider-original-id").value;
  const payload = providerFormPayload();
  try {
    if (originalId) {
      const originalProvider = state.providers.find((item) => item.id === originalId)
        || { id: originalId, display_name: originalId };
      if (!(await confirmSensitiveProviderEdit(originalProvider, payload))) {
        return;
      }
      const { type, ...updates } = payload;
      await api(`/providers/${encodeURIComponent(originalId)}`, {
        method: "PATCH",
        body: JSON.stringify(updates),
      });
    } else {
      const { clear_api_key, ...createPayload } = payload;
      await api("/providers", { method: "POST", body: JSON.stringify(createPayload) });
    }
    closeOverlay("provider-modal");
    toast("Provider 已保存");
    await loadProviders();
    await loadLibraries(false);
  } catch (error) {
    toast(error.message, true);
  }
};

async function testDraftProvider() {
  const payload = providerFormPayload();
  const { clear_api_key, ...draft } = payload;
  const result = await api("/providers/test-draft", {
    method: "POST",
    body: JSON.stringify(draft),
  });
  $("provider-test-result").textContent = JSON.stringify(result, null, 2);
  $("provider-test-result").classList.remove("hidden");
  return result;
}

$("provider-draft-test").onclick = async () => {
  try {
    const result = await testDraftProvider();
    toast(result.available ? "Provider 测试成功" : result.error, !result.available);
  } catch (error) {
    toast(error.message, true);
  }
};

$("provider-detect-dimension").onclick = async () => {
  try {
    const payload = providerFormPayload();
    if (isRerankProvider(payload.type)) {
      toast("Rerank Provider 不需要检测嵌入维度", true);
      return;
    }
    const { clear_api_key, ...draft } = payload;
    draft.dimensions = 0;
    const result = await api("/providers/detect-dimension", {
      method: "POST",
      body: JSON.stringify(draft),
    });
    $("provider-dimensions").value = result.dimensions;
    toast(`检测到 ${result.dimensions} 维`);
  } catch (error) {
    toast(error.message, true);
  }
};

$("provider-detect-context").onclick = async () => {
  const button = $("provider-detect-context");
  button.disabled = true;
  try {
    const payload = providerFormPayload();
    if (isRerankProvider(payload.type)) {
      toast("Rerank Provider 不需要检测最大上下文长度", true);
      return;
    }
    const { clear_api_key, ...draft } = payload;
    draft.max_context_tokens = 0;
    draft.max_context_tokens_source = "";
    const originalId = $("provider-original-id").value.trim();
    const endpoint = originalId
      ? `/providers/${encodeURIComponent(originalId)}/detect-context-length`
      : "/providers/detect-context-length";
    const result = await api(endpoint, {
      method: "POST",
      body: JSON.stringify(draft),
    });
    if (result.max_context_tokens) {
      applyProviderContextLock(
        {
          max_context_tokens: result.max_context_tokens,
          max_context_tokens_source: result.max_context_tokens_source || "",
        },
        { syncDraftOrigin: true },
      );
      toast(`${t("autoDetect")} ${result.max_context_tokens}`);
      return;
    }
    applyProviderContextLock({
      max_context_tokens: Number($("provider-max-context").value) || 0,
      max_context_tokens_source: "",
    });
    toast(t("maxContextManualHelp"), true);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
  }
};

$("provider-api-base")?.addEventListener("input", refreshProviderContextDraftLock);
$("provider-model")?.addEventListener("input", refreshProviderContextDraftLock);

function fillProviderSelect(select, selectedId = "", kind = "embedding", options = {}) {
  const candidates = state.providers.filter(
    (provider) => providerKindOf(provider) === kind
      && (provider.enabled || provider.id === selectedId),
  );
  const allowEmpty = Boolean(options.optional);
  const resolvedSelectedId = candidates.some((provider) => provider.id === selectedId)
    ? selectedId
    : (allowEmpty ? "" : (candidates[0]?.id || ""));
  select.disabled = candidates.length === 0 && !allowEmpty;
  const optionRows = candidates
    .map(
      (provider) => `<option value="${escapeHtml(provider.id)}" ${provider.id === resolvedSelectedId ? "selected" : ""}>${escapeHtml(provider.display_name)} · ${escapeHtml(provider.model)}</option>`,
    );
  if (allowEmpty) {
    optionRows.unshift(`<option value="" ${resolvedSelectedId ? "" : "selected"}>${escapeHtml(options.emptyLabel || t("none"))}</option>`);
  }
  select.innerHTML = optionRows.length
    ? optionRows.join("")
    : `<option value="">${escapeHtml(t("noProviders"))}</option>`;
  if (resolvedSelectedId || allowEmpty) {
    select.value = resolvedSelectedId;
  }
}

$("settings-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const draft = settingsDraft();
  const payload = {
    access_base_url: draft.access_base_url,
    port: draft.port,
    access_port: draft.access_port,
    new_password: draft.new_password || null,
    clear_password: draft.clear_password,
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

$("libraries-refresh")?.addEventListener("click", () => loadLibraries());
$("providers-refresh")?.addEventListener("click", () => loadProviders());
$("graph-refresh")?.addEventListener("click", () => loadGraph());
$("recall-refresh")?.addEventListener("click", () => {
  setRecallK(DEFAULT_RECALL_K);
  setRecallRerankK(DEFAULT_RERANK_K);
  if (!$("recall-query").value.trim()) {
    toast(t("recallRefreshEmpty"), true);
    return;
  }
  $("run-recall").click();
});
$("system-refresh")?.addEventListener("click", () => loadSystem());
$("settings-refresh")?.addEventListener("click", () => loadSettings());
$("logs-refresh")?.addEventListener("click", async () => {
  try {
    await loadTasks(state.tasks.scope);
    await loadLogs({ reset: true });
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
  .catch(() => showLogin());
