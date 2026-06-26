const $ = (id) => document.getElementById(id);

const state = {
  page: "libraries",
  memoryPage: 1,
  memoryHasMore: false,
  stats: null,
  libraries: [],
  providers: [],
  providerTypes: [],
  providerStatuses: {},
  settings: null,
  loginMode: "api_key",
  selectedLibraryId: localStorage.getItem("prag_library_id") || "",
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
    scope: "active",
    polling: false,
    pollTimer: null,
  },
};

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
    runRecall: "运行召回",
    test: "测试",
    indexGeneration: "索引代次",
    rebuild: "完全重建",
    importanceDistribution: "重要性分布",
    atomTypes: "原子类型",
    backups: "备份与迁移归档",
    integrity: "完整性检查",
    cancel: "取消",
    save: "保存并重建索引",
    serviceEyebrow: "人格记忆服务",
    online: "在线",
    loading: "加载中…",
    sortNewest: "最新优先",
    sortOldest: "最早优先",
    sortImportanceDesc: "重要性从高到低",
    sortImportanceAsc: "重要性从低到高",
    previousPage: "上一页",
    nextPage: "下一页",
    topKLabel: "召回条数",
    embeddingProvider: "嵌入模型提供商",
    libraryManagement: "记忆库管理",
    libraryManagementHint: "每个记忆库拥有独立数据库、索引、缓存和模型绑定。",
    createLibrary: "新增记忆库",
    providerManagement: "模型提供商",
    providerManagementHint: "统一管理可供不同记忆库绑定的 Embedding Provider。",
    createProvider: "新增模型提供商",
    currentProviderBinding: "当前模型绑定",
    manageProviders: "管理 Provider",
    changeProvider: "切换并重建",
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
    statsGraphEntries: "图条目",
    statsAtoms: "原子",
    statsMessages: "消息",
    graphNoData: "暂无图谱数据",
    legendPerson: "人物",
    legendTopic: "主题",
    legendFact: "事实",
    legendSummary: "摘要",
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
    confirmRebuild: "确定重建文档索引和图谱索引吗？",
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
    switchProviderTitle: "切换模型并重建索引",
    switchProviderHint: "重建期间继续使用当前索引；新索引完整验证通过后才会原子切换。",
    targetProvider: "目标模型提供商",
    startFullRebuild: "开始完整重建",
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
    runRecall: "Run Recall",
    test: "Test",
    indexGeneration: "Index Generation",
    rebuild: "Full Rebuild",
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
    topKLabel: "Top K",
    embeddingProvider: "Embedding Provider",
    libraryManagement: "Memory Libraries",
    libraryManagementHint: "Each library owns an isolated database, index, cache, and provider binding.",
    createLibrary: "New Library",
    providerManagement: "Model Providers",
    providerManagementHint: "Manage embedding providers shared by memory libraries.",
    createProvider: "New Provider",
    currentProviderBinding: "Current Provider Binding",
    manageProviders: "Manage Providers",
    changeProvider: "Switch & Rebuild",
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
    statsGraphEntries: "Graph entries",
    statsAtoms: "Atoms",
    statsMessages: "Messages",
    graphNoData: "No graph data",
    legendPerson: "Person",
    legendTopic: "Topic",
    legendFact: "Fact",
    legendSummary: "Summary",
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
    confirmRebuild: "Rebuild document and graph indexes?",
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
    switchProviderTitle: "Switch provider and rebuild indexes",
    switchProviderHint: "Queries keep using the current index while rebuilding. The new index and provider switch atomically only after validation.",
    targetProvider: "Target provider",
    startFullRebuild: "Start full rebuild",
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
    runRecall: "Запустить поиск",
    test: "Проверить",
    indexGeneration: "Поколение индекса",
    rebuild: "Полная перестройка",
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
    topKLabel: "Top K",
    embeddingProvider: "Поставщик эмбеддингов",
    libraryManagement: "Библиотеки памяти",
    libraryManagementHint: "У каждой библиотеки отдельные база, индекс, кэш и провайдер.",
    createLibrary: "Новая библиотека",
    providerManagement: "Провайдеры моделей",
    providerManagementHint: "Управление провайдерами эмбеддингов для библиотек.",
    createProvider: "Новый провайдер",
    currentProviderBinding: "Текущая привязка",
    manageProviders: "Провайдеры",
    changeProvider: "Сменить и перестроить",
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
    statsGraphEntries: "Записи графа",
    statsAtoms: "Атомы",
    statsMessages: "Сообщения",
    graphNoData: "Нет данных графа",
    legendPerson: "Персона",
    legendTopic: "Тема",
    legendFact: "Факт",
    legendSummary: "Сводка",
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
    confirmRebuild: "Перестроить индексы документов и графа?",
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
    switchProviderTitle: "Сменить провайдер и перестроить индексы",
    switchProviderHint: "Во время перестройки запросы используют текущий индекс. Новый индекс и провайдер переключатся атомарно после проверки.",
    targetProvider: "Целевой провайдер",
    startFullRebuild: "Начать полную перестройку",
  },
};

Object.assign(strings.zh, {
  settings: "基础设置",
  settingsTitle: "基础设置",
  settingsHint: "管理 WebUI 登录方式和访问端口。端口修改会在下次启动生效；API Token 仍可作为脚本 Bearer 凭据。",
  currentAccessUrl: "当前 WebUI 地址",
  configuredPort: "配置端口",
  actualPort: "本次实际端口",
  loginMode: "登录方式",
  loginModePassword: "登录密码",
  loginModeApiKey: "API Token",
  newLoginPassword: "新登录密码",
  passwordKeepPlaceholder: "留空保持不变",
  clearLoginPassword: "清除登录密码，恢复 API Token 登录",
  saveSettings: "保存基础设置",
  settingsSaved: "基础设置已保存",
  settingsPortRestartHint: "端口修改会在下次启动生效；若端口被占用，启动时会自动回退并写入 warning 日志。",
  loginHintPassword: "请输入你设置的 WebUI 登录密码。",
  confirmAction: "确认",
  confirmTitle: "请确认",
  confirmDeleteLibrary: "确定删除记忆库 {id} 吗？系统会在回收目录保留核心 livingmemory.db 与 conversations.db。",
  confirmDeleteProvider: "确定删除 Provider {id} 吗？",
  confirmClearLogsTitle: "清空实时日志",
  confirmRebuildTitle: "重建索引",
  copyLibrary: "复制",
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
});

Object.assign(strings.en, {
  settings: "Settings",
  settingsTitle: "Basic Settings",
  settingsHint: "Manage WebUI login and access port. Port changes take effect next launch; API token remains available for Bearer automation.",
  currentAccessUrl: "Current WebUI URL",
  configuredPort: "Configured port",
  actualPort: "Actual port",
  loginMode: "Login mode",
  loginModePassword: "Password",
  loginModeApiKey: "API Token",
  newLoginPassword: "New login password",
  passwordKeepPlaceholder: "Leave blank to keep unchanged",
  clearLoginPassword: "Clear password and restore API token login",
  saveSettings: "Save settings",
  settingsSaved: "Settings saved",
  settingsPortRestartHint: "Port changes take effect on next launch. If occupied, startup falls back and writes a warning log.",
  loginHintPassword: "Enter your WebUI login password.",
  confirmAction: "Confirm",
  confirmTitle: "Confirm",
  confirmDeleteLibrary: "Delete library {id}? The trash will retain livingmemory.db and conversations.db.",
  confirmDeleteProvider: "Delete Provider {id}?",
  confirmClearLogsTitle: "Clear live logs",
  confirmRebuildTitle: "Rebuild indexes",
  copyLibrary: "Copy",
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
});

Object.assign(strings.ru, {
  settings: "Настройки",
  settingsTitle: "Основные настройки",
  settingsHint: "Настройка входа WebUI и порта. Порт применяется при следующем запуске; API token остаётся для Bearer-доступа.",
  currentAccessUrl: "Текущий WebUI URL",
  configuredPort: "Настроенный порт",
  actualPort: "Фактический порт",
  loginMode: "Способ входа",
  loginModePassword: "Пароль",
  loginModeApiKey: "API Token",
  newLoginPassword: "Новый пароль",
  passwordKeepPlaceholder: "Оставьте пустым без изменений",
  clearLoginPassword: "Удалить пароль и вернуть вход по API Token",
  saveSettings: "Сохранить настройки",
  settingsSaved: "Настройки сохранены",
  settingsPortRestartHint: "Порт применяется при следующем запуске. Если занят, будет выбран запасной порт и записан warning.",
  loginHintPassword: "Введите пароль WebUI.",
  confirmAction: "Подтвердить",
  confirmTitle: "Подтвердите",
  confirmDeleteLibrary: "Удалить библиотеку {id}? В корзине сохранятся livingmemory.db и conversations.db.",
  confirmDeleteProvider: "Удалить Provider {id}?",
  confirmClearLogsTitle: "Очистить журнал",
  confirmRebuildTitle: "Перестроить индексы",
  copyLibrary: "Копировать",
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
});

Object.assign(strings.zh, {
  rebuildIndex: "\u91cd\u5efa\u7d22\u5f15",
  indexRebuildQueued: "\u5df2\u4e3a {library} \u63d0\u4ea4\u7d22\u5f15\u91cd\u5efa\u4efb\u52a1",
  confirmRebuildLibraryIndex: "\u786e\u5b9a\u8981\u4e3a {library} \u91cd\u5efa\u7d22\u5f15\u5417\uff1f",
  providerSwitchQueued: "\u5df2\u4fdd\u5b58\u5e76\u5f00\u59cb\u4e3a {library} \u5168\u91cf\u91cd\u5efa\u7d22\u5f15",
  logs: "日志与任务列表",
  logsTitle: "日志与任务列表",
  graphPageHint: "查看当前记忆库的关系网络，可手动刷新最新导入或重建后的图谱。",
  recallPageHint: "用当前记忆库执行混合召回，刷新会重新运行上一次查询。",
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
});

Object.assign(strings.en, {
  rebuildIndex: "Rebuild index",
  indexRebuildQueued: "Index rebuild submitted for {library}",
  confirmRebuildLibraryIndex: "Rebuild indexes for {library}?",
  logs: "Logs & Tasks",
  logsTitle: "Logs & Tasks",
  graphPageHint: "View the current library graph and refresh after imports or rebuilds.",
  recallPageHint: "Run hybrid recall for the current library. Refresh reruns the last query.",
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
});

Object.assign(strings.ru, {
  rebuildIndex: "袩械褉械褋褌褉芯懈褌褜 懈薪写械泻褋",
  indexRebuildQueued: "袟邪写邪薪懈械 锌械褉械褋褌褉芯泄泫懈 懈薪写械泫褋邪 写谢褟 {library} 蟹邪锌褍褖械薪芯",
  confirmRebuildLibraryIndex: "袩械褉械褋褌褉芯懈褌褜 懈薪写械泫褋褘 写谢褟 {library}?",
  logs: "Logs & Tasks",
  logsTitle: "Logs & Tasks",
  graphPageHint: "Refresh the graph after imports or rebuilds.",
  recallPageHint: "Refresh reruns the last recall query.",
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
});

let lang = localStorage.getItem("prag_lang") || "zh";

function t(key, replacements = {}) {
  const template = strings[lang]?.[key] ?? strings.zh[key] ?? key;
  return Object.entries(replacements).reduce(
    (result, [name, value]) => result.replaceAll(`{${name}}`, String(value)),
    template,
  );
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

function libraryApi(path, options = {}) {
  if (!state.selectedLibraryId) {
    throw new Error("请先选择记忆库");
  }
  return api(`/libraries/${encodeURIComponent(state.selectedLibraryId)}${path}`, options);
}

function selectedLibrary() {
  return state.libraries.find((item) => item.id === state.selectedLibraryId) || null;
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
  $("sidebar-library-provider").textContent = `${provider.display_name || provider.id || "未绑定"} · r${provider.revision || library.provider_revision || "—"}`;
  box.title = `${library.name || library.id}\n${library.id}`;
  box.classList.remove("hidden");
}

function refreshLibraryContext() {
  refreshSidebarLibrary();
  const button = $("library-context");
  const library = selectedLibrary();
  const libraryPages = ["graph", "memory", "recall", "system"];
  if (!library || !libraryPages.includes(state.page)) {
    button.classList.add("hidden");
    return;
  }
  const provider = library.provider || {};
  const generation = library.indexes?.generation || "尚未构建";
  button.innerHTML = `<strong>${escapeHtml(library.name)}</strong><small>${escapeHtml(provider.display_name || provider.id || "未绑定")} · r${provider.revision || library.provider_revision} · ${escapeHtml(generation)}</small>`;
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
  state.selectedLibraryId = libraryId;
  localStorage.setItem("prag_library_id", state.selectedLibraryId);
  if (options.resetMemoryPage !== false) {
    state.memoryPage = 1;
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
    await loadPage(state.page);
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

document.querySelectorAll(".nav[data-page]").forEach((button) =>
  button.addEventListener("click", async () => {
    document.querySelectorAll(".nav[data-page]").forEach((item) =>
      item.classList.toggle("active", item === button),
    );
    document.querySelectorAll(".page").forEach((page) => page.classList.remove("active"));
    state.page = button.dataset.page;
    $("page-" + state.page).classList.add("active");
    applyLanguage();
    refreshLibraryContext();
    await loadPage(state.page);
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
    await loadTasks("active");
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
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
    drawGraph(snapshot);
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

function drawGraph(snapshot) {
  const nodes = snapshot.nodes || [];
  const edges = snapshot.edges || [];
  const canvas = $("graph-canvas");

  if (!nodes.length) {
    canvas.innerHTML = `<div class="empty">${escapeHtml(t("graphNoData"))}</div>`;
    return;
  }

  const width = 1100;
  const height = 600;
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.min(240, 45 + nodes.length * 4);
  const positions = new Map(
    nodes.map((node, index) => [
      node.id,
      {
        x: cx + Math.cos((index / nodes.length) * Math.PI * 2) * radius * (0.7 + (index % 3) * 0.14),
        y: cy + Math.sin((index / nodes.length) * Math.PI * 2) * radius * (0.7 + (index % 3) * 0.14),
      },
    ]),
  );
  const colors = { person: "#7367d8", topic: "#805bd1", fact: "#d5a20a", summary: "#ef4d86" };
  const legendLabels = {
    person: t("legendPerson"),
    topic: t("legendTopic"),
    fact: t("legendFact"),
    summary: t("legendSummary"),
  };

  let html = `<svg viewBox="0 0 ${width} ${height}"><g class="graph-viewport">`;
  edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (source && target) {
      html += `<line class="graph-edge" data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}"><title>${escapeHtml(edge.relation_type)}</title></line>`;
    }
  });
  nodes.forEach((node) => {
    const point = positions.get(node.id);
    const label = node.label.length > 22 ? `${node.label.slice(0, 22)}…` : node.label;
    html += `<g class="graph-node" data-id="${escapeHtml(node.id)}" transform="translate(${point.x},${point.y})"><circle r="${node.type === "fact" ? 9 : 7}" fill="${colors[node.type] || "#8492a6"}"></circle><text x="12" y="4">${escapeHtml(label)}</text><title>${escapeHtml(node.label)}</title></g>`;
  });
  html += "</g></svg>";
  canvas.innerHTML = html;
  enableGraphDragging(canvas, positions);

  $("graph-legend").innerHTML = Object.entries(colors)
    .map(([key, color]) => `<span><i style="background:${color}"></i> ${escapeHtml(legendLabels[key] || key)}</span>`)
    .join("");
}

function pointerToSvg(svg, event) {
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function enableGraphDragging(canvas, positions) {
  const svg = canvas.querySelector("svg");
  const viewport = canvas.querySelector(".graph-viewport");
  if (!svg || !viewport) return;
  const pan = { x: 0, y: 0 };
  let drag = null;

  const updateViewport = () => {
    viewport.setAttribute("transform", `translate(${pan.x} ${pan.y})`);
  };
  const updateEdges = () => {
    canvas.querySelectorAll(".graph-edge").forEach((line) => {
      const source = positions.get(line.dataset.source);
      const target = positions.get(line.dataset.target);
      if (!source || !target) return;
      line.setAttribute("x1", source.x);
      line.setAttribute("y1", source.y);
      line.setAttribute("x2", target.x);
      line.setAttribute("y2", target.y);
    });
  };

  canvas.querySelectorAll(".graph-node").forEach((nodeElement) => {
    nodeElement.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const id = nodeElement.dataset.id;
      const pos = positions.get(id);
      if (!pos) return;
      const point = pointerToSvg(svg, event);
      drag = {
        type: "node",
        id,
        element: nodeElement,
        offsetX: point.x - pan.x - pos.x,
        offsetY: point.y - pan.y - pos.y,
      };
      canvas.classList.add("dragging");
      nodeElement.setPointerCapture(event.pointerId);
    });
  });

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".graph-node")) return;
    const point = pointerToSvg(svg, event);
    drag = {
      type: "pan",
      startX: point.x,
      startY: point.y,
      baseX: pan.x,
      baseY: pan.y,
    };
    canvas.classList.add("dragging");
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const point = pointerToSvg(svg, event);
    if (drag.type === "pan") {
      pan.x = drag.baseX + point.x - drag.startX;
      pan.y = drag.baseY + point.y - drag.startY;
      updateViewport();
    } else if (drag.type === "node") {
      const pos = positions.get(drag.id);
      if (!pos) return;
      pos.x = point.x - pan.x - drag.offsetX;
      pos.y = point.y - pan.y - drag.offsetY;
      drag.element.setAttribute("transform", `translate(${pos.x},${pos.y})`);
      updateEdges();
    }
  });

  svg.addEventListener("pointerup", () => {
    drag = null;
    canvas.classList.remove("dragging");
  });
  svg.addEventListener("pointercancel", () => {
    drag = null;
    canvas.classList.remove("dragging");
  });
}

async function loadMemories() {
  const params = new URLSearchParams({
    page: state.memoryPage,
    page_size: 20,
    keyword: $("memory-keyword").value,
    persona_id: $("memory-persona").value,
    sort: $("memory-sort").value,
  });
  try {
    const data = await libraryApi("/memories?" + params);
    state.memoryHasMore = data.has_more;
    $("memory-rows").innerHTML =
      data.items
        .map(
          (item) => `<tr>
            <td>${item.id}</td>
            <td class="memory-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</td>
            <td>${escapeHtml(item.metadata.persona_id || "—")}</td>
            <td>${Number(item.metadata.importance ?? 0.5).toFixed(2)}</td>
            <td>${escapeHtml(displayStatus(item.metadata.status || "active"))}</td>
            <td>
              <div class="row-actions">
                <button class="ghost edit-memory" data-id="${item.id}">${escapeHtml(t("tableEdit"))}</button>
                <button class="ghost danger delete-memory" data-id="${item.id}">${escapeHtml(t("tableDelete"))}</button>
              </div>
            </td>
          </tr>`,
        )
        .join("") || `<tr><td colspan="6">${escapeHtml(t("tableEmpty"))}</td></tr>`;
    $("memory-page-info").textContent = t("pageOfTotal", { page: state.memoryPage, total: data.total });
    $("memory-prev").disabled = state.memoryPage <= 1;
    $("memory-next").disabled = !data.has_more;
    document.querySelectorAll(".edit-memory").forEach((button) => {
      button.onclick = () => openEdit(Number(button.dataset.id));
    });
    document.querySelectorAll(".delete-memory").forEach((button) => {
      button.onclick = () => deleteMemory(Number(button.dataset.id));
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
$("memory-sort").onchange = () => loadMemories();
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
  try {
    await libraryApi(id ? "/memories/" + id : "/memories", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify(payload),
    });
    closeModal();
    toast(t("saved"));
    await loadMemories();
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
    loadMemories();
  } catch (error) {
    toast(error.message, true);
  }
}

$("run-recall").onclick = async () => {
  const query = $("recall-query").value.trim();
  if (!query) {
    return;
  }
  try {
    const data = await libraryApi("/recall", {
      method: "POST",
      body: JSON.stringify({
        query,
        k: Number($("recall-k").value),
        persona_id: $("recall-persona").value || null,
        session_id: $("recall-session").value || null,
      }),
    });
    $("recall-summary").textContent = t("recallSummary", {
      total: data.total,
      elapsed: data.elapsed_time_ms,
    });
    $("recall-results").innerHTML =
      data.results
        .map(
          (item, index) => `<article class="result">
            <header>
              <span class="rank">#${index + 1}</span>
              <b>ID ${item.memory_id}</b>
              <span class="score">${Number(item.similarity_score).toFixed(4)}</span>
            </header>
            <div>${escapeHtml(item.content)}</div>
            <p>${escapeHtml(item.metadata.persona_id || "")} · ${escapeHtml(item.metadata.session_id || "")}</p>
            <details>
              <summary>${escapeHtml(t("recallScoreBreakdown"))}</summary>
              <pre>${escapeHtml(JSON.stringify(item.score_breakdown, null, 2))}</pre>
            </details>
          </article>`,
        )
        .join("") || `<div class="panel">${escapeHtml(t("recallNoResult"))}</div>`;
  } catch (error) {
    toast(error.message, true);
  }
};

async function loadSystem() {
  try {
    const data = await libraryApi("/stats");
    state.stats = data;
    statCards($("system-stats"), [
      [t("statsMemories"), data.total_memories],
      [t("statsNodes"), data.graph_nodes],
      [t("statsRelations"), data.graph_edges],
      [t("statsGraphEntries"), data.graph_entries],
      [t("statsAtoms"), data.atom_count],
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
    const data = await api("/settings");
    state.settings = data;
    $("settings-access-url").value = data.access_url || "";
    $("settings-port").value = data.configured_port || 8765;
    $("settings-actual-port").value = data.actual_port || data.configured_port || "";
    $("settings-login-mode").value =
      data.login_password_enabled ? t("loginModePassword") : t("loginModeApiKey");
    $("settings-password").value = "";
    $("settings-clear-password").checked = false;
    $("settings-note").textContent = t("settingsPortRestartHint");
  } catch (error) {
    toast(error.message, true);
  }
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

function renderTasks() {
  const list = $("task-list");
  if (!list) return;
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
    state.tasks.active = data.items || [];
  } else if (scope === "finished") {
    state.tasks.finished = data.items || [];
  } else {
    const items = data.items || [];
    state.tasks.active = items.filter((job) => ["queued", "running"].includes(job.status));
    state.tasks.finished = items.filter((job) => !["queued", "running"].includes(job.status));
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

$("rebuild-index").onclick = async () => {
  if (!(await confirmDialog({
    title: t("confirmRebuildTitle"),
    message: t("confirmRebuild"),
    confirmText: t("rebuild"),
  }))) {
    return;
  }
  try {
    const { job_id: jobId } = await libraryApi("/indexes/rebuild", {
      method: "POST",
      body: '{"reason":"webui"}',
    });
    watchJob(jobId);
  } catch (error) {
    toast(error.message, true);
  }
};

async function watchJob(id) {
  const box = $("job-progress");
  const bar = box.querySelector("div");
  const label = box.querySelector("span");
  box.classList.remove("hidden");
  while (true) {
    const job = await api("/jobs/" + id);
    if (state.page === "logs") {
      await loadTasks("active");
      if (state.tasks.scope === "finished") {
        await loadTasks("finished");
      }
    }
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
      refreshLibraryContext();
      if (state.page === "logs") {
        await loadTasks(state.tasks.scope);
      }
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

async function startLibraryIndexRebuild(libraryId, providerId = "", reason = "manual") {
  const payload = { reason };
  if (providerId) {
    payload.provider_id = providerId;
  }
  const { job_id: jobId } = await api(`/libraries/${encodeURIComponent(libraryId)}/indexes/rebuild`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  watchJob(jobId);
  toast(t("indexRebuildQueued", { library: libraryId }));
  return jobId;
}

function latestProvider(providerId) {
  return state.providers.find((provider) => provider.id === providerId) || null;
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
  if (Number(provider.revision || 0) !== Number(library.provider_revision || 0)) {
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
  document.querySelector(`.nav[data-page="${page}"]`)?.click();
}

$("library-context").onclick = () => navigate("libraries");
$("sidebar-current-library").onclick = () => navigate("libraries");
$("go-providers").onclick = () => navigate("providers");

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

async function loadLibraries(render = true) {
  try {
    const data = await api("/libraries");
    state.libraries = data.items || [];
    await ensureLibrarySelection();
    if (!render) {
      refreshLibraryContext();
      return;
    }
    $("library-cards").innerHTML =
      state.libraries
        .map((library) => {
          const stats = library.stats || {};
          const provider = library.provider || {};
          const indexes = library.indexes || {};
          const isSelected = library.id === state.selectedLibraryId;
          const isEmpty = libraryIsEmpty(library);
          const indexHealthy = Boolean(indexes.generation)
            && Number(indexes.document_vectors || 0) === Number(stats.total_memories || 0)
            && Number(indexes.graph_vectors || 0) === Number(stats.graph_entries || 0);
          const indexHealth = indexHealthy
            ? `<span class="pill success">${escapeHtml(t("indexHealthy"))}</span>`
            : indexes.generation
              ? `<span class="pill danger">${escapeHtml(t("indexMismatch"))}</span>`
              : `<span class="pill warning">${escapeHtml(t("indexPending"))}</span>`;
          return `<article class="management-card library-card ${isSelected ? "active-card" : ""}" data-id="${escapeHtml(library.id)}" role="button" tabindex="0" aria-pressed="${isSelected ? "true" : "false"}">
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
              <div class="mini-metric"><strong>${stats.conversation_counts?.sessions || 0}</strong><span>${escapeHtml(t("statsSessions"))}</span></div>
            </div>
            <dl class="provider-meta">
              <dt>${escapeHtml(t("providerLabel"))}</dt><dd>${escapeHtml(provider.display_name || provider.id || "—")} · r${provider.revision || library.provider_revision}</dd>
              <dt>${escapeHtml(t("modelDimension"))}</dt><dd>${escapeHtml(provider.model || "—")} / ${provider.dimensions || indexes.manifest?.dimension || t("autoDetect")}</dd>
              <dt>${escapeHtml(t("generationLabel"))}</dt><dd>${escapeHtml(indexes.generation || t("indexPending"))}</dd>
              <dt>${escapeHtml(t("indexStatus"))}</dt><dd>${indexHealth}</dd>
              <dt>${escapeHtml(t("defaultPersona"))}</dt><dd>${escapeHtml(library.default_persona_id || t("noLimit"))}</dd>
            </dl>
            <div class="card-actions">
              <button class="primary enter-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("enterLibrary"))}</button>
              <button class="ghost edit-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("edit"))}</button>
              <button class="ghost copy-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("copyLibrary"))}</button>
              ${isEmpty ? `<button class="ghost import-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("importMemory"))}</button>` : ""}
              <button class="ghost rebuild-library-index" data-id="${escapeHtml(library.id)}" data-provider="${escapeHtml(provider.id || library.provider_id || "")}">${escapeHtml(t("rebuildIndex"))}</button>
              <button class="ghost backup-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("backupNow"))}</button>
              ${library.is_default ? "" : `<button class="ghost default-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("setDefault"))}</button><button class="ghost danger delete-library" data-id="${escapeHtml(library.id)}">${escapeHtml(t("delete"))}</button>`}
            </div>
          </article>`;
        })
        .join("") || `<div class="panel">${escapeHtml(t("noLibraries"))}</div>`;
    bindLibraryCardActions();
    refreshLibraryContext();
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
    button.onclick = (event) => {
      event.stopPropagation();
      openLibraryModal(
        state.libraries.find((item) => item.id === button.dataset.id),
      );
    };
  });
  document.querySelectorAll(".copy-library").forEach((button) => {
    button.onclick = async (event) => {
      event.stopPropagation();
      try {
        const result = await api(`/libraries/${encodeURIComponent(button.dataset.id)}/copy`, { method: "POST" });
        toast(t("libraryCopyQueued", { id: button.dataset.id }));
        watchJob(result.job_id);
        if (state.page === "logs") await loadTasks("active");
      } catch (error) {
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

$("library-create").onclick = async () => {
  await loadProviders(false);
  openLibraryModal();
};

function openLibraryModal(library = null) {
  $("library-original-id").value = library?.id || "";
  $("library-id").value = library?.id || "";
  $("library-id").readOnly = Boolean(library);
  $("library-name").value = library?.name || "";
  $("library-description").value = library?.description || "";
  $("library-persona").value = library?.default_persona_id || "";
  $("library-provider-field").classList.remove("hidden");
  fillProviderSelect($("library-provider"), library?.provider_id);
  $("library-modal-title").textContent = library ? `编辑记忆库：${library.name}` : "新增记忆库";
  $("library-modal").classList.remove("hidden");
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
  };
  try {
    if (originalId) {
      const originalLibrary = state.libraries.find((item) => item.id === originalId);
      const providerChanged = Boolean(
        originalLibrary
        && payload.provider_id
        && payload.provider_id !== originalLibrary.provider_id,
      );
      await api(`/libraries/${encodeURIComponent(originalId)}`, {
        method: "PATCH",
        body: JSON.stringify({
          name: payload.name,
          description: payload.description,
          default_persona_id: payload.default_persona_id,
        }),
      });
      await loadProviders(false);
      const selectedProvider = latestProvider(payload.provider_id);
      const rebuildReason = providerChanged
        ? "library_edit_provider_switch"
        : libraryEditNeedsRebuild(originalLibrary, selectedProvider);
      if (rebuildReason) {
        await startLibraryIndexRebuild(originalId, payload.provider_id, rebuildReason);
        toast(t("providerSwitchQueued", { library: originalId }));
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
  try {
    const result = await api(`/libraries/${encodeURIComponent(libraryId)}/imports/livingmemory-db`, {
      method: "POST",
      body: form,
    });
    closeOverlay("import-modal");
    toast(t("importQueued", { library: libraryId }));
    watchJob(result.job_id);
    if (state.page === "logs") await loadTasks("active");
  } catch (error) {
    toast(error.message, true);
  }
});

function providerIcon(type) {
  const mapping = {
    openai_embedding: "/static/icons/openai.svg",
    ollama_embedding: "/static/icons/ollama.svg",
    vllm_embedding: "/static/icons/vllm.svg",
  };
  return mapping[type] || "/static/icons/openai.svg";
}

function providerTypeName(type) {
  return {
    openai_embedding: "OpenAI Embedding",
    ollama_embedding: "Ollama Embedding",
    vllm_embedding: "vLLM Embedding",
  }[type] || type;
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
    $("provider-cards").innerHTML =
      state.providers
        .map((provider) => {
          const status = state.providerStatuses[provider.id];
          const pendingLibraries = (provider.used_by || []).filter(
            (item) => Number(item.provider_revision) !== Number(provider.revision),
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
          return `<article class="management-card">
            <header>
              <div class="provider-card-head">
                <img class="provider-logo" src="${providerIcon(provider.type)}" alt="">
                <div><h3>${escapeHtml(provider.display_name)}</h3><span class="subtle">${escapeHtml(provider.id)} · r${provider.revision}</span></div>
              </div>
              <label class="switch" title="启用或停用"><input class="provider-toggle" data-id="${escapeHtml(provider.id)}" type="checkbox" ${provider.enabled ? "checked" : ""}><i></i></label>
            </header>
            <span class="status-dot ${statusClass}">${escapeHtml(statusText)}</span>
            <dl class="provider-meta">
              <dt>${escapeHtml(t("typeLabel"))}</dt><dd>${escapeHtml(providerTypeName(provider.type))}</dd>
              <dt>${escapeHtml(t("modelLabel"))}</dt><dd>${escapeHtml(provider.model)}</dd>
              <dt>${escapeHtml(t("dimensionLabel"))}</dt><dd>${provider.dimensions || escapeHtml(t("autoDetect"))}</dd>
              <dt>${escapeHtml(t("endpointLabel"))}</dt><dd>${escapeHtml(provider.api_base)}</dd>
              <dt>${escapeHtml(t("usedLibraries"))}</dt><dd>${(provider.used_by || []).map((item) => `${escapeHtml(item.library_name)}(r${item.provider_revision})`).join("、") || escapeHtml(t("none"))}</dd>
            </dl>
            <div class="card-actions">
              <button class="ghost danger delete-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("delete"))}</button>
              <button class="ghost edit-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("edit"))}</button>
              <button class="ghost copy-provider" data-id="${escapeHtml(provider.id)}">${escapeHtml(t("copy"))}</button>
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
  $("provider-type-cards").innerHTML = state.providerTypes
    .map((type) => `<button type="button" class="provider-type-card" data-type="${escapeHtml(type.id)}">
      <img src="${providerIcon(type.id)}" alt=""><b>${escapeHtml(providerTypeName(type.id))}</b>
      <p>${escapeHtml(type.id === "vllm_embedding" ? "适配 vLLM OpenAI-compatible Embedding，自动对齐 served-model-name。" : type.id === "ollama_embedding" ? "连接 Ollama /api/embed 接口。" : "连接 OpenAI 官方或兼容 Embedding 接口。")}</p>
    </button>`)
    .join("");
  document.querySelectorAll(".provider-type-card").forEach((button) => {
    button.onclick = () => {
      closeOverlay("provider-type-modal");
      const template = state.providerTypes.find((item) => item.id === button.dataset.type);
      openProviderEditor({ ...template, id: template.id, enabled: false }, true);
    };
  });
  $("provider-type-modal").classList.remove("hidden");
};

function openProviderEditor(provider, creating = false) {
  $("provider-original-id").value = creating ? "" : provider.id;
  $("provider-type").value = provider.type;
  $("provider-id").value = creating ? provider.id : provider.id;
  $("provider-id").readOnly = !creating;
  $("provider-name").value = provider.display_name || providerTypeName(provider.type);
  $("provider-enabled").checked = Boolean(provider.enabled);
  $("provider-api-key").value = "";
  $("provider-api-base").value = provider.api_base || "";
  $("provider-model").value = provider.model || "";
  $("provider-dimensions").value = provider.dimensions ?? 0;
  $("provider-timeout").value = provider.timeout_seconds || 30;
  $("provider-proxy").value = provider.proxy || "";
  $("provider-batch").value = provider.batch_size || 64;
  $("provider-concurrency").value = provider.concurrency || 2;
  $("provider-retries").value = provider.max_retries || 5;
  $("provider-clear-key").checked = false;
  $("provider-api-key-row").classList.toggle("hidden", provider.type === "ollama_embedding");
  $("provider-clear-key-row").classList.toggle("hidden", creating || !provider.has_api_key || provider.type === "ollama_embedding");
  $("provider-hint").textContent =
    provider.type === "vllm_embedding"
      ? "vLLM Provider 会自动忽略 dimensions，并尝试将模型名对齐到 served-model-name。"
      : provider.type === "ollama_embedding"
        ? "Ollama Provider 使用 /api/tags 获取模型，并通过 /api/embed 生成向量。"
        : "OpenAI Provider 支持官方及兼容接口；维度大于 0 时会发送 dimensions。";
  $("provider-modal-title").textContent = creating ? `新增 ${providerTypeName(provider.type)}` : `编辑 ${provider.display_name}`;
  $("provider-test-result").classList.add("hidden");
  $("provider-modal").classList.remove("hidden");
}

function providerFormPayload() {
  return {
    id: $("provider-id").value.trim(),
    display_name: $("provider-name").value.trim(),
    type: $("provider-type").value,
    enabled: $("provider-enabled").checked,
    api_base: $("provider-api-base").value.trim(),
    api_key: $("provider-api-key").value,
    clear_api_key: $("provider-clear-key").checked,
    model: $("provider-model").value.trim(),
    dimensions: Number($("provider-dimensions").value) || 0,
    timeout_seconds: Number($("provider-timeout").value),
    proxy: $("provider-proxy").value.trim(),
    batch_size: Number($("provider-batch").value),
    concurrency: Number($("provider-concurrency").value),
    max_retries: Number($("provider-retries").value),
  };
}

$("provider-form").onsubmit = async (event) => {
  event.preventDefault();
  const originalId = $("provider-original-id").value;
  const payload = providerFormPayload();
  try {
    if (originalId) {
      const { id, type, ...updates } = payload;
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

function fillProviderSelect(select, selectedId = "") {
  const enabled = state.providers.filter((provider) => provider.enabled);
  select.innerHTML = enabled
    .map((provider) => `<option value="${escapeHtml(provider.id)}" ${provider.id === selectedId ? "selected" : ""}>${escapeHtml(provider.display_name)} · ${escapeHtml(provider.model)} · r${provider.revision}</option>`)
    .join("");
}

$("change-provider").onclick = async () => {
  await loadProviders(false);
  const library = selectedLibrary();
  fillProviderSelect($("provider-switch-select"), library?.provider_id);
  $("provider-switch-modal").classList.remove("hidden");
};

$("provider-switch-form").onsubmit = async (event) => {
  event.preventDefault();
  const providerId = $("provider-switch-select").value;
  closeOverlay("provider-switch-modal");
  try {
    const { job_id: jobId } = await libraryApi("/indexes/rebuild", {
      method: "POST",
      body: JSON.stringify({ reason: "provider_switch", provider_id: providerId }),
    });
    watchJob(jobId);
  } catch (error) {
    toast(error.message, true);
  }
};

$("settings-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    port: Number($("settings-port").value),
    new_password: $("settings-password").value || null,
    clear_password: $("settings-clear-password").checked,
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
  button.onclick = async () => {
    state.tasks.scope = button.dataset.taskScope || "active";
    try {
      await loadTasks(state.tasks.scope);
    } catch (error) {
      toast(error.message, true);
    }
  };
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

applyLanguage();

fetch("/api/v1/auth/status", { credentials: "same-origin" })
  .then((response) => response.json())
  .then((status) => {
    applyLoginMode(status);
    if (status.authenticated) {
      showApp();
      loadPage(state.page);
    } else {
      showLogin();
    }
  })
  .catch(() => showLogin());
