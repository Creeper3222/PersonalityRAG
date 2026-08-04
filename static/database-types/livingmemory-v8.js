import { createGraphController } from "../modules/graph.js";
import { createMemoriesController } from "../modules/memories.js";
import { createRecallController } from "../modules/recall.js";
import { createSystemController } from "../modules/system.js";
import { LIVINGMEMORY_NAV_ICONS } from "./livingmemory-v8-icons.js";

const TYPE_ID = "livingmemory_v8";

export function createDatabaseUiDriver(context = {}) {
  let controllers = null;

  function ensureControllers() {
    if (!controllers) {
      controllers = {
        graph: createGraphController(context),
        memories: createMemoriesController(context),
        recall: createRecallController(context),
        overview: createSystemController(context),
      };
    }
    return controllers;
  }

  const actions = Object.freeze({
    loadGraph: (...args) => ensureControllers().graph.loadGraph(...args),
    renderGraph: (...args) => ensureControllers().graph.render(...args),
    loadMemories: (...args) => ensureControllers().memories.loadMemories(...args),
    setRecallView: (...args) => ensureControllers().recall.setRecallView(...args),
    renderRecallResults: (...args) => ensureControllers().recall.renderRecallResults(...args),
    resetRecallTest: (...args) => ensureControllers().recall.resetRecallTest(...args),
    loadOverview: (...args) => ensureControllers().overview.loadSystem(...args),
    scheduleOverviewRefresh: (...args) => ensureControllers().overview.scheduleSystemPanelsRefresh(...args),
    formatBytes: (...args) => ensureControllers().overview.formatBytes(...args),
  });

  async function load({ pageId, payload = null } = {}) {
    if (pageId === "overview") return actions.loadOverview();
    if (pageId === "graph") return actions.loadGraph(payload);
    if (pageId === "memories") return actions.loadMemories();
    if (pageId === "recall") return actions.renderRecallResults();
    return undefined;
  }

  return {
    typeId: TYPE_ID,
    category: "memory",
    icon: "/static/icons/livingmemory-v8.svg",
    cardLayout: "memory",
    settingsPage: null,
    defaultPage: "overview",
    pages: [
      { id: "overview", scope: "database", labelKey: "libraryOverview", titleKey: "libraryOverview", capability: "backup", viewId: "library-overview", iconMarkup: LIVINGMEMORY_NAV_ICONS.overview, order: 10 },
      { id: "graph", scope: "database", labelKey: "graph", titleKey: "graph", capability: "graph", viewId: "graph", iconMarkup: LIVINGMEMORY_NAV_ICONS.graph, order: 20 },
      { id: "memories", scope: "database", labelKey: "memory", titleKey: "memory", capability: "memory_records", viewId: "memory", iconMarkup: LIVINGMEMORY_NAV_ICONS.memories, order: 30 },
      { id: "recall", scope: "database", labelKey: "recall", titleKey: "recall", capability: "recall", viewId: "recall", iconMarkup: LIVINGMEMORY_NAV_ICONS.recall, order: 40 },
    ],
    actions,
    async mount({ root = document.querySelector("main") } = {}) {
      if (!document.getElementById("page-graph")) {
        const response = await fetch("/static/database-types/livingmemory-v8.html");
        if (!response.ok) throw new Error(`Unable to load LivingMemory WebUI: ${response.status}`);
        document.getElementById("page-system")?.insertAdjacentHTML(
          "beforebegin",
          await response.text(),
        );
      }
      ensureControllers();
    },
    load,
    async onDatabaseChange(database, pageId) {
      if (database) await load({ pageId });
    },
    async onLanguageChange(pageId) {
      await load({ pageId });
    },
    unmount() {
      controllers?.graph.destroy();
    },
  };
}
