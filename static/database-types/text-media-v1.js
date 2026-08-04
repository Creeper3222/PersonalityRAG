import { createTextMediaV1Controller } from "../modules/text-media-v1.js";

const TYPE_ID = "text_media_v1";

function navIcon(path) {
  return `<span class="nav-ico nav-ico-mask" aria-hidden="true" style="--nav-icon-mask:url('${path}')"></span>`;
}

export function createDatabaseUiDriver(context = {}) {
  let controller = null;

  function ensureController() {
    controller ||= createTextMediaV1Controller(context);
    return controller;
  }

  const actions = Object.freeze({
    openCreate: (...args) => ensureController().openCreate(...args),
    openEdit: (...args) => ensureController().openEdit(...args),
    openDatabase: (...args) => ensureController().openWorkspace(...args),
    refresh: (...args) => ensureController().refreshWorkspace(...args),
    refreshPage: (...args) => ensureController().refreshPage(...args),
  });

  return {
    typeId: TYPE_ID,
    category: "knowledge",
    icon: "/static/icons/text-media-v1.svg",
    cardLayout: "knowledge",
    settingsPage: null,
    defaultPage: "content",
    pages: [
      { id: "content", scope: "database", labelKey: "knowledgeContentManagement", titleKey: "knowledgeContentManagement", capability: "content_management", viewId: "database", order: 10, iconMarkup: navIcon("/static/icons/content-management.svg") },
      { id: "search", scope: "database", labelKey: "knowledgeSearchTest", titleKey: "knowledgeSearchTest", capability: "search", viewId: "database", order: 30 },
      { id: "settings", scope: "database_type", labelKey: "knowledgeSettings", titleKey: "knowledgeSettings", viewId: "database", order: 40, iconMarkup: navIcon("/static/icons/knowledge-settings.svg") },
    ],
    actions,
    async mount({ host } = {}) {
      if (host && !document.getElementById("text-media-workspace-modal") && !host.children.length) {
        const response = await fetch("/static/database-types/text-media-v1.html");
        if (!response.ok) throw new Error(`Unable to load text-media WebUI: ${response.status}`);
        host.innerHTML = await response.text();
      }
      ensureController().mountWorkspace(host);
    },
    async load({ scope = "database", pageId, database } = {}) {
      if (scope === "database_type") {
        await ensureController().openTypeSettings();
      } else if (database) {
        await actions.openDatabase(database, pageId || "content");
      }
    },
    async onDatabaseChange(database, pageId) {
      if (database) await actions.openDatabase(database, pageId || "content");
    },
    async onLanguageChange(pageId, scope = "database") {
      ensureController().setPage(pageId || "content");
      if (scope === "database_type") await ensureController().openTypeSettings();
      else await actions.refresh();
    },
    unmount() {},
  };
}
