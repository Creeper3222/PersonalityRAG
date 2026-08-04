const DATABASE_UI_LOADERS = Object.freeze({
  livingmemory_v8: () => import("../database-types/livingmemory-v8.js"),
  text_media_v1: () => import("../database-types/text-media-v1.js"),
});

export function databasePageRoute(databaseType, pageId) {
  return `database:${databaseType}:${pageId}`;
}

export function databaseTypePageRoute(databaseType, pageId) {
  return `database-type:${databaseType}:${pageId}`;
}

export function globalPageRoute(pageId) {
  return `global:${pageId}`;
}

export function parsePageRoute(value) {
  const parts = String(value || "").split(":");
  if (parts[0] === "global" && parts.length === 2 && parts[1]) {
    return { scope: "global", pageId: parts[1] };
  }
  if (parts[0] === "database" && parts.length === 3 && parts[1] && parts[2]) {
    return { scope: "database", databaseType: parts[1], pageId: parts[2] };
  }
  if (parts[0] === "database-type" && parts.length === 3 && parts[1] && parts[2]) {
    return { scope: "database_type", databaseType: parts[1], pageId: parts[2] };
  }
  return null;
}

function validateDriver(driver, expectedType) {
  if (!driver || typeof driver !== "object") {
    throw new Error(`Invalid database UI driver: ${expectedType}`);
  }
  if (driver.typeId !== expectedType) {
    throw new Error(`Database UI driver type mismatch: ${expectedType}`);
  }
  if (!driver.defaultPage || !Array.isArray(driver.pages) || !driver.pages.length) {
    throw new Error(`Database UI driver has no pages: ${expectedType}`);
  }
  const pageIds = new Set();
  for (const page of driver.pages) {
    if (!page?.id || !page?.labelKey || !page?.titleKey) {
      throw new Error(`Invalid database UI page descriptor: ${expectedType}`);
    }
    if (pageIds.has(page.id)) {
      throw new Error(`Duplicate database UI page: ${expectedType}:${page.id}`);
    }
    if (page.scope && !["database", "database_type"].includes(page.scope)) {
      throw new Error(`Invalid database UI page scope: ${expectedType}:${page.id}`);
    }
    pageIds.add(page.id);
  }
  if (!pageIds.has(driver.defaultPage)) {
    throw new Error(`Invalid default database UI page: ${expectedType}:${driver.defaultPage}`);
  }
  for (const method of ["mount", "load", "onDatabaseChange", "onLanguageChange", "unmount"]) {
    if (typeof driver[method] !== "function") {
      throw new Error(`Database UI driver is missing ${method}: ${expectedType}`);
    }
  }
  return driver;
}

export function createDatabaseUiRegistry(context = {}) {
  const drivers = new Map();
  const pending = new Map();

  async function load(databaseType) {
    const typeId = String(databaseType || "");
    if (drivers.has(typeId)) return drivers.get(typeId);
    const loader = DATABASE_UI_LOADERS[typeId];
    if (!loader) return null;
    if (!pending.has(typeId)) {
      pending.set(typeId, loader().then((module) => {
        if (typeof module.createDatabaseUiDriver !== "function") {
          throw new Error(`Database UI module has no factory: ${typeId}`);
        }
        const driver = validateDriver(module.createDatabaseUiDriver(context), typeId);
        drivers.set(typeId, driver);
        pending.delete(typeId);
        return driver;
      }).catch((error) => {
        pending.delete(typeId);
        throw error;
      }));
    }
    return pending.get(typeId);
  }

  async function preload(databaseTypes = Object.keys(DATABASE_UI_LOADERS)) {
    return Promise.all(databaseTypes.map((databaseType) => load(databaseType)));
  }

  function get(databaseType) {
    return drivers.get(String(databaseType || "")) || null;
  }

  function availablePages(database) {
    const driver = get(database?.database_type);
    if (!driver) return [];
    const capabilities = new Set(database?.capabilities || []);
    return driver.pages.filter((page) => (
      (page.scope || "database") === "database"
      && (!page.capability || capabilities.has(page.capability))
    ));
  }

  function availableTypePages(databaseType) {
    const driver = get(databaseType);
    if (!driver) return [];
    return driver.pages.filter((page) => page.scope === "database_type");
  }

  function resolvePage(database, requestedPage = "") {
    const driver = get(database?.database_type);
    if (!driver) return null;
    const pages = availablePages(database);
    if (!pages.length) return null;
    return pages.find((page) => page.id === requestedPage)
      || pages.find((page) => page.id === driver.defaultPage)
      || pages[0];
  }

  function resolveTypePage(databaseType, requestedPage = "") {
    const pages = availableTypePages(databaseType);
    return pages.find((page) => page.id === requestedPage) || pages[0] || null;
  }

  return {
    load,
    preload,
    get,
    has: (databaseType) => Boolean(get(databaseType)),
    availablePages,
    availableTypePages,
    resolvePage,
    resolveTypePage,
    supportedTypes: Object.freeze(Object.keys(DATABASE_UI_LOADERS)),
  };
}
