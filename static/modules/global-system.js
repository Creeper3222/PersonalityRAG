export function createGlobalSystemController({ $, api, t, escapeHtml, statCards }) {
  function countByCategory(databases, category) {
    return databases.filter((item) => item.database_category === category).length;
  }

  async function loadGlobalSystem() {
    const [health, settings, databaseData, providerData, jobData] = await Promise.all([
      api("/health"),
      api("/settings"),
      api("/databases?stats_mode=summary"),
      api("/providers"),
      api("/jobs?scope=active"),
    ]);
    const databases = databaseData.items || [];
    const providers = providerData.items || [];
    const jobs = jobData.items || [];
    statCards($("global-system-stats"), [
      [t("serviceStatus"), health.status || t("unknown")],
      [t("databaseCount"), databases.length],
      [t("providerCount"), providers.length],
      [t("activeJobs"), jobs.length],
    ]);
    const status = health.status || t("unknown");
    $("global-system-stats")
      ?.querySelector(".stat:first-child")
      ?.classList.add("global-system-health-stat");
    $("global-system-service").innerHTML = `
      <div class="global-system-service-hero">
        <div>
          <span>${escapeHtml(t("version"))}</span>
          <strong>${escapeHtml(health.version || settings.version || "-")}</strong>
        </div>
        <span class="pill success">${escapeHtml(status)}</span>
      </div>
      <dl class="global-system-detail-list">
        <div>
          <dt>${escapeHtml(t("webuiAddress"))}</dt>
          <dd><code>${escapeHtml(settings.webui_url || settings.access_url || "-")}</code></dd>
        </div>
        <div>
          <dt>${escapeHtml(t("accessAddress"))}</dt>
          <dd><code>${escapeHtml(settings.recommended_adapter_url || settings.api_access_url || settings.access_url || "-")}</code></dd>
        </div>
      </dl>`;
    $("global-system-resources").innerHTML = `
      <div class="global-system-metric-grid global-system-resource-grid">
        <div><strong>${countByCategory(databases, "memory")}</strong><span>${escapeHtml(t("memoryLibraries"))}</span></div>
        <div><strong>${countByCategory(databases, "knowledge")}</strong><span>${escapeHtml(t("knowledgeLibraries"))}</span></div>
        <div><strong>${providers.filter((item) => item.provider_kind === "embedding").length}</strong><span>${escapeHtml(t("embeddingProviders"))}</span></div>
        <div><strong>${providers.filter((item) => item.provider_kind === "rerank").length}</strong><span>${escapeHtml(t("rerankProviders"))}</span></div>
      </div>`;
    $("global-system-runtime").innerHTML = `
      <div class="global-system-metric-grid global-system-runtime-grid">
        <div><strong>${escapeHtml(settings.runtime_residency?.idle_minutes ?? "-")}</strong><span>${escapeHtml(t("runtimeIdleMinutes"))}</span></div>
        <div><strong>${escapeHtml(settings.runtime_residency?.max_non_default_runtimes ?? "-")}</strong><span>${escapeHtml(t("maxNonDefaultRuntimes"))}</span></div>
        <div><strong>${jobs.length}</strong><span>${escapeHtml(t("activeJobs"))}</span></div>
      </div>`;
  }

  $("global-system-refresh")?.addEventListener("click", () => {
    loadGlobalSystem().catch(() => {});
  });

  return { loadGlobalSystem };
}
