export function createProvidersController({ $, state, t, toast, api, escapeHtml, validateIdentifierInput, confirmSensitiveProviderEdit, navigate, loadLibraries, confirmDialog, closeOverlay, selectLibrary }) {
const PROVIDER_ICON_SVG = {
  gemini_embedding: `<img src="/static/icons/google_gemini.svg" alt="" aria-hidden="true">`,
  nvidia_embedding: `<img src="/static/icons/nvidia.svg" alt="" aria-hidden="true">`,
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
    gemini_embedding: "Gemini Embedding",
    nvidia_embedding: "NVIDIA Embedding",
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
    gemini_embedding: "连接 Google Gemini Embedding API，支持指定输出向量维度。",
    nvidia_embedding: "连接 NVIDIA NIM Embedding API，支持 input_type。",
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
          const contextMode = inferContextLengthMode(provider);
          const maxContextText = provider.max_context_tokens
            ? `${provider.max_context_tokens} · ${contextMode === "auto" ? t("contextLengthModeAuto") : t("contextLengthModeManual")}`
            : t("maxContextAutoPending");
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
  setProviderRowVisible("provider-index-rebuild-section", !rerank);
  setProviderRowVisible("provider-index-embedding-batch-row", !rerank);
  setProviderRowVisible("provider-index-retry-delay-row", !rerank);
  setProviderRowVisible("provider-index-batch-delay-row", !rerank);
  setProviderRowVisible("provider-index-request-delay-row", !rerank);
  setProviderRowVisible("provider-index-failure-ratio-row", !rerank);
  setProviderRowVisible("provider-api-suffix-row", rerank && ["vllm_rerank", "xinference_rerank"].includes(type));
  setProviderRowVisible("provider-return-documents-row", type === "bailian_rerank");
  setProviderRowVisible("provider-instruct-row", type === "bailian_rerank");
  setProviderRowVisible("provider-model-endpoint-row", type === "nvidia_rerank");
  setProviderRowVisible("provider-truncate-row", type === "nvidia_rerank");
  setProviderRowVisible("provider-input-type-row", type === "nvidia_embedding");
  setProviderRowVisible("provider-launch-model-row", type === "xinference_rerank");
}

function inferContextLengthMode(provider = {}) {
  const source = String(provider.max_context_tokens_source || "");
  const tokens = Number(provider.max_context_tokens || 0);
  const mode = String(provider.context_length_mode || "").toLowerCase();
  if (mode === "auto" || mode === "manual") return mode;
  if (source.startsWith("auto:")) return "auto";
  return tokens >= 128 ? "manual" : "auto";
}

function applyProviderContextState(provider, options = {}) {
  const source = String(provider.max_context_tokens_source || "");
  const mode = inferContextLengthMode(provider);
  const autoMode = mode === "auto";
  const input = $("provider-max-context");
  const help = $("provider-context-help");
  const sourceLabel = $("provider-context-source-label");
  const modeSelect = $("provider-context-mode");
  input.value = provider.max_context_tokens ?? (autoMode ? 0 : 512);
  input.dataset.originalValue = String(provider.max_context_tokens ?? 0);
  input.dataset.originalSource = source;
  input.dataset.originalMode = mode;
  if (options.syncDraftOrigin) {
    $("provider-api-base").dataset.originalValue = $("provider-api-base").value;
    $("provider-model").dataset.originalValue = $("provider-model").value;
  }
  $("provider-max-context-source").value = source;
  if (modeSelect) modeSelect.value = mode;
  input.readOnly = autoMode;
  input.classList.toggle("readonly-lock", autoMode);
  input.setAttribute("aria-readonly", autoMode ? "true" : "false");
  if (help) {
    help.textContent = autoMode ? t("maxContextAutoHelp") : t("maxContextManualHelp");
  }
  if (sourceLabel) {
    const sourceText = source || (autoMode ? t("maxContextAutoPending") : "manual:user");
    sourceLabel.textContent = `${t("maxContextSource")}: ${sourceText}`;
  }
}

function applyProviderContextLock(provider, options = {}) {
  applyProviderContextState(provider, options);
}

function refreshProviderContextDraftLock() {
  const input = $("provider-max-context");
  const modeSelect = $("provider-context-mode");
  if (!input || modeSelect?.value !== "auto") return;
  const changed = $("provider-api-base").value !== ($("provider-api-base").dataset.originalValue || "")
    || $("provider-model").value !== ($("provider-model").dataset.originalValue || "");
  input.readOnly = true;
  input.classList.add("readonly-lock");
  input.setAttribute("aria-readonly", "true");
  if (changed) {
    input.value = "0";
    $("provider-max-context-source").value = "";
    $("provider-context-help").textContent = t("maxContextAutoPending");
    $("provider-context-source-label").textContent = `${t("maxContextSource")}: ${t("maxContextAutoPending")}`;
  } else {
    input.value = input.dataset.originalValue || "0";
    $("provider-max-context-source").value = input.dataset.originalSource || "";
    $("provider-context-help").textContent = t("maxContextAutoHelp");
    $("provider-context-source-label").textContent = `${t("maxContextSource")}: ${input.dataset.originalSource || t("maxContextAutoPending")}`;
  }
}

function providerHintText(type) {
  return {
    gemini_embedding: "Gemini Embedding 使用 batchEmbedContents，并按嵌入维度发送 outputDimensionality。",
    nvidia_embedding: "NVIDIA Embedding 使用 OpenAI-compatible /embeddings，并额外发送 input_type 与 float 编码。",
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
  const blankEmbeddingDraft = creating
    && kind !== "rerank"
    && !["gemini_embedding", "nvidia_embedding"].includes(provider.type);
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
  $("provider-input-type").value = provider.input_type || "passage";
  $("provider-launch-model").checked = Boolean(provider.launch_model_if_not_running);
  $("provider-timeout").value = provider.timeout_seconds || 30;
  $("provider-proxy").value = provider.proxy || "";
  const rebuild = provider.index_rebuild_settings || {};
  $("provider-batch").value = rebuild.batch_size || provider.batch_size || 50;
  $("provider-concurrency").value = rebuild.tasks_limit || provider.concurrency || 1;
  $("provider-retries").value = rebuild.max_retries || provider.max_retries || 5;
  $("provider-index-embedding-batch").value = rebuild.embedding_batch_size || 8;
  $("provider-index-retry-delay").value = rebuild.retry_base_delay ?? 30;
  $("provider-index-batch-delay").value = rebuild.batch_delay ?? 5;
  $("provider-index-request-delay").value = rebuild.request_delay ?? 5;
  $("provider-index-failure-ratio").value = rebuild.max_failure_ratio ?? 0.02;
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
    context_length_mode: rerank ? "auto" : ($("provider-context-mode")?.value || "auto"),
    max_context_tokens: rerank ? 0 : (Number($("provider-max-context").value) || 0),
    max_context_tokens_source: rerank ? "" : $("provider-max-context-source").value,
    return_documents: $("provider-return-documents").checked,
    instruct: $("provider-instruct").value.trim(),
    model_endpoint: $("provider-model-endpoint").value.trim(),
    truncate: $("provider-truncate").value.trim(),
    input_type: type === "nvidia_embedding" ? $("provider-input-type").value : "",
    launch_model_if_not_running: $("provider-launch-model").checked,
    timeout_seconds: Number($("provider-timeout").value),
    proxy: $("provider-proxy").value.trim(),
    batch_size: rerank ? 1 : Number($("provider-batch").value),
    concurrency: rerank ? 1 : Number($("provider-concurrency").value),
    max_retries: Number($("provider-retries").value),
    index_rebuild_settings: rerank ? null : {
      batch_size: Number($("provider-batch").value) || 50,
      embedding_batch_size: Number($("provider-index-embedding-batch").value) || 8,
      tasks_limit: Number($("provider-concurrency").value) || 1,
      max_retries: Number($("provider-retries").value) || 5,
      retry_base_delay: Number($("provider-index-retry-delay").value) || 0,
      batch_delay: Number($("provider-index-batch-delay").value) || 0,
      request_delay: Number($("provider-index-request-delay").value) || 0,
      max_failure_ratio: Number($("provider-index-failure-ratio").value) || 0,
    },
  };
}

$("provider-form").onsubmit = async (event) => {
  event.preventDefault();
  const originalId = $("provider-original-id").value;
  if (validateIdentifierInput($("provider-id")) === null) return;
  const payload = providerFormPayload();
  if (!isRerankProvider(payload.type)
    && payload.context_length_mode === "manual"
    && Number(payload.max_context_tokens || 0) < 128) {
    toast(t("maxContextManualInvalid"), true);
    return;
  }
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
    draft.context_length_mode = "auto";
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
      applyProviderContextState(
        {
          context_length_mode: "auto",
          max_context_tokens: result.max_context_tokens,
          max_context_tokens_source: result.max_context_tokens_source || "",
        },
        { syncDraftOrigin: true },
      );
      toast(`${t("autoDetect")} ${result.max_context_tokens}`);
      return;
    }
    applyProviderContextState({
      context_length_mode: "manual",
      max_context_tokens: Math.max(512, Number($("provider-max-context").value) || 0),
      max_context_tokens_source: "manual:fallback-undetected",
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
$("provider-context-mode")?.addEventListener("change", () => {
  const mode = $("provider-context-mode").value;
  if (mode === "manual") {
    applyProviderContextState({
      context_length_mode: "manual",
      max_context_tokens: Math.max(512, Number($("provider-max-context").value) || 0),
      max_context_tokens_source: $("provider-max-context-source").value || "manual:user",
    });
    return;
  }
  $("provider-detect-context")?.click();
});

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

  return {
    loadProviders,
    fillProviderSelect,
  };
}
