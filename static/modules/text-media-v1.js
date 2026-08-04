const TYPE_ID = "text_media_v1";
const DEFAULT_RETRIEVAL_SETTINGS = Object.freeze({
  rrf_k: 60,
  rerank_candidate_limit: 10,
  rerank_fusion_weight: 0.30,
  rerank_rank_bonus_weight: 0,
  rerank_rank_reliability_exponent: 1.5,
  text_lexical_boost: 0.6,
  media_candidate_limit: 30,
  unbound_media_candidate_limit: 10,
  media_relevance_pivot_fallback: 0.35,
  media_pivot_positive_blend: 0.7,
  media_pivot_negative_weight: 0.35,
  media_pivot_negative_attenuation_floor: 0.05,
  media_format_mismatch_factor: 0.1,
  media_content_mismatch_factor: 0.10,
  media_threshold_evidence_limit: 5,
  media_threshold_rank_decay_exponent: 1.5,
  media_threshold_negative_reliability_exponent: 1.5,
  visual_intent_gate_enabled: true,
  media_semantic_floor: 0.35,
  media_semantic_weight: 0.8,
  media_lexical_boost: 0.3,
  media_lexical_coverage_exponent: 1.2,
  media_lexical_common_floor: 0,
  media_lexical_oov_penalty: 0.3,
  media_distinctive_rarity_exponent: 1.5,
  unbound_media_distinctive_boost: 0.35,
  unbound_media_collection_boost: 0.55,
  unbound_media_competition_floor: 0.35,
  unbound_media_reliability_target: 0.25,
  unbound_media_specificity_exponent: 1.0,
  unbound_media_advantage_target: 0.04,
  media_bound_distinctive_boost: 0.1,
  media_bound_distinctive_rescue_min: 0.8,
  media_rank_decay_exponent: 1,
  media_corroboration_weight: 0.5,
  media_corroboration_limit: 5,
});
const VISUAL_INTENT_POLICY_KEYS = Object.freeze([
  "visual_object_terms",
  "lookup_action_terms",
  "generation_action_terms",
  "reference_connector_terms",
]);

function retrievalScoreTierClass(value) {
  const score = Number(value);
  if (score >= 0.7) return "score-high";
  if (score >= 0.4) return "score-mid";
  return "score-low";
}
const VISUAL_INTENT_POLICY_LABEL_KEYS = Object.freeze({
  visual_object_terms: "visualObjectTerms",
  lookup_action_terms: "lookupActionTerms",
  generation_action_terms: "generationActionTerms",
  reference_connector_terms: "referenceConnectorTerms",
});
const MEDIA_DECISION_COLLAPSED_COUNT = 4;

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function createTextMediaV1Controller(deps) {
  const {
    $, state, t, api, responseError, toast, escapeHtml, asyncGuard, confirmDialog,
    closeOverlay, loadDatabases, loadProviders, fillProviderSelect, navigate, selectDatabase,
    addOptimisticTask, removeOptimisticTask, trackQueuedJob,
  } = deps;
  let currentKnowledgeBase = null;
  let entries = [];
  let documents = [];
  let documentOptions = [];
  let chunks = [];
  let assets = [];
  let reviewMode = "documents";
  let documentPage = { offset: 0, limit: 20, total: 0, query: "", sort: "created_desc" };
  let chunkPage = { offset: 0, limit: 20, total: 0, query: "", documentId: "", sort: "ordinal_asc" };
  const selectedDocumentIds = new Set();
  let importPreview = null;
  let batchImportPreview = null;
  let ingestDocuments = [];
  let ingestImages = [];
  let currentTab = "content";
  let searchGeneration = 0;
  let searchRetrievalMode = "standard";
  let searchResultView = "embedding";
  let searchResultCache = {
    embedding: [],
    rerank: [],
    rerankMeta: null,
    summary: null,
  };
  let documentSearchTimer = 0;
  let chunkSearchTimer = 0;
  let detailContentObserver = null;
  let detailContentFrame = 0;
  let searchCollapseFrame = 0;
  let mediaDecisionsExpanded = false;
  let visualIntentPolicyDraft = {};
  let visualIntentPolicyOriginal = {};
  let visualIntentPolicyTypeDefaults = {};
  let protectedVisualBlockerTerms = [];
  let visualIntentPolicyImportPreview = null;
  let retrievalSettingsLayout = "compact";
  let retrievalSettingsExpanded = false;
  let retrievalCollapseFrame = 0;
  let activeIngestPicker = "";

  const path = (databaseId, suffix = "") => (
    `/knowledge-libraries/${TYPE_ID}/${encodeURIComponent(databaseId)}${suffix}`
  );

  async function guarded(key, action, options = {}) {
    const {
      refreshCatalogOnConflict = false,
      ...guardOptions
    } = options;
    try {
      return await asyncGuard.run(key, action, guardOptions);
    } catch (error) {
      if (refreshCatalogOnConflict && Number(error?.status || 0) === 409) {
        await loadDatabases(state.page === "libraries", { force: true });
      }
      toast(error?.message || String(error), true);
      return null;
    }
  }

  function fileSize(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
    return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  }

  function renderEmpty(message) {
    return `<div class="text-media-empty">${escapeHtml(message)}</div>`;
  }

  function ensureImagePreview() {
    let overlay = $("text-media-image-preview");
    if (overlay) return overlay;
    overlay = document.createElement("div");
    overlay.id = "text-media-image-preview";
    overlay.className = "text-media-image-preview hidden";
    overlay.innerHTML = `
      <button type="button" class="text-media-image-preview-close" aria-label="Close">&times;</button>
      <figure>
        <img id="text-media-image-preview-img" alt="">
        <figcaption id="text-media-image-preview-caption"></figcaption>
      </figure>`;
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (event) => {
      if (event.target === overlay || event.target.closest(".text-media-image-preview-close")) {
        closeImagePreview();
      }
    });
    return overlay;
  }

  function closeImagePreview() {
    const overlay = $("text-media-image-preview");
    if (!overlay) return;
    overlay.classList.add("hidden");
    const image = $("text-media-image-preview-img");
    if (image) {
      image.removeAttribute("src");
      image.alt = "";
    }
  }

  function openImagePreview(url, caption) {
    if (!url) return;
    const overlay = ensureImagePreview();
    const image = $("text-media-image-preview-img");
    const label = $("text-media-image-preview-caption");
    image.src = url;
    image.alt = caption || t("knowledgeImage");
    label.textContent = caption || t("knowledgeImage");
    overlay.classList.remove("hidden");
    overlay.querySelector(".text-media-image-preview-close")?.focus();
  }

  function mediaVariantUrl(asset, variant) {
    const directUrl = asset ? asset[variant + "_url"] : "";
    if (directUrl) return directUrl;
    const assetId = asset?.asset_id || asset?.id || "";
    if (!currentKnowledgeBase?.id || !assetId) return "";
    return "/api/v1" + path(
      currentKnowledgeBase.id,
      "/assets/" + encodeURIComponent(assetId) + "/" + variant
    );
  }

  function mediaPreviewButton(asset, className = "text-media-search-media-preview") {
    const caption = asset?.caption || asset?.original_name || t("knowledgeImage");
    const thumbnailUrl = mediaVariantUrl(asset, "thumbnail");
    const contentUrl = mediaVariantUrl(asset, "content");
    const imageMarkup = thumbnailUrl
      ? (
        '<img src="' + escapeHtml(thumbnailUrl)
        + '" alt="' + escapeHtml(asset?.alt_text || caption) + '">'
      )
      : '<span aria-hidden="true"></span>';
    return (
      '<button type="button" class="' + escapeHtml(className)
      + '" data-preview-url="' + escapeHtml(contentUrl || thumbnailUrl)
      + '" data-preview-caption="' + escapeHtml(caption)
      + '" aria-label="' + escapeHtml(caption) + '">'
      + imageMarkup
      + "</button>"
    );
  }

  function clearIngestFiles() {
    ingestImages.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    ingestDocuments = [];
    ingestImages = [];
  }

  function ingestFileIdentity(file) {
    return `${file.name}\u0000${file.size}\u0000${file.lastModified}`;
  }

  function mergeIngestFiles(current, incoming) {
    const seen = new Set(current.map(ingestFileIdentity));
    const merged = [...current];
    incoming.forEach((file) => {
      const identity = ingestFileIdentity(file);
      if (seen.has(identity)) return;
      seen.add(identity);
      merged.push(file);
    });
    return merged;
  }

  function isIngestDocument(file) {
    const name = String(file?.name || "").toLowerCase();
    const type = String(file?.type || "").toLowerCase();
    return [".txt", ".md", ".markdown", ".pdf", ".docx"].some((extension) => name.endsWith(extension))
      || [
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      ].includes(type);
  }

  function isIngestImage(file) {
    const name = String(file?.name || "").toLowerCase();
    return [".png", ".jpg", ".jpeg", ".webp"].some((extension) => name.endsWith(extension))
      || ["image/png", "image/jpeg", "image/webp"].includes(
        String(file?.type || "").toLowerCase(),
      );
  }

  function setIngestPickerSummary(kind, count) {
    const picker = $(`text-media-ingest-${kind === "documents" ? "document" : "image"}-picker`);
    const output = $(`text-media-ingest-${kind === "documents" ? "document" : "image"}-picker-count`);
    picker?.classList.toggle("has-files", count > 0);
    if (output) output.textContent = count > 0 ? t("selectedFileCount", { count }) : t("notSelected");
  }

  function syncIngestParameter(source) {
    const parameter = source?.dataset?.ingestParameter;
    if (!parameter) return;
    document.querySelectorAll(`[data-ingest-parameter="${parameter}"]`).forEach((input) => {
      if (input !== source) input.value = source.value;
    });
  }

  function syncIngestParameterMirrors() {
    document.querySelectorAll(
      "#text-media-ingest-form [data-ingest-parameter]",
    ).forEach(syncIngestParameter);
  }

  function closeIngestPicker() {
    $("text-media-upload-picker-modal")?.classList.add("hidden");
    const picker = activeIngestPicker;
    activeIngestPicker = "";
    if (picker) {
      $(`text-media-ingest-${picker === "documents" ? "document" : "image"}-picker`)?.focus();
    }
  }

  function openIngestPicker(kind) {
    activeIngestPicker = kind === "images" ? "images" : "documents";
    syncIngestParameterMirrors();
    const images = activeIngestPicker === "images";
    $("text-media-upload-picker-title").textContent = t(
      images ? "addBatchImages" : "addBatchDocuments",
    );
    $("text-media-upload-drop-title").textContent = t(
      images ? "dropBatchImages" : "dropBatchDocuments",
    );
    $("text-media-upload-drop-hint").textContent = t(
      images ? "dropBatchImagesHint" : "dropBatchDocumentsHint",
    );
    $("text-media-upload-picker-limit").textContent = t(
      "uploadPickerRemaining",
      { count: Math.max(0, 10 - (images ? ingestImages.length : ingestDocuments.length)) },
    );
    $("text-media-upload-picker-modal").classList.remove("hidden");
    window.setTimeout(() => $("text-media-upload-drop-zone")?.focus(), 0);
  }

  function acceptIngestFiles(kind, files) {
    const incoming = [...(files || [])];
    const images = kind === "images";
    const accepted = incoming.filter(images ? isIngestImage : isIngestDocument);
    if (accepted.length !== incoming.length) {
      toast(t(images ? "unsupportedBatchImage" : "unsupportedBatchDocument"), true);
    }
    if (!accepted.length) return;
    if (images) {
      const existingFiles = ingestImages.map((item) => item.file);
      const merged = mergeIngestFiles(existingFiles, accepted);
      if (merged.length > 10) toast(t("batchImageLimit"), true);
      const retained = new Set(existingFiles.map(ingestFileIdentity));
      ingestImages = [
        ...ingestImages,
        ...merged
          .filter((file) => !retained.has(ingestFileIdentity(file)))
          .slice(0, Math.max(0, 10 - ingestImages.length))
          .map((file) => ({
            file,
            previewUrl: URL.createObjectURL(file),
            bindingMode: ingestDocuments.length ? "all" : "none",
            documentIndexes: new Set(),
            mediaDescription: filenameStem(file.name),
            mediaDescriptions: [filenameStem(file.name)],
          })),
      ];
      renderIngestImages();
    } else {
      const previouslyHadDocuments = Boolean(ingestDocuments.length);
      const merged = mergeIngestFiles(ingestDocuments, accepted);
      if (merged.length > 10) toast(t("batchDocumentLimit"), true);
      ingestDocuments = merged.slice(0, 10);
      ingestImages.forEach((item) => {
        item.documentIndexes = new Set(
          [...item.documentIndexes].filter((index) => index < ingestDocuments.length),
        );
        if (!ingestDocuments.length) item.bindingMode = "none";
        else if (!previouslyHadDocuments && item.bindingMode === "none") item.bindingMode = "all";
      });
      renderIngestDocuments();
      renderIngestImages();
    }
    closeIngestPicker();
  }

  function filenameStem(name) {
    return String(name || "").replace(/\.[^.]+$/, "") || t("knowledgeImage");
  }

  function currentIngestMode() {
    if (ingestDocuments.length && ingestImages.length) return "mixed";
    if (ingestDocuments.length) return "textOnly";
    if (ingestImages.length) return "mediaOnly";
    return "empty";
  }

  function updateIngestControls() {
    const mode = currentIngestMode();
    const output = $("text-media-ingest-mode");
    if (output) output.textContent = t(`ingestMode_${mode}`);
    const hasDocuments = Boolean(ingestDocuments.length);
    ["text-media-ingest-chunk-target", "text-media-ingest-chunk-overlap"].forEach((id) => {
      const input = $(id);
      if (input) input.disabled = !hasDocuments;
    });
    const hasBoundImage = ingestImages.some((item) => item.bindingMode !== "none");
    $("text-media-ingest-semantic-section")?.classList.toggle(
      "hidden", !hasDocuments || !ingestImages.length
    );
    const semantic = $("text-media-ingest-semantic-enabled");
    if (semantic) {
      semantic.disabled = !hasDocuments || !hasBoundImage;
      if (semantic.disabled) semantic.checked = false;
    }
  }

  function renderIngestDocuments() {
    $("text-media-ingest-document-count").textContent = `${ingestDocuments.length} / 10`;
    setIngestPickerSummary("documents", ingestDocuments.length);
    $("text-media-ingest-document-list").innerHTML = ingestDocuments.length
      ? ingestDocuments.map((file, index) => `
        <article class="text-media-ingest-file-row">
          <span><strong>${escapeHtml(file.name)}</strong><small>${escapeHtml(fileSize(file.size))}</small></span>
          <button type="button" class="icon text-media-remove-ingest-document" data-index="${index}" aria-label="${escapeHtml(t("removeFile"))}">×</button>
        </article>`).join("")
      : renderEmpty(t("noBatchDocuments"));
    document.querySelectorAll(".text-media-remove-ingest-document").forEach((button) => {
      button.onclick = () => {
        const removed = Number(button.dataset.index);
        ingestDocuments.splice(removed, 1);
        ingestImages.forEach((item) => {
          item.documentIndexes = new Set(
            [...item.documentIndexes]
              .filter((index) => index !== removed)
              .map((index) => index > removed ? index - 1 : index)
          );
          if (!ingestDocuments.length) item.bindingMode = "none";
        });
        renderIngestDocuments();
        renderIngestImages();
      };
    });
    updateIngestControls();
  }

  function renderIngestImages() {
    $("text-media-ingest-image-count").textContent = `${ingestImages.length} / 10`;
    setIngestPickerSummary("images", ingestImages.length);
    $("text-media-ingest-image-list").innerHTML = ingestImages.length
      ? ingestImages.map((item, imageIndex) => {
        const descriptions = item.mediaDescriptions || [item.mediaDescription || filenameStem(item.file.name)];
        item.mediaDescriptions = descriptions;
        return `
        <article class="text-media-ingest-image-row">
          <img src="${escapeHtml(item.previewUrl)}" alt="${escapeHtml(item.file.name)}">
          <div class="text-media-ingest-image-details">
            <header><span><strong>${escapeHtml(item.file.name)}</strong><small>${escapeHtml(fileSize(item.file.size))}</small></span><button type="button" class="icon text-media-remove-ingest-image" data-index="${imageIndex}" aria-label="${escapeHtml(t("removeFile"))}">×</button></header>
            <div class="text-media-ingest-scope">
              <label><input type="radio" name="text-media-image-scope-${imageIndex}" value="none" data-image-index="${imageIndex}"${item.bindingMode === "none" ? " checked" : ""}>${escapeHtml(t("noDocumentBinding"))}</label>
              ${ingestDocuments.length ? `<label><input type="radio" name="text-media-image-scope-${imageIndex}" value="all" data-image-index="${imageIndex}"${item.bindingMode === "all" ? " checked" : ""}>${escapeHtml(t("allBatchDocuments"))}</label>
              <label><input type="radio" name="text-media-image-scope-${imageIndex}" value="selected" data-image-index="${imageIndex}"${item.bindingMode === "selected" ? " checked" : ""}>${escapeHtml(t("selectedBatchDocuments"))}</label>` : ""}
            </div>
            <div class="text-media-ingest-document-links${item.bindingMode === "selected" ? "" : " hidden"}">
              ${ingestDocuments.map((file, documentIndex) => `<label><input type="checkbox" data-image-index="${imageIndex}" data-document-index="${documentIndex}"${item.documentIndexes.has(documentIndex) ? " checked" : ""}>${escapeHtml(file.name)}</label>`).join("") || `<small>${escapeHtml(t("chooseBatchDocumentsFirst"))}</small>`}
            </div>
            <section class="text-media-description-list">
              <header><strong>${escapeHtml(t("mediaDescriptions"))}</strong><button type="button" class="ghost text-media-add-description" data-image-index="${imageIndex}"${descriptions.length >= 20 ? " disabled" : ""}>＋ ${escapeHtml(t("addMediaDescription"))}</button></header>
              ${descriptions.map((description, descriptionIndex) => `
                <div class="text-media-description-row">
                  <label class="text-media-description"><span>${escapeHtml(descriptionIndex === 0 ? t("primaryMediaDescription") : t("mediaDescriptionNumber", { number: descriptionIndex + 1 }))}</span><textarea class="text-media-ingest-description" data-image-index="${imageIndex}" data-description-index="${descriptionIndex}" rows="2" maxlength="2000" required placeholder="${escapeHtml(t("mediaDescriptionPlaceholder"))}">${escapeHtml(description)}</textarea></label>
                  <div class="text-media-description-actions">
                    ${descriptionIndex > 0 ? `<button type="button" class="ghost text-media-make-primary-description" data-image-index="${imageIndex}" data-description-index="${descriptionIndex}">${escapeHtml(t("setAsPrimaryDescription"))}</button>` : ""}
                    ${descriptions.length > 1 ? `<button type="button" class="ghost danger text-media-remove-description" data-image-index="${imageIndex}" data-description-index="${descriptionIndex}">${escapeHtml(t("remove"))}</button>` : ""}
                  </div>
                </div>`).join("")}
            </section>
          </div>
        </article>`;
      }).join("")
      : renderEmpty(t("noBatchImages"));
    document.querySelectorAll(".text-media-remove-ingest-image").forEach((button) => {
      button.onclick = () => {
        const index = Number(button.dataset.index);
        URL.revokeObjectURL(ingestImages[index].previewUrl);
        ingestImages.splice(index, 1);
        renderIngestImages();
      };
    });
    document.querySelectorAll('.text-media-ingest-scope input[type="radio"]').forEach((input) => {
      input.onchange = () => {
        const item = ingestImages[Number(input.dataset.imageIndex)];
        item.bindingMode = input.value;
        if (item.bindingMode === "selected" && !item.documentIndexes.size && ingestDocuments.length) {
          item.documentIndexes.add(0);
        }
        renderIngestImages();
      };
    });
    document.querySelectorAll('.text-media-ingest-document-links input[type="checkbox"]').forEach((input) => {
      input.onchange = () => {
        const item = ingestImages[Number(input.dataset.imageIndex)];
        const documentIndex = Number(input.dataset.documentIndex);
        if (input.checked) item.documentIndexes.add(documentIndex);
        else item.documentIndexes.delete(documentIndex);
      };
    });
    document.querySelectorAll(".text-media-ingest-description").forEach((input) => {
      input.oninput = () => {
        const item = ingestImages[Number(input.dataset.imageIndex)];
        item.mediaDescriptions[Number(input.dataset.descriptionIndex)] = input.value;
        item.mediaDescription = item.mediaDescriptions[0];
      };
    });
    document.querySelectorAll(".text-media-add-description").forEach((button) => {
      button.onclick = () => {
        const item = ingestImages[Number(button.dataset.imageIndex)];
        if (item.mediaDescriptions.length >= 20) return;
        item.mediaDescriptions.push("");
        renderIngestImages();
      };
    });
    document.querySelectorAll(".text-media-remove-description").forEach((button) => {
      button.onclick = () => {
        const item = ingestImages[Number(button.dataset.imageIndex)];
        if (item.mediaDescriptions.length <= 1) return;
        item.mediaDescriptions.splice(Number(button.dataset.descriptionIndex), 1);
        item.mediaDescription = item.mediaDescriptions[0];
        renderIngestImages();
      };
    });
    document.querySelectorAll(".text-media-make-primary-description").forEach((button) => {
      button.onclick = () => {
        const item = ingestImages[Number(button.dataset.imageIndex)];
        const index = Number(button.dataset.descriptionIndex);
        const [description] = item.mediaDescriptions.splice(index, 1);
        item.mediaDescriptions.unshift(description);
        item.mediaDescription = item.mediaDescriptions[0];
        renderIngestImages();
      };
    });
    updateIngestControls();
  }

  function openBatchIngest() {
    clearIngestFiles();
    closeIngestPicker();
    $("text-media-ingest-form").reset();
    const defaults = currentKnowledgeBase?.ingest_defaults || {};
    $("text-media-ingest-chunk-target").value = defaults.chunk_target || 1200;
    $("text-media-ingest-chunk-overlap").value = defaults.chunk_overlap || 150;
    syncIngestParameterMirrors();
    $("text-media-ingest-progress").textContent = "";
    renderIngestDocuments();
    renderIngestImages();
    $("text-media-ingest-modal").classList.remove("hidden");
  }

  function typeLibraries() {
    return (state.databases || []).filter((item) => item.database_type === TYPE_ID);
  }

  function selectedBatchIds() {
    return [...document.querySelectorAll(".text-media-batch-library-check:checked")]
      .map((input) => input.value);
  }

  function updateBatchSelection() {
    const count = selectedBatchIds().length;
    $("text-media-batch-selected-count").textContent = t("selectedLibraries", { count });
    $("text-media-batch-export").disabled = count === 0;
  }

  function renderTransferLibraries() {
    const libraries = typeLibraries();
    const select = $("text-media-single-export-library");
    select.innerHTML = libraries.length
      ? libraries.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name || item.id)} · ${escapeHtml(item.id)}</option>`).join("")
      : `<option value="">${escapeHtml(t("noKnowledgeLibraries"))}</option>`;
    $("text-media-single-export").disabled = libraries.length === 0;
    $("text-media-batch-library-list").innerHTML = libraries.length
      ? libraries.map((item) => `
        <label class="text-media-batch-library-row">
          <input class="text-media-batch-library-check" type="checkbox" value="${escapeHtml(item.id)}">
          <span><strong>${escapeHtml(item.name || item.id)}</strong><small>${escapeHtml(item.id)}</small></span>
          <em class="pill">${escapeHtml(t(item.status || "ready"))}</em>
        </label>`).join("")
      : renderEmpty(t("noKnowledgeLibrariesRestoreHint"));
    document.querySelectorAll(".text-media-batch-library-check").forEach((input) => {
      input.onchange = updateBatchSelection;
    });
    updateBatchSelection();
  }

  async function openCreate() {
    await loadProviders(false);
    $("text-media-create-form").reset();
    fillProviderSelect($("text-media-create-provider"), "", "embedding");
    fillProviderSelect($("text-media-create-rerank-provider"), "", "rerank", {
      optional: true,
      emptyLabel: t("none"),
    });
    $("text-media-create-modal").classList.remove("hidden");
    window.setTimeout(() => $("text-media-create-id")?.focus(), 0);
  }

  function cloneVisualIntentPolicy(policy) {
    const source = policy || {};
    return Object.fromEntries(VISUAL_INTENT_POLICY_KEYS.map((key) => [
      key,
      Array.isArray(source[key]) ? source[key].map((item) => String(item)) : [],
    ]));
  }

  function normalizedVisualIntentPolicyDraft() {
    return Object.fromEntries(VISUAL_INTENT_POLICY_KEYS.map((key) => {
      const seen = new Set();
      const terms = [];
      (visualIntentPolicyDraft[key] || []).forEach((value) => {
        const term = String(value || "").normalize("NFKC").trim().replace(/\s+/g, " ");
        const identity = term.toLocaleLowerCase();
        if (!term || seen.has(identity)) return;
        seen.add(identity);
        terms.push(term);
      });
      return [key, terms];
    }));
  }

  function policyModalCategories(attribute) {
    return [...document.querySelectorAll(`[${attribute}]:checked`)]
      .map((input) => input.getAttribute(attribute))
      .filter((key) => VISUAL_INTENT_POLICY_KEYS.includes(key));
  }

  function closePolicyModal(id) {
    $(id)?.classList.add("hidden");
  }

  function updatePolicyTransferConfirmButtons() {
    const exportConfirm = $("text-media-visual-policy-export-confirm");
    if (exportConfirm) {
      exportConfirm.disabled = policyModalCategories("data-policy-export-category").length === 0;
    }
    const importConfirm = $("text-media-visual-policy-import-confirm");
    if (importConfirm) {
      importConfirm.disabled = policyModalCategories("data-policy-import-category").length === 0;
    }
  }

  function openVisualPolicyExport() {
    document.querySelectorAll("[data-policy-export-category]").forEach((input) => {
      input.checked = true;
    });
    updatePolicyTransferConfirmButtons();
    $("text-media-visual-policy-export-modal")?.classList.remove("hidden");
  }

  async function createVisualPolicySaveTarget(filename) {
    if (typeof window.showSaveFilePicker === "function") {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        excludeAcceptAllOption: true,
        types: [{
          description: "CSV",
          accept: { "text/csv": [".csv"] },
        }],
      });
      return async (blob) => {
        const writable = await handle.createWritable();
        try {
          await writable.write(blob);
          await writable.close();
        } catch (error) {
          await writable.abort().catch(() => {});
          throw error;
        }
      };
    }
    return async (blob) => {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    };
  }

  function openVisualPolicyImportPreview(preview) {
    visualIntentPolicyImportPreview = preview;
    const categories = Array.isArray(preview?.categories) ? preview.categories : [];
    $("text-media-visual-policy-import-summary").textContent = t(
      "importLexiconSummary",
      { filename: preview?.filename || "—", count: categories.length },
    );
    $("text-media-visual-policy-import-categories").innerHTML = categories.map((item) => {
      const key = String(item.key || "");
      const labelKey = VISUAL_INTENT_POLICY_LABEL_KEYS[key] || key;
      return `<label>
        <input type="checkbox" data-policy-import-category="${escapeHtml(key)}" checked>
        <span><strong>${escapeHtml(t(labelKey))}</strong><small>${escapeHtml(t("lexiconTermCount", { count: Number(item.count || 0) }))}</small></span>
      </label>`;
    }).join("");
    updatePolicyTransferConfirmButtons();
    $("text-media-visual-policy-import-modal")?.classList.remove("hidden");
  }

  function renderVisualIntentPolicy() {
    VISUAL_INTENT_POLICY_KEYS.forEach((key) => {
      const host = document.querySelector(`[data-policy-list="${key}"]`);
      if (!host) return;
      const terms = visualIntentPolicyDraft[key] || [];
      host.innerHTML = terms.length
        ? terms.map((term, index) => `
          <label class="text-media-policy-term-row">
            <input type="text" maxlength="64" value="${escapeHtml(term)}" data-policy-term="${escapeHtml(key)}" data-policy-index="${index}" aria-label="${escapeHtml(t("visualIntentTerm"))}">
            <button type="button" class="icon text-media-policy-remove" data-policy-remove="${escapeHtml(key)}" data-policy-index="${index}" aria-label="${escapeHtml(t("remove"))}">×</button>
          </label>`).join("")
        : `<p class="form-help">${escapeHtml(t("visualIntentCategoryEmpty"))}</p>`;
      const addButton = document.querySelector(`[data-policy-add="${key}"]`);
      if (addButton) addButton.disabled = terms.length >= 256;
    });
    const blockers = $("text-media-protected-blocker-list");
    if (blockers) blockers.innerHTML = protectedVisualBlockerTerms.length
      ? protectedVisualBlockerTerms.map((term) => `<code>${escapeHtml(term)}</code>`).join("")
      : `<span>—</span>`;
  }

  function fillRetrievalSettings(value = DEFAULT_RETRIEVAL_SETTINGS) {
    const retrieval = { ...DEFAULT_RETRIEVAL_SETTINGS, ...(value || {}) };
    $("text-media-edit-rrf-k").value = retrieval.rrf_k;
    $("text-media-edit-rerank-candidate-limit").value = retrieval.rerank_candidate_limit;
    $("text-media-edit-rerank-fusion-weight").value = retrieval.rerank_fusion_weight;
    $("text-media-edit-rerank-rank-bonus-weight").value = retrieval.rerank_rank_bonus_weight;
    $("text-media-edit-rerank-reliability-exponent").value = retrieval.rerank_rank_reliability_exponent;
    $("text-media-edit-text-lexical-boost").value = retrieval.text_lexical_boost;
    $("text-media-edit-candidate-limit").value = retrieval.media_candidate_limit;
    $("text-media-edit-unbound-candidate-limit").value = retrieval.unbound_media_candidate_limit;
    $("text-media-edit-score-threshold-fallback").value = retrieval.media_relevance_pivot_fallback ?? retrieval.media_score_threshold_fallback;
    $("text-media-edit-threshold-evidence-limit").value = retrieval.media_threshold_evidence_limit;
    $("text-media-edit-threshold-rank-exponent").value = retrieval.media_threshold_rank_decay_exponent;
    $("text-media-edit-threshold-negative-exponent").value = retrieval.media_threshold_negative_reliability_exponent;
    $("text-media-edit-threshold-reinforcement-weight").value = retrieval.media_pivot_positive_blend ?? retrieval.media_threshold_reinforcement_weight;
    $("text-media-edit-threshold-weakening-weight").value = retrieval.media_pivot_negative_weight ?? retrieval.media_threshold_weakening_weight;
    $("text-media-edit-pivot-negative-floor").value = retrieval.media_pivot_negative_attenuation_floor;
    $("text-media-edit-format-mismatch-factor").value = retrieval.media_format_mismatch_factor;
    $("text-media-edit-content-mismatch-factor").value = retrieval.media_content_mismatch_factor;
    $("text-media-edit-intent-gate").checked = retrieval.visual_intent_gate_enabled;
    $("text-media-edit-semantic-floor").value = retrieval.media_semantic_floor;
    $("text-media-edit-semantic-weight").value = retrieval.media_semantic_weight;
    $("text-media-edit-lexical-boost").value = retrieval.media_lexical_boost;
    $("text-media-edit-lexical-exponent").value = retrieval.media_lexical_coverage_exponent;
    $("text-media-edit-lexical-common-floor").value = retrieval.media_lexical_common_floor;
    $("text-media-edit-lexical-oov-penalty").value = retrieval.media_lexical_oov_penalty;
    $("text-media-edit-distinctive-rarity-exponent").value = retrieval.media_distinctive_rarity_exponent;
    $("text-media-edit-unbound-distinctive-boost").value = retrieval.unbound_media_distinctive_boost;
    $("text-media-edit-unbound-collection-boost").value = retrieval.unbound_media_collection_boost;
    $("text-media-edit-unbound-competition-floor").value = retrieval.unbound_media_competition_floor;
    $("text-media-edit-unbound-reliability-target").value = retrieval.unbound_media_reliability_target;
    $("text-media-edit-unbound-specificity-exponent").value = retrieval.unbound_media_specificity_exponent;
    $("text-media-edit-unbound-advantage-target").value = retrieval.unbound_media_advantage_target;
    $("text-media-edit-bound-distinctive-boost").value = retrieval.media_bound_distinctive_boost;
    $("text-media-edit-bound-distinctive-rescue-min").value = retrieval.media_bound_distinctive_rescue_min;
    $("text-media-edit-rank-exponent").value = retrieval.media_rank_decay_exponent;
    $("text-media-edit-corroboration-weight").value = retrieval.media_corroboration_weight;
    $("text-media-edit-corroboration-limit").value = retrieval.media_corroboration_limit;
    return retrieval;
  }

  function applyRetrievalCollapseState() {
    const section = $("text-media-retrieval-settings");
    const primary = $("text-media-retrieval-primary");
    const toggle = $("text-media-retrieval-toggle");
    const topToggle = $("text-media-retrieval-collapse-top");
    if (!section || !primary || !toggle || !topToggle || section.closest(".modal-overlay")?.classList.contains("hidden")) return;
    const collapsedHeight = Math.round(window.innerHeight * 0.75);
    primary.style.setProperty("--text-media-retrieval-collapsed-height", `${collapsedHeight}px`);
    primary.style.setProperty("--text-media-retrieval-expanded-height", `${primary.scrollHeight}px`);
    const collapsible = primary.scrollHeight > collapsedHeight + 8;
    if (!collapsible) retrievalSettingsExpanded = false;
    section.classList.toggle("text-media-retrieval-collapsible", collapsible);
    section.classList.toggle("text-media-retrieval-collapsed", collapsible && !retrievalSettingsExpanded);
    section.classList.toggle("text-media-retrieval-expanded", collapsible && retrievalSettingsExpanded);
    toggle.classList.toggle("hidden", !collapsible);
    toggle.classList.toggle("expanded", collapsible && retrievalSettingsExpanded);
    toggle.setAttribute("aria-expanded", retrievalSettingsExpanded ? "true" : "false");
    topToggle.classList.toggle("hidden", !collapsible || !retrievalSettingsExpanded);
    const label = t(retrievalSettingsExpanded ? "collapseLibraryCard" : "expandLibraryCard");
    toggle.title = label;
    toggle.setAttribute("aria-label", label);
    const collapseLabel = t("collapseLibraryCard");
    topToggle.title = collapseLabel;
    topToggle.setAttribute("aria-label", collapseLabel);
  }

  function scheduleRetrievalCollapseRefresh() {
    if (retrievalCollapseFrame) window.cancelAnimationFrame(retrievalCollapseFrame);
    retrievalCollapseFrame = window.requestAnimationFrame(() => {
      retrievalCollapseFrame = 0;
      applyRetrievalCollapseState();
    });
  }

  function setRetrievalSettingsLayout(layout) {
    retrievalSettingsLayout = layout === "detailed" ? "detailed" : "compact";
    const section = $("text-media-retrieval-settings");
    if (section) section.dataset.layout = retrievalSettingsLayout;
    document.querySelectorAll("[data-retrieval-layout]").forEach((button) => {
      const active = button.dataset.retrievalLayout === retrievalSettingsLayout;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    $("text-media-retrieval-settings")?.querySelector("[role='group']")
      ?.setAttribute("aria-label", t("retrievalLayoutLabel"));
    scheduleRetrievalCollapseRefresh();
  }

  async function openEdit(library) {
    await loadProviders(false);
    currentKnowledgeBase = library;
    $("text-media-edit-original-id").value = library.id;
    $("text-media-edit-original-provider").value = library.provider_id || library.provider?.id || "";
    $("text-media-edit-id").value = library.id;
    $("text-media-edit-name").value = library.name || "";
    $("text-media-edit-description").value = library.description || "";
    $("text-media-edit-uniform-strength").value = Number(
      library.uniform_media_strength ?? 0.5,
    ).toFixed(2);
    fillProviderSelect(
      $("text-media-edit-provider"),
      library.provider_id || library.provider?.id || "",
      "embedding",
    );
    fillProviderSelect(
      $("text-media-edit-rerank-provider"),
      library.rerank_provider_id || "",
      "rerank",
      { optional: true, emptyLabel: t("none") },
    );
    const rerankStatus = library.rerank_binding || {};
    $("text-media-edit-rerank-status").textContent = rerankStatus.needs_recalibration
      ? t("rerankNeedsRecalibration")
      : (rerankStatus.available ? t("rerankAvailable") : t("rerankUnavailableHint"));
    const retrieval = fillRetrievalSettings(library.retrieval_settings);
    retrievalSettingsExpanded = false;
    setRetrievalSettingsLayout("compact");
    $("text-media-retrieval-status").textContent = "";
    visualIntentPolicyDraft = cloneVisualIntentPolicy(
      library.visual_intent_policy || retrieval.visual_intent_policy,
    );
    visualIntentPolicyOriginal = cloneVisualIntentPolicy(visualIntentPolicyDraft);
    visualIntentPolicyTypeDefaults = cloneVisualIntentPolicy(
      library.visual_intent_policy_type_defaults || visualIntentPolicyDraft,
    );
    protectedVisualBlockerTerms = Array.isArray(library.protected_visual_blocker_terms)
      ? [...library.protected_visual_blocker_terms]
      : [];
    renderVisualIntentPolicy();
    visualIntentPolicyImportPreview = null;
    if ($("text-media-visual-policy-import-file")) {
      $("text-media-visual-policy-import-file").value = "";
    }
    $("text-media-visual-policy-status").textContent = library.visual_intent_policy_fingerprint
      ? t("visualIntentPolicyFingerprint", { fingerprint: String(library.visual_intent_policy_fingerprint).slice(0, 12) })
      : "";
    $("text-media-edit-modal").classList.remove("hidden");
    scheduleRetrievalCollapseRefresh();
    window.setTimeout(() => $("text-media-edit-name")?.focus(), 0);
  }

  function setTab(tab) {
    if (tab === "media") tab = "content";
    currentTab = tab || "content";
    document.querySelectorAll("[data-text-media-tab]").forEach((button) => {
      const active = button.dataset.textMediaTab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-text-media-view]").forEach((view) => {
      view.classList.toggle("active", view.dataset.textMediaView === tab);
    });
  }

  function mountWorkspace(host) {
    const overlay = $("text-media-workspace-modal");
    const workspace = overlay?.querySelector(".text-media-workspace-modal");
    if (!host || !workspace) return;
    workspace.classList.remove("modal", "text-media-workspace-modal");
    workspace.classList.add("text-media-workspace-page");
    workspace.removeAttribute("role");
    workspace.removeAttribute("aria-modal");
    workspace.querySelector(".modal-dismiss")?.remove();
    workspace.querySelector(".text-media-tabs")?.remove();
    const editModal = $("text-media-edit-modal");
    if (editModal) document.body.append(editModal);
    const ingestModal = $("text-media-ingest-modal");
    if (ingestModal) document.body.append(ingestModal);
    const uploadPickerModal = $("text-media-upload-picker-modal");
    if (uploadPickerModal) document.body.append(uploadPickerModal);
    const policyExportModal = $("text-media-visual-policy-export-modal");
    if (policyExportModal) document.body.append(policyExportModal);
    const policyImportModal = $("text-media-visual-policy-import-modal");
    if (policyImportModal) document.body.append(policyImportModal);
    const detailOverlay = $("text-media-detail-overlay");
    if (detailOverlay) document.body.append(detailOverlay);
    const detailPanel = $("text-media-detail-panel");
    if (detailPanel) document.body.append(detailPanel);
    host.replaceChildren(workspace);
    overlay.remove();
  }

  async function openWorkspace(library, tab = "content") {
    const changedLibrary = !currentKnowledgeBase || currentKnowledgeBase.id !== library.id;
    currentKnowledgeBase = library;
    if (changedLibrary) {
      selectedDocumentIds.clear();
      documentPage.offset = 0;
      documentPage.query = "";
      chunkPage.offset = 0;
      chunkPage.query = "";
      chunkPage.documentId = "";
      if ($("text-media-document-search")) $("text-media-document-search").value = "";
      if ($("text-media-chunk-search")) $("text-media-chunk-search").value = "";
      closeDetail();
    }
    document.querySelector(".text-media-workspace-header")?.classList.remove("hidden");
    $("text-media-workspace-title").textContent = library.name;
    $("text-media-workspace-id").textContent = library.id;
    configureSearchRerankDefault();
    setTab(tab);
    await refreshWorkspace();
  }

  async function openTypeSettings() {
    currentKnowledgeBase = null;
    document.querySelector(".text-media-workspace-header")?.classList.add("hidden");
    $("text-media-workspace-title").textContent = t("knowledgeSettings");
    $("text-media-workspace-id").textContent = "";
    setTab("settings");
    await loadDatabases(false);
    renderTransferLibraries();
  }

  function formatDate(value) {
    if (!value) return "—";
    const numeric = Number(value);
    const normalized = Number.isFinite(numeric) && numeric > 0 && numeric < 1e12
      ? numeric * 1000
      : value;
    const date = new Date(normalized);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function formatConfidence(value) {
    const number = Number(value || 0);
    return number > 0 && number < 0.001 ? "<0.001" : number.toFixed(3);
  }

  function headingPath(text) {
    const headings = String(text || "").split("\n").filter((line) => /^#{1,6}\s+/.test(line));
    return headings.map((line) => line.replace(/^#{1,6}\s+/, "").trim()).slice(0, 4).join(" / ");
  }

  function contentExcerpt(text, limit = 240) {
    const value = String(text || "").replace(/^#{1,6}\s+.*$/gm, "").replace(/\s+/g, " ").trim();
    return value.length > limit ? `${value.slice(0, limit)}…` : value;
  }

  function metaItem(label, value) {
    return `<div class="memory-detail-meta-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "—")}</strong></div>`;
  }

  function collapsibleDetailContent(content) {
    return `
      <div class="text-media-detail-content-panel text-media-detail-content-collapsed">
        <button type="button" class="collapse-top-toggle text-media-detail-content-collapse-top hidden">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m6 15 6-6 6 6"></path></svg>
        </button>
        <div class="text-media-detail-content-primary">
          <div class="memory-detail-content">${escapeHtml(content || "")}</div>
        </div>
        <button type="button" class="library-expand-toggle text-media-detail-content-toggle hidden" aria-expanded="false">
          <svg viewBox="0 0 240 24" aria-hidden="true" focusable="false">
            <circle cx="92" cy="12" r="3"></circle>
            <circle cx="120" cy="12" r="3"></circle>
            <circle cx="148" cy="12" r="3"></circle>
          </svg>
        </button>
      </div>`;
  }

  function disconnectDetailContentCollapse() {
    detailContentObserver?.disconnect();
    detailContentObserver = null;
    if (detailContentFrame) cancelAnimationFrame(detailContentFrame);
    detailContentFrame = 0;
  }

  function setupDetailContentCollapse() {
    disconnectDetailContentCollapse();
    const host = $("text-media-detail-body");
    const container = host?.querySelector(".text-media-detail-content-panel");
    const primary = container?.querySelector(".text-media-detail-content-primary");
    const toggle = container?.querySelector(".text-media-detail-content-toggle");
    const topToggle = container?.querySelector(".text-media-detail-content-collapse-top");
    if (!container || !primary || !toggle || !topToggle) return;

    let expanded = false;
    const applyState = () => {
      detailContentFrame = 0;
      const collapsedHeight = Math.max(280, Math.round(window.innerHeight * 0.5));
      container.style.setProperty("--text-media-detail-collapsed-height", `${collapsedHeight}px`);
      container.style.setProperty("--text-media-detail-expanded-height", `${Math.ceil(primary.scrollHeight)}px`);
      const overflowing = primary.scrollHeight > collapsedHeight + 24;
      container.classList.toggle("text-media-detail-content-collapsible", overflowing);
      toggle.classList.toggle("hidden", !overflowing);
      topToggle.classList.toggle("hidden", !overflowing || !expanded);
      if (!overflowing) {
        expanded = false;
        container.classList.remove("text-media-detail-content-collapsed", "text-media-detail-content-expanded");
        toggle.classList.remove("expanded");
        toggle.setAttribute("aria-expanded", "false");
        toggle.removeAttribute("title");
        toggle.removeAttribute("aria-label");
        topToggle.removeAttribute("title");
        topToggle.removeAttribute("aria-label");
        return;
      }
      container.classList.toggle("text-media-detail-content-collapsed", !expanded);
      container.classList.toggle("text-media-detail-content-expanded", expanded);
      toggle.classList.toggle("expanded", expanded);
      toggle.setAttribute("aria-expanded", String(expanded));
      const label = t(expanded ? "collapseLibraryCard" : "expandLibraryCard");
      toggle.title = label;
      toggle.setAttribute("aria-label", label);
      const collapseLabel = t("collapseLibraryCard");
      topToggle.title = collapseLabel;
      topToggle.setAttribute("aria-label", collapseLabel);
    };
    const scheduleApply = () => {
      if (detailContentFrame) cancelAnimationFrame(detailContentFrame);
      detailContentFrame = requestAnimationFrame(applyState);
    };
    toggle.onclick = () => {
      expanded = !expanded;
      applyState();
    };
    topToggle.onclick = () => {
      expanded = false;
      applyState();
    };
    if (typeof ResizeObserver !== "undefined") {
      detailContentObserver = new ResizeObserver(scheduleApply);
      detailContentObserver.observe(primary.querySelector(".memory-detail-content") || primary);
    }
    scheduleApply();
  }

  function closeDetail() {
    disconnectDetailContentCollapse();
    const panel = $("text-media-detail-panel");
    const overlay = $("text-media-detail-overlay");
    panel?.classList.remove("visible");
    panel?.setAttribute("aria-hidden", "true");
    overlay?.classList.add("hidden");
    overlay?.setAttribute("aria-hidden", "true");
  }

  function showDetail(title, markup) {
    disconnectDetailContentCollapse();
    $("text-media-detail-title").textContent = title;
    $("text-media-detail-body").innerHTML = markup;
    const panel = $("text-media-detail-panel");
    const overlay = $("text-media-detail-overlay");
    overlay.classList.remove("hidden");
    overlay.setAttribute("aria-hidden", "false");
    panel.setAttribute("aria-hidden", "false");
    requestAnimationFrame(() => panel.classList.add("visible"));
    setupDetailContentCollapse();
  }

  function renderPagination(hostId, page, handler) {
    const host = $(hostId);
    if (!host) return;
    const pageCount = Math.max(1, Math.ceil(page.total / page.limit));
    const current = Math.min(pageCount, Math.floor(page.offset / page.limit) + 1);
    host.innerHTML = `
      <button type="button" class="ghost" data-page="previous"${current <= 1 ? " disabled" : ""}>${escapeHtml(t("previousPage"))}</button>
      <span>${escapeHtml(t("pageOf", { current, total: pageCount, count: page.total }))}</span>
      <button type="button" class="ghost" data-page="next"${current >= pageCount ? " disabled" : ""}>${escapeHtml(t("nextPage"))}</button>`;
    host.querySelectorAll("button").forEach((button) => {
      button.onclick = () => handler(button.dataset.page === "previous" ? -1 : 1);
    });
  }

  function updateDocumentSelection() {
    const visibleIds = documents.map((item) => String(item.id));
    const selectedVisible = visibleIds.filter((id) => selectedDocumentIds.has(id));
    const selectPage = $("text-media-document-select-page");
    selectPage.checked = Boolean(visibleIds.length) && selectedVisible.length === visibleIds.length;
    selectPage.indeterminate = selectedVisible.length > 0 && selectedVisible.length < visibleIds.length;
    $("text-media-document-selected-count").textContent = t("selectedDocumentsCount", { count: selectedDocumentIds.size });
    $("text-media-document-batch-delete").disabled = selectedDocumentIds.size === 0;
  }

  function renderDocuments() {
    const host = $("text-media-document-list");
    host.innerHTML = documents.length ? `
      <div class="text-media-review-head"><span></span><span>${escapeHtml(t("document"))}</span><span>${escapeHtml(t("chunkCountLabel"))}</span><span>${escapeHtml(t("imageCountLabel"))}</span><span>${escapeHtml(t("uploadedAt"))}</span><span></span></div>
      ${documents.map((item) => `
        <article class="text-media-review-row text-media-document-row" data-document-id="${escapeHtml(item.id)}">
          <label class="text-media-row-check"><input type="checkbox" data-document-select="${escapeHtml(item.id)}"${selectedDocumentIds.has(String(item.id)) ? " checked" : ""}><span class="sr-only">${escapeHtml(t("selectDocument"))}</span></label>
          <button type="button" class="text-media-review-primary" data-open-document="${escapeHtml(item.id)}"><strong>${escapeHtml(item.title || item.original_name || item.id)}</strong><small>${escapeHtml(item.original_name || item.parser_id || item.id)}</small></button>
          <span class="text-media-review-stat"><strong>${Number(item.chunk_count || 0)}</strong><small>${escapeHtml(t("chunks"))}</small></span>
          <span class="text-media-review-stat"><strong>${Number(item.image_count || 0)}</strong><small>${escapeHtml(t("images"))}</small></span>
          <time>${escapeHtml(formatDate(item.created_at))}</time>
          <button type="button" class="icon text-media-row-open" data-open-document="${escapeHtml(item.id)}" aria-label="${escapeHtml(t("viewDetails"))}">›</button>
        </article>`).join("")}` : renderEmpty(t("noSourceDocuments"));
    host.querySelectorAll("[data-document-select]").forEach((input) => {
      input.onchange = () => {
        if (input.checked) selectedDocumentIds.add(String(input.dataset.documentSelect));
        else selectedDocumentIds.delete(String(input.dataset.documentSelect));
        updateDocumentSelection();
      };
    });
    host.querySelectorAll("[data-open-document]").forEach((button) => {
      button.onclick = () => openDocumentDetail(button.dataset.openDocument);
    });
    updateDocumentSelection();
    renderPagination("text-media-document-pagination", documentPage, async (direction) => {
      documentPage.offset = Math.max(0, documentPage.offset + direction * documentPage.limit);
      await loadDocuments();
    });
  }

  function renderChunks() {
    const host = $("text-media-chunk-list");
    host.innerHTML = chunks.length ? `
      <div class="text-media-review-head text-media-chunk-head"><span>#</span><span>${escapeHtml(t("chunkContent"))}</span><span>${escapeHtml(t("sourceDocument"))}</span><span>${escapeHtml(t("characterCount"))}</span><span>${escapeHtml(t("associatedImages"))}</span><span></span></div>
      ${chunks.map((item) => `
        <article class="text-media-review-row text-media-chunk-row">
          <span class="text-media-chunk-index">${Number(item.ordinal || 0) + 1}</span>
          <button type="button" class="text-media-review-primary" data-open-chunk="${Number(item.id)}"><strong>${escapeHtml(headingPath(item.text) || item.entry_title || t("chunk"))}</strong><small>${escapeHtml(contentExcerpt(item.text))}</small></button>
          <span class="text-media-review-source">${escapeHtml(item.document_title || item.document_id)}</span>
          <span class="text-media-review-stat"><strong>${Number(item.char_count || 0)}</strong><small>${escapeHtml(t("characters"))}</small></span>
          <span class="text-media-review-stat"><strong>${Number(item.media_count || 0)}</strong><small>${escapeHtml(t("images"))}</small></span>
          <button type="button" class="icon text-media-row-open" data-open-chunk="${Number(item.id)}" aria-label="${escapeHtml(t("viewDetails"))}">›</button>
        </article>`).join("")}` : renderEmpty(t("noKnowledgeChunks"));
    host.querySelectorAll("[data-open-chunk]").forEach((button) => {
      button.onclick = () => openChunkDetail(Number(button.dataset.openChunk));
    });
    renderPagination("text-media-chunk-pagination", chunkPage, async (direction) => {
      chunkPage.offset = Math.max(0, chunkPage.offset + direction * chunkPage.limit);
      await loadChunks();
    });
  }

  function renderAssets() {
    $("text-media-asset-grid").innerHTML = assets.length ? assets.map((asset) => `
      <article class="text-media-asset-card">
        <button type="button" class="text-media-asset-preview" data-preview-url="${escapeHtml(asset.content_url || asset.thumbnail_url)}" data-preview-caption="${escapeHtml(asset.original_name || t("knowledgeImage"))}"><img src="${escapeHtml(asset.thumbnail_url)}" alt="${escapeHtml(asset.original_name || t("knowledgeImage"))}" loading="lazy"></button>
        <div><strong>${escapeHtml(asset.original_name || asset.id)}</strong><small>${escapeHtml(`${asset.width} × ${asset.height} · ${fileSize(asset.size_bytes)}`)}</small><button type="button" class="ghost text-media-asset-detail" data-asset-id="${escapeHtml(asset.id)}">${escapeHtml(t("viewDetails"))}</button></div>
      </article>`).join("") : renderEmpty(t("noImageAssets"));
    document.querySelectorAll(".text-media-asset-preview").forEach((button) => {
      button.onclick = () => openImagePreview(button.dataset.previewUrl, button.dataset.previewCaption);
    });
    document.querySelectorAll(".text-media-asset-detail").forEach((button) => {
      button.onclick = () => openAssetDetail(button.dataset.assetId);
    });
  }

  async function loadDocuments() {
    if (!currentKnowledgeBase) return;
    const params = new URLSearchParams({ query: documentPage.query, offset: documentPage.offset, limit: documentPage.limit, sort: documentPage.sort });
    const result = await api(path(currentKnowledgeBase.id, `/documents?${params}`));
    documents = result.items || [];
    documentPage.total = Number(result.total || 0);
    if (documentPage.offset >= documentPage.total && documentPage.offset > 0) {
      documentPage.offset = Math.max(0, Math.floor(Math.max(0, documentPage.total - 1) / documentPage.limit) * documentPage.limit);
      return loadDocuments();
    }
    renderDocuments();
  }

  async function loadChunks() {
    if (!currentKnowledgeBase) return;
    const params = new URLSearchParams({ query: chunkPage.query, document_id: chunkPage.documentId, offset: chunkPage.offset, limit: chunkPage.limit, sort: chunkPage.sort });
    const result = await api(path(currentKnowledgeBase.id, `/chunks?${params}`));
    chunks = result.items || [];
    chunkPage.total = Number(result.total || 0);
    if (chunkPage.offset >= chunkPage.total && chunkPage.offset > 0) {
      chunkPage.offset = Math.max(0, Math.floor(Math.max(0, chunkPage.total - 1) / chunkPage.limit) * chunkPage.limit);
      return loadChunks();
    }
    renderChunks();
  }

  function renderDocumentFilter() {
    const select = $("text-media-chunk-document-filter");
    select.innerHTML = `<option value="">${escapeHtml(t("allDocuments"))}</option>${documentOptions.map((item) => `<option value="${escapeHtml(item.id)}"${chunkPage.documentId === String(item.id) ? " selected" : ""}>${escapeHtml(item.title || item.original_name || item.id)}</option>`).join("")}`;
  }

  async function refreshWorkspace() {
    if (!currentKnowledgeBase) return;
    const [optionData, entryData, assetData] = await Promise.all([
      api(path(currentKnowledgeBase.id, "/documents?limit=200&sort=title_asc")),
      api(path(currentKnowledgeBase.id, "/entries")),
      api(path(currentKnowledgeBase.id, "/assets")),
    ]);
    documentOptions = optionData.items || [];
    entries = entryData.items || [];
    assets = assetData.items || [];
    renderDocumentFilter();
    renderAssets();
    await Promise.all([loadDocuments(), loadChunks()]);
  }

  async function openDocumentDetail(documentId) {
    const item = await guarded(`text-media:${currentKnowledgeBase?.id}:document:${documentId}`, () => api(path(currentKnowledgeBase.id, `/documents/${encodeURIComponent(documentId)}`)));
    if (!item) return;
    const media = item.associated_media || [];
    showDetail(item.title || item.original_name || item.id, `
      <div class="memory-detail-actions"><button type="button" class="ghost danger" id="text-media-detail-delete-document">${escapeHtml(t("deleteDocument"))}</button></div>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("documentMetadata"))}</span><div class="memory-detail-meta-grid">${metaItem(t("sourceFile"), item.original_name)}${metaItem(t("documentId"), item.id)}${metaItem(t("parser"), item.parser_id)}${metaItem(t("fileSize"), fileSize(item.size_bytes))}${metaItem(t("chunkCountLabel"), item.chunk_count)}${metaItem(t("imageCountLabel"), item.image_count)}${metaItem(t("uploadedAt"), formatDate(item.created_at))}${metaItem("SHA-256", item.source_sha256)}</div></section>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("documentContent"))}</span>${collapsibleDetailContent(item.content)}</section>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("associatedImages"))}</span><div class="text-media-detail-media-list">${media.length ? media.map((asset) => `<article class="text-media-detail-media">${mediaPreviewButton(asset, "text-media-detail-thumbnail")}<span><strong>${escapeHtml(asset.original_name || asset.asset_id)}</strong><small>${escapeHtml(t("strengthRange", { minimum: Number(asset.minimum_strength || 0).toFixed(3), maximum: Number(asset.maximum_strength || 0).toFixed(3) }))}</small></span></article>`).join("") : renderEmpty(t("noImageAssets"))}</div></section>`);
    $("text-media-detail-delete-document").onclick = () => deleteDocuments([documentId]);
    $("text-media-detail-body").querySelectorAll(".text-media-detail-thumbnail").forEach((button) => {
      button.onclick = () => openImagePreview(button.dataset.previewUrl, button.dataset.previewCaption);
    });
  }

  async function openChunkDetail(chunkId) {
    const item = await guarded(`text-media:${currentKnowledgeBase?.id}:chunk:${chunkId}`, () => api(path(currentKnowledgeBase.id, `/chunks/${chunkId}`)));
    if (!item) return;
    const media = item.associated_media || [];
    showDetail(headingPath(item.text) || `${t("chunk")} #${item.id}`, `
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("chunkMetadata"))}</span><div class="memory-detail-meta-grid">${metaItem(t("chunkId"), item.id)}${metaItem(t("sourceDocument"), item.document_title || item.document_id)}${metaItem(t("entry"), item.entry_title || item.entry_id)}${metaItem(t("documentOrder"), Number(item.ordinal || 0) + 1)}${metaItem(t("characterCount"), item.char_count)}${metaItem("SHA-256", item.content_sha256)}</div></section>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("chunkContent"))}</span>${collapsibleDetailContent(item.text)}</section>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("associatedImages"))}</span><div class="text-media-detail-media-list">${media.length ? media.map((asset) => `<article class="text-media-detail-media">${mediaPreviewButton(asset, "text-media-detail-thumbnail")}<span><strong>${escapeHtml(asset.original_name || asset.asset_id)}</strong><small>${escapeHtml(t("mediaRelationStrength", { strength: Number(asset.semantic_strength ?? asset.relation_weight ?? 0).toFixed(3), scope: t(`${asset.scope || "document"}Level`) }))}</small><small>${escapeHtml(t("relationSummary", { weight: Number(asset.relation_weight || 0).toFixed(2), policy: relationPolicyLabel(asset.output_policy) }))}</small>${asset.caption ? `<small>${escapeHtml(asset.caption)}</small>` : ""}</span></article>`).join("") : renderEmpty(t("noMediaCandidates"))}</div></section>`);
    $("text-media-detail-body").querySelectorAll(".text-media-detail-thumbnail").forEach((button) => {
      button.onclick = () => openImagePreview(button.dataset.previewUrl, button.dataset.previewCaption);
    });
  }

  function relationTargetOptions(scope) {
    const items = scope === "document" ? documentOptions : entries;
    return items.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title || item.id)}</option>`).join("");
  }

  function relationPolicyLabel(policy) {
    return t({
      auto: "policyAuto",
      with_result: "policyWithResult",
      metadata_only: "policyMetadataOnly",
      disabled: "policyDisabled",
    }[policy] || "policyAuto");
  }

  async function openAssetDetail(assetId) {
    const item = await guarded(`text-media:${currentKnowledgeBase?.id}:asset:${assetId}`, () => api(path(currentKnowledgeBase.id, `/assets/${encodeURIComponent(assetId)}`)));
    if (!item) return;
    const relations = item.relations || [];
    const mediaMetadata = item.media_metadata || {};
    const descriptions = (item.media_descriptions || []).length
      ? item.media_descriptions.map((description) => ({ ...description }))
      : [{ media_description: mediaMetadata.media_description || filenameStem(item.original_name || ""), sort_order: 0, vector_status: mediaMetadata.vector_status || "missing" }];
    const descriptionEditor = `
      <form id="text-media-detail-descriptions-form" class="text-media-description-manager">
        <header><div><strong>${escapeHtml(t("mediaDescriptions"))}</strong><small>${escapeHtml(t("mediaDescriptionsAssetHint"))}</small></div><button type="button" class="ghost" id="text-media-detail-add-description"${descriptions.length >= 20 ? " disabled" : ""}>＋ ${escapeHtml(t("addMediaDescription"))}</button></header>
        <div id="text-media-detail-description-list">
          ${descriptions.map((description, index) => `
            <article class="text-media-description-row" data-description-index="${index}">
              <label class="text-media-description"><span>${escapeHtml(index === 0 ? t("primaryMediaDescription") : t("mediaDescriptionNumber", { number: index + 1 }))}</span><textarea rows="2" maxlength="2000" required>${escapeHtml(description.media_description || "")}</textarea></label>
              <div class="text-media-description-actions"><small>${escapeHtml(t(`mediaVectorStatus_${description.vector_status || "missing"}`))}</small>${index > 0 ? `<button type="button" class="ghost text-media-detail-make-primary">${escapeHtml(t("setAsPrimaryDescription"))}</button>` : ""}${descriptions.length > 1 ? `<button type="button" class="ghost danger text-media-detail-remove-description">${escapeHtml(t("remove"))}</button>` : ""}</div>
            </article>`).join("")}
        </div>
        <footer><button class="primary">${escapeHtml(t("saveAndRebuildMediaIndex"))}</button></footer>
      </form>`;
    showDetail(item.original_name || item.id, `
      <button type="button" class="text-media-detail-image" data-preview-url="${escapeHtml(item.content_url)}" data-preview-caption="${escapeHtml(item.original_name || t("knowledgeImage"))}"><img src="${escapeHtml(item.thumbnail_url)}" alt="${escapeHtml(item.original_name || t("knowledgeImage"))}"></button>
      <div class="memory-detail-actions"><button type="button" class="ghost danger" id="text-media-detail-delete-asset">${escapeHtml(t("deleteImage"))}</button></div>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("mediaMetadata"))}</span><div class="memory-detail-meta-grid">${metaItem(t("imageAsset"), item.id)}${metaItem(t("dimensions"), `${item.width} × ${item.height}`)}${metaItem(t("fileSize"), fileSize(item.size_bytes))}${metaItem(t("relationCount"), item.relation_count)}${metaItem(t("mediaDescriptionCount"), descriptions.length)}${metaItem(t("mediaVectorStatus"), t(`mediaVectorStatus_${mediaMetadata.vector_status || "missing"}`))}${metaItem(t("embeddingProviderFingerprint"), mediaMetadata.provider_fingerprint || "—")}${metaItem(t("uploadedAt"), formatDate(item.created_at))}${metaItem("SHA-256", item.sha256)}</div>${descriptionEditor}</section>
      <section class="memory-detail-section"><span class="memory-detail-section-title">${escapeHtml(t("mediaRelations"))}</span><div class="memory-detail-list">${relations.length ? relations.map((relation, index) => `<article class="memory-detail-list-item text-media-relation-card" data-relation-index="${index}"><header><div><strong>${escapeHtml(relation.target_title || relation.target_id)}</strong><small>${escapeHtml(t(`${relation.scope}Level`))}</small></div><button type="button" class="ghost danger text-media-detail-unlink">${escapeHtml(t("removeRelation"))}</button></header><p>${escapeHtml(t("relationSummary", { weight: Number(relation.relation_weight || 0).toFixed(2), policy: relationPolicyLabel(relation.output_policy) }))}</p>${relation.scope === "document" ? `<p>${escapeHtml(t("usesAssetMediaDescriptions", { count: descriptions.length }))}</p><label class="text-media-calibration-toggle"><input class="text-media-detail-calibration-enabled" type="checkbox"${relation.semantic_mode === "calibrated" ? " checked" : ""}><span>${escapeHtml(t("enableMediaSecondaryEmbedding"))}</span></label><button type="button" class="ghost text-media-detail-calibrate">${escapeHtml(t("recalibrateMediaLink"))}</button>` : ""}</article>`).join("") : renderEmpty(t("noMediaSemanticConnections"))}</div></section>
      <form id="text-media-detail-relation-form" class="memory-detail-section text-media-detail-relation-form"><span class="memory-detail-section-title">${escapeHtml(t("linkImage"))}</span><div class="memory-detail-edit-grid"><label><span>${escapeHtml(t("relationLevel"))}</span><select id="text-media-detail-relation-scope"><option value="document">${escapeHtml(t("documentLevel"))}</option><option value="entry">${escapeHtml(t("entryLevel"))}</option><option value="chunk">${escapeHtml(t("chunkLevel"))}</option></select></label><label><span>${escapeHtml(t("relationTarget"))}</span><select id="text-media-detail-relation-target">${relationTargetOptions("document")}</select><input id="text-media-detail-relation-chunk" class="hidden" type="number" min="1" placeholder="${escapeHtml(t("chunkId"))}"></label><label><span>${escapeHtml(t("outputPolicy"))}</span><select id="text-media-detail-relation-policy"><option value="auto">${escapeHtml(t("policyAuto"))}</option><option value="with_result">${escapeHtml(t("policyWithResult"))}</option><option value="metadata_only">${escapeHtml(t("policyMetadataOnly"))}</option><option value="disabled">${escapeHtml(t("policyDisabled"))}</option></select></label><label><span>${escapeHtml(t("relationWeight"))}</span><input id="text-media-detail-relation-weight" type="number" min="0" max="1" step="0.05" value="1"></label><label class="wide"><span>${escapeHtml(t("imageCaption"))}</span><input id="text-media-detail-relation-caption"></label></div><footer><button class="primary">${escapeHtml(t("saveRelation"))}</button></footer></form>`);
    const renderDetailDescriptions = () => {
      $("text-media-detail-description-list").innerHTML = descriptions.map((description, index) => `
        <article class="text-media-description-row" data-description-index="${index}">
          <label class="text-media-description"><span>${escapeHtml(index === 0 ? t("primaryMediaDescription") : t("mediaDescriptionNumber", { number: index + 1 }))}</span><textarea rows="2" maxlength="2000" required>${escapeHtml(description.media_description || "")}</textarea></label>
          <div class="text-media-description-actions"><small>${escapeHtml(t(`mediaVectorStatus_${description.vector_status || "missing"}`))}</small>${index > 0 ? `<button type="button" class="ghost text-media-detail-make-primary">${escapeHtml(t("setAsPrimaryDescription"))}</button>` : ""}${descriptions.length > 1 ? `<button type="button" class="ghost danger text-media-detail-remove-description">${escapeHtml(t("remove"))}</button>` : ""}</div>
        </article>`).join("");
      $("text-media-detail-add-description").disabled = descriptions.length >= 20;
      $("text-media-detail-description-list").querySelectorAll("textarea").forEach((input) => {
        input.oninput = () => {
          descriptions[Number(input.closest("[data-description-index]").dataset.descriptionIndex)].media_description = input.value;
        };
      });
      $("text-media-detail-description-list").querySelectorAll(".text-media-detail-make-primary").forEach((button) => {
        button.onclick = () => {
          const index = Number(button.closest("[data-description-index]").dataset.descriptionIndex);
          const [description] = descriptions.splice(index, 1);
          descriptions.unshift(description);
          renderDetailDescriptions();
        };
      });
      $("text-media-detail-description-list").querySelectorAll(".text-media-detail-remove-description").forEach((button) => {
        button.onclick = () => {
          if (descriptions.length <= 1) return;
          const index = Number(button.closest("[data-description-index]").dataset.descriptionIndex);
          descriptions.splice(index, 1);
          renderDetailDescriptions();
        };
      });
    };
    $("text-media-detail-add-description").onclick = () => {
      if (descriptions.length >= 20) return;
      descriptions.push({ media_description: "", sort_order: descriptions.length, vector_status: "missing" });
      renderDetailDescriptions();
    };
    renderDetailDescriptions();
    $("text-media-detail-descriptions-form").onsubmit = async (event) => {
      event.preventDefault();
      const values = descriptions.map((description) => description.media_description.trim());
      if (!values.length || values.some((value) => !value)) return toast(t("mediaDescriptionRequired"), true);
      await guarded(`text-media:${currentKnowledgeBase.id}:descriptions:${item.id}`, async () => {
        const queued = await api(path(currentKnowledgeBase.id, `/assets/${encodeURIComponent(item.id)}/media-descriptions`), { method: "PUT", body: JSON.stringify({ media_descriptions: values }) });
        trackQueuedJob?.(queued, {
          kind: "text_media_media_descriptions_update",
          databaseId: currentKnowledgeBase.id,
          databaseType: TYPE_ID,
        });
        await waitJob(queued.job_id);
        toast(t("mediaDescriptionsUpdated"));
        await refreshWorkspace();
        await openAssetDetail(item.id);
      }, { form: event.currentTarget, button: event.submitter, busyText: t("rebuildingMediaIndex") });
    };
    $("text-media-detail-body").querySelector(".text-media-detail-image").onclick = (event) => openImagePreview(event.currentTarget.dataset.previewUrl, event.currentTarget.dataset.previewCaption);
    $("text-media-detail-delete-asset").onclick = () => deleteAsset(item);
    $("text-media-detail-relation-scope").onchange = (event) => {
      const scope = event.currentTarget.value;
      $("text-media-detail-relation-target").classList.toggle("hidden", scope === "chunk");
      $("text-media-detail-relation-chunk").classList.toggle("hidden", scope !== "chunk");
      $("text-media-detail-relation-target").innerHTML = relationTargetOptions(scope);
    };
    $("text-media-detail-relation-form").onsubmit = async (event) => {
      event.preventDefault();
      const scope = $("text-media-detail-relation-scope").value;
      const target = scope === "chunk" ? $("text-media-detail-relation-chunk").value : $("text-media-detail-relation-target").value;
      if (!target) return toast(t("relationTargetRequired"), true);
      await guarded(`text-media:${currentKnowledgeBase.id}:relation:${item.id}`, async () => {
        await api(path(currentKnowledgeBase.id, `/relations/${scope}/${encodeURIComponent(target)}/assets/${encodeURIComponent(item.id)}`), { method: "PUT", body: JSON.stringify({ role: "illustration", relation_weight: Number($("text-media-detail-relation-weight").value || 1), caption: $("text-media-detail-relation-caption").value.trim(), output_policy: $("text-media-detail-relation-policy").value }) });
        toast(t("relationSaved"));
        await refreshWorkspace();
        await openAssetDetail(item.id);
      }, { form: event.currentTarget, button: event.submitter, busyText: t("loading") });
    };
    $("text-media-detail-body").querySelectorAll(".text-media-relation-card").forEach((card) => {
      const relation = relations[Number(card.dataset.relationIndex)];
      card.querySelector(".text-media-detail-unlink").onclick = async (event) => {
        await guarded(`text-media:${currentKnowledgeBase.id}:unlink:${item.id}:${relation.target_id}`, async () => {
          await api(path(currentKnowledgeBase.id, `/relations/${relation.scope}/${encodeURIComponent(relation.target_id)}/assets/${encodeURIComponent(item.id)}`), { method: "DELETE" });
          toast(t("relationRemoved"));
          await refreshWorkspace();
          await openAssetDetail(item.id);
        }, { button: event.currentTarget, busyText: t("loading") });
      };
      const calibrate = card.querySelector(".text-media-detail-calibrate");
      if (calibrate) calibrate.onclick = async () => {
        const enabled = card.querySelector(".text-media-detail-calibration-enabled").checked;
        await guarded(`text-media:${currentKnowledgeBase.id}:calibrate:${relation.target_id}:${item.id}`, async () => {
          const queued = await api(path(currentKnowledgeBase.id, `/document-media-relations/${encodeURIComponent(relation.target_id)}/${encodeURIComponent(item.id)}/semantic-calibration`), { method: "PUT", body: JSON.stringify({ enabled, media_description: descriptions[0]?.media_description || "" }) });
          trackQueuedJob?.(queued, {
            kind: "text_media_media_calibration",
            databaseId: currentKnowledgeBase.id,
            databaseType: TYPE_ID,
          });
          await waitJob(queued.job_id);
          toast(t("mediaCalibrationCompleted"));
          await refreshWorkspace();
          await openAssetDetail(item.id);
        }, { button: calibrate, busyText: t("calibratingMediaLink") });
      };
    });
  }

  async function deleteDocuments(ids) {
    const normalized = [...new Set(ids.map(String))];
    if (!normalized.length || !(await confirmDialog({ title: t("deleteDocumentsTitle"), message: t("deleteDocumentsConfirm", { count: normalized.length }), confirmText: t("delete"), danger: true }))) return;
    await guarded(`text-media:${currentKnowledgeBase?.id}:delete-documents`, async () => {
      const queued = await api(path(currentKnowledgeBase.id, "/documents/batch-delete"), { method: "POST", body: JSON.stringify({ document_ids: normalized }) });
      trackQueuedJob?.(queued, {
        kind: "text_media_document_delete",
        databaseId: currentKnowledgeBase.id,
        databaseType: TYPE_ID,
      });
      await waitJob(queued.job_id);
      normalized.forEach((id) => selectedDocumentIds.delete(id));
      closeDetail();
      toast(t("documentsDeleted", { count: normalized.length }));
      await refreshWorkspace();
      await loadDatabases(false);
    }, { button: $("text-media-document-batch-delete"), busyText: t("loading") });
  }

  async function deleteAsset(item) {
    if (!(await confirmDialog({ title: t("deleteImageTitle"), message: t("deleteImageConfirm", { name: item.original_name || item.id, count: item.relation_count || 0 }), confirmText: t("delete"), danger: true }))) return;
    await guarded(`text-media:${currentKnowledgeBase?.id}:delete:asset:${item.id}`, async () => {
      await api(path(currentKnowledgeBase.id, `/assets/${encodeURIComponent(item.id)}`), { method: "DELETE" });
      closeDetail();
      toast(t("knowledgeDeleted"));
      await refreshWorkspace();
      await loadDatabases(false);
    });
  }

  async function canonicalUploadImage(file) {
    if (!file) throw new Error(t("chooseImageFirst"));
    if (file.size > 25 * 1024 * 1024) throw new Error(t("imageTooLarge"));
    const bitmap = await createImageBitmap(file);
    try {
      const scale = Math.min(1, 1536 / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      canvas.getContext("2d", { alpha: false }).drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve, reject) => canvas.toBlob(
        (value) => value ? resolve(value) : reject(new Error(t("imageCompressFailed"))),
        "image/webp",
        0.8,
      ));
      return new File([blob], `${file.name.replace(/\.[^.]+$/, "") || "image"}.webp`, { type: "image/webp" });
    } finally {
      bitmap.close();
    }
  }

  async function waitJob(jobId, onProgress = null) {
    for (;;) {
      const job = await api(`/jobs/${encodeURIComponent(jobId)}`, { pageScoped: false });
      if (onProgress) onProgress(job);
      if (job.status === "completed") return job.result || {};
      if (["failed", "stopped", "cancelled"].includes(job.status)) {
        throw new Error(job.error || job.message || t("jobFailed"));
      }
      await sleep(650);
    }
  }

  async function downloadTransfer(token, fallbackName) {
    const response = await fetch(`/api/v1/knowledge-libraries/${TYPE_ID}/transfer-batches/exports/${encodeURIComponent(token)}`, { credentials: "same-origin" });
    if (!response.ok) throw new Error(await response.text() || response.statusText);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    const disposition = response.headers.get("content-disposition") || "";
    const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
    anchor.download = encoded ? decodeURIComponent(encoded) : fallbackName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function searchHasRerankResult() {
    const meta = searchResultCache.rerankMeta || {};
    return Boolean(
      meta.applied
      || meta.failed
      || meta.effective_enabled
      || meta.requested === true
    );
  }

  function applySearchRetrievalMode(mode, { clearResults = false } = {}) {
    searchRetrievalMode = ["standard", "text_only", "media_only"].includes(mode)
      ? mode
      : "standard";
    document.querySelectorAll("[data-text-media-retrieval-mode]").forEach((button) => {
      const active = button.dataset.textMediaRetrievalMode === searchRetrievalMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    const topK = $("text-media-search-top-k");
    if (topK) topK.disabled = searchRetrievalMode === "media_only";
    document.querySelectorAll(".text-media-search-text-parameter").forEach((element) => {
      element.classList.toggle("hidden", searchRetrievalMode === "media_only");
    });
    document.querySelectorAll(".text-media-search-media-parameter").forEach((element) => {
      element.classList.toggle("hidden", searchRetrievalMode === "text_only");
    });
    $("text-media-search-chunks-section")?.classList.toggle(
      "hidden", searchRetrievalMode === "media_only"
    );
    $("text-media-search-media-section")?.classList.toggle(
      "hidden", searchRetrievalMode === "text_only"
    );
    $("text-media-search-decisions-section")?.classList.toggle(
      "hidden", searchRetrievalMode === "text_only"
    );
    if (clearResults) {
      $("text-media-search-media").innerHTML = "";
      $("text-media-search-decisions").innerHTML = "";
      mediaDecisionsExpanded = false;
      searchResultCache = {
        embedding: [], rerank: [], rerankMeta: null, summary: null,
      };
      renderSearchChunks();
      scheduleSearchCollapseRefresh();
    }
  }

  function updateSearchResultViewControls() {
    const rerankAvailable = Boolean(currentKnowledgeBase?.rerank_binding?.available);
    const rerankButton = $("text-media-search-view-rerank");
    const rerankResultAvailable = !searchResultCache.summary || searchHasRerankResult();
    rerankButton?.classList.toggle("hidden", !rerankAvailable);
    if (rerankButton) {
      rerankButton.disabled = !rerankAvailable || !rerankResultAvailable;
      rerankButton.setAttribute("aria-hidden", rerankAvailable ? "false" : "true");
    }
    if ((!rerankAvailable || !rerankResultAvailable) && searchResultView === "rerank") {
      searchResultView = "embedding";
    }
    document.querySelectorAll("[data-text-media-search-view]").forEach((button) => {
      const active = button.dataset.textMediaSearchView === searchResultView;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
  }

  function setSearchResultView(view) {
    searchResultView = view === "rerank" && searchHasRerankResult()
      ? "rerank"
      : "embedding";
    updateSearchResultViewControls();
    renderSearchChunks();
  }

  function renderSearchChunks() {
    updateSearchResultViewControls();
    if (!searchResultCache.summary) {
      $("text-media-search-results").innerHTML = "";
      scheduleSearchCollapseRefresh();
      return;
    }
    const items = searchResultView === "rerank"
      ? searchResultCache.rerank
      : searchResultCache.embedding;
    $("text-media-search-results").innerHTML = items.length ? items.map((item) => `
      <article class="text-media-result text-media-result-collapsed">
        <button type="button" class="collapse-top-toggle text-media-result-collapse-top hidden">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="m6 15 6-6 6 6"></path></svg>
        </button>
        <div class="text-media-result-primary">
          <header><span class="rank">#${item.rank}</span><strong>ID ${Number(item.chunk_id)}</strong><strong>${escapeHtml(item.title)}</strong><span class="score ${retrievalScoreTierClass(item.score)}" title="${escapeHtml(t("textRelevanceScoreHint"))}">${escapeHtml(t("combinedScore"))} ${Number(item.score).toFixed(4)}</span><span class="score">${escapeHtml(t("denseSimilarity"))} ${Number(item.dense_score).toFixed(4)}</span><span class="score">${escapeHtml(t("textLexicalRelevance"))} ${Number(item.lexical_relevance || 0).toFixed(4)}</span><span class="score">RRF #${Number(item.rrf_rank || item.initial_rank || item.rank)}</span>${item.rerank_raw_score == null ? "" : `<span class="score">${escapeHtml(t("initialRetrievalRank"))} #${Number(item.initial_rank)}</span><span class="score">${escapeHtml(t("rerankRawScore"))} ${Number(item.rerank_raw_score).toFixed(4)}</span>`}</header>
          <p>${escapeHtml(item.text)}</p>
          <div class="text-media-result-evidence">${(item.associated_media || []).map((asset) => `<span class="pill" title="${escapeHtml(asset.media_description || "")}">${escapeHtml(asset.original_name || asset.asset_id.slice(0, 8))} · ${escapeHtml(t("mediaStrengthEvidence", { strength: Number(asset.semantic_strength ?? 1).toFixed(3), value: Number(asset.media_evidence_score).toFixed(3), rank: Number(asset.ranked_evidence_score).toFixed(3) }))}</span>`).join("")}</div>
        </div>
        <button type="button" class="library-expand-toggle text-media-result-toggle hidden" aria-expanded="false">
          <svg viewBox="0 0 240 24" aria-hidden="true" focusable="false"><circle cx="92" cy="12" r="3"></circle><circle cx="120" cy="12" r="3"></circle><circle cx="148" cy="12" r="3"></circle></svg>
        </button>
      </article>`).join("") : renderEmpty(t("noSearchResults"));
    scheduleSearchCollapseRefresh();
  }

  function searchChunkCollapsedHeight() {
    return Math.round(window.innerHeight / 3);
  }

  function applySearchChunkCollapseState(card) {
    const primary = card.querySelector(".text-media-result-primary");
    const toggle = card.querySelector(".text-media-result-toggle");
    const topToggle = card.querySelector(".text-media-result-collapse-top");
    if (!primary || !toggle || !topToggle) return;
    const collapsedHeight = searchChunkCollapsedHeight();
    card.style.setProperty("--text-media-result-collapsed-height", `${collapsedHeight}px`);
    card.style.setProperty("--text-media-result-expanded-height", `${Math.ceil(primary.scrollHeight)}px`);
    const overflowing = primary.scrollHeight > collapsedHeight + 24;
    card.classList.toggle("text-media-result-collapsible", overflowing);
    toggle.classList.toggle("hidden", !overflowing);
    if (!overflowing) {
      card.classList.remove("text-media-result-collapsed", "text-media-result-expanded");
      toggle.classList.remove("expanded");
      topToggle.classList.add("hidden");
      toggle.setAttribute("aria-expanded", "false");
      toggle.removeAttribute("title");
      toggle.removeAttribute("aria-label");
      return;
    }
    const expanded = card.classList.contains("text-media-result-expanded");
    card.classList.toggle("text-media-result-collapsed", !expanded);
    toggle.classList.toggle("expanded", expanded);
    toggle.setAttribute("aria-expanded", String(expanded));
    topToggle.classList.toggle("hidden", !expanded);
    const label = t(expanded ? "collapseLibraryCard" : "expandLibraryCard");
    toggle.title = label;
    toggle.setAttribute("aria-label", label);
    const collapseLabel = t("collapseLibraryCard");
    topToggle.title = collapseLabel;
    topToggle.setAttribute("aria-label", collapseLabel);
  }

  function applyMediaDecisionsCollapseState() {
    const panel = $("text-media-search-decisions-section");
    const content = $("text-media-search-decisions");
    const toggle = $("text-media-search-decisions-toggle");
    const topToggle = $("text-media-search-decisions-collapse-top");
    if (!panel || !content || !toggle || !topToggle) return;
    const decisions = [...content.querySelectorAll(".text-media-decision")];
    const overflowing = decisions.length > MEDIA_DECISION_COLLAPSED_COUNT;
    panel.classList.toggle("text-media-search-decisions-collapsible", overflowing);
    toggle.classList.toggle("hidden", !overflowing);
    if (!overflowing) {
      mediaDecisionsExpanded = false;
      panel.classList.remove("text-media-search-decisions-collapsed", "text-media-search-decisions-expanded");
      toggle.classList.remove("expanded");
      topToggle.classList.add("hidden");
      toggle.setAttribute("aria-expanded", "false");
      toggle.removeAttribute("title");
      toggle.removeAttribute("aria-label");
      return;
    }
    const lastVisibleDecision = decisions[MEDIA_DECISION_COLLAPSED_COUNT - 1];
    const contentRect = content.getBoundingClientRect();
    const decisionRect = lastVisibleDecision.getBoundingClientRect();
    const collapsedHeight = Math.ceil(decisionRect.bottom - contentRect.top);
    panel.style.setProperty("--text-media-decisions-collapsed-height", `${collapsedHeight}px`);
    panel.style.setProperty("--text-media-decisions-expanded-height", `${Math.ceil(content.scrollHeight)}px`);
    panel.classList.toggle("text-media-search-decisions-collapsed", !mediaDecisionsExpanded);
    panel.classList.toggle("text-media-search-decisions-expanded", mediaDecisionsExpanded);
    toggle.classList.toggle("expanded", mediaDecisionsExpanded);
    toggle.setAttribute("aria-expanded", String(mediaDecisionsExpanded));
    topToggle.classList.toggle("hidden", !mediaDecisionsExpanded);
    const label = t(mediaDecisionsExpanded ? "collapseLibraryCard" : "expandLibraryCard");
    toggle.title = label;
    toggle.setAttribute("aria-label", label);
    const collapseLabel = t("collapseLibraryCard");
    topToggle.title = collapseLabel;
    topToggle.setAttribute("aria-label", collapseLabel);
  }

  function applySearchCollapseStates() {
    searchCollapseFrame = 0;
    document.querySelectorAll("#text-media-search-results .text-media-result").forEach(
      applySearchChunkCollapseState,
    );
    applyMediaDecisionsCollapseState();
  }

  function scheduleSearchCollapseRefresh() {
    if (searchCollapseFrame) cancelAnimationFrame(searchCollapseFrame);
    searchCollapseFrame = requestAnimationFrame(applySearchCollapseStates);
  }

  function renderSearchResponse(result) {
    applySearchRetrievalMode(result.retrieval_mode || searchRetrievalMode);
    const rerank = result.rerank || {};
    const rerankStatus = $("text-media-search-rerank-status");
    if (rerankStatus) {
      rerankStatus.textContent = rerank.failed
        ? t("rerankFallbackStatus", { reason: rerank.fallback_reason || "—" })
        : rerank.applied
          ? t("rerankAppliedStatus", { count: Number(rerank.provider_candidates || 0), elapsed: Number(rerank.elapsed_ms || 0).toFixed(1) })
          : t("rerankDisabledStatus");
      rerankStatus.classList.toggle("warning", Boolean(rerank.failed));
    }
    const outputs = result.media_outputs || [];
    $("text-media-search-media").innerHTML = outputs.length ? outputs.map((asset) => `
      <figure class="text-media-search-media-card">
        <button type="button" class="text-media-search-media-preview" data-preview-url="${escapeHtml(asset.content_url || asset.thumbnail_url || "")}" data-preview-caption="${escapeHtml(asset.caption || asset.original_name || t("knowledgeImage"))}" aria-label="${escapeHtml(asset.caption || asset.original_name || t("knowledgeImage"))}">
          ${asset.thumbnail_url ? `<img src="${escapeHtml(asset.thumbnail_url)}" alt="${escapeHtml(asset.alt_text || asset.caption || asset.original_name || t("knowledgeImage"))}">` : ""}
          <figcaption><strong>${escapeHtml(asset.caption || asset.original_name || t("knowledgeImage"))}</strong><small>${escapeHtml(t("mediaConfidenceValue", { value: formatConfidence(asset.output_confidence) }))}</small></figcaption>
        </button>
      </figure>`).join("") : renderEmpty(t("noMediaOutput"));
    mediaDecisionsExpanded = false;
    $("text-media-search-decisions").innerHTML = result.media_decisions?.length
      ? result.media_decisions.map((decision) => {
        const confidenceThreshold = Number(decision.output_confidence_threshold ?? result.thresholds?.media_output_confidence_threshold ?? 0);
        const associationScore = Number(decision.association_score || 0);
        const outputConfidence = Number(decision.output_confidence || 0);
        const confidenceMet = decision.confidence_threshold_met ?? outputConfidence >= confidenceThreshold;
        const relevancePivot = Number(
          decision.media_relevance_pivot
          ?? decision.media_score_threshold
          ?? result.thresholds?.media_relevance_pivot
          ?? result.thresholds?.media_score_threshold
          ?? 0,
        );
        const thresholdAdjustment = Number(decision.evidence_threshold_adjustment || 0);
        const thresholdAdjustmentText = `${thresholdAdjustment >= 0 ? "+" : ""}${thresholdAdjustment.toFixed(3)}`;
        const thresholdSource = decision.media_relevance_pivot_source || decision.media_score_threshold_source || result.thresholds?.media_relevance_pivot_source || result.thresholds?.media_score_threshold_source || "request";
        const evidenceRows = (decision.evidence || []).map((item) => `<li>#${Number(item.rank)} · initial=${Number(item.initial_rank || item.rank)} · e=${Number(item.media_evidence_score || 0).toFixed(3)} · rerank=${item.rerank_raw_score == null ? "—" : Number(item.rerank_raw_score).toFixed(3)} · r=${item.threshold_rank_weight == null ? "—" : Number(item.threshold_rank_weight).toFixed(3)} · +=${Number(item.threshold_positive_contribution || 0).toFixed(3)} · −=${Number(item.threshold_negative_contribution || 0).toFixed(3)} · ${escapeHtml(item.threshold_effect || "neutral")}</li>`).join("");
        const frequencyTokenText = (decision.frequency_token_details || []).map((item) => (
          `${item.token}: DF=${Number(item.document_frequency || 0)}, IDF=${Number(item.idf || 0).toFixed(3)}, rarity=${Number(item.rarity || 0).toFixed(3)}${item.oov ? " (OOV)" : ""}`
        )).join(" · ") || "—";
        const matchedDescription = decision.matched_media_description
          ? `#${Number(decision.matched_media_description_sort_order || 0) + 1} / ${Number(decision.media_description_count || 1)} · ${decision.matched_media_description}`
          : "—";
        const descriptionDiagnostics = (decision.direct_relations || [])
          .filter((item) => item.media_description_id != null)
          .sort((left, right) => Number(left.media_description_sort_order || 0) - Number(right.media_description_sort_order || 0))
          .map((item) => {
            const embeddingScore = Number(item.embedding_calibrated_semantic_score ?? item.calibrated_semantic_score ?? 0).toFixed(3);
            const rerankScore = item.rerank_raw_score == null ? "—" : Number(item.rerank_raw_score).toFixed(3);
            const fusedScore = Number(item.calibrated_semantic_score || 0).toFixed(3);
            return `#${Number(item.media_description_sort_order || 0) + 1} ${escapeHtml(item.media_description || "")} · Embedding ${embeddingScore} · Rerank ${rerankScore} · ${escapeHtml(t("combinedScore"))} ${fusedScore}`;
          }).join("<br>") || "—";
        const visualCategoryText = Object.entries(
          decision.visual_intent_matched_categories || {},
        ).filter(([, terms]) => Array.isArray(terms) && terms.length)
          .map(([category, terms]) => `${t(`visualIntentCategory_${category}`)}: ${terms.join(", ")}`)
          .join(" · ") || "—";
        const visualDetectorText = decision.visual_intent_detector_version
          ? `${decision.visual_intent_detector_version} · ${String(decision.visual_intent_policy_fingerprint || "").slice(0, 12)}`
          : "—";
        const metric = (label, value, className = "") => `<div class="text-media-decision-metric${className ? ` ${className}` : ""}"><dt>${escapeHtml(label)}</dt><dd>${value}</dd></div>`;
        const summaryMetrics = [
          metric(t("outputConfidence"), `${formatConfidence(outputConfidence)} / ${confidenceThreshold.toFixed(3)} · ${escapeHtml(t(confidenceMet ? "thresholdMet" : "thresholdNotMet"))}`, "primary"),
          metric(t("mediaRelevancePivot"), `${relevancePivot.toFixed(3)} · ${escapeHtml(t(`thresholdSource_${thresholdSource}`))}`, "wide"),
          metric(t("qualifiedChunks"), Number(decision.qualifying_chunk_count || 0)),
          metric(t("weakeningChunks"), Number(decision.weakening_chunk_count || 0)),
          metric(t("bestEvidenceRank"), decision.best_rank || "—"),
          metric(t("maxMediaEvidence"), Number(decision.max_evidence_score || 0).toFixed(3)),
          metric(t("visualIntent"), escapeHtml(t(decision.visual_intent ? "yes" : "no"))),
          metric(t("matchedTokens"), escapeHtml((decision.matched_tokens || []).join(", ") || "—")),
        ].join("");
        const detailedMetrics = [
          metric(t("originalQuery"), escapeHtml(decision.original_query || result.query || "—"), "wide"),
          metric(t("effectiveMediaQuery"), escapeHtml(decision.effective_media_query || result.media_query || "—"), "wide"),
          metric(t("visualIntentKind"), escapeHtml(t(`visualIntentKind_${decision.visual_intent_kind || "none"}`))),
          metric(t("referenceSpan"), escapeHtml(decision.visual_intent_reference_span || "—")),
          metric(t("generationSpan"), escapeHtml(decision.visual_intent_generation_span || "—")),
          metric(t("ignoredOutputStyleTerms"), escapeHtml((decision.visual_intent_ignored_output_terms || []).join(", ") || "—"), "wide"),
          metric(t("visualIntentMatches"), escapeHtml(visualCategoryText), "wide"),
          metric(t("visualIntentDetector"), escapeHtml(visualDetectorText), "wide"),
          metric(t("queryProjection"), escapeHtml(t(decision.visual_intent_projection_applied ? "queryProjectionApplied" : "queryProjectionNotApplied"))),
          metric(t("confidenceAlgorithm"), escapeHtml(t(`confidenceAlgorithm_${decision.confidence_algorithm || "bound_chunk_grounded"}`))),
          metric(t("preThresholdConfidence"), Number(decision.pre_threshold_output_confidence || 0).toFixed(3)),
          metric(t("mediaAssociationScore"), associationScore.toFixed(3)),
          metric(t("rawRelevanceScore"), Number(decision.raw_relevance_score || 0).toFixed(3)),
          metric(t("pivotCalibratedRelevance"), Number(decision.pivot_calibrated_relevance_score || 0).toFixed(3)),
          metric(t("pivotCalibratedDirectRelevance"), Number(decision.pivot_calibrated_direct_relevance || 0).toFixed(3)),
          metric(t("structuralAttenuationFactor"), Number(decision.structural_attenuation_factor || 0).toFixed(3)),
          metric(t("formatAttenuationFactor"), Number(decision.format_attenuation_factor || 0).toFixed(3)),
          metric(t("contentAttenuationFactor"), Number(decision.content_attenuation_factor || 0).toFixed(3)),
          metric(t("outputConfidenceDelta"), Number(decision.output_confidence_delta || 0).toFixed(3)),
          metric(t("evidenceThresholdAdjustment"), thresholdAdjustmentText),
          metric(t("positiveSupport"), Number(decision.positive_support || 0).toFixed(3)),
          metric(t("negativePressure"), Number(decision.negative_pressure || 0).toFixed(3)),
          metric(t("negativeAttenuationFactor"), Number(decision.negative_attenuation_factor || 0).toFixed(3)),
          metric(t("tailNegativePressure"), Number(decision.tail_negative_pressure || 0).toFixed(3)),
          metric(t("thresholdEvidenceWindow"), `${Number(decision.threshold_evidence_count || 0)} / ${Number(decision.threshold_evidence_limit || 0)} · Z=${Number(decision.threshold_normalization_constant || 0).toFixed(3)}`),
          metric(t("groundingScore"), `${Number(decision.grounding_score || 0).toFixed(3)} → ${Number(decision.adjusted_grounding_score || 0).toFixed(3)}`),
          metric(t("mediaVectorSimilarity"), decision.media_vector_similarity == null ? "—" : Number(decision.media_vector_similarity).toFixed(3)),
          metric(t("calibratedSemanticScore"), Number(decision.calibrated_semantic_score || 0).toFixed(3)),
          metric(t("lexicalCoverage"), Number(decision.lexical_coverage || 0).toFixed(3)),
          metric(t("mediaFrequencyScope"), `${escapeHtml(decision.media_frequency_scope || "—")} · N=${Number(decision.media_frequency_corpus_size || 0)}`),
          metric(t("frequencyWeightedCompleteness"), Number(decision.frequency_weighted_completeness || 0).toFixed(3)),
          metric(t("frequencyInformativeness"), Number(decision.frequency_informativeness || 0).toFixed(3)),
          metric(t("distinctiveSupport"), Number(decision.distinctive_support || 0).toFixed(3)),
          metric(t("distinctiveMembershipSupport"), Number(decision.distinctive_membership_support || 0).toFixed(3)),
          metric(t("collectionSupport"), Number(decision.collection_support || 0).toFixed(3)),
          metric(t("collectionMembershipSupport"), Number(decision.collection_membership_support || 0).toFixed(3)),
          metric(t("distinctiveComponent"), Number(decision.distinctive_component || 0).toFixed(3)),
          metric(t("collectionComponent"), Number(decision.collection_component || 0).toFixed(3)),
          metric(t("groundingReliability"), Number(decision.grounding_reliability || 0).toFixed(3)),
          metric(t("descriptorRescue"), escapeHtml(t(decision.descriptor_rescue ? "yes" : "no"))),
          metric(t("frequencyTokenDetails"), escapeHtml(frequencyTokenText), "wide"),
          metric(t("mediaEvidenceScope"), escapeHtml(t(decision.media_evidence_scope === "asset_metadata_only" ? "assetMetadataOnly" : "assetBoundOnly"))),
          metric(t("mediaRankingScope"), escapeHtml(t(decision.media_ranking_scope === "asset_index" ? "assetIndexRanking" : "perAssetRanking"))),
          metric(t("boundChunkCount"), Number(decision.bound_chunk_count || 0)),
          metric(t("mediaCandidateChunkCount"), Number(decision.candidate_chunk_count || 0)),
          metric(t("mediaGate"), escapeHtml(t(decision.media_gate_open ? "open" : "closed"))),
          metric(t("subjectAnchor"), decision.subject_anchor_required ? escapeHtml(t(decision.subject_anchor_met ? "matched" : "notMatched")) : "—"),
          metric(t("rerankRawScore"), decision.rerank_raw_score == null ? "—" : Number(decision.rerank_raw_score).toFixed(3)),
          metric(t("rerankFusedScore"), decision.rerank_fused_score == null ? "—" : Number(decision.rerank_fused_score).toFixed(3)),
          metric(t("rerankConfidenceDelta"), decision.rerank_output_confidence_delta == null ? "—" : Number(decision.rerank_output_confidence_delta).toFixed(3)),
          metric(t("matchedMediaDescription"), escapeHtml(matchedDescription), "wide"),
          metric(t("mediaDescriptionDiagnostics"), descriptionDiagnostics, "wide"),
          ...(decision.confidence_algorithm === "unbound_asset_direct" ? [
            metric(t("unboundPreCompetitionConfidence"), Number(decision.unbound_pre_competition_output_confidence ?? decision.media_only_pre_competition_output_confidence ?? 0).toFixed(3)),
            metric(t("unboundDirectConfidence"), Number(decision.unbound_direct_score ?? decision.media_only_direct_score ?? 0).toFixed(3)),
            metric(t("unboundDirectReliability"), Number(decision.unbound_direct_reliability ?? decision.media_only_direct_reliability ?? 0).toFixed(3)),
            metric(t("unboundRelativeSpecificity"), Number(decision.unbound_relative_specificity ?? decision.media_only_relative_specificity ?? 0).toFixed(3)),
            metric(t("unboundStrongestCompetitor"), Number(decision.unbound_strongest_competitor_score ?? decision.media_only_strongest_competitor_score ?? 0).toFixed(3)),
            metric(t("unboundAdvantageMargin"), Number(decision.unbound_advantage_margin ?? decision.media_only_advantage_margin ?? 0).toFixed(3)),
            metric(t("unboundAdvantageWeight"), Number(decision.unbound_advantage_weight ?? decision.media_only_advantage_weight ?? 0).toFixed(3)),
            metric(t("unboundWinnerSupport"), Number(decision.unbound_winner_support ?? decision.media_only_winner_support ?? 0).toFixed(3)),
            metric(t("unboundCompetitionFloor"), Number(decision.unbound_competition_floor ?? decision.media_only_competition_floor ?? 0).toFixed(3)),
            metric(t("unboundCollectionMembershipWeight"), Number(decision.unbound_collection_membership_weight ?? decision.media_only_collection_membership_weight ?? 0).toFixed(3)),
            metric(t("unboundConfidenceFactor"), Number(decision.unbound_confidence_factor ?? decision.media_only_confidence_factor ?? 0).toFixed(3)),
            metric(t("unboundCollectionIntent"), escapeHtml(t((decision.unbound_collection_intent ?? decision.media_only_collection_intent) ? "yes" : "no"))),
            metric(t("unboundReliabilityTarget"), Number(decision.unbound_reliability_target ?? 0).toFixed(3)),
            metric(t("unboundSpecificityExponent"), Number(decision.unbound_specificity_exponent ?? 0).toFixed(3)),
            metric(t("unboundAdvantageTarget"), Number(decision.unbound_advantage_target ?? 0).toFixed(3)),
          ] : []),
          metric(t("calibrationStrengthSource"), escapeHtml(t(`calibrationStrengthSource_${decision.calibration_strength_source || "embedding_baseline"}`))),
          metric(t("rerankCalibrationFallbackReason"), decision.rerank_calibration_stale ? escapeHtml(t("rerankCalibrationStaleReason")) : "—"),
        ].join("");
        return `
        <article class="text-media-decision${decision.output ? " accepted" : " rejected"}">
          ${mediaPreviewButton(decision, "text-media-search-media-preview text-media-decision-preview")}
          <div><strong>${escapeHtml(decision.caption || decision.original_name || decision.asset_id)}</strong><small>${escapeHtml(t(`mediaDecision_${decision.reason}`))}</small></div>
          <dl class="text-media-decision-summary">${summaryMetrics}</dl>
          <details class="text-media-diagnostic-details"><summary>${escapeHtml(t("detailedDiagnosticFactors"))}</summary><dl>${detailedMetrics}</dl></details>
          ${evidenceRows ? `<details class="text-media-threshold-evidence"><summary>${escapeHtml(t("thresholdEvidenceDetails"))}</summary><ol>${evidenceRows}</ol></details>` : ""}
        </article>`;
      }).join("")
      : renderEmpty(t("noMediaCandidates"));
    scheduleSearchCollapseRefresh();
    searchResultCache = {
      embedding: result.baseline_items || result.items || [],
      rerank: result.items || [],
      rerankMeta: result.rerank || null,
      summary: result,
    };
    if (!searchHasRerankResult()) searchResultView = "embedding";
    renderSearchChunks();
  }

  function resetSearchTest() {
    searchGeneration += 1;
    const query = $("text-media-search-query").value;
    $("text-media-search-form").reset();
    $("text-media-search-query").value = query;
    applySearchRetrievalMode("standard");
    configureSearchRerankDefault();
    $("text-media-search-media").innerHTML = "";
    $("text-media-search-decisions").innerHTML = "";
    mediaDecisionsExpanded = false;
    scheduleSearchCollapseRefresh();
    searchResultView = "embedding";
    searchResultCache = {
      embedding: [],
      rerank: [],
      rerankMeta: null,
      summary: null,
    };
    renderSearchChunks();
    $("text-media-search-rerank-status").textContent = "";
  }

  function configureSearchRerankDefault() {
    const input = $("text-media-search-rerank");
    const hint = $("text-media-search-rerank-hint");
    if (!input) return;
    const available = Boolean(currentKnowledgeBase?.rerank_binding?.available);
    input.disabled = !available;
    input.checked = available;
    if (hint) hint.classList.toggle("hidden", available);
    updateSearchResultViewControls();
  }

  async function runSearch(button) {
    const form = $("text-media-search-form");
    const query = $("text-media-search-query").value.trim();
    if (!query) {
      $("text-media-search-query").focus();
      await refreshWorkspace();
      return true;
    }
    const generation = searchGeneration;
    const result = await guarded(`text-media:${currentKnowledgeBase?.id}:search`, async () => {
      const result = await api(path(currentKnowledgeBase.id, "/search"), {
        method: "POST",
        body: JSON.stringify({
          query,
          retrieval_mode: searchRetrievalMode,
          top_k: Number($("text-media-search-top-k").value || 10),
          media_output_confidence_threshold: Number($("text-media-search-confidence-threshold").value || 0),
          media_relevance_pivot: Number($("text-media-search-score-threshold").value || 0),
          max_media_outputs: Number($("text-media-search-max-outputs").value || 0),
          rerank: Boolean($("text-media-search-rerank")?.checked),
        }),
      });
      if (generation !== searchGeneration) return;
      renderSearchResponse(result);
      toast(t("knowledgeSearchToastSuccess", {
        count: (result.items || []).length,
        media: (result.media_outputs || []).length,
      }));
    }, { form, button, busyText: t("searching") });
    return result !== null;
  }

  async function refreshPage(pageId = "", options = {}) {
    const targetPage = pageId || currentTab;
    let refreshed = true;
    if (targetPage === "search") {
      resetSearchTest();
    } else if (targetPage === "settings") {
      const result = await guarded("text-media:type-settings:refresh", openTypeSettings, {
        button: options.button,
        busyText: t("loading"),
      });
      refreshed = result !== null;
    } else {
      const result = await guarded(`text-media:${currentKnowledgeBase?.id}:refresh`, refreshWorkspace, {
        button: options.button,
        busyText: t("loading"),
      });
      refreshed = result !== null;
    }
    if (refreshed) {
      toast(t("pageRefreshed"));
    }
  }

  $("text-media-page-refresh")?.addEventListener("click", async (event) => {
    await refreshPage(currentTab, { button: event.currentTarget });
  });

  $("text-media-create-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await guarded("text-media:create", async () => {
      await api(`/knowledge-libraries/${TYPE_ID}`, {
        method: "POST",
        body: JSON.stringify({
          id: $("text-media-create-id").value.trim(),
          name: $("text-media-create-name").value.trim(),
          description: $("text-media-create-description").value.trim(),
          provider_id: $("text-media-create-provider").value,
          rerank_provider_id: $("text-media-create-rerank-provider").value || null,
        }),
      });
      closeOverlay("text-media-create-modal");
      toast(t("knowledgeCreated"));
      await loadDatabases();
    }, { form: event.currentTarget, button: event.submitter, busyText: t("loading") });
  });

  $("text-media-visual-policy-groups")?.addEventListener("input", (event) => {
    const input = event.target.closest("[data-policy-term]");
    if (!input) return;
    const key = input.dataset.policyTerm;
    const index = Number(input.dataset.policyIndex);
    if (!VISUAL_INTENT_POLICY_KEYS.includes(key) || !Number.isInteger(index)) return;
    visualIntentPolicyDraft[key][index] = input.value;
  });

  $("text-media-visual-policy-groups")?.addEventListener("click", (event) => {
    const add = event.target.closest("[data-policy-add]");
    if (add) {
      const key = add.dataset.policyAdd;
      if (!VISUAL_INTENT_POLICY_KEYS.includes(key)) return;
      if ((visualIntentPolicyDraft[key] || []).length >= 256) return;
      visualIntentPolicyDraft[key].push("");
      renderVisualIntentPolicy();
      document.querySelector(`[data-policy-term="${key}"][data-policy-index="${visualIntentPolicyDraft[key].length - 1}"]`)?.focus();
      return;
    }
    const remove = event.target.closest("[data-policy-remove]");
    if (!remove) return;
    const key = remove.dataset.policyRemove;
    const index = Number(remove.dataset.policyIndex);
    if (!VISUAL_INTENT_POLICY_KEYS.includes(key) || !Number.isInteger(index)) return;
    visualIntentPolicyDraft[key].splice(index, 1);
    renderVisualIntentPolicy();
  });

  document.querySelectorAll("[data-retrieval-layout]").forEach((button) => {
    button.addEventListener("click", () => setRetrievalSettingsLayout(button.dataset.retrievalLayout));
  });

  $("text-media-retrieval-toggle")?.addEventListener("click", () => {
    retrievalSettingsExpanded = !retrievalSettingsExpanded;
    applyRetrievalCollapseState();
  });
  $("text-media-retrieval-collapse-top")?.addEventListener("click", () => {
    retrievalSettingsExpanded = false;
    applyRetrievalCollapseState();
  });

  $("text-media-retrieval-reset")?.addEventListener("click", async () => {
    if (!(await confirmDialog({
      title: t("confirmRetrievalDefaultsTitle"),
      message: t("confirmRetrievalDefaultsMessage"),
      confirmText: t("restoreRetrievalDefaults"),
      danger: true,
    }))) return;
    fillRetrievalSettings(DEFAULT_RETRIEVAL_SETTINGS);
    $("text-media-retrieval-status").textContent = t("retrievalDefaultsPending");
    scheduleRetrievalCollapseRefresh();
  });

  $("text-media-visual-policy-reset")?.addEventListener("click", () => {
    visualIntentPolicyDraft = cloneVisualIntentPolicy(visualIntentPolicyTypeDefaults);
    renderVisualIntentPolicy();
    $("text-media-visual-policy-status").textContent = t("visualIntentPolicyResetPending");
  });

  $("text-media-visual-policy-export")?.addEventListener("click", openVisualPolicyExport);

  $("text-media-visual-policy-import")?.addEventListener("click", () => {
    $("text-media-visual-policy-import-file")?.click();
  });

  $("text-media-visual-policy-import-file")?.addEventListener("change", async (event) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    const databaseId = $("text-media-edit-original-id")?.value || currentKnowledgeBase?.id;
    const body = new FormData();
    body.append("file", file, file.name);
    const preview = await guarded(`text-media:${databaseId}:visual-policy-import`, () => api(
      path(databaseId, "/visual-intent-policy/imports/inspect"),
      { method: "POST", body },
    ), { busyText: t("loading") });
    input.value = "";
    if (preview) openVisualPolicyImportPreview(preview);
  });

  document.querySelectorAll("[data-policy-modal-close]").forEach((button) => {
    button.addEventListener("click", () => closePolicyModal(button.dataset.policyModalClose));
  });

  $("text-media-visual-policy-export-modal")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closePolicyModal(event.currentTarget.id);
  });
  $("text-media-visual-policy-import-modal")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closePolicyModal(event.currentTarget.id);
  });
  $("text-media-visual-policy-export-form")?.addEventListener("change", updatePolicyTransferConfirmButtons);
  $("text-media-visual-policy-import-form")?.addEventListener("change", updatePolicyTransferConfirmButtons);

  $("text-media-visual-policy-export-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const categories = policyModalCategories("data-policy-export-category");
    if (!categories.length) return;
    const databaseId = $("text-media-edit-original-id")?.value || currentKnowledgeBase?.id;
    const filename = `${databaseId}-visual-intent-policy.csv`;
    let save;
    try {
      save = await createVisualPolicySaveTarget(filename);
    } catch (error) {
      if (error?.name !== "AbortError") toast(error?.message || String(error), true);
      return;
    }
    await guarded(`text-media:${databaseId}:visual-policy-export`, async () => {
      const response = await fetch(
        `/api/v1${path(databaseId, "/visual-intent-policy/export")}`,
        {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            categories,
            policy: normalizedVisualIntentPolicyDraft(),
          }),
        },
      );
      if (!response.ok) throw await responseError(response);
      await save(await response.blob());
      closePolicyModal("text-media-visual-policy-export-modal");
      toast(t("lexiconExported"));
    }, { form: event.currentTarget, button: event.submitter, busyText: t("loading") });
  });

  $("text-media-visual-policy-import-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const categories = policyModalCategories("data-policy-import-category");
    if (!categories.length || !visualIntentPolicyImportPreview) return;
    categories.forEach((key) => {
      visualIntentPolicyDraft[key] = [
        ...(visualIntentPolicyImportPreview.policy?.[key] || []),
      ];
    });
    renderVisualIntentPolicy();
    $("text-media-visual-policy-status").textContent = t(
      "visualIntentPolicyImportPending",
      { count: categories.length },
    );
    visualIntentPolicyImportPreview = null;
    closePolicyModal("text-media-visual-policy-import-modal");
  });

  window.addEventListener("resize", scheduleRetrievalCollapseRefresh);

  $("text-media-edit-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const originalId = $("text-media-edit-original-id").value;
    const nextId = $("text-media-edit-id").value.trim();
    const originalProviderId = $("text-media-edit-original-provider").value;
    const nextProviderId = $("text-media-edit-provider").value;
    const providerChanged = Boolean(nextProviderId && nextProviderId !== originalProviderId);
    if (nextId !== originalId && !(await confirmDialog({
      title: t("confirmKnowledgeRenameTitle"),
      message: t("confirmKnowledgeRenameMessage"),
      confirmText: t("savePlain"),
    }))) return;
    if (providerChanged && !(await confirmDialog({
      title: t("confirmLibraryEditSensitiveTitle"),
      message: t("textMediaProviderSwitchConfirm"),
      confirmText: t("savePlain"),
    }))) return;
    await guarded(`text-media:${originalId}:edit`, async () => {
      const visualIntentPolicy = normalizedVisualIntentPolicyDraft();
      const visualIntentPolicyChanged = JSON.stringify(visualIntentPolicy)
        !== JSON.stringify(cloneVisualIntentPolicy(visualIntentPolicyOriginal));
      const updated = await api(path(originalId), {
        method: "PATCH",
        body: JSON.stringify({
          id: nextId,
          name: $("text-media-edit-name").value.trim(),
          description: $("text-media-edit-description").value.trim(),
          uniform_media_strength: Number($("text-media-edit-uniform-strength").value),
          rerank_provider_id: $("text-media-edit-rerank-provider").value || null,
          retrieval_settings: {
            rrf_k: Number($("text-media-edit-rrf-k").value),
            rerank_candidate_limit: Number($("text-media-edit-rerank-candidate-limit").value),
            rerank_fusion_weight: Number($("text-media-edit-rerank-fusion-weight").value),
            rerank_rank_bonus_weight: Number($("text-media-edit-rerank-rank-bonus-weight").value),
            rerank_rank_reliability_exponent: Number($("text-media-edit-rerank-reliability-exponent").value),
            text_lexical_boost: Number($("text-media-edit-text-lexical-boost").value),
            media_candidate_limit: Number($("text-media-edit-candidate-limit").value),
            unbound_media_candidate_limit: Number($("text-media-edit-unbound-candidate-limit").value),
            media_relevance_pivot_fallback: Number($("text-media-edit-score-threshold-fallback").value),
            media_threshold_evidence_limit: Number($("text-media-edit-threshold-evidence-limit").value),
            media_threshold_rank_decay_exponent: Number($("text-media-edit-threshold-rank-exponent").value),
            media_threshold_negative_reliability_exponent: Number($("text-media-edit-threshold-negative-exponent").value),
            media_pivot_positive_blend: Number($("text-media-edit-threshold-reinforcement-weight").value),
            media_pivot_negative_weight: Number($("text-media-edit-threshold-weakening-weight").value),
            media_pivot_negative_attenuation_floor: Number($("text-media-edit-pivot-negative-floor").value),
            media_format_mismatch_factor: Number($("text-media-edit-format-mismatch-factor").value),
            media_content_mismatch_factor: Number($("text-media-edit-content-mismatch-factor").value),
            visual_intent_gate_enabled: $("text-media-edit-intent-gate").checked,
            media_semantic_floor: Number($("text-media-edit-semantic-floor").value),
            media_semantic_weight: Number($("text-media-edit-semantic-weight").value),
            media_lexical_boost: Number($("text-media-edit-lexical-boost").value),
            media_lexical_coverage_exponent: Number($("text-media-edit-lexical-exponent").value),
            media_lexical_common_floor: Number($("text-media-edit-lexical-common-floor").value),
            media_lexical_oov_penalty: Number($("text-media-edit-lexical-oov-penalty").value),
            media_distinctive_rarity_exponent: Number($("text-media-edit-distinctive-rarity-exponent").value),
            unbound_media_distinctive_boost: Number($("text-media-edit-unbound-distinctive-boost").value),
            unbound_media_collection_boost: Number($("text-media-edit-unbound-collection-boost").value),
            unbound_media_competition_floor: Number($("text-media-edit-unbound-competition-floor").value),
            unbound_media_reliability_target: Number($("text-media-edit-unbound-reliability-target").value),
            unbound_media_specificity_exponent: Number($("text-media-edit-unbound-specificity-exponent").value),
            unbound_media_advantage_target: Number($("text-media-edit-unbound-advantage-target").value),
            media_bound_distinctive_boost: Number($("text-media-edit-bound-distinctive-boost").value),
            media_bound_distinctive_rescue_min: Number($("text-media-edit-bound-distinctive-rescue-min").value),
            media_rank_decay_exponent: Number($("text-media-edit-rank-exponent").value),
            media_corroboration_weight: Number($("text-media-edit-corroboration-weight").value),
            media_corroboration_limit: Number($("text-media-edit-corroboration-limit").value),
          },
        }),
      });
      const savedKnowledgeBaseId = updated.id || nextId || originalId;
      if (visualIntentPolicyChanged) {
        const policyResult = await api(path(savedKnowledgeBaseId, "/visual-intent-policy"), {
          method: "PUT",
          body: JSON.stringify(visualIntentPolicy),
        });
        visualIntentPolicyOriginal = cloneVisualIntentPolicy(visualIntentPolicy);
        $("text-media-visual-policy-status").textContent = t(
          "visualIntentPolicyUpdated",
          { job: policyResult.job_id || "—" },
        );
      }
      closeOverlay("text-media-edit-modal");
      selectDatabase(savedKnowledgeBaseId, { databaseType: TYPE_ID, resetMemoryPage: false });
      if (providerChanged) {
        const tempJobId = addOptimisticTask?.({
          kind: "text_media_index_rebuild",
          databaseId: savedKnowledgeBaseId,
          databaseType: TYPE_ID,
          markLibraryConflict: true,
        });
        try {
          const queued = await api(path(savedKnowledgeBaseId, "/indexes/rebuild"), {
            method: "POST",
            body: JSON.stringify({
              provider_id: nextProviderId,
              reason: "library_edit_provider_switch",
            }),
          });
          trackQueuedJob?.(queued, {
            kind: "text_media_index_rebuild",
            databaseId: savedKnowledgeBaseId,
            databaseType: TYPE_ID,
            tempId: tempJobId,
          });
          toast(t("providerSwitchQueued", { library: savedKnowledgeBaseId }));
        } catch (error) {
          if (tempJobId) removeOptimisticTask?.(tempJobId);
          throw error;
        }
      } else {
        toast(t(visualIntentPolicyChanged ? "visualIntentPolicySaved" : "knowledgeSettingsSaved"));
      }
      await loadDatabases();
    }, {
      form: event.currentTarget,
      button: event.submitter,
      busyText: t("loading"),
      refreshCatalogOnConflict: true,
    });
  });

  document.querySelectorAll("[data-text-media-tab]").forEach((button) => {
    button.onclick = () => setTab(button.dataset.textMediaTab);
  });

  function setReviewMode(mode) {
    reviewMode = mode === "chunks" ? "chunks" : "documents";
    const documentsActive = reviewMode === "documents";
    $("text-media-document-mode").classList.toggle("active", documentsActive);
    $("text-media-document-mode").setAttribute("aria-selected", documentsActive ? "true" : "false");
    $("text-media-chunk-mode").classList.toggle("active", !documentsActive);
    $("text-media-chunk-mode").setAttribute("aria-selected", documentsActive ? "false" : "true");
    $("text-media-document-review").classList.toggle("active", documentsActive);
    $("text-media-chunk-review").classList.toggle("active", !documentsActive);
  }

  $("text-media-document-mode")?.addEventListener("click", () => setReviewMode("documents"));
  $("text-media-chunk-mode")?.addEventListener("click", () => setReviewMode("chunks"));
  $("text-media-document-select-page")?.addEventListener("change", (event) => {
    documents.forEach((item) => {
      if (event.currentTarget.checked) selectedDocumentIds.add(String(item.id));
      else selectedDocumentIds.delete(String(item.id));
    });
    renderDocuments();
  });
  $("text-media-document-batch-delete")?.addEventListener("click", () => deleteDocuments([...selectedDocumentIds]));
  $("text-media-document-search")?.addEventListener("input", (event) => {
    window.clearTimeout(documentSearchTimer);
    documentSearchTimer = window.setTimeout(async () => {
      documentPage.query = event.target.value.trim();
      documentPage.offset = 0;
      await guarded(`text-media:${currentKnowledgeBase?.id}:documents:filter`, loadDocuments);
    }, 260);
  });
  $("text-media-document-sort")?.addEventListener("change", async (event) => {
    documentPage.sort = event.target.value;
    documentPage.offset = 0;
    await guarded(`text-media:${currentKnowledgeBase?.id}:documents:sort`, loadDocuments);
  });
  $("text-media-document-page-size")?.addEventListener("change", async (event) => {
    documentPage.limit = Number(event.target.value || 20);
    documentPage.offset = 0;
    await guarded(`text-media:${currentKnowledgeBase?.id}:documents:size`, loadDocuments);
  });
  $("text-media-chunk-search")?.addEventListener("input", (event) => {
    window.clearTimeout(chunkSearchTimer);
    chunkSearchTimer = window.setTimeout(async () => {
      chunkPage.query = event.target.value.trim();
      chunkPage.offset = 0;
      await guarded(`text-media:${currentKnowledgeBase?.id}:chunks:filter`, loadChunks);
    }, 260);
  });
  $("text-media-chunk-document-filter")?.addEventListener("change", async (event) => {
    chunkPage.documentId = event.target.value;
    chunkPage.offset = 0;
    await guarded(`text-media:${currentKnowledgeBase?.id}:chunks:document`, loadChunks);
  });
  $("text-media-chunk-sort")?.addEventListener("change", async (event) => {
    chunkPage.sort = event.target.value;
    chunkPage.offset = 0;
    await guarded(`text-media:${currentKnowledgeBase?.id}:chunks:sort`, loadChunks);
  });
  $("text-media-chunk-page-size")?.addEventListener("change", async (event) => {
    chunkPage.limit = Number(event.target.value || 20);
    chunkPage.offset = 0;
    await guarded(`text-media:${currentKnowledgeBase?.id}:chunks:size`, loadChunks);
  });
  $("text-media-detail-close")?.addEventListener("click", closeDetail);
  $("text-media-detail-overlay")?.addEventListener("click", closeDetail);
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const uploadPicker = $("text-media-upload-picker-modal");
    if (uploadPicker && !uploadPicker.classList.contains("hidden")) {
      closeIngestPicker();
      return;
    }
    const imagePreview = $("text-media-image-preview");
    if (imagePreview && !imagePreview.classList.contains("hidden")) {
      closeImagePreview();
      return;
    }
    closeDetail();
  });

  $("text-media-ingest-open")?.addEventListener("click", openBatchIngest);
  document.querySelectorAll("[data-ingest-parameter]").forEach((input) => {
    input.addEventListener("input", () => syncIngestParameter(input));
    input.addEventListener("change", () => syncIngestParameter(input));
  });
  document.querySelectorAll("[data-ingest-picker]").forEach((button) => {
    button.addEventListener("click", () => openIngestPicker(button.dataset.ingestPicker));
  });
  $("text-media-upload-picker-close")?.addEventListener("click", closeIngestPicker);
  $("text-media-upload-picker-modal")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeIngestPicker();
  });
  $("text-media-upload-browse")?.addEventListener("click", () => {
    $(`text-media-ingest-${activeIngestPicker}`)?.click();
  });
  const ingestDropZone = $("text-media-upload-drop-zone");
  ["dragenter", "dragover"].forEach((type) => {
    ingestDropZone?.addEventListener(type, (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      ingestDropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((type) => {
    ingestDropZone?.addEventListener(type, (event) => {
      event.preventDefault();
      ingestDropZone.classList.remove("drag-over");
      if (type === "drop") acceptIngestFiles(activeIngestPicker, event.dataTransfer.files);
    });
  });
  ingestDropZone?.addEventListener("click", () => {
    $(`text-media-ingest-${activeIngestPicker}`)?.click();
  });
  ingestDropZone?.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    $(`text-media-ingest-${activeIngestPicker}`)?.click();
  });
  $("text-media-ingest-documents")?.addEventListener("change", (event) => {
    acceptIngestFiles("documents", event.target.files);
    event.target.value = "";
  });
  $("text-media-ingest-images")?.addEventListener("change", (event) => {
    acceptIngestFiles("images", event.target.files);
    event.target.value = "";
  });
  $("text-media-ingest-semantic-enabled")?.addEventListener("change", renderIngestImages);
  $("text-media-import-file")?.addEventListener("change", (event) => {
    $("text-media-import-name").textContent = event.target.files?.[0]?.name || "";
  });
  $("text-media-batch-import-file")?.addEventListener("change", (event) => {
    $("text-media-batch-import-name").textContent = event.target.files?.[0]?.name || "";
  });

  $("text-media-ingest-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await guarded(`text-media:${currentKnowledgeBase?.id}:ingest-batch`, async () => {
      if (!currentKnowledgeBase) throw new Error(t("chooseKnowledgeLibrary"));
      if (!ingestDocuments.length && !ingestImages.length) throw new Error(t("chooseDocumentOrImage"));
      const chunkTarget = Number($("text-media-ingest-chunk-target").value);
      const chunkOverlap = Number($("text-media-ingest-chunk-overlap").value);
      if (ingestDocuments.length && chunkOverlap > chunkTarget / 2) throw new Error(t("chunkOverlapTooLarge"));
      if (ingestImages.some((item) => item.bindingMode === "selected" && !item.documentIndexes.size)) {
        throw new Error(t("imageDocumentMappingRequired"));
      }
      const semanticEnabled = Boolean($("text-media-ingest-semantic-enabled").checked);
      if (ingestImages.some((item) => (
        !item.mediaDescriptions.length
        || item.mediaDescriptions.some((description) => !description.trim())
      ))) {
        throw new Error(t("mediaDescriptionRequired"));
      }
      const normalizedImages = [];
      for (const item of ingestImages) normalizedImages.push(await canonicalUploadImage(item.file));
      const body = new FormData();
      ingestDocuments.forEach((file) => body.append("documents[]", file, file.name));
      normalizedImages.forEach((file) => body.append("images[]", file, file.name));
      body.append("manifest", JSON.stringify({
        chunk_target: chunkTarget,
        chunk_overlap: chunkOverlap,
        embedding_batch_size: Number($("text-media-ingest-batch-size").value),
        concurrency: Number($("text-media-ingest-concurrency").value),
        max_retries: Number($("text-media-ingest-retries").value),
        media_semantic_calibration_enabled: semanticEnabled,
        images: ingestImages.map((item) => ({
          document_indexes: item.bindingMode === "all"
            ? ingestDocuments.map((_, index) => index)
            : item.bindingMode === "none"
            ? []
            : [...item.documentIndexes].sort((left, right) => left - right),
          media_description: item.mediaDescriptions[0].trim() || filenameStem(item.file.name),
          media_descriptions: item.mediaDescriptions.map((description) => description.trim()),
        })),
      }));
      const queued = await api(path(currentKnowledgeBase.id, "/ingest-batches"), { method: "POST", body });
      trackQueuedJob?.(queued, {
        kind: "text_media_ingest_batch",
        databaseId: currentKnowledgeBase.id,
        databaseType: TYPE_ID,
      });
      await waitJob(queued.job_id, (job) => {
        $("text-media-ingest-progress").textContent = t("batchProgress", {
          progress: Math.round(Number(job.progress || 0) * 100),
          message: job.message || t("batchIngesting"),
        });
      });
      closeOverlay("text-media-ingest-modal");
      clearIngestFiles();
      event.currentTarget.reset();
      toast(t("batchIngestCompleted"));
      await refreshWorkspace();
      await loadDatabases(false);
    }, { form: event.currentTarget, button: event.submitter, busyText: t("batchIngesting") });
  });

  document.querySelectorAll("[data-text-media-search-view]").forEach((button) => {
    button.addEventListener("click", () => {
      setSearchResultView(button.dataset.textMediaSearchView || "embedding");
    });
  });
  document.querySelectorAll("[data-text-media-retrieval-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      applySearchRetrievalMode(
        button.dataset.textMediaRetrievalMode || "standard",
        { clearResults: true },
      );
    });
  });
  $("text-media-search-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runSearch(event.submitter);
  });
  $("text-media-search-media")?.addEventListener("click", (event) => {
    const button = event.target instanceof Element
      ? event.target.closest(".text-media-search-media-preview")
      : null;
    if (!button) return;
    openImagePreview(button.dataset.previewUrl, button.dataset.previewCaption);
  });
  $("text-media-search-results")?.addEventListener("click", (event) => {
    const topToggle = event.target instanceof Element
      ? event.target.closest(".text-media-result-collapse-top")
      : null;
    const topCard = topToggle?.closest(".text-media-result");
    if (topToggle && topCard) {
      topCard.classList.remove("text-media-result-expanded");
      topCard.classList.add("text-media-result-collapsed");
      applySearchChunkCollapseState(topCard);
      return;
    }
    const toggle = event.target instanceof Element
      ? event.target.closest(".text-media-result-toggle")
      : null;
    const card = toggle?.closest(".text-media-result");
    if (!toggle || !card) return;
    const expanded = !card.classList.contains("text-media-result-expanded");
    card.classList.toggle("text-media-result-expanded", expanded);
    card.classList.toggle("text-media-result-collapsed", !expanded);
    applySearchChunkCollapseState(card);
  });
  $("text-media-search-decisions-toggle")?.addEventListener("click", () => {
    mediaDecisionsExpanded = !mediaDecisionsExpanded;
    applyMediaDecisionsCollapseState();
  });
  $("text-media-search-decisions-collapse-top")?.addEventListener("click", () => {
    mediaDecisionsExpanded = false;
    applyMediaDecisionsCollapseState();
  });
  $("text-media-search-decisions")?.addEventListener("toggle", () => {
    scheduleSearchCollapseRefresh();
  }, true);
  $("text-media-search-decisions")?.addEventListener("click", (event) => {
    const button = event.target instanceof Element
      ? event.target.closest(".text-media-search-media-preview")
      : null;
    if (!button) return;
    openImagePreview(button.dataset.previewUrl, button.dataset.previewCaption);
  });
  window.addEventListener("resize", scheduleSearchCollapseRefresh);

  $("text-media-single-export")?.addEventListener("click", async (event) => {
    const databaseId = $("text-media-single-export-library").value;
    if (!databaseId) return;
    await guarded(`text-media:${databaseId}:export`, async () => {
      if (!(await confirmDialog({ title: t("exportTmkb"), message: t("tmkbAllPlaintextWarning"), confirmText: t("exportTmkb") }))) return;
      const queued = await api(`/knowledge-libraries/${TYPE_ID}/transfer-batches/exports`, {
        method: "POST",
        body: JSON.stringify({ database_ids: [databaseId] }),
      });
      const result = await waitJob(queued.job_id);
      await downloadTransfer(result.download_token, `${databaseId}.tmkb`);
      toast(t("tmkbExported"));
    }, { button: event.currentTarget, busyText: t("exporting") });
  });

  $("text-media-batch-select-all")?.addEventListener("click", () => {
    document.querySelectorAll(".text-media-batch-library-check").forEach((input) => { input.checked = true; });
    updateBatchSelection();
  });

  $("text-media-batch-clear")?.addEventListener("click", () => {
    document.querySelectorAll(".text-media-batch-library-check").forEach((input) => { input.checked = false; });
    updateBatchSelection();
  });

  $("text-media-batch-export")?.addEventListener("click", async (event) => {
    const databaseIds = selectedBatchIds();
    if (!databaseIds.length) return;
    await guarded("text-media:batch-export", async () => {
      if (!(await confirmDialog({ title: t("exportSelectedLibraries"), message: t("tmkbAllPlaintextWarning"), confirmText: t("exportSelectedLibraries") }))) return;
      const queued = await api(`/knowledge-libraries/${TYPE_ID}/transfer-batches/exports`, {
        method: "POST",
        body: JSON.stringify({ database_ids: databaseIds }),
      });
      const result = await waitJob(queued.job_id);
      const extension = databaseIds.length === 1 ? "tmkb" : "tmkbs";
      await downloadTransfer(result.download_token, `text-media-libraries.${extension}`);
      toast(t(databaseIds.length === 1 ? "tmkbExported" : "tmkbsExported"));
    }, { button: event.currentTarget, busyText: t("exporting") });
  });

  $("text-media-import-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await guarded("text-media:inspect", async () => {
      const file = $("text-media-import-file").files?.[0];
      if (!file) throw new Error(t("chooseTmkbFirst"));
      const body = new FormData();
      body.append("file", file, file.name);
      importPreview = await api(`/knowledge-libraries/${TYPE_ID}/imports/inspect`, { method: "POST", body });
      const manifest = importPreview.manifest || {};
      $("text-media-import-preview").innerHTML = `
        <h3>${escapeHtml(t("tmkbPreview"))}</h3>
        <p class="warning-copy">${escapeHtml(t("tmkbAllPlaintextWarning"))}</p>
        <dl class="provider-meta"><dt>${escapeHtml(t("databaseId"))}</dt><dd>${escapeHtml(manifest.database_id || "—")}</dd><dt>${escapeHtml(t("displayName"))}</dt><dd>${escapeHtml(manifest.name || "—")}</dd><dt>${escapeHtml(t("providerFingerprint"))}</dt><dd><code>${escapeHtml(manifest.provider?.fingerprint || "—")}</code></dd></dl>
        <form id="text-media-import-commit-form" class="form-grid"><label><span>${escapeHtml(t("targetKnowledgeId"))}</span><input id="text-media-import-target-id" value="${escapeHtml(manifest.database_id || "")}" required pattern="[a-zA-Z0-9_-]+"></label><label><span>${escapeHtml(t("displayName"))}</span><input id="text-media-import-target-name" value="${escapeHtml(manifest.name || "")}"></label><footer class="wide"><button class="primary">${escapeHtml(t("importKnowledge"))}</button></footer></form>`;
      $("text-media-import-preview").classList.remove("hidden");
      $("text-media-import-commit-form").onsubmit = commitImport;
    }, { form: event.currentTarget, button: event.submitter, busyText: t("inspecting") });
  });

  async function commitImport(event) {
    event.preventDefault();
    await guarded("text-media:import", async () => {
      if (!importPreview?.import_token) return;
      const queued = await api(`/knowledge-libraries/${TYPE_ID}/imports/${encodeURIComponent(importPreview.import_token)}/commit`, {
        method: "POST",
        body: JSON.stringify({ target_id: $("text-media-import-target-id").value.trim(), name: $("text-media-import-target-name").value.trim() || null }),
      });
      await waitJob(queued.job_id);
      toast(t("tmkbImported"));
      importPreview = null;
      $("text-media-import-preview").classList.add("hidden");
      await loadDatabases();
      renderTransferLibraries();
    }, { form: event.currentTarget, button: event.submitter, busyText: t("importing") });
  }

  function validateBatchImportRows() {
    const rows = [...document.querySelectorAll(".text-media-batch-import-row")];
    const selected = rows.filter((row) => row.querySelector(".text-media-batch-import-check").checked);
    const targetIds = selected.map((row) => row.querySelector(".text-media-batch-target-id").value.trim());
    const duplicates = new Set(targetIds.filter((value, index) => value && targetIds.indexOf(value) !== index));
    const existing = new Set(typeLibraries().map((item) => item.id));
    let conflicts = 0;
    rows.forEach((row) => {
      const checked = row.querySelector(".text-media-batch-import-check").checked;
      const targetId = row.querySelector(".text-media-batch-target-id").value.trim();
      const invalid = checked && (!/^[a-zA-Z0-9_-]+$/.test(targetId) || duplicates.has(targetId) || existing.has(targetId));
      row.classList.toggle("has-conflict", invalid);
      row.querySelector(".text-media-batch-row-status").textContent = invalid
        ? t("targetIdConflict")
        : checked ? t("readyToImport") : t("notSelected");
      if (invalid) conflicts += 1;
    });
    const button = $("text-media-batch-import-commit");
    if (button) button.disabled = selected.length === 0 || conflicts > 0;
    const summary = $("text-media-batch-import-summary");
    if (summary) summary.textContent = t("batchImportSelectionSummary", { count: selected.length, conflicts });
  }

  function renderBatchImportPreview(preview) {
    const rows = (preview.libraries || []).map((item) => `
      <div class="text-media-batch-import-row${item.id_conflict ? " has-conflict" : ""}" data-source-id="${escapeHtml(item.source_id)}">
        <label class="text-media-batch-include"><input class="text-media-batch-import-check" type="checkbox" checked><span>${escapeHtml(t("includeInImport"))}</span></label>
        <div class="text-media-batch-source"><strong>${escapeHtml(item.name || item.source_id)}</strong><small>${escapeHtml(item.source_id)} · ${escapeHtml(t("chunkCount", { count: item.counts?.chunks || 0 }))}</small></div>
        <label><span>${escapeHtml(t("targetKnowledgeId"))}</span><input class="text-media-batch-target-id" value="${escapeHtml(item.source_id)}" pattern="[a-zA-Z0-9_-]+" required></label>
        <label><span>${escapeHtml(t("displayName"))}</span><input class="text-media-batch-target-name" value="${escapeHtml(item.name || item.source_id)}"></label>
        <span class="text-media-batch-row-status">${escapeHtml(item.id_conflict ? t("targetIdConflict") : t("readyToImport"))}</span>
      </div>`).join("");
    $("text-media-batch-import-preview").innerHTML = `
      <header><div><h3>${escapeHtml(t("tmkbsPreview"))}</h3><p id="text-media-batch-import-summary" class="subtle"></p></div></header>
      <p class="warning-copy">${escapeHtml(t("batchImportAtomicHint"))}</p>
      <form id="text-media-batch-import-commit-form" class="text-media-batch-import-list">${rows}<footer><button id="text-media-batch-import-commit" class="primary">${escapeHtml(t("importSelectedLibraries"))}</button></footer></form>`;
    $("text-media-batch-import-preview").classList.remove("hidden");
    document.querySelectorAll(".text-media-batch-import-check,.text-media-batch-target-id").forEach((input) => {
      input.addEventListener("input", validateBatchImportRows);
      input.addEventListener("change", validateBatchImportRows);
    });
    $("text-media-batch-import-commit-form").onsubmit = commitBatchImport;
    validateBatchImportRows();
  }

  $("text-media-batch-import-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await guarded("text-media:batch-inspect", async () => {
      const file = $("text-media-batch-import-file").files?.[0];
      if (!file) throw new Error(t("chooseTmkbsFirst"));
      const body = new FormData();
      body.append("file", file, file.name);
      batchImportPreview = await api(`/knowledge-libraries/${TYPE_ID}/transfer-batches/imports/inspect`, { method: "POST", body });
      renderBatchImportPreview(batchImportPreview);
    }, { form: event.currentTarget, button: event.submitter, busyText: t("inspecting") });
  });

  async function commitBatchImport(event) {
    event.preventDefault();
    await guarded("text-media:batch-import", async () => {
      if (!batchImportPreview?.import_token) return;
      const libraries = [...document.querySelectorAll(".text-media-batch-import-row")]
        .filter((row) => row.querySelector(".text-media-batch-import-check").checked)
        .map((row) => ({
          source_id: row.dataset.sourceId,
          target_id: row.querySelector(".text-media-batch-target-id").value.trim(),
          name: row.querySelector(".text-media-batch-target-name").value.trim() || null,
        }));
      const queued = await api(`/knowledge-libraries/${TYPE_ID}/transfer-batches/imports/${encodeURIComponent(batchImportPreview.import_token)}/commit`, {
        method: "POST",
        body: JSON.stringify({ libraries }),
      });
      await waitJob(queued.job_id);
      toast(t("tmkbsImported", { count: libraries.length }));
      batchImportPreview = null;
      $("text-media-batch-import-preview").classList.add("hidden");
      await loadDatabases();
      renderTransferLibraries();
    }, { form: event.currentTarget, button: event.submitter, busyText: t("importing") });
  }

  return {
    typeId: TYPE_ID,
    openCreate,
    openEdit,
    openWorkspace,
    openTypeSettings,
    refreshWorkspace,
    refreshPage,
    resetSearchTest,
    mountWorkspace,
    setPage: setTab,
    currentKnowledgeBase: () => currentKnowledgeBase,
  };
}
