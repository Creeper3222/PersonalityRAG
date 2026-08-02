const IMAGE_EXTENSIONS = new Set([".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"]);

export function createFileManagerController({
  $,
  state,
  t,
  api,
  toast,
  responseError,
  parseDownloadFilename,
  confirmDialog,
  escapeHtml,
  asyncGuard,
}) {
  state.files ||= {
    root: "",
    roots: [],
    path: "",
    parentPath: "",
    canGoUp: false,
    items: [],
    selectedPaths: new Set(),
    loading: false,
    uploading: false,
    moving: false,
    downloading: false,
    createType: "file",
    renameItem: null,
    movePaths: [],
    moveSelectedPath: "",
    moveNodes: new Map(),
    moveExpanded: new Set([""]),
    editPayload: null,
    editPassword: "",
    imageItems: [],
    imageIndex: -1,
    imageObjectUrl: "",
  };
  const files = state.files;

  function normalizePath(value = "") {
    return String(value || "")
      .replaceAll("\\", "/")
      .split("/")
      .filter((part) => part && part !== ".")
      .join("/");
  }

  function formatPath(value = "") {
    const normalized = normalizePath(value);
    return normalized ? `/${normalized}` : "/";
  }

  function joinPath(base = "", child = "") {
    const left = normalizePath(base);
    const right = normalizePath(child);
    return left && right ? `${left}/${right}` : right || left;
  }

  function extension(item = {}) {
    if (item.extension) return String(item.extension).toLowerCase();
    const name = String(item.name || item.path || "");
    const index = name.lastIndexOf(".");
    return index >= 0 ? name.slice(index).toLowerCase() : "";
  }

  function isImage(item = {}) {
    return item.preview_type === "image" || IMAGE_EXTENSIONS.has(extension(item));
  }

  function previewUrl(item) {
    const query = new URLSearchParams({ root: files.root, path: normalizePath(item.path) });
    return `/api/v1/files/preview?${query}`;
  }

  function formatSize(value, isDirectory = false) {
    if (isDirectory) return "-";
    const size = Number(value);
    if (!Number.isFinite(size) || size < 0) return "-";
    if (size < 1024) return `${size} B`;
    if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 ** 3) return `${(size / 1024 ** 2).toFixed(1)} MB`;
    return `${(size / 1024 ** 3).toFixed(2)} GB`;
  }

  function formatTime(value) {
    const date = value ? new Date(value) : null;
    return date && !Number.isNaN(date.getTime()) ? date.toLocaleString() : "-";
  }

  function validateBaseName(value) {
    const name = String(value || "").trim();
    if (!name) throw new Error(t("fileNameRequired"));
    if (name === "." || name === ".." || name.includes("/") || name.includes("\\")) {
      throw new Error(t("fileNameNoPath"));
    }
    if (/^[A-Za-z]:/.test(name) || /[<>:"|?*]/.test(name)) {
      throw new Error(t("fileNameInvalid"));
    }
    return name;
  }

  function fileIconVariant(item) {
    const suffix = extension(item);
    if (suffix === ".txt") return "text";
    if ([".json", ".py", ".md"].includes(suffix)) return "code";
    if (suffix === ".pdf") return "pdf";
    if ([".doc", ".docx"].includes(suffix)) return "word";
    return "generic";
  }

  function documentIcon(variant) {
    const shell = (className, body) => `<span class="file-icon file-icon--file ${className}" aria-hidden="true"><svg viewBox="0 0 24 24" focusable="false">${body}</svg></span>`;
    const page = '<path class="file-icon-page" d="M6.8 2.8h7.1l5.3 5.3v13.1H6.8c-1.1 0-2-.9-2-2V4.8c0-1.1.9-2 2-2Z"/><path class="file-icon-fold" d="M13.8 2.8v5.3c0 .6.5 1.1 1.1 1.1h5.3L13.8 2.8Z"/>';
    if (variant === "text") {
      return shell("file-icon--file-text", `${page}<path class="file-icon-mark" d="M8.2 11.6h7.4M8.2 14.4h7.4M8.2 17.2h6.1"/>`);
    }
    if (variant === "code") {
      return shell("file-icon--file-code", `${page}<path class="file-icon-mark" d="m10.2 11.3-2.2 2.2 2.2 2.2m3.6-4.4 2.2 2.2-2.2 2.2"/>`);
    }
    if (variant === "pdf") {
      return shell("file-icon--file-pdf", `${page}<text x="12" y="17.2" text-anchor="middle">PDF</text>`);
    }
    if (variant === "word") {
      return shell("file-icon--file-word", `${page}<text x="12" y="17.4" text-anchor="middle">W</text>`);
    }
    return shell("file-icon--file-generic", page);
  }

  function renderFileIcon(item) {
    if (item.is_directory) {
      return `<span class="file-icon file-icon--folder" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M2.8 7.2c0-1.3 1-2.3 2.3-2.3h5.3c.7 0 1.3.3 1.7.8l1.1 1.3h5.7c1.4 0 2.5 1.1 2.5 2.5v7.9c0 1.4-1.1 2.5-2.5 2.5H5.1c-1.3 0-2.3-1-2.3-2.3V7.2Z"/></svg></span>`;
    }
    if (isImage(item)) {
      if (item.requires_password) {
        return `<span class="file-icon file-icon--image file-icon--image-locked" aria-hidden="true">${toolIcon("image")}</span>`;
      }
      return `<span class="file-icon file-icon--image" aria-hidden="true"><img src="${escapeHtml(previewUrl(item))}" alt="" loading="lazy"></span>`;
    }
    return documentIcon(fileIconVariant(item));
  }

  function actionIcon(name) {
    const paths = {
      rename: '<path d="M4.6 7.1h6.8M4.6 12h5.2M4.6 16.9h4.2M13.4 17.6l5.3-5.3a1.7 1.7 0 0 0-2.4-2.4L11 15.2l-.7 3.1 3.1-.7Z"/>',
      move: '<path d="m12 3.8 3.2 3.2-3.2 3.2M12 3.8 8.8 7l3.2 3.2m0 3.6 3.2 3.2-3.2 3.2M12 13.8 8.8 17l3.2 3.2M3.8 12 7 8.8l3.2 3.2M3.8 12 7 15.2l3.2-3.2M13.8 12 17 8.8l3.2 3.2M13.8 12l3.2 3.2 3.2-3.2"/>',
      copy: '<rect x="8.2" y="8.2" width="10.2" height="10.2" rx="1.7"/><path d="M6 14.6h-.3c-1 0-1.7-.8-1.7-1.7V5.7C4 4.7 4.8 4 5.7 4h7.2c1 0 1.7.8 1.7 1.7V6"/>',
      download: '<path d="M12 4.6v10.1m-4.6-4.7 4.6 4.6 4.6-4.6M5.2 15.7v2.6c0 1 .8 1.9 1.9 1.9h9.8c1 0 1.9-.8 1.9-1.9v-2.6"/>',
      trash: '<path d="M4.8 7h14.4M9.4 7V4.8h5.2V7M7 7.2 7.7 20h8.6l.7-12.8M10 10.7v5.5m4-5.5v5.5"/>',
    };
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || ""}</svg>`;
  }

  function toolIcon(name) {
    const paths = {
      up: '<path d="m6 14 6-6 6 6"/><path d="M12 8v10"/>',
      plus: '<path d="M12 5v14M5 12h14"/>',
      refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
      upload: '<path d="M12 16V4m-4 4 4-4 4 4M5 15v4h14v-4"/>',
      image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="9" r="1.5"/><path d="m4 17 5-5 3 3 2-2 6 5"/>',
    };
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || ""}</svg>`;
  }

  function rootLabel(rootId) {
    if (files.roots.length === 1) return t("fileRootMerged");
    return rootId === "state" ? t("fileRootState") : t("fileRootProject");
  }

  function typeLabel(item) {
    let base = t("fileTypeGeneric");
    if (item.is_directory) base = t("fileTypeDirectory");
    else if (isImage(item)) base = t("fileTypeImage");
    else if (extension(item) === ".pdf") base = t("fileTypePdf");
    else if ([".doc", ".docx"].includes(extension(item))) base = t("fileTypeWord");
    else if (item.preview_type === "text") base = t("fileTypeText");
    else if (item.preview_type === "binary") base = t("fileTypeBinary");
    const flags = [base];
    if (item.requires_password) flags.push(t("fileRequiresAuth"));
    if (item.is_protected || (!item.is_directory && item.can_edit === false)) flags.push(t("fileReadonly"));
    return flags.join(" · ");
  }

  function selectedItems() {
    return files.items.filter((item) => files.selectedPaths.has(normalizePath(item.path)));
  }

  function renderSelection() {
    const selected = selectedItems();
    const count = selected.length;
    const hasProtected = selected.some((item) => item.is_protected);
    const busy = files.loading || files.uploading || files.moving || files.downloading;
    $("file-delete-selected")?.classList.toggle("hidden", count === 0);
    $("file-move-selected")?.classList.toggle("hidden", count === 0);
    $("file-download-selected")?.classList.toggle("hidden", count === 0);
    if ($("file-selected-count")) $("file-selected-count").textContent = String(count);
    if ($("file-delete-selected")) $("file-delete-selected").disabled = busy || hasProtected;
    if ($("file-move-selected")) $("file-move-selected").disabled = busy || hasProtected;
    if ($("file-download-selected")) $("file-download-selected").disabled = busy;
    const selectAll = $("file-select-all");
    if (selectAll) {
      selectAll.checked = files.items.length > 0 && count === files.items.length;
      selectAll.indeterminate = count > 0 && count < files.items.length;
      selectAll.disabled = busy || files.items.length === 0;
    }
  }

  function renderBreadcrumb() {
    const parts = normalizePath(files.path).split("/").filter(Boolean);
    let current = "";
    const crumbs = [`<button type="button" data-file-breadcrumb="">${escapeHtml(rootLabel(files.root))}</button>`];
    parts.forEach((part) => {
      current = joinPath(current, part);
      crumbs.push(`<span>/</span><button type="button" data-file-breadcrumb="${escapeHtml(current)}">${escapeHtml(part)}</button>`);
    });
    $("file-breadcrumb").innerHTML = crumbs.join("");
  }

  function renderRoots() {
    const select = $("file-root-select");
    select.innerHTML = files.roots.map((root) => `<option value="${escapeHtml(root.id)}">${escapeHtml(rootLabel(root.id))}</option>`).join("");
    select.value = files.root;
    $("file-root-switcher").classList.toggle("hidden", files.roots.length <= 1);
  }

  function renderRows() {
    const body = $("file-table-body");
    if (files.loading) {
      body.innerHTML = `<tr><td colspan="6" class="file-table-message">${escapeHtml(t("fileLoading"))}</td></tr>`;
      return;
    }
    if (!files.items.length) {
      body.innerHTML = `<tr><td colspan="6" class="file-table-message">${escapeHtml(t("fileEmpty"))}</td></tr>`;
      return;
    }
    body.innerHTML = files.items.map((item) => {
      const path = normalizePath(item.path);
      const selected = files.selectedPaths.has(path);
      const protectedActions = item.is_protected ? "disabled" : "";
      const badges = [
        item.requires_password ? `<span class="file-badge">${escapeHtml(t("fileRequiresAuth"))}</span>` : "",
        item.is_protected ? `<span class="file-badge file-badge-readonly">${escapeHtml(t("fileReadonly"))}</span>` : "",
      ].join("");
      return `<tr class="${selected ? "file-row-selected" : ""}">
        <td class="file-select-cell"><input class="file-checkbox" type="checkbox" data-file-select="${escapeHtml(path)}" ${selected ? "checked" : ""}></td>
        <td><button class="file-name-button" type="button" data-file-open="${escapeHtml(path)}">${renderFileIcon(item)}<span class="file-name-copy"><strong>${escapeHtml(item.name)}</strong><small>${badges}</small></span></button></td>
        <td>${escapeHtml(formatSize(item.size, item.is_directory))}</td>
        <td>${escapeHtml(formatTime(item.mtime))}</td>
        <td>${escapeHtml(typeLabel(item))}</td>
        <td class="file-actions-cell"><div class="file-row-actions">
          <button type="button" data-file-action="rename" data-file-path="${escapeHtml(path)}" title="${escapeHtml(t("fileRename"))}" ${protectedActions}>${actionIcon("rename")}</button>
          <button type="button" data-file-action="move" data-file-path="${escapeHtml(path)}" title="${escapeHtml(t("fileMove"))}" ${protectedActions}>${actionIcon("move")}</button>
          <button type="button" data-file-action="copy" data-file-path="${escapeHtml(path)}" title="${escapeHtml(t("fileCopyPath"))}">${actionIcon("copy")}</button>
          <button type="button" data-file-action="download" data-file-path="${escapeHtml(path)}" title="${escapeHtml(t("fileDownload"))}">${actionIcon("download")}</button>
          <button class="danger" type="button" data-file-action="delete" data-file-path="${escapeHtml(path)}" title="${escapeHtml(t("delete"))}" ${protectedActions}>${actionIcon("trash")}</button>
        </div></td>
      </tr>`;
    }).join("");
  }

  function renderFiles() {
    renderRoots();
    renderBreadcrumb();
    renderRows();
    $("file-current-path").textContent = `${rootLabel(files.root)} · ${formatPath(files.path)}`;
    $("file-up").disabled = files.loading || !files.canGoUp;
    $("file-refresh").disabled = files.loading;
    $("file-create").disabled = files.loading;
    $("file-upload-toggle").disabled = files.loading;
    renderSelection();
  }

  async function loadFiles({ root = files.root, path = files.path, silent = false } = {}) {
    files.loading = true;
    if (!silent) renderFiles();
    try {
      const query = new URLSearchParams({ root: root || "", path: normalizePath(path) });
      const payload = await api(`/files?${query}`);
      files.roots = Array.isArray(payload.roots) ? payload.roots : [];
      files.root = payload.root || root || files.roots[0]?.id || "project";
      files.path = normalizePath(payload.path);
      files.parentPath = normalizePath(payload.parent_path);
      files.canGoUp = Boolean(payload.can_go_up);
      files.items = Array.isArray(payload.items) ? payload.items : [];
      files.selectedPaths.clear();
    } catch (error) {
      if (error?.name === "AbortError") return;
      toast(error.message, true);
    } finally {
      files.loading = false;
      renderFiles();
    }
  }

  function itemForPath(path) {
    const normalized = normalizePath(path);
    return files.items.find((item) => normalizePath(item.path) === normalized) || null;
  }

  function openOverlay(id) {
    $(id)?.classList.remove("hidden");
  }

  function closeOverlay(id) {
    $(id)?.classList.add("hidden");
  }

  function requestPassword(item, action = t("fileOpen")) {
    return new Promise((resolve) => {
      const overlay = $("file-auth-modal");
      const form = $("file-auth-form");
      const input = $("file-auth-password");
      $("file-auth-message").textContent = t("fileAuthMessage", { action, path: formatPath(item.path) });
      input.value = "";
      overlay.classList.remove("hidden");
      const finish = (value) => {
        overlay.classList.add("hidden");
        form.onsubmit = null;
        overlay.onclick = null;
        overlay.querySelectorAll(".modal-dismiss").forEach((button) => { button.onclick = null; });
        resolve(value);
      };
      form.onsubmit = (event) => {
        event.preventDefault();
        if (!input.value) {
          toast(t("filePasswordRequired"), true);
          return;
        }
        finish(input.value);
      };
      overlay.querySelectorAll(".modal-dismiss").forEach((button) => { button.onclick = () => finish(null); });
      overlay.onclick = (event) => { if (event.target === overlay) finish(null); };
      setTimeout(() => input.focus(), 0);
    });
  }

  async function readItem(item) {
    let password = "";
    if (item.requires_password) {
      password = await requestPassword(item, t("fileOpen"));
      if (password === null) return null;
    }
    try {
      return await api("/files/read", {
        method: "POST",
        body: JSON.stringify({ root: files.root, path: item.path, password }),
        suppressUnauthorizedHandler: true,
      });
    } catch (error) {
      toast(error.message, true);
      return null;
    }
  }

  function openPreview(payload) {
    $("file-preview-title").textContent = payload.name || t("filePreview");
    $("file-preview-meta").textContent = `${formatSize(payload.size)} · ${formatTime(payload.mtime)}${payload.truncated ? ` · ${t("filePreviewTruncated")}` : ""}`;
    $("file-preview-content").textContent = payload.content || "";
    openOverlay("file-preview-modal");
  }

  function updateLineNumbers() {
    const count = Math.max(1, $("file-edit-content").value.split("\n").length);
    $("file-edit-lines").textContent = Array.from({ length: count }, (_, index) => index + 1).join("\n");
  }

  function openEditor(payload) {
    files.editPayload = payload;
    $("file-edit-title").textContent = payload.name || t("fileEdit");
    $("file-edit-path").textContent = formatPath(payload.path);
    $("file-edit-content").value = payload.content || "";
    updateLineNumbers();
    openOverlay("file-edit-modal");
    setTimeout(() => $("file-edit-content").focus(), 0);
  }

  async function openItem(item) {
    if (!item) return;
    if (item.is_directory) {
      await loadFiles({ path: item.path });
      return;
    }
    if (isImage(item)) {
      await openImageViewer(item);
      return;
    }
    if (item.preview_type !== "text") {
      toast(t("fileCannotPreview"), true);
      return;
    }
    const payload = await readItem(item);
    if (!payload) return;
    if (payload.can_edit) openEditor(payload);
    else openPreview(payload);
  }

  function setCreateType(type) {
    files.createType = type === "directory" ? "directory" : "file";
    document.querySelectorAll("[data-file-create-type]").forEach((button) => {
      button.classList.toggle("active", button.dataset.fileCreateType === files.createType);
    });
  }

  function openCreateModal() {
    setCreateType("file");
    $("file-create-name").value = "";
    openOverlay("file-create-modal");
    setTimeout(() => $("file-create-name").focus(), 0);
  }

  async function createItem() {
    return asyncGuard.run("files:create", async () => {
    try {
      const name = validateBaseName($("file-create-name").value);
      await api("/files/create", {
        method: "POST",
        body: JSON.stringify({ root: files.root, path: joinPath(files.path, name), type: files.createType }),
      });
      closeOverlay("file-create-modal");
      await loadFiles({ silent: true });
      toast(t("fileCreated"));
    } catch (error) {
      toast(error.message, true);
    }
    }, {
      form: $("file-create-form"),
      busyText: t("loading"),
    });
  }

  function toggleUpload() {
    $("file-upload-zone").classList.toggle("hidden");
  }

  async function upload(fileList) {
    const selected = Array.from(fileList || []);
    if (!selected.length || files.uploading) return;
    files.uploading = true;
    renderSelection();
    try {
      const form = new FormData();
      selected.forEach((file) => form.append("files", file, file.webkitRelativePath || file.name));
      const query = new URLSearchParams({ root: files.root, path: files.path });
      const result = await api(`/files/upload?${query}`, { method: "POST", body: form });
      await loadFiles({ silent: true });
      toast(t("fileUploaded", { count: result.uploaded || selected.length }));
    } catch (error) {
      toast(error.message, true);
    } finally {
      files.uploading = false;
      $("file-upload-input").value = "";
      renderFiles();
    }
  }

  async function deletePaths(paths, button = null) {
    if (!paths.length) return;
    return asyncGuard.run(`files:delete:${paths.join("|")}`, async () => {
    const confirmed = await confirmDialog({
      title: t("fileDeleteTitle"),
      message: t("fileDeleteConfirm", { count: paths.length }),
      confirmText: t("delete"),
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api("/files/delete", {
        method: "POST",
        body: JSON.stringify({ root: files.root, paths }),
      });
      await loadFiles({ silent: true });
      toast(t("fileDeleted", { count: paths.length }));
    } catch (error) {
      toast(error.message, true);
    }
    }, {
      button,
      busyText: t("loading"),
    });
  }

  function openRename(item) {
    if (!item || item.is_protected) return;
    files.renameItem = item;
    $("file-rename-name").value = item.name || "";
    openOverlay("file-rename-modal");
    setTimeout(() => $("file-rename-name").select(), 0);
  }

  async function renameItem() {
    return asyncGuard.run("files:rename", async () => {
    try {
      const name = validateBaseName($("file-rename-name").value);
      await api("/files/rename", {
        method: "POST",
        body: JSON.stringify({ root: files.root, path: files.renameItem.path, name }),
      });
      closeOverlay("file-rename-modal");
      await loadFiles({ silent: true });
      toast(t("fileRenamed"));
    } catch (error) {
      toast(error.message, true);
    }
    }, {
      form: $("file-rename-form"),
      busyText: t("loading"),
    });
  }

  async function loadMoveNode(path) {
    const normalized = normalizePath(path);
    if (files.moveNodes.has(normalized)) return;
    const query = new URLSearchParams({ root: files.root, path: normalized });
    const payload = await api(`/files?${query}`);
    files.moveNodes.set(normalized, (payload.items || []).filter((item) => item.is_directory));
  }

  function renderMoveTree() {
    const rows = [];
    const walk = (path, label, depth, item = null) => {
      const expanded = files.moveExpanded.has(path);
      const selected = files.moveSelectedPath === path;
      const protectedTarget = Boolean(item?.is_protected);
      rows.push(`<div class="file-move-tree-row ${selected ? "selected" : ""}" style="--depth:${depth}">
        <button type="button" class="file-move-toggle" data-file-move-toggle="${escapeHtml(path)}">${expanded ? "−" : "+"}</button>
        <button type="button" class="file-move-node" data-file-move-path="${escapeHtml(path)}" ${protectedTarget ? "disabled" : ""}>${renderFileIcon({ is_directory: true })}<span>${escapeHtml(label)}</span></button>
      </div>`);
      if (!expanded) return;
      const children = files.moveNodes.get(path) || [];
      children.forEach((child) => walk(normalizePath(child.path), child.name, depth + 1, child));
    };
    walk("", rootLabel(files.root), 0);
    $("file-move-tree").innerHTML = rows.join("");
    $("file-move-selected-path").textContent = formatPath(files.moveSelectedPath);
  }

  async function openMove(paths) {
    files.movePaths = paths;
    files.moveSelectedPath = "";
    files.moveNodes = new Map();
    files.moveExpanded = new Set([""]);
    try {
      await loadMoveNode("");
      renderMoveTree();
      openOverlay("file-move-modal");
    } catch (error) {
      toast(error.message, true);
    }
  }

  async function submitMove() {
    if (!files.movePaths.length || files.moving) return;
    files.moving = true;
    try {
      await api("/files/move", {
        method: "POST",
        body: JSON.stringify({ root: files.root, paths: files.movePaths, target_path: files.moveSelectedPath }),
      });
      closeOverlay("file-move-modal");
      await loadFiles({ silent: true });
      toast(t("fileMoved", { count: files.movePaths.length }));
    } catch (error) {
      toast(error.message, true);
    } finally {
      files.moving = false;
    }
  }

  async function copyPath(item) {
    const value = `${rootLabel(files.root)}:${formatPath(item.path)}`;
    try {
      await navigator.clipboard.writeText(value);
      toast(t("filePathCopied"));
    } catch {
      toast(t("fileCopyFailed"), true);
    }
  }

  async function fetchDownload(items, password = "") {
    const response = await fetch("/api/v1/files/download", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root: files.root, paths: items.map((item) => item.path), password }),
    });
    if (!response.ok) throw await responseError(response);
    return {
      blob: await response.blob(),
      name: parseDownloadFilename(response.headers.get("Content-Disposition"), items.length === 1 ? items[0].name : "files.zip"),
    };
  }

  async function downloadItems(items) {
    if (!items.length || files.downloading) return;
    let password = "";
    const sensitive = items.find((item) => item.requires_password);
    if (sensitive) {
      password = await requestPassword(sensitive, t("fileDownload"));
      if (password === null) return;
    }
    files.downloading = true;
    renderSelection();
    try {
      const result = await fetchDownload(items, password);
      const url = URL.createObjectURL(result.blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = result.name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast(t("fileDownloadReady"));
    } catch (error) {
      toast(error.message, true);
    } finally {
      files.downloading = false;
      renderSelection();
    }
  }

  async function imageUrl(item) {
    if (!item.requires_password) return { url: previewUrl(item), temporary: false };
    const password = await requestPassword(item, t("filePreview"));
    if (password === null) return null;
    const result = await fetchDownload([item], password);
    return { url: URL.createObjectURL(result.blob), temporary: true };
  }

  async function renderImageViewer() {
    const item = files.imageItems[files.imageIndex];
    if (!item) return;
    if (files.imageObjectUrl) {
      URL.revokeObjectURL(files.imageObjectUrl);
      files.imageObjectUrl = "";
    }
    try {
      const result = await imageUrl(item);
      if (!result) return closeImageViewer();
      $("file-image-viewer-image").src = result.url;
      if (result.temporary) files.imageObjectUrl = result.url;
      $("file-image-viewer-count").textContent = `${files.imageIndex + 1} / ${files.imageItems.length}`;
      $("file-image-viewer-name").textContent = item.name;
      $("file-image-prev").classList.toggle("hidden", files.imageItems.length <= 1);
      $("file-image-next").classList.toggle("hidden", files.imageItems.length <= 1);
    } catch (error) {
      toast(error.message, true);
      closeImageViewer();
    }
  }

  async function openImageViewer(item) {
    files.imageItems = files.items.filter((candidate) => isImage(candidate));
    files.imageIndex = Math.max(0, files.imageItems.findIndex((candidate) => candidate.path === item.path));
    $("file-image-viewer").classList.remove("hidden");
    document.body.classList.add("file-image-viewer-open");
    await renderImageViewer();
  }

  function closeImageViewer() {
    $("file-image-viewer").classList.add("hidden");
    document.body.classList.remove("file-image-viewer-open");
    $("file-image-viewer-image").removeAttribute("src");
    if (files.imageObjectUrl) URL.revokeObjectURL(files.imageObjectUrl);
    files.imageObjectUrl = "";
  }

  async function saveEditor() {
    if (!files.editPayload) return;
    return asyncGuard.run(`files:write:${files.editPayload.path}`, async () => {
    const confirmed = await confirmDialog({
      title: t("fileSaveTitle"),
      message: t("fileSaveConfirm", { path: formatPath(files.editPayload.path) }),
      confirmText: t("savePlain"),
    });
    if (!confirmed) return;
    try {
      await api("/files/write", {
        method: "POST",
        body: JSON.stringify({
          root: files.root,
          path: files.editPayload.path,
          content: $("file-edit-content").value,
          password: files.editPassword,
        }),
      });
      closeOverlay("file-edit-modal");
      await loadFiles({ silent: true });
      toast(t("fileSaved"));
    } catch (error) {
      toast(error.message, true);
    }
    }, {
      button: $("file-edit-save"),
      busyText: t("loading"),
    });
  }

  function bind() {
    $("file-root-select")?.addEventListener("change", (event) => loadFiles({ root: event.target.value, path: "" }));
    $("file-up")?.addEventListener("click", () => loadFiles({ path: files.parentPath }));
    $("file-refresh")?.addEventListener("click", () => loadFiles({ silent: true }));
    $("file-create")?.addEventListener("click", openCreateModal);
    $("file-upload-toggle")?.addEventListener("click", toggleUpload);
    $("file-upload-input")?.addEventListener("change", (event) => upload(event.target.files));
    $("file-upload-zone")?.addEventListener("dragover", (event) => { event.preventDefault(); event.currentTarget.classList.add("drag-over"); });
    $("file-upload-zone")?.addEventListener("dragleave", (event) => event.currentTarget.classList.remove("drag-over"));
    $("file-upload-zone")?.addEventListener("drop", (event) => {
      event.preventDefault();
      event.currentTarget.classList.remove("drag-over");
      upload(event.dataTransfer?.files);
    });
    $("file-select-all")?.addEventListener("change", (event) => {
      files.selectedPaths.clear();
      if (event.target.checked) files.items.forEach((item) => files.selectedPaths.add(normalizePath(item.path)));
      renderFiles();
    });
    $("file-delete-selected")?.addEventListener("click", (event) => deletePaths(selectedItems().map((item) => item.path), event.currentTarget));
    $("file-move-selected")?.addEventListener("click", () => openMove(selectedItems().map((item) => item.path)));
    $("file-download-selected")?.addEventListener("click", () => downloadItems(selectedItems()));
    $("file-create-form")?.addEventListener("submit", (event) => { event.preventDefault(); createItem(); });
    $("file-rename-form")?.addEventListener("submit", (event) => { event.preventDefault(); renameItem(); });
    $("file-move-form")?.addEventListener("submit", (event) => { event.preventDefault(); submitMove(); });
    $("file-edit-save")?.addEventListener("click", saveEditor);
    $("file-edit-content")?.addEventListener("input", updateLineNumbers);
    $("file-edit-content")?.addEventListener("scroll", (event) => { $("file-edit-lines").scrollTop = event.target.scrollTop; });
    $("file-edit-content")?.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveEditor();
      }
    });
    $("file-image-close")?.addEventListener("click", closeImageViewer);
    $("file-image-prev")?.addEventListener("click", () => {
      files.imageIndex = (files.imageIndex - 1 + files.imageItems.length) % files.imageItems.length;
      renderImageViewer();
    });
    $("file-image-next")?.addEventListener("click", () => {
      files.imageIndex = (files.imageIndex + 1) % files.imageItems.length;
      renderImageViewer();
    });
    document.querySelectorAll("[data-file-create-type]").forEach((button) => button.addEventListener("click", () => setCreateType(button.dataset.fileCreateType)));
    document.querySelectorAll("[data-file-modal-close]").forEach((button) => button.addEventListener("click", () => closeOverlay(button.dataset.fileModalClose)));
    $("file-breadcrumb")?.addEventListener("click", (event) => {
      const button = event.target.closest("[data-file-breadcrumb]");
      if (button) loadFiles({ path: button.dataset.fileBreadcrumb });
    });
    $("file-table-body")?.addEventListener("change", (event) => {
      const input = event.target.closest("[data-file-select]");
      if (!input) return;
      if (input.checked) files.selectedPaths.add(input.dataset.fileSelect);
      else files.selectedPaths.delete(input.dataset.fileSelect);
      renderFiles();
    });
    $("file-table-body")?.addEventListener("click", async (event) => {
      const open = event.target.closest("[data-file-open]");
      if (open) return openItem(itemForPath(open.dataset.fileOpen));
      const action = event.target.closest("[data-file-action]");
      if (!action || action.disabled) return;
      const item = itemForPath(action.dataset.filePath);
      if (!item) return;
      if (action.dataset.fileAction === "rename") openRename(item);
      if (action.dataset.fileAction === "move") openMove([item.path]);
      if (action.dataset.fileAction === "copy") copyPath(item);
      if (action.dataset.fileAction === "download") downloadItems([item]);
      if (action.dataset.fileAction === "delete") deletePaths([item.path], action);
    });
    $("file-move-tree")?.addEventListener("click", async (event) => {
      const toggle = event.target.closest("[data-file-move-toggle]");
      if (toggle) {
        const path = normalizePath(toggle.dataset.fileMoveToggle);
        if (files.moveExpanded.has(path)) files.moveExpanded.delete(path);
        else {
          files.moveExpanded.add(path);
          try { await loadMoveNode(path); } catch (error) { toast(error.message, true); }
        }
        renderMoveTree();
        return;
      }
      const node = event.target.closest("[data-file-move-path]");
      if (node && !node.disabled) {
        files.moveSelectedPath = normalizePath(node.dataset.fileMovePath);
        renderMoveTree();
      }
    });
  }

  bind();
  return { loadFiles, renderFiles, closeImageViewer };
}
