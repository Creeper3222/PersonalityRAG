const EXPANDED_GRAPH_LIMITS = Object.freeze({
  limit_memories: 24,
  limit_entries: 80,
  limit_nodes: 80,
  limit_edges: 120,
});

const GRAPH_TYPE_COLORS = Object.freeze({
  person: "#59c2ff",
  topic: "#5d35c7",
  fact: "#d5a20a",
  summary: "var(--graph-summary)",
  other: "#8492a6",
});

export function createGraphController({ $, state, t, selectedDatabaseApi, statCards, toast, escapeHtml, asyncGuard }) {
  const graphView = {
    payload: null,
    index: null,
    selectedNodeId: null,
    selectedMemoryId: null,
    rendererReady: false,
  };

  function numericId(value) {
    if (value === null || value === undefined || String(value).trim() === "") return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
  }

  function graphTypeColor(type) {
    const color = GRAPH_TYPE_COLORS[type] || GRAPH_TYPE_COLORS.other;
    if (color !== "var(--graph-summary)") return color;
    return getComputedStyle(document.documentElement).getPropertyValue("--graph-summary").trim() || "#2f86bd";
  }

  function normalizeSnapshot(snapshot = {}) {
    return {
      nodes: (snapshot.nodes || []).map((node) => ({
        ...node,
        id: numericId(node.id),
      })).filter((node) => node.id !== null),
      edges: (snapshot.edges || []).map((edge) => ({
        ...edge,
        id: numericId(edge.id) ?? `${edge.source}:${edge.target}:${edge.memory_id ?? edge.source_memory_id ?? ""}`,
        source: numericId(edge.source),
        target: numericId(edge.target),
        memory_id: numericId(edge.memory_id ?? edge.source_memory_id) ?? 0,
      })).filter((edge) => edge.source !== null && edge.target !== null),
      entries: (snapshot.entries || []).map((entry) => ({
        ...entry,
        memory_id: numericId(entry.memory_id ?? entry.source_memory_id) ?? 0,
        node_ids: (entry.node_ids || []).map(numericId).filter((id) => id !== null),
      })),
      memories: (snapshot.memories || []).map((memory) => ({
        ...memory,
        memory_id: numericId(memory.memory_id ?? memory.id),
      })).filter((memory) => memory.memory_id !== null),
    };
  }

  function graphSummary(snapshot) {
    const nodeTypeBreakdown = {};
    const relationBreakdown = {};
    snapshot.nodes.forEach((node) => {
      const key = node.type || "other";
      nodeTypeBreakdown[key] = (nodeTypeBreakdown[key] || 0) + 1;
    });
    snapshot.edges.forEach((edge) => {
      const key = edge.relation_type || "related";
      relationBreakdown[key] = (relationBreakdown[key] || 0) + 1;
    });
    return {
      node_type_breakdown: nodeTypeBreakdown,
      relation_breakdown: relationBreakdown,
    };
  }

  function normalizePayload(data = {}, requestPayload = null) {
    const snapshot = normalizeSnapshot(data.snapshot || {});
    const retrievalItems = Array.isArray(data.retrieval)
      ? data.retrieval
      : Array.isArray(data.retrieval?.items)
        ? data.retrieval.items
        : [];
    return {
      ...data,
      enabled: data.enabled !== false,
      mode: data.mode === "full_graph"
        ? "overview"
        : data.mode || (requestPayload?.memory_id ? "memory_focus" : requestPayload ? "query" : "overview"),
      snapshot,
      summary: data.summary || graphSummary(snapshot),
      retrieval: { ...(Array.isArray(data.retrieval) ? {} : data.retrieval), items: retrievalItems },
      requested_memory_id: numericId(requestPayload?.memory_id),
    };
  }

  function setCanvasMessage(message) {
    const element = $("graph-canvas-state");
    if (!element) return;
    element.textContent = message || "";
    element.classList.toggle("hidden", !message);
  }

  function ensureRenderer() {
    const renderer = window.Graph2D;
    const container = $("graph-canvas");
    if (!renderer || typeof renderer.init !== "function") {
      throw new Error(t("graphRendererUnavailable"));
    }
    if (renderer._initialized && renderer.container !== container) renderer.destroy();
    if (!renderer._initialized) {
      renderer.init(container, {
        onNodeClick: (nodeId) => selectGraphNode(nodeId),
        onNodeDblClick: (nodeId) => selectGraphNode(nodeId),
        onBackgroundClick: () => clearGraphSelection(),
      });
    }
    graphView.rendererReady = true;
    return renderer;
  }

  function addExpandedLimits(target) {
    return { ...target, ...EXPANDED_GRAPH_LIMITS };
  }

  function currentSessionId() {
    return $("graph-session")?.value.trim() || null;
  }

  async function loadGraph(payload = null, options = {}) {
    const key = payload ? "graph:query" : "graph:overview";
    return asyncGuard.run(key, async () => {
      setCanvasMessage(payload ? t("graphLoadingQuery") : t("graphLoadingOverview"));
      try {
        let data;
        if (payload) {
          const requestPayload = addExpandedLimits({ ...payload });
          const sessionId = requestPayload.session_id || currentSessionId();
          if (sessionId) requestPayload.session_id = sessionId;
          else delete requestPayload.session_id;
          data = await selectedDatabaseApi("/graph/query", {
            method: "POST",
            body: JSON.stringify(requestPayload),
          });
          renderPayload(normalizePayload(data, requestPayload));
        } else {
          const overviewParams = new URLSearchParams({ full_graph: "true" });
          const overviewSessionId = currentSessionId();
          if (overviewSessionId) overviewParams.set("session_id", overviewSessionId);
          data = await selectedDatabaseApi("/graph/overview?" + overviewParams.toString());
          renderPayload(normalizePayload(data));
        }
      } catch (error) {
        if (error?.name === "AbortError") return;
        setCanvasMessage(error.message || t("graphNoData"));
        toast(error.message, true);
      }
    }, {
      button: options.button,
      busyText: t("loading"),
    });
  }

  function runQuery(event) {
    const query = $("graph-query")?.value.trim() || "";
    if (!query) return loadGraph(null, { button: event?.currentTarget });
    return loadGraph({ query }, { button: event?.currentTarget });
  }

  function focusMemory(event) {
    const memoryId = numericId($("graph-memory-id")?.value);
    if (memoryId === null || !Number.isInteger(memoryId)) {
      toast(t("graphFocusInvalid"), true);
      return undefined;
    }
    setCanvasMessage(t("graphLoadingFocus").replace("{id}", String(memoryId)));
    return loadGraph({ memory_id: memoryId }, { button: event?.currentTarget });
  }

  function renderPayload(payload) {
    graphView.payload = payload;
    graphView.index = buildGraphIndex(payload.snapshot);
    const snapshot = payload.snapshot;
    const stats = payload.stats || state.stats || {};
    statCards($("graph-stats"), [
      [t("statsMemories"), stats.total_memories ?? snapshot.memories.length],
      [t("statsNodes"), stats.graph_nodes ?? snapshot.nodes.length],
      [t("statsRelations"), stats.graph_edges ?? snapshot.edges.length],
      [t("statsSessions"), Object.keys(stats.sessions || {}).length],
    ]);

    const renderer = ensureRenderer();
    if (payload.mode === "overview") clearGraphSelection(false);
    renderer.loadData(payload);
    renderGraphLegend(payload);

    if (!snapshot.nodes.length) {
      graphView.selectedNodeId = null;
      graphView.selectedMemoryId = null;
      setCanvasMessage(t("graphNoData"));
      return;
    }

    setCanvasMessage("");
    if (payload.mode !== "overview") ensureSelection(payload);
  }

  function ensureSelection(payload) {
    const index = graphView.index;
    const requestedMemoryId = numericId(payload.requested_memory_id);
    if (requestedMemoryId !== null && index.memoryMap.has(requestedMemoryId)) {
      selectGraphMemory(requestedMemoryId, false);
      return;
    }
    const matchedNodeId = (payload.matched_node_ids || [])
      .map(numericId)
      .find((id) => id !== null && index.nodeMap.has(id));
    if (matchedNodeId !== undefined) {
      selectGraphNode(matchedNodeId, false);
      return;
    }
    const retrievedMemoryId = (payload.retrieval?.items || [])
      .map((item) => numericId(item.memory_id))
      .find((id) => id !== null && index.memoryMap.has(id));
    if (retrievedMemoryId !== undefined) {
      selectGraphMemory(retrievedMemoryId, false);
      return;
    }
    const memory = payload.snapshot.memories[0];
    if (memory) selectGraphMemory(memory.memory_id, false);
  }

  function buildGraphIndex(snapshot = {}) {
    const nodeMap = new Map(snapshot.nodes.map((node) => [node.id, node]));
    const memoryMap = new Map(snapshot.memories.map((memory) => [memory.memory_id, memory]));
    const memoryToNodes = new Map();
    const nodeToMemories = new Map();
    const nodeToEntries = new Map();
    const neighborMap = new Map();
    const ensureSet = (map, key) => {
      if (!map.has(key)) map.set(key, new Set());
      return map.get(key);
    };
    snapshot.entries.forEach((entry) => {
      entry.node_ids.forEach((nodeId) => {
        ensureSet(memoryToNodes, entry.memory_id).add(nodeId);
        ensureSet(nodeToMemories, nodeId).add(entry.memory_id);
        if (!nodeToEntries.has(nodeId)) nodeToEntries.set(nodeId, []);
        nodeToEntries.get(nodeId).push(entry);
      });
    });
    snapshot.edges.forEach((edge) => {
      ensureSet(memoryToNodes, edge.memory_id).add(edge.source);
      ensureSet(memoryToNodes, edge.memory_id).add(edge.target);
      ensureSet(nodeToMemories, edge.source).add(edge.memory_id);
      ensureSet(nodeToMemories, edge.target).add(edge.memory_id);
      ensureSet(neighborMap, edge.source).add(edge.target);
      ensureSet(neighborMap, edge.target).add(edge.source);
    });
    return { nodeMap, memoryMap, memoryToNodes, nodeToMemories, nodeToEntries, neighborMap };
  }

  function renderGraphLegend(payload) {
    const summary = payload.summary || {};
    const nodeTypes = summary.node_type_breakdown || {};
    const relationTypes = summary.relation_breakdown || {};
    const labels = {
      person: t("legendPerson"),
      topic: t("legendTopic"),
      fact: t("legendFact"),
      summary: t("legendSummary"),
      other: t("legendOther"),
    };
    const nodeChips = Object.entries(nodeTypes)
      .sort((left, right) => right[1] - left[1])
      .map(([type, count]) => `<span class="legend-chip"><i class="dot" style="background:${graphTypeColor(type)}"></i>${escapeHtml(labels[type] || type)} · ${escapeHtml(count)}</span>`);
    const relationChips = Object.entries(relationTypes)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 4)
      .map(([type, count]) => `<span class="legend-chip">${escapeHtml(relationLabel(type))} · ${escapeHtml(count)}</span>`);
    $("graph-legend").innerHTML = [...nodeChips, ...relationChips].join("");
  }

  function relationLabel(value) {
    return String(value || "related")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function clearGraphSelection(closePeek = true) {
    graphView.selectedNodeId = null;
    graphView.selectedMemoryId = null;
    if (window.Graph2D?._initialized) window.Graph2D.clearSelection();
    if (closePeek) closeGraphPeek();
  }

  function selectGraphNode(nodeId, openPeek = true) {
    const id = numericId(nodeId);
    const node = graphView.index?.nodeMap.get(id);
    if (!node) return;
    graphView.selectedNodeId = id;
    graphView.selectedMemoryId = null;
    window.Graph2D?.selectNode(id);
    if (openPeek) openGraphNodePeek(node);
  }

  function selectGraphMemory(memoryId, openPeek = true) {
    const id = numericId(memoryId);
    const memory = graphView.index?.memoryMap.get(id);
    if (!memory) return;
    graphView.selectedMemoryId = id;
    graphView.selectedNodeId = null;
    window.Graph2D?.selectMemory(id);
    if (openPeek) openGraphMemoryPeek(memory);
  }

  function openGraphNodePeek(node) {
    const typeClass = String(node.type || "other").replace(/[^a-z0-9_-]/gi, "") || "other";
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
    $("graph-peek-badge").textContent = t("memory");
    $("graph-peek-badge").className = "graph-peek-badge memory";
    $("graph-peek-title").textContent = `#${memory.memory_id}`;
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

  $("graph-search")?.addEventListener("click", runQuery);
  $("graph-focus")?.addEventListener("click", focusMemory);
  $("graph-overview")?.addEventListener("click", (event) => loadGraph(null, { button: event.currentTarget }));
  $("graph-query")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runQuery(event);
    }
  });
  $("graph-memory-id")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      focusMemory(event);
    }
  });
  $("graph-peek-close")?.addEventListener("click", closeGraphPeek);
  $("graph-peek-overlay")?.addEventListener("click", closeGraphPeek);

  return {
    loadGraph,
    render: () => window.Graph2D?.animator?.wake(),
    diagnostics: () => window.Graph2D?.getDiagnostics?.() || null,
    destroy: () => {
      if (window.Graph2D?._initialized) window.Graph2D.destroy();
      graphView.payload = null;
      graphView.index = null;
      graphView.selectedNodeId = null;
      graphView.selectedMemoryId = null;
      graphView.rendererReady = false;
      closeGraphPeek();
    },
  };
}
