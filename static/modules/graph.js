export function createGraphController({ $, state, t, libraryApi, statCards, toast, escapeHtml }) {
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
  person: "#59c2ff",
  topic: "#5d35c7",
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
  return {
    loadGraph,
    render: () => graphView.renderer?.render(),
  };
}
