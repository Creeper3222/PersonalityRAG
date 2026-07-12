export function createRecallController({ $, state, t, toast, libraryApi, escapeHtml, selectedLibrary, selectedLibraryHasRerank, setRecallK, setRecallRerankK, updateRecallRerankControls }) {
function setRecallView(view) {
  state.recallView = view === "rerank" && selectedLibraryHasRerank() ? "rerank" : "embedding";
  updateRecallRerankControls();
  document.querySelectorAll("[data-recall-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.recallView === state.recallView);
  });
  renderRecallResults();
}

function recallResultScore(item) {
  const breakdown = item.score_breakdown || {};
  if (state.recallView === "rerank" && breakdown.rerank_score != null) {
    return Number(breakdown.rerank_score);
  }
  return Number(item.similarity_score || 0);
}

function recallSummaryHtml(items = []) {
  const summary = state.recallCache.summary;
  if (!summary) return "";
  const modeLabel = state.recallView === "rerank"
    ? t("recallWithRerank")
    : t("recallOnlyEmbedding");
  const parts = [
    t("recallSummary", {
      total: items.length,
      elapsed: summary.elapsed_time_ms,
    }),
    `<strong>${escapeHtml(modeLabel)}</strong>`,
  ];
  const meta = state.recallCache.rerankMeta || {};
  if (state.recallView === "rerank") {
    if (meta.applied) {
      parts.push(escapeHtml(t("rerankAppliedText", {
        provider: meta.provider_id || meta.provider_type || t("rerankProvider"),
      })));
      if (meta.candidate_count != null) {
        parts.push(escapeHtml(t("rerankCandidateText", { count: meta.candidate_count })));
      }
    } else if (meta.failed || meta.requested) {
      parts.push(`<span class="danger">${escapeHtml(t("rerankFallbackText", {
        reason: meta.error || "unknown",
      }))}</span>`);
    } else {
      parts.push(escapeHtml(t("rerankNotConfiguredText")));
    }
  }
  return parts.join(" · ");
}

function renderRecallResults() {
  updateRecallRerankControls();
  document.querySelectorAll("[data-recall-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.recallView === state.recallView);
  });
  const items = state.recallView === "rerank"
    ? state.recallCache.rerank
    : state.recallCache.embedding;
  $("recall-summary").innerHTML = recallSummaryHtml(items);
  $("recall-results").innerHTML =
    items
      .map(
        (item, index) => `<article class="result">
          <header>
            <span class="rank">#${index + 1}</span>
            <b>ID ${item.memory_id}</b>
            <span class="score ${recallResultScore(item) >= 0.7 ? "score-high" : recallResultScore(item) >= 0.4 ? "score-mid" : "score-low"}">${recallResultScore(item).toFixed(4)}</span>
          </header>
          <div>${escapeHtml(item.content)}</div>
          <p>${escapeHtml(item.metadata?.persona_id || "")} · ${escapeHtml(item.metadata?.session_id || "")}</p>
          <details>
            <summary>${escapeHtml(t("recallScoreBreakdown"))}</summary>
            <pre>${escapeHtml(JSON.stringify(item.score_breakdown || {}, null, 2))}</pre>
          </details>
        </article>`,
      )
      .join("") || `<div class="panel">${escapeHtml(t("recallNoResult"))}</div>`;
}

$("run-recall").onclick = async () => {
  const query = $("recall-query").value.trim();
  if (!query) {
    return;
  }
  const embeddingK = setRecallK($("recall-k").value);
  const hasRerank = selectedLibraryHasRerank();
  const rerankK = hasRerank ? setRecallRerankK($("recall-rerank-k").value) : embeddingK;
  const payload = {
    query,
    k: embeddingK,
    persona_id: $("recall-persona").value || null,
    session_id: $("recall-session").value || null,
    rerank: hasRerank,
  };
  if (hasRerank) {
    payload.rerank_k = rerankK;
    payload.include_baseline = true;
  }
  try {
    const data = await libraryApi("/recall", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.recallCache.summary = data;
    state.recallCache.embedding = data.baseline_results || data.results || [];
    state.recallCache.rerank = hasRerank ? data.results || [] : [];
    state.recallCache.rerankMeta = data.rerank || null;
    if (!hasRerank) {
      state.recallView = "embedding";
    }
    renderRecallResults();
    const resultItems = state.recallView === "rerank"
      ? state.recallCache.rerank
      : state.recallCache.embedding;
    const meta = state.recallCache.rerankMeta || {};
    let msg = t("recallToastSuccess", { count: resultItems.length });
    if (hasRerank) {
      msg += meta.applied ? t("recallToastRerank") : t("recallToastRerankFail");
    } else {
      msg += t("recallToastNoRerank");
    }
    toast(msg);
  } catch (error) {
    toast(error.message, true);
  }
};

$("recall-k")?.addEventListener("input", (event) => {
  setRecallK(event.target.value);
});

$("recall-rerank-k")?.addEventListener("input", (event) => {
  setRecallRerankK(event.target.value);
});

  return { setRecallView, renderRecallResults };
}
