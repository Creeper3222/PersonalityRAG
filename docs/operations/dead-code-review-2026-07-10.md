# Dead-code review (2026-07-10)

The review used repository-wide reference search, Ruff name/import analysis,
targeted tests, and the full test suite. It was limited to orchestration and UI
code; memory, graph, retrieval, rerank, storage, and index semantics were not
refactored.

## Removed

- `app._enqueue_index_rebuild()`: definition only; incremental memory jobs and
  explicit rebuild routes use the shared `LibraryManager.jobs` queue directly.
- `PersonalityRAGService.jobs`: constructed per runtime but never started,
  queried, awaited, or closed. The application has one shared serial queue on
  `LibraryManager`.
- WebUI `openEdit()`: definition only; current editing is handled by the memory
  detail panel and the create modal has its own path.
- Unused Python imports and locals reported by Ruff.
- Repeated runtime `Object.assign(strings.*)` locale overrides. Their final
  effective values now live once in dedicated locale modules.

## Deliberately retained

The migration, recovery, public API, provider alias, and historical provider
configuration paths in `compatibility-inventory.md` remain supported. A symbol
that looks old is not removable while it owns one of those duties.
