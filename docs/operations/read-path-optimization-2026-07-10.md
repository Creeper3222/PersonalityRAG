# Control and status read-path optimization (2026-07-10)

The control database remains schema-compatible. SQL reads are now organized by
Provider, Library, Adapter, Job, and Snapshot repositories while `ControlStore`
continues to expose the existing façade.

## Query changes

- `ControlStore.list_libraries()` decodes the rows from one query instead of
  opening one additional connection per library.
- Library card loading resolves every exact embedding revision and latest
  rerank revision in one provider query.
- Adapter and long-job summaries remain one query per library list, with active
  in-memory jobs overlaid by `JobManager`.
- `stats_mode=summary` reads only scalar card counters. It does not materialize
  document metadata, status distributions, importance buckets, atom groups, or
  a session map.

## Provider status

Provider health probes use a 60-second single-flight cache. Concurrent requests
share one probe. Adapter status requests and status reads during a long library
task only consume cached state and cannot trigger a model request.

## Local measurement

On the live three-library data set, ten direct manager reads averaged 29.31 ms
in full mode and 18.97 ms in summary mode (18.34 ms median). The earlier live
HTTP `/libraries` baseline was 87.21 ms mean. These figures are diagnostic and
machine-dependent; query-count tests provide the deterministic regression gate.
