# PersonalityRAG compatibility inventory

This inventory distinguishes deliberate compatibility code from removable
internal leftovers. An item listed here must not be deleted as dead code
without a dedicated migration and public compatibility decision.

## Data and layout compatibility

- Legacy single-library layout migration in `LibraryManager._migrate_legacy_layout()`.
- Its backup verification, commit marker, and rollback path.
- LivingMemory v8 database validation and import compatibility.
- Existing `livingmemory.db`, `conversations.db`, and FAISS generation formats.

## Provider compatibility

- Conversion of the historical `openai_compatible` provider type to
  `vllm_embedding` while loading configuration.
- Public provider class alias
  `OpenAICompatibleEmbeddingProvider = VLLMEmbeddingProvider`.
- Compatibility endpoint `POST /api/v1/providers/test`.
- Existing provider request payload fields and revision bindings.

## HTTP compatibility

- Unscoped endpoints such as `/api/v1/memories`, `/api/v1/recall`, and
  `/api/v1/indexes` remain aliases for default-library operations.
- Library-scoped endpoints under `/api/v1/libraries/{library_id}` remain the
  canonical multi-library API.
- WebUI and adapter authentication headers, status codes, and response core
  fields are contract-tested in `tests/test_operational_contract.py`.

## Review rule

Internal helpers may be removed only after repository-wide reference search,
static analysis, and tests show no caller. Public routes, aliases, migration
paths, recovery paths, and serialized fields are not classified as dead code.
