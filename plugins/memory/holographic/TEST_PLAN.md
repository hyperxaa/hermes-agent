# Holographic Memory — Test Plan

## Scope

Cobertura completa del plugin `plugins/memory/holographic/`: HRR algebra, `MemoryStore`, `FactRetriever`, `HolographicMemoryProvider`.

## Philosophy

- **Aislamiento total**: cada test usa su propia DB en `tmp_path`, nunca toca producción.
- **Primero unitario, luego integración**: cada función del `holographic.py` se prueba en aislamiento con inputs/outputs conocidos antes de probarla dentro del pipeline.
- **Cobertura de ramas**: explícitamente test paths con numpy y sin numpy.
- **Propiedades algebraicas over assertions numéricos exactos**: HRR es probabilístico — se testean invariantes (bind/unbind roundtrip, sim(self,self)≈1, sim(random,random)≈0 ± epsilon) con tolerancias numéricas.
- **One assertion per test when practicable**: cada test verifica una propiedad. Tests con múltiples asserts solo cuando las verificaciones son del mismo concepto (setup → action → verify).
- **Given/When/Then** naming: `test_given_X_when_Y_then_Z`.

## Structure

```
plugins/memory/holographic/tests/
├── __init__.py
├── conftest.py                 # fixtures: store, retriever, provider, numpy mock
├── test_hrr_algebra.py         # holographic.py — encode, bind/unbind, bundle, similarity
├── test_hrr_encoding.py        # holographic.py — encode_text, encode_fact, serialization, SNR
├── test_store_crud.py          # store.py — add/update/remove/list, dedup, trust
├── test_store_entities.py      # store.py — entity extraction patterns, aliases, stops
├── test_store_fts.py           # store.py — FTS5 triggers, rebuild, sync
├── test_store_banks.py         # store.py — memory bank create/rebuild
├── test_retriever_search.py    # retrieval.py — search, weights, decay, min_trust, category
├── test_retriever_hrr.py       # retrieval.py — probe, related, reason, contradict (HRR paths)
├── test_retriever_fallback.py  # retrieval.py — no-numpy fallback behavior
├── test_retriever_fts.py       # retrieval.py — FTS5 OR semantics, sanitization, ranking
├── test_provider_lifecycle.py  # __init__.py — init, hooks, tool handling, on_memory_write
├── test_provider_integration.py # end-to-end: add fact → search/probe/reason → get results
└── test_norun_paths.py         # tests that deliberately run without numpy
```

## Test Files Detail

### conftest.py

Fixtures compartidas:

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `db_path` | function | Temp SQLite DB path |
| `store` | function | `MemoryStore(db_path, default_trust=0.5)` |
| `retriever` | function | `FactRetriever(store)` |
| `provider` | function | `HolographicMemoryProvider(config=...)` w/ mock config |
| `seeded_facts` | function | 5-10 facts across categories with known entities |
| `patch_no_numpy` | function | `monkeypatch` setter that sets `hrr._HAS_NUMPY = False` |
| `ephemeral_provider` | function | Provider with isolated plugin config dict (no YAML reads) |

### test_hrr_algebra.py — HRR core algebra

Tests for `holographic.py` primitives. Each test verifies a mathematical invariant.

| # | Test name | What | Assertion |
|---|-----------|------|-----------|
| 1 | `test_encode_atom_deterministic` | `encode_atom("foo")` returns same vector twice | `np.array_equal(v1, v2)` |
| 2 | `test_encode_atom_different_inputs` | `encode_atom("foo")` ≠ `encode_atom("bar")` | `similarity < 0.15` |
| 3 | `test_encode_atom_phase_range` | All phase values in [0, 2π) | `0 ≤ v < 2π` |
| 4 | `test_encode_atom_dim_matches` | Output shape matches dim param | `len(v) == dim` |
| 5 | `test_bind_unbind_roundtrip` | `unbind(bind(a, b), a) ≈ b` | `similarity ≈ 0.85+` (degraded by b's overlap noise) |
| 6 | `test_bind_commutes_in_structure` | bind is element-wise addition, order doesn't matter | `np.allclose(bind(a,b), bind(b,a))` |
| 7 | `test_bundle_similarity_to_inputs` | `bundle(a, b)` is similar to each input | `sim(bundle, a) > sim(a, random)` |
| 8 | `test_bundle_multi_similarity` | `bundle(a,b,c)` stays similar to each | similarity > threshold for all |
| 9 | `test_similarity_self_perfect` | `similarity(a, a)` | `≈ 1.0 (±1e-10)` |
| 10 | `test_similarity_orthogonal_near_zero` | `similarity(encode_atom("foo"), encode_atom("bar"))` at dim=1024 | `|sim| < 0.10` |
| 11 | `test_similarity_range` | Sim always in [-1, 1] | always |
| 12 | `test_bind_quasi_orthogonal` | `bind(a, b)` dissimilar to both inputs | `sim < 0.10` for both |
| 13 | `test_dimension_scaling` | Orthogonality improves with dim | `sim(foo,bar)` at dim=64 > dim=1024 |

### test_hrr_encoding.py — Text and fact encoding

| # | Test name | What |
|---|-----------|------|
| 1 | `test_encode_text_single_token` | `"hello"` → bundled 1-token vector |
| 2 | `test_encode_text_multi_token` | `"hello world"` → 2 tokens bundled |
| 3 | `test_encode_text_punctuation_stripped` | `"Hello, world!"` → same tokens as `"hello world"` |
| 4 | `test_encode_text_empty_returns_empty_atom` | `""` / `"   "` / `"..."` → `encode_atom("__hrr_empty__")` |
| 5 | `test_encode_text_case_insensitive` | `"HELLO"` → same as `"hello"` (lowercased) |
| 6 | `test_encode_fact_with_entities` | Structured encoding with role binding |
| 7 | `test_encode_fact_no_entities` | Fact with empty entity list |
| 8 | `test_encode_fact_entity_lowercased` | Entity `"Xavi"` and `"xavi"` produce same vector |
| 9 | `test_phases_to_bytes_roundtrip` | `bytes_to_phases(phases_to_bytes(v)) ≈ v` |
| 10 | `test_serialization_size` | `len(phases_to_bytes(v))` == `dim * 8` bytes |
| 11 | `test_serialization_mutable_copy` | Returned array from `bytes_to_phases` is writable |
| 12 | `test_snr_estimate_zero_items` | `snr(dim, 0)` == `inf` |
| 13 | `test_snr_estimate_decreases_with_items` | SNR shrinks as n_items grows |
| 14 | `test_snr_warning_at_capacity` | Log warning when `n_items > dim/4` |
| 15 | `test_encode_fact_algebraic_extract` | `unbind(fact_vec, bind(entity_atom, ROLE_ENTITY))` ≈ content vector |

### test_store_crud.py — MemoryStore CRUD

| # | Test name | What |
|---|-----------|------|
| 1 | `test_add_fact_returns_id` | Returns positive int |
| 2 | `test_add_fact_stores_content` | DB contains exact content |
| 3 | `test_add_fact_default_category` | Default is `"general"` |
| 4 | `test_add_fact_explicit_category` | Stores specified category |
| 5 | `test_add_fact_default_trust` | New fact has configured default trust |
| 6 | `test_add_fact_with_tags` | Tags stored verbatim |
| 7 | `test_add_fact_dedup_returns_same_id` | Duplicate content → same fact_id |
| 8 | `test_add_fact_dedup_does_not_modify` | Duplicate doesn't change trust/tags |
| 9 | `test_add_fact_empty_raises` | `ValueError` on empty content |
| 10 | `test_add_fact_strips_whitespace` | `"  hello  "` → `"hello"` in DB |
| 11 | `test_add_fact_creates_directory` | Parent dirs created when db path is in nested temp |
| 12 | `test_update_fact_content` | Content update reflected in DB |
| 13 | `test_update_fact_trust_delta_positive` | Trust increases, clamped at 1.0 |
| 14 | `test_update_fact_trust_delta_negative` | Trust decreases, clamped at 0.0 |
| 15 | `test_update_fact_trust_clamp_upper` | `trust_delta=999` → trust=1.0 |
| 16 | `test_update_fact_trust_clamp_lower` | `trust_delta=-999` → trust=0.0 |
| 17 | `test_update_fact_tags` | Tags replaced on update |
| 18 | `test_update_fact_category` | Category changed |
| 19 | `test_update_fact_not_found` | Returns `False` for non-existent fact_id |
| 20 | `test_update_fact_content_re_rextracts_entities` | New entities from updated content are added |
| 21 | `test_update_fact_content_recomputes_hrr` | HRR vector updated when content changes |
| 22 | `test_remove_fact_deletes_row` | Row gone from facts table |
| 23 | `test_remove_fact_deletes_entity_links` | fact_entities rows removed |
| 24 | `test_remove_fact_not_found` | Returns `False` |
| 25 | `test_list_facts_ordered_by_trust` | Highest trust first |
| 26 | `test_list_facts_category_filter` | Only returns matching category |
| 27 | `test_list_facts_min_trust_filter` | Only returns above threshold |
| 28 | `test_list_facts_limit` | Returns at most N |
| 29 | `test_list_facts_includes_entities` | Entity names in result dict |
| 30 | `test_search_facts_fts_basic` | FTS5 returns matching facts |
| 31 | `test_search_facts_empty_query` | `""` query returns `[]` |
| 32 | `test_search_facts_category_filter` | Category filter works in FTS search |
| 33 | `test_search_facts_bumps_retrieval_count` | retrieval_count incremented after search |
| 34 | `test_record_feedback_helpful` | Trust += 0.05, helpful_count += 1 |
| 35 | `test_record_feedback_unhelpful` | Trust -= 0.10 |
| 36 | `test_record_feedback_not_found` | `KeyError` for non-existent fact_id |
| 37 | `test_row_to_dict_includes_entities` | `_row_to_dict` returns fact with entity list |
| 38 | `test_thread_safety` | Concurrent add_fact from multiple threads doesn't corrupt DB |
| 39 | `test_wal_mode_enabled` | `PRAGMA journal_mode` == `"wal"` (or fallback) |

### test_store_entities.py — Entity extraction

| # | Test name | What |
|---|-----------|------|
| 1 | `test_extract_multi_capitalized` | `"Oracle Cloud"` detected |
| 2 | `test_extract_acronym` | `"OCI"` detected |
| 3 | `test_extract_single_name` | `"Xavi"` detected |
| 4 | `test_extract_double_quoted` | `"Python"` detected |
| 5 | `test_extract_single_quoted` | `'pytest'` detected |
| 6 | `test_extract_aka_pattern` | `"Guido aka BDFL"` → two entities |
| 7 | `test_stop_words_excluded_single` | `"The"` NOT extracted |
| 8 | `test_stop_words_excluded_spanish` | `"Cuando"` NOT extracted |
| 9 | `test_acronym_stop_words` | `"EL"`, `"SI"`, `"NO"` NOT extracted as acronyms |
| 10 | `test_entity_alias_normalization` | `"Usuario Xavi"` → `"Xavi"` |
| 11 | `test_multiple_entities_one_fact` | `"Xavi works at OCI on gateway"` → [Xavi, OCI] |
| 12 | `test_reused_entity_shared_id` | Same entity name in 2 facts → same entity_id |
| 13 | `test_extract_lowercase_only_skips` | `"hello world"` → no capitalized entities (except quoted) |
| 14 | `test_extract_ignores_gateway` | `"Gateway"` NOT extracted (stop word) |

### test_store_fts.py — FTS5 triggers and sync

| # | Test name | What |
|---|-----------|------|
| 1 | `test_fts_sync_on_insert` | After `add_fact`, FTS5 contains it |
| 2 | `test_fts_sync_on_delete` | After `remove_fact`, FTS5 doesn't contain it |
| 3 | `test_fts_sync_on_update` | After `update_fact(content)`, FTS5 has new text |
| 4 | `test_fts_matches_tags` | Tag tokens searchable in FTS |
| 5 | `test_fts_rebuild_after_bulk` | `INSERT INTO facts_fts(facts_fts) VALUES('rebuild')` works |

### test_store_banks.py — Memory banks

| # | Test name | What |
|---|-----------|------|
| 1 | `test_bank_created_on_fact_add` | Bank exists after first fact |
| 2 | `test_bank_rebuilt_on_fact_update` | Bank updated when fact content changes |
| 3 | `test_bank_rebuilt_on_fact_delete` | Bank updated after remove |
| 4 | `test_separate_banks_per_category` | `cat:user_pref` ≠ `cat:project` |
| 5 | `test_bank_vector_dimension` | Vector length == `dim * 8` bytes |
| 6 | `test_bank_fact_count_updated` | fact_count reflects category total |
| 7 | `test_rebuild_all_vectors` | All facts with entities get HRR vectors, all banks rebuilt |

### test_retriever_search.py — Hybrid search pipeline

| # | Test name | What |
|---|-----------|------|
| 1 | `test_search_returns_results` | Query against seeded facts returns matches |
| 2 | `test_search_no_match` | Unrelated query returns `[]` |
| 3 | `test_search_score_field_present` | Each result has `"score"` > 0 |
| 4 | `test_search_sorted_by_score_desc` | Results ordered highest→lowest |
| 5 | `test_search_respects_limit` | Returns at most N |
| 6 | `test_search_category_filter` | Only matching category |
| 7 | `test_search_min_trust_filter` | Only facts above threshold |
| 8 | `test_search_strips_hrr_vector` | No `hrr_vector` key in results (not JSON-serializable) |
| 9 | `test_search_bumps_retrieval_count` | retrieval_count incremented |
| 10 | `test_search_temporal_decay_applied` | With half_life > 0, older facts score lower |
| 11 | `test_search_temporal_decay_disabled` | With half_life=0, age doesn't matter |
| 12 | `test_search_query_includes_tag_tokens` | Tag tokens participate in Jaccard scoring |
| 13 | `test_search_empty_query_returns_none` | `""` → `[]` |
| 14 | `test_search_weights_affect_order` | Different weight configs change result order |
| 15 | `test_candidate_expansion_limit` | Fetches limit*3 candidates for reranking headroom |

### test_retriever_hrr.py — HRR-based retrieval methods

| # | Test name | What |
|---|-----------|------|
| 1 | `test_probe_finds_entity_facts` | `probe("Xavi")` finds facts about Xavi |
| 2 | `test_probe_scores_higher_than_unrelated` | Entity facts score > unrelated facts |
| 3 | `test_probe_returns_results_sorted` | Sorted by score DESC |
| 4 | `test_related_finds_shared_context` | Facts sharing entity context found |
| 5 | `test_related_different_from_probe` | `related` returns broader set than `probe` (both roles) |
| 6 | `test_reason_finds_multi_entity_facts` | `reason(["Xavi", "gateway"])` finds facts with both |
| 7 | `test_reason_and_semantics_min_score` | Score is MIN across entities, not average |
| 8 | `test_reason_empty_entities_list` | `[]` → falls back to search |
| 9 | `test_contradict_facts_found` | Facts with shared entities + different content detected |
| 10 | `test_contradict_score_formula` | Score = entity_overlap × (1 - content_similarity) |
| 11 | `test_contradict_threshold_filters` | Pairs with entity_jaccard < 0.3 skipped |
| 12 | `test_contradict_500_limit` | Only processes first 500 facts for O(n²) safety |

### test_retriever_fallback.py — No-numpy fallback behavior

| # | Test name | What |
|---|-----------|------|
| 1 | `test_probe_fallback_to_search` | Without numpy, probe → keyword search |
| 2 | `test_related_fallback_to_search` | Without numpy, related → keyword search |
| 3 | `test_reason_fallback_to_search` | Without numpy, reason → FTS search |
| 4 | `test_contradict_returns_empty` | Without numpy, contradict → `[]` |
| 5 | `test_search_weights_redistributed` | fts=0.6, jaccard=0.4, hrr=0.0 without numpy |
| 6 | `test_hrr_weight_zero_no_numpy` | Retriever sets hrr_weight=0.0 |
| 7 | `test_rebuild_all_vectors_noop` | Returns 0 when no numpy |
| 8 | `test_add_fact_works_without_numpy` | Fact added, HRR vector is NULL |
| 9 | `test_encode_text_raises_without_numpy` | `encode_text()` → RuntimeError |
| 10 | `test_similarity_raises_without_numpy` | `similarity()` → RuntimeError |
| 11 | `test_bind_raises_without_numpy` | `bind()` → RuntimeError |
| 12 | `test_encode_fact_raises_without_numpy` | `encode_fact()` → RuntimeError |

### test_retriever_fts.py — FTS5 specific behavior

| # | Test name | What |
|---|-----------|------|
| 1 | `test_fts_or_semantics` | `"editor config"` matches facts with EITHER word |
| 2 | `test_fts_special_chars_sanitized` | Query with `()-~^/+'` → sanitized, doesn't explode |
| 3 | `test_fts_special_chars_all_eaten_tries_raw` | If sanitization eats everything, raw query attempted |
| 4 | `test_fts_rank_positional` | Ranks based on position, not SQLite version-dependent raw rank |
| 5 | `test_fts_candidate_limit_expansion` | Returns up to limit*3 candidates |
| 6 | `test_fts_returns_empty_on_error` | FTS5 MATCH exception → `[]`, never throws |

### test_provider_lifecycle.py — HolographicMemoryProvider hooks

| # | Test name | What |
|---|-----------|------|
| 1 | `test_provider_name` | `name == "holographic"` |
| 2 | `test_provider_is_available` | Always `True` |
| 3 | `test_provider_initialize_creates_store` | After init, `._store` and `._retriever` set |
| 4 | `test_provider_initialize_expands_hermes_home` | `$HERMES_HOME` expanded in db_path |
| 5 | `test_provider_system_prompt_block_empty` | Returns "empty" prompt when 0 facts |
| 6 | `test_provider_system_prompt_block_has_facts` | Returns count summary when facts exist |
| 7 | `test_provider_prefetch_returns_top5` | Returns up to 5 results as formatted string |
| 8 | `test_provider_prefetch_empty_query` | `""` → `""` |
| 9 | `test_provider_prefetch_no_results` | No matching facts → `""` |
| 10 | `test_provider_prefetch_exception_returns_empty` | Any exception → `""`, never raises |
| 11 | `test_provider_sync_turn_noop` | Does nothing |
| 12 | `test_provider_get_tool_schemas` | Returns 2 schemas (fact_store + fact_feedback) |
| 13 | `test_provider_handle_fact_store_add` | Routes to store.add_fact, returns JSON |
| 14 | `test_provider_handle_fact_store_search` | Routes to retriever.search, returns JSON with count |
| 15 | `test_provider_handle_fact_store_probe` | Routes to retriever.probe |
| 16 | `test_provider_handle_fact_store_related` | Routes to retriever.related |
| 17 | `test_provider_handle_fact_store_reason` | Routes to retriever.reason |
| 18 | `test_provider_handle_fact_store_contradict` | Routes to retriever.contradict |
| 19 | `test_provider_handle_fact_store_update` | Routes to store.update_fact |
| 20 | `test_provider_handle_fact_store_remove` | Routes to store.remove_fact |
| 21 | `test_provider_handle_fact_store_list` | Routes to store.list_facts |
| 22 | `test_provider_handle_fact_feedback_helpful` | Routes to store.record_feedback(True) |
| 23 | `test_provider_handle_fact_feedback_unhelpful` | Routes to store.record_feedback(False) |
| 24 | `test_provider_handle_unknown_tool` | Returns error via tool_error() |
| 25 | `test_provider_on_memory_write_user_target` | `target="user"` → `category="user_pref"` |
| 26 | `test_provider_on_memory_write_memory_target` | `target="memory"` → `category="general"` |
| 27 | `test_provider_on_memory_write_only_adds` | `action="replace"` → no-op |
| 28 | `test_provider_on_memory_write_ignores_remove` | `action="remove"` → no-op |
| 29 | `test_provider_on_memory_write_empty_content` | Empty content → no fact created |
| 30 | `test_provider_shutdown_clears_refs` | `._store` and `._retriever` set to None |
| 31 | `test_provider_get_config_schema` | Returns 4 fields |
| 32 | `test_provider_save_config_writes_yaml` | Persists to config.yaml format |

### test_provider_integration.py — End-to-end

| # | Test name | What |
|---|-----------|------|
| 1 | `test_e2e_add_and_search` | Add fact → search finds it |
| 2 | `test_e2e_add_and_probe` | Add fact with entity → probe finds it |
| 3 | `test_e2e_add_and_reason` | Add facts with multiple entities → reason finds intersection |
| 4 | `test_e2e_add_and_feedback` | Add → feedback → trust changes |
| 5 | `test_e2e_add_and_list` | Add multiple → list returns all ordered by trust |
| 6 | `test_e2e_full_lifecycle` | Add → search → probe → update → feedback → search again (trust/order changed) |
| 7 | `test_e2e_provider_tool_dispatch` | Initialize provider → handle_tool_call("fact_store", add) → fact stored |
| 8 | `test_e2e_memory_write_mirror` | Create provider → on_memory_write("add", "user", ...) → fact exists |

## Execution

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
pytest plugins/memory/holographic/tests/ -xvs           # all
pytest plugins/memory/holographic/tests/test_hrr_algebra.py -xvs  # one file
pytest plugins/memory/holographic/tests/ -k "no_numpy" -xvs       # fallback paths
```

## Priorities

1. **P0**: `test_hrr_algebra.py` (invariantes matemáticos, base de todo lo demás)
2. **P1**: `test_store_crud.py` (la capa de datos, más surface area)
3. **P2**: `test_retriever_search.py` + `test_retriever_hrr.py` (core retrieval)
4. **P3**: `test_provider_lifecycle.py` (hooks y tool dispatch)
5. **P4**: `test_store_entities.py` (entity extraction regexes)
6. **P5**: `test_retriever_fallback.py` (no-numpy paths)
7. **P6**: Resto — FTS5, banks, integration

## Dependencies

- `pytest` (ya en el venv del proyecto)
- `numpy` (required for most tests; fallback tests explicitly mock it out)
- `pyyaml` (ya en el venv)

## What's NOT tested (gaps a cubrir después)

- `auto_extract` at session end (regex-based, no LLM — solo pattern matching)
- `_load_plugin_config()` reading from external YAML (mocked in provider tests)
- `save_config()` with real filesystem writes beyond tempdir
- WAL mode on actual NFS/SMB mounts (would need real infra)
