# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval. Implements the `MemoryProvider` ABC.

## Table of Contents

- [Architecture](#architecture)
- [Components](#components)
- [Database Schema](#database-schema)
- [Plugin Lifecycle](#plugin-lifecycle)
- [Tool API](#tool-api)
- [Encoding: HRR (holographic.py)](#encoding-hrr-holographicpy)
- [Store API (store.py)](#store-api-storepy)
- [Retrieval Pipeline (retrieval.py)](#retrieval-pipeline-retrievalpy)
- [Query Flow](#query-flow)
- [MemoryProvider Integration](#memoryprovider-integration)
- [Holographic vs Built-in Memory](#holographic-vs-built-in-memory)
- [Hook Coverage](#hook-coverage)
- [Configuration](#configuration)
- [Performance & Capacity](#performance--capacity)
- [Maintenance & Reindexing](#maintenance--reindexing)
- [Testing](#testing)
- [Known Bugs & Pitfalls](#known-bugs--pitfalls)
- [Diagnosis Skill](#diagnosis-skill)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent / Gateway                   │
│  ┌─────────────┐     ┌───────────────────────────┐  │
│  │ config.yaml │────▶│ plugins.hermes-memory-store│  │
│  └─────────────┘     └───────────────────────────┘  │
│                     │                               │
│                     ▼                               │
│  ┌──────────────────────────────────────────────┐   │
│  │        HolographicMemoryProvider             │   │
│  │  ┌──────────────┐  ┌─────────────────────┐   │   │
│  │  │ MemoryStore   │  │   FactRetriever     │   │   │
│  │  │ (store.py)    │  │   (retrieval.py)    │   │   │
│  │  │               │  │                     │   │   │
│  │  │ • add_fact    │  │ • search()          │   │   │
│  │  │ • update_fact │  │ • probe()           │   │   │
│  │  │ • remove_fact │  │ • related()         │   │   │
│  │  │ • list_facts  │  │ • reason()          │   │   │
│  │  │ • entities    │  │ • contradict()      │   │   │
│  │  └───────┬───────┘  └────────┬────────────┘   │   │
│  └──────────┼───────────────────┼────────────────┘   │
│             │                   │                     │
└─────────────┼───────────────────┼─────────────────────┘
              │                   │
              ▼                   ▼
┌─────────────────────┐  ┌─────────────────────┐
│   SQLite Database   │  │    HRR Module       │
│  memory_store.db    │  │  (holographic.py)   │
│                     │  │                     │
│  • facts            │  │  • encode_atom()    │
│  • entities         │  │  • bind/unbind      │
│  • fact_entities    │  │  • bundle()         │
│  • memory_banks     │  │  • encode_fact()    │
│  • facts_fts (FTS5) │  │  • similarity()     │
└─────────────────────┘  └─────────────────────┘
```

### Design principles

- **Explicit over implicit**: facts are stored via tool calls (`fact_store`), not auto-extracted from conversations (unless `auto_extract: true`).
- **SQLite-first**: no external services. SQLite is always available.
- **HRR optional**: if numpy is unavailable, the system falls back to FTS5 + Jaccard only.
- **Profile-scoped**: the DB path resolves to the active profile's `$HERMES_HOME`.
- **Dual-storage compatible**: mirrors `memory` tool writes through `on_memory_write()`.

---

## Components

| File | LOC | Purpose |
|------|-----|---------|
| `plugin.yaml` | 5 | Plugin metadata: name, version, hooks |
| `__init__.py` | 408 | Provider entry point, tool schemas, config loader |
| `holographic.py` | 203 | HRR algebra: encode/bind/unbind/bundle/similarity |
| `store.py` | 627 | SQLite CRUD, entity extraction/resolution, HRR vector compute |
| `retrieval.py` | 638 | Hybrid search: FTS5 + Jaccard + HRR + trust + decay |

### plugin.yaml

Registers the plugin with name `holographic`, version `0.1.0`, and declares the `on_session_end` hook for auto-extraction.

### `__init__.py`

The plugin entry point. Contains:

- **`HolographicMemoryProvider`** — implements the `MemoryProvider` ABC.
- **Tool schemas** — `FACT_STORE_SCHEMA` (9 actions) and `FACT_FEEDBACK_SCHEMA`.
- **Config loading** — reads from `plugins.hermes-memory-store` in config.yaml, expands `$HERMES_HOME` in paths.
- **MemoryProvider hooks** — `initialize()`, `system_prompt_block()`, `prefetch()`, `sync_turn()`, `handle_tool_call()`, `on_session_end()`, `shutdown()`, `on_memory_write()`, `get_config_schema()`, `save_config()`.
- **`register(ctx)`** — plugin entry point, called by the plugin system.

### `holographic.py`

Holographic Reduced Representations (HRR) with phase encoding. A vector symbolic architecture for encoding compositional structure into fixed-width distributed representations.

Key concept: each concept is a vector of angles in [0, 2π). Operations are:

- **bind** — element-wise phase addition (associates two concepts)
- **unbind** — element-wise phase subtraction (retrieves a bound value)
- **bundle** — circular mean of complex exponentials (merges multiple concepts)

Atoms are generated deterministically from SHA-256 — identical across processes and machines.

### `store.py`

SQLite-backed fact store with entity resolution and trust scoring.

Handles:
- Schema creation and migration
- Fact CRUD (add, update, remove, list)
- Entity extraction from text via regex patterns
- Entity resolution (find existing or create new)
- HRR vector computation per fact
- Memory bank rebuild per category
- Trust scoring and feedback

### `retrieval.py`

Multi-strategy fact retrieval. Combines:
1. FTS5 full-text search with OR semantics
2. Jaccard similarity reranking
3. HRR vector similarity
4. Trust-weighted scoring
5. Optional temporal decay

---

## Database Schema

### Tables

**`facts`** — The core fact table

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `fact_id` | INTEGER PK AUTOINCREMENT | — | Unique identifier |
| `content` | TEXT NOT NULL UNIQUE | — | Fact text |
| `category` | TEXT | `'general'` | One of: `user_pref`, `project`, `tool`, `general` |
| `tags` | TEXT | `''` | Comma-separated tags |
| `trust_score` | REAL | `0.5` | Confidence [0.0, 1.0] |
| `retrieval_count` | INTEGER | `0` | Times retrieved |
| `helpful_count` | INTEGER | `0` | Positive feedback count |
| `created_at` | TIMESTAMP | CURRENT | Creation timestamp |
| `updated_at` | TIMESTAMP | CURRENT | Last modification timestamp |
| `hrr_vector` | BLOB | NULL | Phase-encoded HRR vector (8 KB at dim=1024) |

**`entities`** — Named entities extracted from facts

| Column | Type | Default |
|--------|------|---------|
| `entity_id` | INTEGER PK AUTOINCREMENT | — |
| `name` | TEXT NOT NULL | — |
| `entity_type` | TEXT | `'unknown'` |
| `aliases` | TEXT | `''` |
| `created_at` | TIMESTAMP | CURRENT |

**`fact_entities`** — Many-to-many link between facts and entities

| Column | Type | Description |
|--------|------|-------------|
| `fact_id` | INTEGER FK → facts | |
| `entity_id` | INTEGER FK → entities | Composite PK (fact_id, entity_id) |

**`memory_banks`** — Category-level bundled HRR vectors for compositional queries

| Column | Type | Description |
|--------|------|-------------|
| `bank_id` | INTEGER PK AUTOINCREMENT | |
| `bank_name` | TEXT NOT NULL UNIQUE | Format: `cat:<category>` |
| `vector` | BLOB NOT NULL | Bundled HRR vector |
| `dim` | INTEGER NOT NULL | HRR dimension |
| `fact_count` | INTEGER | Facts in this bank |
| `updated_at` | TIMESTAMP | |

**`facts_fts`** — FTS5 virtual table (auto-mirrors facts via triggers)

- Triggers: `facts_ai` (after insert), `facts_ad` (after delete), `facts_au` (after update)
- Indexes: `facts(trust_score)`, `facts(category)`, `entities(name)`
- WAL mode enabled (with NFS fallback via `apply_wal_with_fallback`)

---

## Plugin Lifecycle

The plugin implements the `MemoryProvider` ABC. Lifecycle managed by the MemoryManager and the agent/gateway:

1. **`register(ctx)`** — Plugin system calls this on startup. Creates `HolographicMemoryProvider` instance, passes it to `ctx.register_memory_provider()`.
2. **`initialize(session_id, ...)`** — Called by MemoryManager when a session starts. Loads config, creates `MemoryStore` and `FactRetriever`.
3. **`system_prompt_block()`** — Returns a short summary injected into the system prompt at startup.
4. **`prefetch(query)`** — Called with the user's query before the model responds. Returns top 5 facts as formatted text for context injection.
5. **`sync_turn(user, assistant)`** — No-op. Holographic stores explicit facts via tools, not auto-sync.
6. **`handle_tool_call(tool_name, args)`** — Routes `fact_store` and `fact_feedback` to appropriate methods.
7. **`on_memory_write(action, target, content)`** — Mirrors `memory(action='add')` calls as facts for dual-storage compatibility.
8. **`on_session_end(messages)`** — Auto-extracts facts from conversation if `auto_extract: true` (disabled by default).
9. **`shutdown()`** — Clears store and retriever references.

### Initialization config path

```
config.yaml → _load_plugin_config() → HolographicMemoryProvider(config)
  → initialize() → MemoryStore(db_path, default_trust, hrr_dim)
                → FactRetriever(store, temporal_decay, hrr_weight, hrr_dim)
```

The `db_path` is expanded: `$HERMES_HOME` and `${HERMES_HOME}` are replaced with the active profile's home directory. `~` expansion also works.

---

## Tool API

### `fact_store`

9 actions for structured memory operations:

**`add`** — Store a new fact

```
action: "add"
content: "Xavi prefers concise responses"           # required
category: "user_pref" | "project" | "tool" | "general"  # default: "general"
tags: "preferences,communication"                   # default: ""
```

**`search`** — Keyword search with hybrid ranking

```
action: "search"
query: "editor config"                              # required
category: null                                      # optional filter
min_trust: 0.3                                      # default: 0.3
limit: 10                                           # default: 10
```

**`probe`** — Entity recall: all facts about a specific entity

```
action: "probe"
entity: "Xavi"                                      # required
category: null
limit: 10
```

**`related`** — Structural adjacency: facts connected to an entity

```
action: "related"
entity: "gateway"                                   # required
category: null
limit: 10
```

**`reason`** — Compositional: facts connected to MULTIPLE entities (AND semantics)

```
action: "reason"
entities: ["Xavi", "gateway"]                       # required, list
category: null
limit: 10
```

**`contradict`** — Find potentially contradictory facts

```
action: "contradict"
category: null
limit: 10
```

**`update`** — Modify an existing fact

```
action: "update"
fact_id: 42                                         # required
content: "Updated fact text"                        # optional
trust_delta: 0.05                                   # optional, clamped to [0,1]
tags: "new,tags"                                    # optional
category: "project"                                 # optional
```

**`remove`** — Delete a fact

```
action: "remove"
fact_id: 42                                         # required
```

**`list`** — Browse facts ordered by trust score

```
action: "list"
category: null
min_trust: 0.0
limit: 50
```

### `fact_feedback`

Rate facts to train trust scores:

| Input | Result |
|-------|--------|
| `helpful` | trust += 0.05, helpful_count += 1 |
| `unhelpful` | trust -= 0.10 (penalizes more heavily) |

---

## Encoding: HRR (holographic.py)

### Core algebra

| Operation | Math | Purpose |
|-----------|------|---------|
| `bind(a, b)` | `(a + b) % 2π` | Associate two concepts |
| `unbind(mem, key)` | `(mem - key) % 2π` | Retrieve bound value |
| `bundle(*vectors)` | `angle(Σ e^(j*v))` | Merge multiple concepts |
| `similarity(a, b)` | `mean(cos(a - b))` | Phase cosine similarity [-1, 1] |

### Atoms

Deterministic phase vectors generated from SHA-256:

```
atom("Xavi") = SHA-256("Xavi:0") + SHA-256("Xavi:1") + ... → phases
```

Each SHA-256 digest (32 bytes) yields 16 uint16 values. Enough blocks are generated to fill `dim` elements, scaled to [0, 2π).

### encode_fact

Structured encoding with role vectors:

```
fact_vector = bundle(
    bind(encode_text(content), ROLE_CONTENT),
    bind(encode_atom(entity1), ROLE_ENTITY),
    bind(encode_atom(entity2), ROLE_ENTITY),
    ...
)
```

Roles:

- `__hrr_role_content__` — marks the content component
- `__hrr_role_entity__` — marks entity components

This enables algebraic extraction: `unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector`

### encode_text

Bag-of-words: tokenize → encode each token → bundle all atom vectors. Empty text produces `encode_atom("__hrr_empty__", dim)`.

Tokenization: lowercase, split whitespace, strip `.,!?;:"'()[]{}`.

### Serialization

- `phases_to_bytes(vector)` — `np.float64.tobytes()` (8 KB at dim=1024)
- `bytes_to_phases(data)` — `np.frombuffer(data, dtype=np.float64).copy()`

### SNR Estimation

`snr_estimate(dim, n_items) = sqrt(dim / n_items)`

SNR falls below 2.0 when `n_items > dim / 4`. At that point, retrieval errors become likely. Logs a warning.

---

## Store API (store.py)

### MemoryStore

```python
MemoryStore(
    db_path: str | Path | None = None,    # default: $HERMES_HOME/memory_store.db
    default_trust: float = 0.5,
    hrr_dim: int = 1024,
)
```

#### Public methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_fact(content, category, tags)` | `int` (fact_id) | Insert fact. Deduplicates by content (UNIQUE). Auto-extracts entities and computes HRR vector. |
| `search_facts(query, category, min_trust, limit)` | `list[dict]` | FTS5 full-text search. Increments retrieval_count. |
| `update_fact(fact_id, content, trust_delta, tags, category)` | `bool` | Partial update. Re-extracts entities and recomputes HRR if content changed. |
| `remove_fact(fact_id)` | `bool` | Delete fact and entity links. Rebuilds category bank. |
| `list_facts(category, min_trust, limit)` | `list[dict]` | Browse by trust score. |
| `record_feedback(fact_id, helpful)` | `dict` | Adjust trust asymmetrically. Raises KeyError if not found. |
| `rebuild_all_vectors(dim)` | `int` | Recompute all HRR vectors + banks. For recovery/migration. |

#### Entity extraction (`_extract_entities`)

6 regex-based rules applied in order:

1. **Capitalized multi-word phrases** — `"John Doe"`, `"Oracle Cloud"` — `\b([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15}){1,3})\b`
2. **Acronyms / ALL-CAPS** — `"OCI"`, `"TTS"`, `"CLI"` — 2-6 chars, exclude Spanish words via `_ACRONYM_STOPS`
3. **Single capitalized words** — `"Xavi"`, `"Jack"` — 1-15 chars after first, excludes ~60 stop words in EN+ES
4. **Double-quoted terms** — `"Python"`
5. **Single-quoted terms** — `'pytest'`
6. **AKA patterns** — `"Guido aka BDFL"` → two entities

Stop word lists cover English + Spanish + generic tech words (`Gateway`, `Agente`, `Usuario`...).

Entity alias normalization: `"Usuario Xavi"` → `"Xavi"`.

---

## Retrieval Pipeline (retrieval.py)

### FactRetriever

```python
FactRetriever(
    store: MemoryStore,
    temporal_decay_half_life: int = 0,   # days, 0 = disabled
    fts_weight: float = 0.4,
    jaccard_weight: float = 0.3,
    hrr_weight: float = 0.3,
    hrr_dim: int = 1024,
)
```

### search() — Hybrid pipeline

```
User query: "editor config"
         │
         ▼
┌─────────────────────────────────┐
│ Stage 1: FTS5 candidates        │
│ • Tokenize → "editor" "config"  │
│ • Sanitize FTS5 special chars   │
│ • MATCH "editor OR config"      │
│ • Get top limit*3 candidates    │
│ • Normalize rank by position    │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Stage 2: Per-candidate scoring  │
│ ┌───────────────────────────┐   │
│ │ FTS rank      × 0.4       │   │
│ │ Jaccard sim   × 0.3       │   │
│ │ HRR sim       × 0.3       │   │
│ │ ─────────────────────     │   │
│ │ relevance = weighted sum  │   │
│ │ score = relevance × trust │   │
│ │ (× temporal decay if set)│   │
│ └───────────────────────────┘   │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Stage 3: Sort by score DESC     │
│ Return top limit                │
└─────────────────────────────────┘
```

### probe() — Entity recall

Uses HRR algebra, not keyword matching:

1. Encode entity as `bind(atom(entity), ROLE_ENTITY)`
2. Try category-specific memory bank first, unbind the probe key from the bank vector
3. Score individual facts by comparing residual to content vector
4. Falls back to FTS5 keyword search if numpy unavailable, no bank, or no vectors

### related() — Structural adjacency

Finds facts connected through shared context:

1. Encode entity as bare atom (not role-bound — wants ANY structural match)
2. Unbind from each fact vector
3. Compare residual against BOTH role vectors (`ROLE_ENTITY` and `ROLE_CONTENT`)
4. Takes max similarity — entity could appear in either role
5. Falls back to FTS5 if no numpy or empty result set

### reason() — Compositional (multi-entity AND)

Finds facts related to ALL entities simultaneously (like a vector-space JOIN):

1. Encode each entity as `bind(atom(entity), ROLE_ENTITY)` probe key
2. Unbind each probe key from each fact vector
3. Compare each residual to `ROLE_CONTENT`
4. Score = **min** across entities (AND semantics — all must be present, vs OR which would use mean/max)
5. Falls back to FTS5 with all keywords if no numpy or empty entities

### contradict() — Memory hygiene

Facts that share entities but make different claims:

1. Get all facts with HRR vectors (max 500 for O(n²) safety)
2. Build entity sets per fact
3. Compute entity Jaccard overlap per pair — skip pairs with overlap < 0.3
4. Remaining pairs: `contradiction_score = entity_overlap × (1 - content_similarity)`
5. Higher score = more contradictory
6. Returns `[]` silently if numpy unavailable

### _fts_candidates() — FTS5 with OR semantics

- Tokenizes query, sanitizes FTS5-special characters (`"()-~^/+'`)
- Joins tokens with `OR` for wider recall: `"editor OR config"` (vs FTS5 default AND)
- If sanitization eats everything, tries raw query
- Normalizes rank by position: `(n - i) / n` (version-independent)
- Falls back to `[]` if FTS5 MATCH explodes (never throws)
- Guarded query with trust_score and category filters

---

## Query Flow

### Example: `fact_store(action="search", query="editor config")`

1. **Tool dispatch** — `HolographicMemoryProvider._handle_fact_store()` receives `fact_store` call
2. **Retriever.search()** — `FactRetriever.search("editor config", min_trust=0.3, limit=10)`
3. **FTS5 candidate fetch** — `_fts_candidates("editor config", None, 0.3, 30)`
   - Tokenize → `{"editor", "config"}`
   - Sanitize → `["editor", "config"]` (no special chars)
   - FTS5 MATCH: `"editor OR config"` → up to 30 candidates from facts_fts
   - Normalize ranks positionally
4. **Per-candidate scoring** (for each of the ≤30 candidates)
   - Jaccard(query_tokens, content_tokens ∪ tag_tokens)
   - HRR similarity: `similarity(encode_text("editor config"), fact_hrr_vector)`
   - `relevance = 0.4 × fts + 0.3 × jaccard + 0.3 × hrr`
   - `score = relevance × trust_score`
5. **Sort and return** — top 10 by score DESC
6. **Bump retrieval count** — `UPDATE facts SET retrieval_count = ... WHERE fact_id IN (...)`
7. **Strip HRR vectors** — remove raw bytes from output (not JSON serializable)
8. **Return** — `json.dumps({"results": [...], "count": 10})`

---

## MemoryProvider Integration

### Registration flow

The plugin system discovers all `plugins/memory/*/` directories and calls each `register(ctx)` function:

```
plugin_loader → register(ctx) → ctx.register_memory_provider(provider)
                              → MemoryManager.add_provider(provider)
```

### Single provider constraint

The MemoryManager enforces **one external memory provider active at a time**. When `memory.provider: holographic` is set:

1. MemoryManager checks available providers via `provider.is_available()`
2. Only the selected provider receives `initialize()`, tool registrations, and lifecycle hooks
3. The built-in `memory` tool (SQLite text entries in `~/.hermes/memories/`) **always runs alongside** — it is not a provider, it is a core tool
4. If you switch to a different provider (`memory.provider: honcho`), `holographic` is deactivated — no tool registration, no lifecycle callbacks

### Provider lifecycle in the agent

The agent calls providers in this sequence per session:

```
Agent startup:
  MemoryManager.select(provider="holographic")
  → provider.initialize(session_id, hermes_home, platform)
  → Register tool schemas via provider.get_tool_schemas()

Each turn:
  → provider.on_turn_start(turn, message)  # no-op for holographic
  → provider.prefetch(query)                # returns context to inject
  → Model generates response (tools available)
  → provider.sync_turn(user, assistant)     # no-op for holographic

Context compression:
  → provider.on_pre_compress(messages)      # no-op for holographic

Subagent completion:
  → provider.on_delegation(task, result)    # no-op for holographic

Session switch (/reset, /branch, /new):
  → provider.on_session_switch(new_id, ...) # no-op for holographic

Session end (exit, timeout):
  → provider.on_session_end(messages)       # auto-extract if configured
  → provider.shutdown()
```

### Tool registration

`get_tool_schemas()` returns `[FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]`. These schemas are merged with the agent's tool set at startup. Tool calls are dispatched via `handle_tool_call()`.

---

## Holographic vs Built-in Memory

Hermes has **two memory systems that run in parallel**:

| | `memory` (built-in) | `fact_store` (holographic) |
|---|---|---|
| **Type** | Flat text entries | Structured SQLite facts |
| **Actions** | `add`, `replace`, `remove` | `add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list` |
| **Location** | `~/.hermes/MEMORY.md` (markdown files) | `~/.hermes/memory_store.db` (SQLite) |
| **Retrieval** | Injected as static system prompt blocks | Hybrid: FTS5 + Jaccard + HRR + trust |
| **Structure** | Category: `user` or `memory` | Category: `user_pref`, `project`, `tool`, `general` |
| **Entities** | None | Automatic extraction + resolution |
| **Compositional** | No | Yes (HRR algebra) |
| **Trust** | No | Yes (0.0–1.0, trained via feedback) |
| **Capacity** | Soft limit by context injection size | ~256 facts before HRR degradation (dim=1024) |
| **Profile-scoped** | Per-profile files | Per-profile SQLite DB |

### Bridge: `on_memory_write()`

The holographic provider bridges to built-in memory via `on_memory_write()`. When the agent calls `memory(action='add', ...)`:

1. MemoryManager notifies all active memory providers via `on_memory_write()`
2. `HolographicMemoryProvider.on_memory_write()` intercepts the write
3. Creates a corresponding fact:
   - `target="user"` → `category="user_pref"`
   - `target="memory"` → `category="general"`
4. The `memory` entry and `fact_store` entry **coexist independently** — deleting one does NOT delete the other

⚠️ **Limitation**: The bridge only mirrors `add`. It does NOT mirror `replace` or `remove` — if you delete a memory entry, the corresponding fact (if any) stays behind.

### When to use which

- **`memory`** — general knowledge, preferences that should always be in the system prompt, user profile info
- **`fact_store`** — structured facts you'll query later, anything needing entity search, multi-entity reasoning, contradiction detection

They complement each other. Use both.

---

## Hook Coverage

The `MemoryProvider` ABC defines these hooks. Here's what holographic implements vs what it ignores:

| Hook | Implemented? | Behavior |
|------|-------------|----------|
| `initialize()` | ✅ Yes | Creates MemoryStore + FactRetriever |
| `system_prompt_block()` | ✅ Yes | Returns fact count summary |
| `prefetch()` | ✅ Yes | Searches and returns top 5 facts |
| `queue_prefetch()` | ❌ No | Default no-op (background prefetch for next turn) |
| `sync_turn()` | ✅ No-op | Explicit facts only — intentionally skipped |
| `handle_tool_call()` | ✅ Yes | Routes fact_store + fact_feedback |
| `get_tool_schemas()` | ✅ Yes | 2 schemas |
| `shutdown()` | ✅ Yes | Clears references |
| `get_config_schema()` | ✅ Yes | 4 config fields |
| `save_config()` | ✅ Yes | Writes to config.yaml |
| `is_available()` | ✅ Yes | Always True (SQLite) |
| `on_turn_start()` | ❌ No | Default no-op (inherited) |
| `on_session_switch()` | ❌ No | Default no-op — mid-session session_id changes don't affect the store |
| `on_pre_compress()` | ❌ No | Default no-op — no pre-compression extraction |
| `on_memory_write()` | ✅ Yes | Mirrors memory adds as facts |
| `on_session_end()` | ✅ Yes | Auto-extraction if configured |
| `on_delegation()` | ❌ No | Default no-op — no subagent observation |

---

## Configuration

### Setup

```bash
hermes memory setup              # interactive — select "holographic"
hermes config set memory.provider holographic   # direct
```

### Config keys (`plugins.hermes-memory-store` in config.yaml)

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path (supports `$HERMES_HOME`, `${HERMES_HOME}`, and `~` expansion) |
| `auto_extract` | `false` | Auto-extract facts at end of session from conversation messages |
| `default_trust` | `0.5` | Trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions (8 KB per vector) |

### Not exposed in config schema (code defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_trust_threshold` | `0.3` | Minimum trust for search results |
| `hrr_weight` | `0.3` | HRR weight in retrieval scoring |
| `temporal_decay_half_life` | `0` | Days, 0 = disabled |
| `fts_weight` | `0.4` | FTS5 weight in retrieval (auto-adjusts if no numpy) |
| `jaccard_weight` | `0.3` | Jaccard weight in retrieval (auto-adjusts if no numpy) |

If numpy is unavailable, weights auto-redistribute: `fts=0.6, jaccard=0.4, hrr=0.0`.

---

## Performance & Capacity

### Capacity limits

| Dimension | Value | Note |
|-----------|-------|------|
| **HRR practical capacity** | ~256 facts | `dim / 4 = 1024 / 4`. Beyond this, SNR < 2.0 and retrieval accuracy degrades |
| **HRR storage per fact** | 8 KB | float64 × 1024 dimensions |
| **Memory bank per category** | 8 KB | same size as individual fact vectors |
| **FTS5** | No hard limit | SQLite FTS5 handles millions of rows efficiently |
| **DB size at 256 facts** | ~3 MB | facts data + HRR blobs + FTS5 index + entity tables |
| **contradict() max input** | 500 facts | O(n²) guard — beyond this only checks most recent 500 |

### Expected latency (approximate, local disk)

| Operation | Typical time | Notes |
|-----------|-------------|-------|
| `add_fact()` | 5-20 ms | Entity extraction + HRR compute (numpy-dependent) |
| `search()` with 30 facts | 10-50 ms | FTS5 candidate fetch dominates at low counts |
| `search()` with 256 facts | 30-100 ms | HRR similarity on 30 candidates; scales mostly with candidate count, not total facts |
| `probe()` | 20-80 ms | Bank unbind + per-fact scoring |
| `reason()` with 2 entities | 30-100 ms | Similar to search, two unbind ops per fact |
| `contradict()` at 30 facts | 5-20 ms | 435 pairs, fast |
| `contradict()` at 256 facts | 200-800 ms | ~33K pairs, O(n²) is real cost |
| `prefetch()` | < 50 ms | Returns top 5, same pipeline as search |
| `_auto_extract()` at end of session | 10-100 ms | Pattern matching over user messages only |

### No-numpy fallback

When numpy is not available:

- `bind`, `unbind`, `bundle`, `similarity`, `encode_text`, `encode_fact`, `encode_atom` → **not available** (raise RuntimeError)
- `MemoryStore.add_fact()` → still works, but HRR vector column stays NULL
- `FactRetriever.search()` → uses FTS5 + Jaccard only (weights shift to 0.6/0.4)
- `probe()`, `related()`, `reason()` → fall back to FTS5 keyword search
- `contradict()` → returns `[]` silently
- `rebuild_all_vectors()` → returns 0 (no-op)

### Scaling beyond dim/4

Options if you exceed ~256 facts:

1. **Increase `hrr_dim`** to 2048 or 4096 (costs more storage per vector: 16 KB / 32 KB)
2. **Use categories aggressively** — memory banks are per-category, so facts are distributed across separate vector spaces
3. **Rely on FTS5 for bulk search** — FTS5 doesn't degrade with fact count; only HRR does

---

## Maintenance & Reindexing

### When to reindex

- After changing the entity extractor logic
- After database migration or schema changes
- After bulk operations that bypass triggers
- After FTS5 desync (delete/insert outside triggers)

### Verify health

```sql
-- Facts without entities
SELECT f.fact_id, substr(f.content,1,80)
FROM facts f
WHERE NOT EXISTS (SELECT 1 FROM fact_entities fe WHERE fe.fact_id = f.fact_id);

-- Facts without HRR vectors
SELECT fact_id, substr(content,1,80) FROM facts
WHERE hrr_vector IS NULL OR length(hrr_vector) = 0;

-- DB stats
SELECT COUNT(*) FROM facts;
SELECT COUNT(*) FROM fact_entities;
SELECT COUNT(*) FROM memory_banks;

-- Integrity check
PRAGMA integrity_check;
```

### Rebuild FTS5 (after bulk ops)

```sql
INSERT INTO facts_fts(facts_fts) VALUES('rebuild');
```

**Critical**: FTS5 virtual table does NOT auto-sync after direct INSERT/DELETE on `facts`. Always rebuild after bulk operations.

### Gateway restart

**NOT needed.** Changes to the plugin Python code are loaded on the next session start. The gateway imports plugins fresh per session.

---

## Testing

**Status: No dedicated unit tests exist for this plugin.**

The plugin was contributed via external PR (#2351 by dusterbloom) and integrated without a dedicated test suite. This is a gap that should be addressed.

### Recommended test coverage

| Component | Test class | What to test |
|-----------|-----------|-------------|
| `holographic.py` | `test_hrr_encode` | Atom determinism, bind/unbind roundtrip, bundle identity, similarity edge cases, serialization roundtrip, SNR warning |
| `holographic.py` | `test_hrr_encode_fact` | Role binding correctness, entity encoding, unbind extraction |
| `store.py` | `test_store_crud` | add/update/remove/list, deduplication on UNIQUE, trust clamping |
| `store.py` | `test_store_entities` | Entity extraction regex coverage, alias normalization, stop word exclusion, acronym filter |
| `store.py` | `test_store_banks` | Bank creation, rebuild on update, rebuild on delete |
| `store.py` | `test_store_fts_triggers` | FTS5 synced after insert/delete/update |
| `retrieval.py` | `test_retriever_search` | Hybrid pipeline, category filter, min_trust filter, limit |
| `retrieval.py` | `test_retriever_probe` | Bank unbind, direct vector scoring, keyword fallback |
| `retrieval.py` | `test_retriever_reason` | Multi-entity AND semantics, min scoring |
| `retrieval.py` | `test_retriever_contradict` | Entity overlap threshold, O(n²) guard |
| `retrieval.py` | `test_fts_candidates` | OR semantics, special char sanitization, positional ranking |
| `__init__.py` | `test_provider_lifecycle` | initialize, handle_tool_call, on_memory_write mapping, on_session_end |

### Running existing agent tests

The broader Hermes test suite uses pytest:

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
pytest tests/ -xvs -k "memory"
```

To run the full suite (~17k tests):

```bash
pytest tests/ -x
```

---

## Known Bugs & Pitfalls

### Bug 1: `fts_rank` inverted (FIXED)

**Symptom**: FTS5 raw rank sign varies between SQLite versions. `abs(rank)/max_rank` inverted the order — worst match got 1.0, best got 0.0.

**Fix**: Positional ranking: `fact["fts_rank"] = (n - i) / n` — independent of SQLite version. Applied in `_fts_candidates()`.

### Bug 2: FTS5 MATCH fails silently with special chars (FIXED)

**Symptom**: Queries with `C++`, `()`, `"`, `-`, `~`, `^`, `/` → FTS5 MATCH explodes → returns `[]` with no error.

**Fix**: Token sanitization: `tok.translate(...'\"()-~^/+'...)`. Applied in `_fts_candidates()`.

### Bug 3: `_bump_retrieval` without lock (PENDING)

**Symptom**: `search()` writes `retrieval_count` without `self.store._lock`.

**Severity**: Low in CLI (single-threaded), medium in concurrent gateway.

### Bug 4: `contradict()` fails silently without numpy (PENDING)

**Symptom**: Returns `[]` with no warning.

**Severity**: Low. User sees zero contradictions but no error.

### Bug 5: FTS5 desync after bulk ops

**Symptom**: After direct INSERT/DELETE on `facts` table, `facts_fts` virtual table is stale.

**Fix**: `db.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")` after bulk operations.

### Pitfall: `rebuild_all_vectors()` != full reindex

`rebuild_all_vectors()` only re-encodes HRR vectors from TEXT. It does NOT extract entities from facts that are missing them.

For a full reindex, the order matters:
1. Extract entities → `_extract_entities()` + linker
2. Compute HRR vectors → `_compute_hrr_vector()` (uses the entities)
3. Rebuild memory banks → `_rebuild_bank()`
4. Rebuild FTS5 if bulk ops were used

### Pitfall: `memory_banks` column name

The column is `bank_name`, NOT `name`. Common mistake in SQL queries.

### Pitfall: `on_memory_write` only mirrors `add`

`replace` and `remove` actions on the `memory` tool are NOT mirrored. If you update or delete a memory entry, the corresponding fact (if ever created by the bridge) remains unchanged.

### Pitfall: auto_extract is regex-based, not LLM-based

The `_auto_extract_facts()` method uses simple regex patterns (`I prefer...`, `we decided...`) — it does NOT call the LLM to intelligently extract facts. It misses anything that doesn't match the literal patterns.

---

## Diagnosis Skill

For troubleshooting, diagnosis, and repair procedures, see the `holographic-memory-diagnosis` skill:

```
skill_view(name='holographic-memory-diagnosis')
```

Contains:
- Current state of DBs and configuration
- Entity extractor patterns and stop words
- FTS5 OR logic implementation details
- Known bugs with fix history
- Reindex procedure script
- Commands for health checks

---

## References

- Plate, T. A. (1995). *Holographic Reduced Representations*. IEEE Transactions on Neural Networks.
- Gayler, R. W. (2004). *Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience*. arXiv:cs/0402059.
- Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.
