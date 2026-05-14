"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def _bump_retrieval(self, results: list) -> None:
        """Increment retrieval_count for retrieved facts."""
        if not results:
            return
        conn = self.store._conn
        ids = [r["fact_id"] for r in results if "fact_id" in r]
        if ids:
            conn.execute(
                "UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ("
                + ",".join("?" for _ in ids)
                + ")",
                ids,
            )
        conn.commit()

    @staticmethod
    def _fact_entity_names(conn, fact_id: int) -> list[str]:
        """Return entity names linked to a fact_id."""
        rows = conn.execute(
            """
            SELECT e.name FROM entities e
            JOIN fact_entities fe ON fe.entity_id = e.entity_id
            WHERE fe.fact_id = ?
            ORDER BY e.name
            """,
            (fact_id,),
        ).fetchall()
        return [r["name"] for r in rows]

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        self._bump_retrieval(results)
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Query facts by entity — direct lookup first, HRR fallback.

        1. Look up the entity name in the entities table (exact or LIKE match).
        2. If found, return facts linked via fact_entities, ranked by
           retrieval_count * trust_score (most-used & most-trusted first).
        3. If no direct entity match, fall back to FTS5 + HRR scoring.

        Results include an "entities" key with linked entity names for
        downstream consumers that need to know what each fact is about.
        """
        conn = self.store._conn

        # Stage 1: Direct entity lookup in entities table
        entity_rows = conn.execute(
            """
            SELECT fe.fact_id
            FROM fact_entities fe
            JOIN entities e ON e.entity_id = fe.entity_id
            WHERE LOWER(e.name) = LOWER(?)
            """,
            (entity,),
        ).fetchall()

        if entity_rows:
            # Direct hits — fetch full facts
            fact_ids = [r["fact_id"] for r in entity_rows]
            where = "WHERE f.fact_id IN ({})".format(",".join("?" * len(fact_ids)))
            params: list = list(fact_ids)
            if category:
                where += " AND f.category = ?"
                params.append(category)

            rows = conn.execute(
                f"""
                SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.retrieval_count, f.helpful_count, f.created_at, f.updated_at,
                       f.hrr_vector
                FROM facts f
                {where}
                ORDER BY f.retrieval_count * f.trust_score DESC
                LIMIT ?
                """,
                params + [limit],
            ).fetchall()

            results = []
            for row in rows:
                fact = dict(row)
                fact.pop("hrr_vector", None)
                fact["entities"] = self._fact_entity_names(conn, fact["fact_id"])
                # Direct lookup = guaranteed entity match, score = 1.0 × trust
                fact["score"] = fact["trust_score"]
                results.append(fact)

            self._bump_retrieval(results)
            return results

        # Stage 2: No direct entity match — try FTS5 + HRR fallback
        # First try keyword search (catches partial matches)
        results = self.search(entity, category=category, limit=limit)

        # If nothing found, attempt full HRR probe over all facts
        if not results and hrr._HAS_NUMPY:
            results = self._hrr_probe(entity, category, limit)

        return results

    def _hrr_probe(
        self,
        entity: str,
        category: str | None,
        limit: int,
    ) -> list[dict]:
        """HRR-based entity probe — fallback only.

        Scores facts by structural entity presence using HRR algebra.
        Noisy by nature; prefer direct entity lookup via probe().
        """
        conn = self.store._conn

        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            residual = hrr.unbind(fact_vec, probe_key)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        self._bump_retrieval(results)
        return results

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        self._bump_retrieval(results)
        return results

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — relational JOIN over entity table.

        Finds facts linked to ALL specified entities simultaneously via the
        fact_entities relational table. This is structural reasoning: a fact
        must be bound to every entity to be a match (AND semantics).

        Scoring: uses trust_score primarily, with HRR vector similarity as
        a secondary signal to rank within the matched set.

        Example: reason(["erik", "OCI"]) finds facts where both erik AND OCI
        are linked entities.

        Falls back to FTS5 search if numpy unavailable or no exact match.
        """
        if not entities:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn

        # Step 1: Relational JOIN — find fact_ids linked to ALL entities
        # For each entity, get the set of fact_ids it appears in
        entity_fact_sets: list[set[int]] = []
        entity_id_map: dict[str, set[int]] = {}

        for entity in entities:
            rows = conn.execute(
                """
                SELECT e.entity_id, fe.fact_id
                FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE e.name = ?
                """,
                (entity,),
            ).fetchall()
            if not rows:
                # Entity not found — try case-insensitive match
                rows = conn.execute(
                    """
                    SELECT e.entity_id, fe.fact_id
                    FROM entities e
                    JOIN fact_entities fe ON fe.entity_id = e.entity_id
                    WHERE LOWER(e.name) = LOWER(?)
                    """,
                    (entity,),
                ).fetchall()

            entity_fact_sets.append({r["fact_id"] for r in rows})
            entity_id_map[entity] = {r["entity_id"] for r in rows}

        # Intersection: facts that have ALL entities
        if not entity_fact_sets or not all(entity_fact_sets):
            # At least one entity has no facts — fallback to search
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        matched_ids = set.intersection(*entity_fact_sets)

        if not matched_ids:
            # No fact has ALL entities — try partial match (OR) with scoring
            # This is softer: facts with more entities score higher
            entity_fact_counts: dict[int, int] = {}
            for fact_set in entity_fact_sets:
                for fid in fact_set:
                    entity_fact_counts[fid] = entity_fact_counts.get(fid, 0) + 1

            # Only consider facts with at least half the entities
            min_hits = max(1, len(entities) // 2)
            matched_ids = {
                fid for fid, count in entity_fact_counts.items()
                if count >= min_hits
            }

            if not matched_ids:
                query = " ".join(entities)
                return self.search(query, category=category, limit=limit)

        # Step 2: Fetch matched facts with vectors
        placeholders = ",".join("?" for _ in matched_ids)
        where = f"WHERE hrr_vector IS NOT NULL AND fact_id IN ({placeholders})"
        params: list = list(matched_ids)

        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Step 3: Score — primary: trust_score, secondary: HRR content overlap with query
        if hrr._HAS_NUMPY:
            query_vec = hrr.encode_text(" ".join(entities), self.hrr_dim)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
            query_with_role = hrr.bind(query_vec, role_content)

            scored = []
            for row in rows:
                fact = dict(row)
                fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
                # How many of the entities are bound in this fact's vector
                sim = hrr.similarity(fact_vec, query_with_role)
                # Count exact entity matches for prioritization
                matched_entities = []
                for entity in entities:
                    eids = entity_id_map.get(entity, set())
                    if eids:
                        # Check if any of this entity's IDs are linked to this fact
                        fact_eids = set(
                            r[0] for r in conn.execute(
                                "SELECT entity_id FROM fact_entities WHERE fact_id = ?",
                                (fact["fact_id"],),
                            )
                        )
                        if isinstance(eids, set) and fact_eids & eids:
                            matched_entities.append(entity)

                fact["entities"] = matched_entities
                entity_match_ratio = len(matched_entities) / len(entities)
                fact["score"] = entity_match_ratio * fact["trust_score"]
                scored.append(fact)

            scored.sort(key=lambda x: (x["score"], x.get("trust_score", 0)), reverse=True)
        else:
            # No numpy — just return by trust_score
            scored = []
            for row in rows:
                fact = dict(row)
                fact["score"] = fact["trust_score"]
                scored.append(fact)

        results = scored[:limit]
        self._bump_retrieval(results)
        return results

    def contradict(
        self,
        entity: str | None = None,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        If 'entity' is specified, only checks facts linked to that entity.
        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Build WHERE clause and FROM/JOIN parts
        joins = ""
        conditions = ["f.hrr_vector IS NOT NULL"]
        params: list = []

        if entity:
            joins = """
                JOIN fact_entities fe2 ON fe2.fact_id = f.fact_id
                JOIN entities e2 ON e2.entity_id = fe2.entity_id
            """
            conditions.append("LOWER(e2.name) = LOWER(?)")
            params.append(entity)

        if category:
            conditions.append("f.category = ?")
            params.append(category)

        where_clause = " AND ".join(conditions)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {joins}
            WHERE {where_clause}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        self._bump_retrieval(results)
        return results

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.

        Tokenizes the query and uses OR semantics so that facts matching
        ANY of the query terms are returned (wider recall), not just those
        matching ALL terms (AND — the FTS5 default).

        Tokens containing FTS5-special characters (parentheses, quotes,
        operators) are sanitized to avoid silent MATCH failures.
        """
        conn = self.store._conn

        # Tokenize query, sanitize each token for FTS5 safety
        raw_tokens = self._tokenize(query)
        safe_tokens = []
        for tok in raw_tokens:
            # Strip FTS5-reserved characters: ( ) " - ^ ~ / +
            clean = tok.translate(str.maketrans("", "", '"()-~^/+'))
            if clean:
                safe_tokens.append(clean)

        if safe_tokens:
            fts_query = " OR ".join(safe_tokens)
        else:
            # If sanitization ate everything, try the raw query (let FTS5 handle it)
            fts_query = query

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(fts_query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rows are already sorted by rank ASC (best first),
        # so we use position-based ranking instead of raw rank values, which
        # can vary in sign across SQLite versions.
        # First (best) result → 1.0, last result → 1/n.
        results = []
        n = len(rows)
        for i, row in enumerate(rows):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = (n - i) / n
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
