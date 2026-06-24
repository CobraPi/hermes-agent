"""Built-in memory provider — FTS5 relevance recall over MEMORY.md / USER.md.

The curated memory files are injected into the system prompt as a frozen
whole-file snapshot (:class:`tools.memory_tool.MemoryStore`). That snapshot is
bounded by a per-target char budget and frozen for prefix-cache stability, so it
cannot grow and cannot be relevance-ranked. This provider complements it:

  * it indexes *every* live memory entry — including ones that overflow the
    snapshot's char budget — into a local FTS5 sidecar
    (``~/.hermes/memories/.memory_index.db``);
  * each turn it recalls the few entries most relevant to the user's message and
    returns them through the existing cache-safe ``<memory-context>`` channel
    (the user message, never the cached prefix — see
    ``agent/memory_manager.build_memory_context_block``);
  * entries already in the frozen snapshot *core* are excluded, so nothing is
    duplicated. Under the char budget the core IS the whole file, so recall finds
    nothing new and the prompt is byte-identical (cache preserved); over the
    budget, recall surfaces the relevant overflow.

Zero network, zero embeddings — a single local FTS5 query per turn, which fits
the single-local-model weak-hardware target. It is registered as the always-first
provider so it coexists with at most one external plugin
(``agent/memory_manager.MemoryManager``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


class BuiltinMemoryProvider(MemoryProvider):
    """Indexes curated memory entries and recalls the relevant overflow per turn."""

    def __init__(self, memory_store: Any, *, top_k: int = 5,
                 relevance_ratio: float = 0.5, db_path: "str | Path | None" = None) -> None:
        self._store = memory_store
        self._top_k = max(1, int(top_k))
        # Keep only entries scoring within this fraction of the best match, so
        # weak OR-matches aren't injected into every turn's context.
        self._relevance_ratio = float(relevance_ratio)
        self._db_path = Path(db_path).expanduser() if db_path else None
        self._index: Optional[Any] = None  # FtsLite

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        # Always ready when a memory store exists — purely local, no creds.
        return self._store is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        from agent.fts_index import FtsLite

        if self._db_path is None:
            hermes_home = kwargs.get("hermes_home")
            if hermes_home:
                base = Path(hermes_home) / "memories"
            else:
                from tools.memory_tool import get_memory_dir
                base = get_memory_dir()
            self._db_path = base / ".memory_index.db"
        try:
            self._index = FtsLite(
                self._db_path,
                table="entries",
                columns=[
                    ("target", "TEXT"),
                    ("content", "TEXT"),
                    ("content_hash", "TEXT"),
                ],
                fts_columns=["content"],
            )
            self._reindex_all()
        except Exception as exc:
            logger.debug("Built-in memory index init failed (recall disabled): %s", exc)
            self._index = None

    def _reindex_all(self) -> None:
        """Rebuild the index from the store's current live entries.

        Cheap (entries are few and short) and always correct — the store's lists
        already reflect any just-applied write by the time we are called. Poison
        entries are scanned out here so they never enter the index.
        """
        if self._index is None:
            return
        from tools.memory_tool import _entry_hash
        from tools.threat_patterns import scan_for_threats

        rows: List[Dict[str, Any]] = []
        seen: set = set()
        targets = (
            ("memory", getattr(self._store, "memory_entries", []) or []),
            ("user", getattr(self._store, "user_entries", []) or []),
        )
        for target, entries in targets:
            for entry in entries:
                text = (entry or "").strip()
                if not text or text.startswith("[BLOCKED:"):
                    continue
                if scan_for_threats(text, scope="strict"):
                    continue
                h = _entry_hash(text)
                if h in seen:
                    continue
                seen.add(h)
                rows.append({"target": target, "content": text, "content_hash": h})
        self._index.rebuild_from(rows)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall the top relevant overflow entries for ``query`` (cache-safe)."""
        if self._index is None or not query or not query.strip():
            return ""
        try:
            core = self._store.snapshot_core_hashes()
        except Exception:
            core = set()
        try:
            # Over-fetch so excluded (in-core) entries don't starve the result.
            hits = self._index.search(query, limit=self._top_k + len(core) + 4)
        except Exception:
            return ""

        from tools.threat_patterns import scan_for_threats

        candidates: List[tuple] = []  # (content, score)
        for row in hits:
            if row.get("content_hash") in core:
                continue  # already shown in the frozen snapshot — don't duplicate
            content = (row.get("content") or "").strip()
            if not content or content.startswith("[BLOCKED:"):
                continue
            if scan_for_threats(content, scope="strict"):
                continue  # backstop: never surface poison via recall
            candidates.append((content, row.get("_score")))
            if len(candidates) >= self._top_k:
                break

        if not candidates:
            return ""

        # Relevance gate: drop weak OR-matches. FTS5 ``rank`` is negative
        # (more negative = better), and hits arrive best-first, so the first
        # candidate is the strongest. Keep entries scoring within
        # ``relevance_ratio`` of it. LIKE-fallback hits (score None) bypass the
        # gate. Always keep at least the best match.
        best = candidates[0][1]
        gated = (
            isinstance(best, (int, float)) and best < 0 and self._relevance_ratio > 0
        )
        picked: List[str] = []
        for content, score in candidates:
            if gated and isinstance(score, (int, float)) and score > best * self._relevance_ratio:
                continue
            picked.append(content)
        if not picked:
            picked = [candidates[0][0]]

        body = "\n".join(f"- {c}" for c in picked)
        return "Saved memory relevant to this turn:\n" + body

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Recall is synchronous and cheap; nothing to pre-compute.
        return

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Keep the index in lockstep with the curated files.

        The memory tool has already mutated the store's live lists by the time
        this fires, so a full resync is both simplest and always correct. Entries
        are few, so the cost is negligible and off the inference path.
        """
        if self._index is None:
            return
        try:
            self._reindex_all()
        except Exception as exc:
            logger.debug("Built-in memory index resync failed: %s", exc)

    def on_session_switch(self, new_session_id: str, *, reset: bool = False, **kwargs) -> None:
        # On /reset the store reloads from disk; resync so recall matches.
        if reset and self._index is not None:
            try:
                self._reindex_all()
            except Exception:
                pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        if self._index is not None:
            try:
                self._index.close()
            except Exception:
                pass
            self._index = None
