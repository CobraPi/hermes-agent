"""``skill_search`` — relevance-ranked skill lookup over the skills snapshot.

Complements the existing skills tools: ``skills_list`` dumps every skill,
``skill_view`` loads one. ``skill_search`` lets the agent find the *few* skills
relevant to a task by keyword/topic relevance instead of scanning the whole
index — useful as libraries grow and when categories are demoted to names-only
in the cached prompt. Zero-LLM: a local FTS5 query via
:class:`agent.fts_index.FtsLite`.

The index is built from the SAME on-disk snapshot
(``~/.hermes/.skills_prompt_snapshot.json``) that backs the cached system-prompt
skills index (``agent/prompt_builder.py``), so results stay in lockstep with what
the agent already sees. It is rebuilt only when that snapshot's mtime/size
manifest changes, so steady-state calls are a single local query. Platform- and
config-disabled skills are filtered out of results at query time, exactly as the
prompt index filters them, so search never surfaces an unusable skill.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_SKILL_INDEX_LOCK = threading.Lock()
_SKILL_INDEX: Optional[Any] = None  # FtsLite, lazily created


def _index_path() -> Path:
    return get_hermes_home() / ".skill_index.db"


def _get_index() -> Tuple[Optional[Any], Optional[dict]]:
    """Return ``(FtsLite, snapshot)``, refreshing the index if the snapshot moved.

    The snapshot is the single source of truth. If it is missing or stale we let
    ``build_skills_system_prompt`` rebuild it (its cold path writes the snapshot),
    then load it. Returns ``(None, None)`` when there are no skills at all.
    """
    global _SKILL_INDEX
    from agent.fts_index import FtsLite
    from agent.prompt_builder import (
        _load_skills_snapshot,
        build_skills_system_prompt,
        clear_skills_system_prompt_cache,
    )
    from agent.skill_utils import get_skills_dir

    skills_dir = get_skills_dir()
    snapshot = _load_skills_snapshot(skills_dir)
    if snapshot is None:
        # Snapshot missing or stale (its mtime/size manifest no longer matches).
        # build_skills_system_prompt's in-process LRU is keyed by tools/platform,
        # NOT the manifest, so it can short-circuit the rewrite when skills change
        # mid-process. Clear that LRU first so the cold path runs and rewrites the
        # snapshot on disk; then reload it.
        try:
            clear_skills_system_prompt_cache()
            build_skills_system_prompt()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("skill_search could not materialize snapshot: %s", exc)
        snapshot = _load_skills_snapshot(skills_dir)
    if not snapshot:
        return None, None

    with _SKILL_INDEX_LOCK:
        if _SKILL_INDEX is None:
            _SKILL_INDEX = FtsLite(
                _index_path(),
                table="skills",
                columns=[
                    ("name", "TEXT"),          # frontmatter name — the loadable id
                    ("skill_name", "TEXT"),    # directory name
                    ("category", "TEXT"),
                    ("description", "TEXT"),
                    ("tags", "TEXT"),
                    ("platforms", "TEXT"),     # JSON list, for query-time filtering
                ],
                fts_columns=["name", "description", "tags"],
            )
        _refresh_index(_SKILL_INDEX, snapshot)
        return _SKILL_INDEX, snapshot


def _refresh_index(idx: Any, snapshot: dict) -> None:
    """Rebuild the FTS index from the snapshot iff its manifest changed."""
    manifest_json = json.dumps(snapshot.get("manifest"), sort_keys=True)
    if idx.get_meta("manifest") == manifest_json:
        return
    rows: List[Dict[str, Any]] = []
    for entry in snapshot.get("skills", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("frontmatter_name") or entry.get("skill_name") or ""
        if not name:
            continue
        rows.append({
            "name": str(name),
            "skill_name": str(entry.get("skill_name") or ""),
            "category": str(entry.get("category") or "general"),
            "description": str(entry.get("description") or ""),
            "tags": " ".join(str(t) for t in (entry.get("tags") or [])),
            "platforms": json.dumps(entry.get("platforms") or []),
        })
    idx.rebuild_from(rows)
    idx.set_meta("manifest", manifest_json)


def skill_search(query: str, limit: int = 8, task_id: str = None) -> str:
    """Relevance-rank skills against ``query`` and return the top matches.

    Args:
        query: Keywords / topic / task description to match.
        limit: Maximum number of results (default 8).
        task_id: Unused; accepted for tool-handler signature parity.

    Returns:
        JSON string ``{"success", "query", "results": [{name, category,
        description, score}], "count"}``. Empty query → empty results (not an
        error).
    """
    try:
        q = (query or "").strip()
        if not q:
            return json.dumps(
                {"success": True, "query": "", "results": [], "count": 0},
                ensure_ascii=False,
            )
        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 8

        idx, snapshot = _get_index()
        if idx is None or snapshot is None:
            return json.dumps(
                {"success": True, "query": q, "results": [], "count": 0,
                 "message": "No skills available to search."},
                ensure_ascii=False,
            )

        from agent.skill_utils import (
            get_disabled_skill_names,
            skill_matches_platform,
        )
        try:
            from gateway.session_context import get_session_env
            platform_hint = (
                os.environ.get("HERMES_PLATFORM")
                or get_session_env("HERMES_SESSION_PLATFORM")
                or ""
            )
        except Exception:
            platform_hint = os.environ.get("HERMES_PLATFORM") or ""
        disabled = get_disabled_skill_names(platform_hint or None)

        # Over-fetch then filter, so platform/disabled drops don't shrink the
        # result set below ``limit``.
        raw = idx.search(q, limit=max(limit * 4, limit))
        results: List[Dict[str, Any]] = []
        for row in raw:
            name = row.get("name") or ""
            skill_name = row.get("skill_name") or ""
            if not name or name in disabled or skill_name in disabled:
                continue
            try:
                platforms = json.loads(row.get("platforms") or "[]")
            except Exception:
                platforms = []
            if not skill_matches_platform({"platforms": platforms}):
                continue
            results.append({
                "name": name,
                "category": row.get("category") or "general",
                "description": row.get("description") or "",
                "score": round(float(row.get("_score") or 0.0), 4),
            })
            if len(results) >= limit:
                break

        return json.dumps(
            {
                "success": True,
                "query": q,
                "results": results,
                "count": len(results),
                "hint": "Use skill_view(name) to load a match's full content.",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return tool_error(str(e), success=False)


def _clear_index_cache() -> None:
    """Drop the in-process FtsLite handle (used by tests)."""
    global _SKILL_INDEX
    with _SKILL_INDEX_LOCK:
        if _SKILL_INDEX is not None:
            try:
                _SKILL_INDEX.close()
            except Exception:
                pass
        _SKILL_INDEX = None


SKILL_SEARCH_SCHEMA = {
    "name": "skill_search",
    "description": (
        "Search available skills by relevance (keywords, topic, or task "
        "description) and get the top matches with name + description. Faster "
        "than skills_list when you know what you're after; load a match with "
        "skill_view(name)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you're looking for — keywords, topic, or a short task description.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 8).",
            },
        },
        "required": ["query"],
    },
}


def _check_skills_requirements() -> bool:
    # Deferred import keeps tools/ auto-import order independent of skills_tool.
    from tools.skills_tool import check_skills_requirements
    return check_skills_requirements()


registry.register(
    name="skill_search",
    toolset="skills",
    schema=SKILL_SEARCH_SCHEMA,
    handler=lambda args, **kw: skill_search(
        query=args.get("query", ""),
        limit=args.get("limit", 8),
        task_id=kw.get("task_id"),
    ),
    check_fn=_check_skills_requirements,
    emoji="🔎",
)
