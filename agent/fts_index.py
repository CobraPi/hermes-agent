"""Lightweight single-file SQLite + FTS5 index, factored out of SessionDB.

This is a small, reusable full-text store for sidecar indexes (built-in memory
recall, skill search). It deliberately does NOT touch ``state.db`` — each index
is its own file so a corrupt index can never threaten session history and the
indexes travel as plain sidecars with ``hermes backup``.

The query sanitizer and CJK helpers are *imported and delegated* to
:class:`hermes_state.SessionDB` (never copied) so the FTS5 query-escaping and
Chinese/Japanese/Korean handling can never drift from the battle-tested session
search.  The three-tier search strategy (unicode61 FTS5 → trigram → LIKE) is the
same one ``SessionDB.search_messages`` uses, generalized over arbitrary columns.

Unlike ``SessionDB`` (which uses external-content FTS5 tables maintained by
triggers), ``FtsLite`` uses *standalone* FTS5 tables maintained explicitly.  The
indexed datasets here are tiny (a few KB of memory entries; a few hundred skill
descriptions) and fully rebuildable, so the small storage duplication buys much
simpler, trigger-free insert/delete/rebuild semantics.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Delegate to SessionDB's implementations so the logic never drifts. hermes_state
# is a core module always imported by the running agent, so this is not an extra
# cold-start cost. SessionDB does not import anything under agent/, so no cycle.
from hermes_state import SessionDB, apply_wal_with_fallback

logger = logging.getLogger(__name__)

_OPERATORS = {"AND", "OR", "NOT"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitize_fts5_query(query: str) -> str:
    """Escape user text for safe use in an FTS5 ``MATCH`` (delegates to SessionDB)."""
    return SessionDB._sanitize_fts5_query(query)


def contains_cjk(text: str) -> bool:
    """True if ``text`` contains any CJK character (delegates to SessionDB)."""
    return SessionDB._contains_cjk(text)


def count_cjk(text: str) -> int:
    """Count CJK characters in ``text`` (delegates to SessionDB)."""
    return SessionDB._count_cjk(text)


def _check_identifier(name: str) -> str:
    """Guard table/column names that get interpolated into SQL.

    Every identifier passed to :class:`FtsLite` comes from *code* (column specs
    defined by the memory/skill modules), never user input, but validating keeps
    the string-formatted SQL injection-proof against future misuse.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


class FtsLite:
    """A standalone SQLite + FTS5 store over one table of short text rows.

    Parameters
    ----------
    db_path:
        Sidecar database file (created with parents).
    table:
        Base table name. An integer-PK row table plus ``<table>_fts`` (and, when
        the trigram tokenizer is available, ``<table>_trigram``) and a
        ``<table>_meta`` key/value table are created.
    columns:
        Ordered ``(name, sql_type)`` pairs for the stored columns (the integer
        primary key ``rowid`` is implied and prepended automatically).
    fts_columns:
        Subset of column names to full-text index. BM25 naturally weights the
        first columns higher, so order them most-significant first (e.g. name
        before description).
    """

    def __init__(
        self,
        db_path: "str | Path",
        *,
        table: str,
        columns: Sequence[Tuple[str, str]],
        fts_columns: Sequence[str],
        rowid: str = "id",
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._table = _check_identifier(table)
        self._rowid = _check_identifier(rowid)
        self._columns: List[Tuple[str, str]] = [
            (_check_identifier(n), t) for n, t in columns
        ]
        self._col_names = [n for n, _ in self._columns]
        self._fts_columns = [_check_identifier(c) for c in fts_columns]
        for c in self._fts_columns:
            if c not in self._col_names:
                raise ValueError(f"fts column {c!r} not in columns")
        self._lock = threading.RLock()
        self._fts_enabled = True
        self._trigram_enabled = False
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=1.0
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # -- Schema --------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            # Shared WAL-with-fallback so the sidecar degrades gracefully on
            # NFS/SMB/FUSE-mounted HERMES_HOME, exactly like state.db/kanban.db.
            apply_wal_with_fallback(self._conn, db_label=self.db_path.name)
            cur = self._conn.cursor()

            # Probe FTS5 once (mirror SessionDB._sqlite_supports_fts5).
            try:
                cur.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)")
                cur.execute("DROP TABLE temp._fts5_probe")
            except sqlite3.OperationalError as exc:
                self._fts_enabled = False
                logger.warning(
                    "SQLite FTS5 unavailable for %s; %s search falls back to LIKE "
                    "(underlying error: %s)",
                    self.db_path, self._table, exc,
                )

            col_defs = ", ".join(f"{n} {t}" for n, t in self._columns)
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} "
                f"({self._rowid} INTEGER PRIMARY KEY, {col_defs})"
            )
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table}_meta "
                f"(key TEXT PRIMARY KEY, value TEXT)"
            )

            if self._fts_enabled:
                fts_cols = ", ".join(self._fts_columns)
                cur.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_fts "
                    f"USING fts5({fts_cols})"
                )
                # Trigram tokenizer is optional (SQLite >= 3.34). Probe by
                # creating the table; a missing tokenizer raises and we fall
                # back to LIKE for CJK substring queries.
                try:
                    cur.execute(
                        f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table}_trigram "
                        f"USING fts5({fts_cols}, tokenize='trigram')"
                    )
                    self._trigram_enabled = True
                except sqlite3.OperationalError as exc:
                    if "no such tokenizer: trigram" in str(exc).lower():
                        logger.info(
                            "trigram tokenizer unavailable for %s; CJK/substring "
                            "search falls back to LIKE", self.db_path,
                        )
                    else:
                        raise
            self._conn.commit()

    # -- Write ---------------------------------------------------------------

    def insert(self, values: Dict[str, Any], *, dedup_column: Optional[str] = None) -> Optional[int]:
        """Insert one row; return its rowid, or ``None`` if deduplicated.

        When ``dedup_column`` is given and a row already has that column value,
        the insert is skipped (returns ``None``).
        """
        with self._lock:
            if dedup_column:
                _check_identifier(dedup_column)
                existing = self._conn.execute(
                    f"SELECT {self._rowid} FROM {self._table} WHERE {dedup_column} = ?",
                    (values.get(dedup_column),),
                ).fetchone()
                if existing is not None:
                    return None
            rowid = self._insert_row(values)
            self._conn.commit()
            return rowid

    def _insert_row(self, values: Dict[str, Any]) -> int:
        cols = [c for c in self._col_names if c in values]
        placeholders = ", ".join("?" for _ in cols)
        cur = self._conn.execute(
            f"INSERT INTO {self._table} ({', '.join(cols)}) VALUES ({placeholders})",
            [values[c] for c in cols],
        )
        rowid = int(cur.lastrowid)
        self._index_row(rowid, values)
        return rowid

    def _index_row(self, rowid: int, values: Dict[str, Any]) -> None:
        if not self._fts_enabled:
            return
        fts_vals = [str(values.get(c, "") or "") for c in self._fts_columns]
        cols = ", ".join(self._fts_columns)
        ph = ", ".join("?" for _ in self._fts_columns)
        self._conn.execute(
            f"INSERT INTO {self._table}_fts (rowid, {cols}) VALUES (?, {ph})",
            [rowid, *fts_vals],
        )
        if self._trigram_enabled:
            self._conn.execute(
                f"INSERT INTO {self._table}_trigram (rowid, {cols}) VALUES (?, {ph})",
                [rowid, *fts_vals],
            )

    def delete(self, column: str, value: Any) -> int:
        """Delete every row whose ``column`` equals ``value``; return the count."""
        _check_identifier(column)
        with self._lock:
            ids = [
                int(r[0])
                for r in self._conn.execute(
                    f"SELECT {self._rowid} FROM {self._table} WHERE {column} = ?",
                    (value,),
                ).fetchall()
            ]
            for rid in ids:
                self._unindex_row(rid)
            self._conn.execute(
                f"DELETE FROM {self._table} WHERE {column} = ?", (value,)
            )
            self._conn.commit()
            return len(ids)

    def _unindex_row(self, rowid: int) -> None:
        if not self._fts_enabled:
            return
        self._conn.execute(f"DELETE FROM {self._table}_fts WHERE rowid = ?", (rowid,))
        if self._trigram_enabled:
            self._conn.execute(
                f"DELETE FROM {self._table}_trigram WHERE rowid = ?", (rowid,)
            )

    def rebuild_from(self, rows: Sequence[Dict[str, Any]]) -> None:
        """Replace the entire index transactionally from ``rows``."""
        with self._lock:
            self._conn.execute(f"DELETE FROM {self._table}")
            if self._fts_enabled:
                self._conn.execute(f"DELETE FROM {self._table}_fts")
                if self._trigram_enabled:
                    self._conn.execute(f"DELETE FROM {self._table}_trigram")
            for values in rows:
                self._insert_row(values)
            self._conn.commit()

    # -- Read ----------------------------------------------------------------

    def search(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Relevance-ranked recall. Returns base rows (dicts) + ``_score``.

        Recall-oriented **OR** semantics: any query term may match and BM25 ranks
        rows that match more (and rarer) terms higher. This differs from
        ``SessionDB.search_messages`` (implicit AND), because here the query is an
        automatically-derived user message / task description, not a deliberate
        search expression — requiring every term to co-occur would recall almost
        nothing. Three tiers still mirror session search: unicode61 FTS5, then the
        trigram table for >=3-char CJK tokens, then a LIKE fallback (also the path
        when FTS5 is unavailable). Empty / punctuation-only queries return ``[]``.
        """
        if not query or not query.strip():
            return []
        if not self._fts_enabled:
            return self._like_search(query, limit)
        if contains_cjk(query):
            return self._cjk_search(query, limit)
        or_query = self._build_or_query(query)
        if not or_query:
            return []
        return self._match_search(f"{self._table}_fts", or_query, limit)

    @staticmethod
    def _quote_token(tok: str) -> str:
        """Wrap a token as an FTS5 phrase, neutralizing every special char."""
        return '"' + tok.replace('"', '""') + '"'

    def _build_or_query(self, text: str) -> str:
        """OR-join the quoted word tokens of ``text`` for recall matching.

        Quoting each token makes the FTS5 query injection-proof without needing
        the AND-preserving :func:`sanitize_fts5_query` (which we reserve for
        deliberate search expressions). Boolean operators are dropped so a user
        message containing the bare word "or" can't corrupt the query.
        """
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        terms = [
            self._quote_token(t) for t in tokens if t.upper() not in _OPERATORS
        ]
        return " OR ".join(terms)

    def _match_search(self, fts_table: str, match_query: str, limit: int) -> List[Dict[str, Any]]:
        sql = (
            f"SELECT b.*, f.rank AS _score "
            f"FROM {self._table} b "
            f"JOIN {fts_table} f ON f.rowid = b.{self._rowid} "
            f"WHERE {fts_table} MATCH ? "
            f"ORDER BY f.rank "
            f"LIMIT ?"
        )
        with self._lock:
            try:
                cur = self._conn.execute(sql, [match_query, limit])
            except sqlite3.OperationalError:
                # FTS5 syntax error despite sanitization — degrade to empty.
                return []
            return [dict(r) for r in cur.fetchall()]

    def _cjk_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        raw = query.strip('"').strip()
        tokens = [t for t in raw.split() if t.upper() not in _OPERATORS]
        if not tokens:
            return []
        cjk_total = sum(count_cjk(t) for t in tokens)
        # Per-token guard (#20494): the trigram tokenizer needs >=3 CJK chars
        # *per token*; a query like "广西 桂林" has 4 CJK chars total but 2 per
        # token and trigram returns nothing — route those to LIKE.
        any_short_cjk = any(0 < count_cjk(t) < 3 for t in tokens)
        if self._trigram_enabled and cjk_total >= 3 and not any_short_cjk:
            or_query = " OR ".join(self._quote_token(t) for t in tokens)
            rows = self._match_search(f"{self._table}_trigram", or_query, limit)
            if rows:
                return rows
        return self._like_search(raw, limit)

    def _like_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        raw = (query or "").strip()
        if raw:
            raw = raw.strip('"').strip()
        if not raw:
            return []
        non_op = [t for t in raw.split() if t.upper() not in _OPERATORS] or [raw]
        clauses: List[str] = []
        params: List[Any] = []
        for tok in non_op:
            esc = tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            col_likes = " OR ".join(
                f"{c} LIKE ? ESCAPE '\\'" for c in self._fts_columns
            )
            clauses.append(f"({col_likes})")
            params.extend([f"%{esc}%"] * len(self._fts_columns))
        where = " OR ".join(clauses)
        sql = f"SELECT b.* FROM {self._table} b WHERE {where} LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # -- Meta / lifecycle ----------------------------------------------------

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {self._table}_meta WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._table}_meta (key, value) VALUES (?, ?) "
                f"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]
            )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
