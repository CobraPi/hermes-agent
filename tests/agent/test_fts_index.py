"""Unit tests for the shared FtsLite full-text store (agent/fts_index.py)."""

import pytest

from agent.fts_index import FtsLite, contains_cjk, count_cjk, sanitize_fts5_query


@pytest.fixture
def idx(tmp_path):
    store = FtsLite(
        tmp_path / "t.db",
        table="entries",
        columns=[("target", "TEXT"), ("content", "TEXT"), ("content_hash", "TEXT")],
        fts_columns=["content"],
    )
    yield store
    store.close()


def test_insert_and_search_with_score(idx):
    idx.insert({"target": "memory", "content": "Joey runs on weak hardware", "content_hash": "h1"})
    idx.insert({"target": "memory", "content": "single flight mode, no parallel subagents", "content_hash": "h2"})
    rows = idx.search("weak hardware", limit=5)
    assert rows and rows[0]["content_hash"] == "h1"
    assert "_score" in rows[0]


def test_or_recall_semantics(idx):
    # OR semantics: a multi-word query matches a doc sharing only some terms.
    idx.insert({"content": "deploy the release script nightly", "content_hash": "h1", "target": "m"})
    rows = idx.search("how do I deploy something", limit=5)
    assert rows and "deploy" in rows[0]["content"]


def test_dedup_insert(idx):
    assert idx.insert({"content": "x", "content_hash": "h1", "target": "m"}, dedup_column="content_hash") is not None
    assert idx.insert({"content": "y", "content_hash": "h1", "target": "m"}, dedup_column="content_hash") is None
    assert idx.count() == 1


def test_delete_removes_from_index(idx):
    idx.insert({"content": "alpha apples", "content_hash": "h1", "target": "m"})
    idx.insert({"content": "beta bananas", "content_hash": "h2", "target": "m"})
    assert idx.delete("content_hash", "h1") == 1
    assert idx.count() == 1
    assert not idx.search("apples", limit=5)
    assert idx.search("bananas", limit=5)


def test_rebuild_from_replaces_everything(idx):
    idx.insert({"content": "old stale entry", "content_hash": "h1", "target": "m"})
    idx.rebuild_from([
        {"content": "fresh one", "content_hash": "r1", "target": "m"},
        {"content": "fresh two", "content_hash": "r2", "target": "m"},
    ])
    assert idx.count() == 2
    assert not idx.search("stale", limit=5)
    assert len(idx.search("fresh", limit=5)) == 2


def test_empty_query_returns_empty(idx):
    idx.insert({"content": "something", "content_hash": "h1", "target": "m"})
    assert idx.search("", limit=5) == []
    assert idx.search("   ", limit=5) == []
    assert idx.search(None, limit=5) == []


def test_special_chars_do_not_crash(idx):
    idx.insert({"content": "fix the chat-send bug in P2.2", "content_hash": "h1", "target": "m"})
    for q in ["TODO: fix", 'a "quote', "(unbalanced", "chat-send", "P2.2", "a AND", "OR b"]:
        assert isinstance(idx.search(q, limit=5), list)
    # the hyphenated identifier is still findable
    assert idx.search("chat-send", limit=5)


def test_meta_roundtrip(idx):
    assert idx.get_meta("manifest") is None
    idx.set_meta("manifest", "v1")
    assert idx.get_meta("manifest") == "v1"
    idx.set_meta("manifest", "v2")
    assert idx.get_meta("manifest") == "v2"


def test_cjk_trigram_and_like_fallback(tmp_path):
    store = FtsLite(tmp_path / "cjk.db", table="d", columns=[("content", "TEXT")], fts_columns=["content"])
    try:
        store.insert({"content": "大别山项目的部署说明"})
        store.insert({"content": "广西桂林漓江风景"})
        # 3+ CJK chars -> trigram (or LIKE if trigram unavailable); both find it.
        assert store.search("部署说明", limit=5)
        # 2-char CJK -> LIKE substring fallback.
        assert store.search("项目", limit=5)
    finally:
        store.close()


def test_like_fallback_when_fts_disabled(tmp_path):
    store = FtsLite(
        tmp_path / "f.db",
        table="d",
        columns=[("content", "TEXT"), ("content_hash", "TEXT")],
        fts_columns=["content"],
    )
    try:
        store.insert({"content": "findable via like", "content_hash": "h1"})
        store._fts_enabled = False  # simulate a build without FTS5
        rows = store.search("findable", limit=5)
        assert rows and rows[0]["content"] == "findable via like"
    finally:
        store.close()


def test_helpers():
    assert contains_cjk("项目") is True
    assert contains_cjk("hello") is False
    assert count_cjk("a项b目") == 2
    assert ":" not in sanitize_fts5_query("TODO: fix")


def test_invalid_identifier_rejected(tmp_path):
    with pytest.raises(ValueError):
        FtsLite(tmp_path / "x.db", table="bad name;", columns=[("c", "TEXT")], fts_columns=["c"])
    with pytest.raises(ValueError):
        FtsLite(tmp_path / "y.db", table="ok", columns=[("c", "TEXT")], fts_columns=["not_a_column"])
