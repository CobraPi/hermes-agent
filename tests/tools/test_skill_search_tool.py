"""Tests for the skill_search tool (tools/skill_search_tool.py)."""

import json

import pytest


def _make_skill(skills_dir, category, name, description, tags=()):
    d = skills_dir / category / name
    d.mkdir(parents=True, exist_ok=True)
    tag_block = ""
    if tags:
        tag_lines = "\n".join(f"      - {t}" for t in tags)
        tag_block = f"\nmetadata:\n  hermes:\n    tags:\n{tag_lines}"
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}{tag_block}\n---\n\n# {name}\n\n{description}\n",
        encoding="utf-8",
    )


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import agent.prompt_builder as pb
    import tools.skill_search_tool as sst

    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    sst._clear_index_cache()

    skills = tmp_path / "skills"
    _make_skill(skills, "mlops", "axolotl", "Fine-tune large language models with a low memory footprint", ["fine-tuning", "llm"])
    _make_skill(skills, "github", "pr-review", "Review pull requests and leave inline comments", ["git", "review"])
    _make_skill(skills, "web", "scrape", "Scrape and extract content from websites", ["web", "scraping"])
    yield tmp_path

    sst._clear_index_cache()
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def _search(query, **kw):
    from tools.skill_search_tool import skill_search
    return json.loads(skill_search(query, **kw))


def test_relevance_ranking_by_description(skills_home):
    result = _search("fine tune a model")
    assert result["success"] is True
    assert result["results"], "expected at least one match"
    assert result["results"][0]["name"] == "axolotl"


def test_tag_match(skills_home):
    result = _search("training llm")
    assert any(r["name"] == "axolotl" for r in result["results"])


def test_empty_query_returns_empty(skills_home):
    result = _search("")
    assert result["success"] is True
    assert result["results"] == []


def test_no_match_returns_empty(skills_home):
    result = _search("zzzqqq totally unrelated topic")
    assert result["results"] == []


def test_limit_is_respected(skills_home):
    result = _search("review pull request and code", limit=1)
    assert len(result["results"]) <= 1


def test_result_shape(skills_home):
    result = _search("scrape a website")
    top = result["results"][0]
    assert set(["name", "category", "description", "score"]).issubset(top.keys())
    assert top["name"] == "scrape"


def test_manifest_rebuild_picks_up_new_skill(skills_home):
    from tools.skill_search_tool import skill_search

    assert not json.loads(skill_search("kubernetes rollout"))["results"]

    # Add a skill on disk; the mtime/size manifest changes, so the next search
    # rebuilds both the prompt snapshot and the FTS index.
    _make_skill(skills_home / "skills", "ops", "k8s", "Kubernetes deployment and rollout management")

    result = json.loads(skill_search("kubernetes rollout"))
    assert any(r["name"] == "k8s" for r in result["results"])


def test_disabled_skill_filtered(skills_home, monkeypatch):
    # A skill present on disk but disabled in config must not appear in results.
    monkeypatch.setattr(
        "agent.skill_utils.get_disabled_skill_names", lambda platform=None: {"scrape"}
    )
    result = _search("scrape a website")
    assert all(r["name"] != "scrape" for r in result["results"])
