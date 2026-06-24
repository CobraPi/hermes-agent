"""Tests for the cheap review triage gate (agent/review_triage.py)."""

import pytest

from agent.review_triage import (
    ReviewIntent,
    _is_trivial_ack,
    heuristic_should_review,
    parse_triage_response,
    triage_llm_should_review,
)


def _turn(user, *, tools=False):
    msgs = [
        {"role": "user", "content": "previous turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": user},
    ]
    if tools:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "terminal"}}]})
        msgs.append({"role": "tool", "name": "terminal", "content": "done"})
    msgs.append({"role": "assistant", "content": "reply"})
    return msgs


def test_trivial_ack_cancels_both():
    intent = heuristic_should_review(
        _turn("thanks!"), review_memory=True, review_skills=True, user_text="thanks!"
    )
    assert not intent.review_memory and not intent.review_skills
    assert not intent.any


def test_substantive_no_tools_keeps_memory_drops_skills():
    intent = heuristic_should_review(
        _turn("My database is at /srv/pg and I prefer tabs"),
        review_memory=True, review_skills=True,
        user_text="My database is at /srv/pg and I prefer tabs",
    )
    assert intent.review_memory
    assert not intent.review_skills  # no tool activity -> no procedure to capture


def test_tool_turn_keeps_both_even_with_ack_text():
    intent = heuristic_should_review(
        _turn("ok", tools=True), review_memory=True, review_skills=True, user_text="ok"
    )
    assert intent.review_memory and intent.review_skills


def test_nothing_requested_stays_off():
    intent = heuristic_should_review(
        _turn("hi"), review_memory=False, review_skills=False, user_text="hi"
    )
    assert not intent.any


@pytest.mark.parametrize("text,expected", [
    ("thanks", True),
    ("ok thanks", True),
    ("", True),
    ("   ", True),
    ("deploy the app", False),
    ("what is the capital of France", False),
])
def test_is_trivial_ack(text, expected):
    assert _is_trivial_ack(text) is expected


def test_parse_triage_response():
    fb = ReviewIntent(True, True)
    assert parse_triage_response('{"save": false}', fb) == ReviewIntent(False, False)
    assert parse_triage_response('x {"save": true, "which": ["memory"]} y', fb) == ReviewIntent(True, False)
    assert parse_triage_response('{"save": true}', fb) == ReviewIntent(True, True)  # no "which" -> keep fallback
    assert parse_triage_response("not json at all", fb) == fb  # fail-open


def test_parse_triage_respects_fallback_narrowing():
    # If the heuristic already dropped skills, a "skill" verdict can't revive it.
    fb = ReviewIntent(review_memory=True, review_skills=False)
    out = parse_triage_response('{"save": true, "which": ["memory", "skill"]}', fb)
    assert out == ReviewIntent(True, False)


def test_triage_llm_fails_open_on_error():
    def boom(_prompt):
        raise RuntimeError("model down")
    out = triage_llm_should_review(_turn("x"), ReviewIntent(True, True), complete_fn=boom)
    assert out == ReviewIntent(True, True)


def test_triage_llm_cancels_on_no():
    out = triage_llm_should_review(
        _turn("x"), ReviewIntent(True, True), complete_fn=lambda p: '{"save": false}'
    )
    assert out == ReviewIntent(False, False)


def test_triage_llm_noop_when_intent_empty():
    calls = []
    triage_llm_should_review(
        _turn("x"), ReviewIntent(False, False), complete_fn=lambda p: calls.append(p) or "{}"
    )
    assert calls == []  # never calls the model when there's nothing to gate
