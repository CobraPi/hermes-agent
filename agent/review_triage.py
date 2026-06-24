"""Cheap pre-flight gate for the background memory/skill review.

The background review (``agent/background_review.py``) spawns a tool-using LLM
fork after eligible turns to decide whether anything should be saved to memory or
captured as a skill. On weak hardware running a single local model under
single-flight, that fork competes with the user's next turn for the one
inference slot — and most turns have nothing worth saving.

This module adds a two-tier gate in front of that spawn:

  1. :func:`heuristic_should_review` — zero-LLM, conservative. It narrows or
     cancels the review only when we are highly confident there is nothing to
     save (a trivial acknowledgment with no tool work). Today's default —
     "always review" — is the safe baseline, so the heuristic only ever removes
     work it is sure is wasted. This is the big weak-hardware win: chit-chat
     ("thanks", "ok") now costs zero review calls.

  2. :func:`triage_llm_should_review` — optional (default off), one tiny
     structured completion on a *digest* of the turn that escalates to the full
     fork only on a "yes". The actual model call is injected by the caller
     (``background_review``) so it can route through the same aux-model selector
     the review uses, and so this module stays trivially testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ReviewIntent:
    """Which review paths should still run after gating."""

    review_memory: bool
    review_skills: bool

    @property
    def any(self) -> bool:
        return bool(self.review_memory or self.review_skills)


# Bare acknowledgments / pleasantries that never carry a durable fact on their
# own. Matched only when the turn also did no tool work (see heuristic).
_ACK_PHRASES = frozenset({
    "thanks", "thank you", "thanks!", "thank you!", "thx", "ty", "tysm",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "good", "fine",
    "got it", "gotcha", "sounds good", "makes sense", "understood", "right",
    "yes", "yep", "yeah", "yup", "no", "nope", "nah", "sure", "done",
    "perfect", "awesome", "lgtm", "ack", "+1", "👍", "🙏",
})

_TRIAGE_PROMPT = (
    "You are a triage gate deciding whether a just-finished assistant turn "
    "contains anything worth saving to long-term memory or capturing as a "
    "reusable skill.\n\n"
    "Save to MEMORY only for durable facts about the user or their environment "
    "(preferences, names, conventions, recurring constraints). Capture a SKILL "
    "only for a repeatable multi-step procedure the assistant worked out.\n"
    "Most turns warrant NEITHER — be strict.\n\n"
    "Turn digest:\n{digest}\n\n"
    "Respond with ONLY a JSON object: "
    '{{"save": true|false, "which": ["memory"|"skill", ...]}}'
)


def _text_of(message: Dict[str, Any]) -> str:
    """Best-effort plain text of an OpenAI-style message (str or parts list)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text") or part.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return ""


def _current_turn(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Messages from the last user message to the end (the turn being finalized)."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return messages[i:]
    return list(messages)


def _had_tool_activity(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if m.get("role") == "tool":
            return True
        if m.get("tool_calls"):
            return True
    return False


def _normalize_ack(text: str) -> str:
    return re.sub(r"[\s.!?,;:]+$", "", (text or "").strip().lower())


def _is_trivial_ack(text: str) -> bool:
    """True for empty / bare-acknowledgment user turns (no durable content)."""
    norm = _normalize_ack(text)
    if not norm:
        return True
    if norm in _ACK_PHRASES:
        return True
    # Short combinations like "ok thanks" / "great, thank you".
    words = norm.replace(",", " ").split()
    if len(words) <= 3 and all(
        _normalize_ack(w) in _ACK_PHRASES or w in {"and", "but", "so", "really", "much", "you"}
        for w in words
    ):
        return True
    return False


def heuristic_should_review(
    messages_snapshot: List[Dict[str, Any]],
    *,
    review_memory: bool,
    review_skills: bool,
    user_text: str = "",
    agent: Any = None,
) -> ReviewIntent:
    """Zero-LLM gate. Returns the (possibly narrowed) review intent.

    Conservative by design — it only removes review work it is confident is
    wasted, because the inherited default is to always review:

      * **Skills** capture tool-driven procedures. If the entire snapshot has no
        tool activity there is nothing procedural to distill, so drop skills.
        (When the skill nudge fired this is near-impossible, so it is mostly a
        safety net; the real skill win comes from the optional LLM tier.)
      * **Memory** is dropped only for a trivial acknowledgment turn with no tool
        work — "thanks", "ok", "sounds good" — which cannot carry a durable fact.

    Anything substantive (a real question, a statement, any tool use) is left
    intact for the full review (or the optional LLM triage).
    """
    intent = ReviewIntent(bool(review_memory), bool(review_skills))
    if not intent.any:
        return intent

    messages = messages_snapshot or []
    turn = _current_turn(messages)
    had_tools_this_turn = _had_tool_activity(turn)

    if intent.review_skills and not _had_tool_activity(messages):
        intent.review_skills = False

    if intent.review_memory and not had_tools_this_turn and _is_trivial_ack(user_text):
        intent.review_memory = False

    return intent


def _extract_json(raw: str) -> str:
    """Pull the first ``{...}`` object out of a model response."""
    if not raw:
        return ""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return match.group(0) if match else raw


def parse_triage_response(raw: str, fallback: ReviewIntent) -> ReviewIntent:
    """Parse the triage model's JSON; fail open to ``fallback`` on anything odd."""
    try:
        data = json.loads(_extract_json(raw))
    except Exception:
        return fallback
    if not isinstance(data, dict):
        return fallback
    if not data.get("save"):
        return ReviewIntent(False, False)
    which = data.get("which")
    if not isinstance(which, list) or not which:
        # "save" with no specifics — keep whatever the heuristic still wanted.
        return ReviewIntent(fallback.review_memory, fallback.review_skills)
    which_l = {str(w).strip().lower() for w in which}
    return ReviewIntent(
        fallback.review_memory and "memory" in which_l,
        fallback.review_skills and bool({"skill", "skills"} & which_l),
    )


def triage_llm_should_review(
    messages_snapshot: List[Dict[str, Any]],
    intent: ReviewIntent,
    *,
    complete_fn: Callable[[str], str],
    digest_fn: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> ReviewIntent:
    """Optional tiny-LLM tier. ``complete_fn(prompt) -> raw_text``; fail-open.

    The caller injects ``complete_fn`` (the actual aux-model call) and optionally
    ``digest_fn`` (defaults to a compact tail digest). Any error — model failure,
    unparseable output — falls back to ``intent`` so triage can never *lose* a
    review the heuristic wanted; it can only cancel one the model is sure about.
    """
    if not intent.any:
        return intent
    try:
        digest = (digest_fn or _default_digest)(messages_snapshot)
        raw = complete_fn(_TRIAGE_PROMPT.format(digest=digest))
        return parse_triage_response(raw, intent)
    except Exception:
        return intent


def _default_digest(messages: List[Dict[str, Any]], *, tail: int = 8, cap: int = 280) -> str:
    """Compact role-prefixed tail digest for the triage prompt."""
    lines: List[str] = []
    for m in messages[-tail:]:
        role = m.get("role", "?")
        if role == "tool":
            name = m.get("name") or m.get("tool_name") or "tool"
            lines.append(f"[tool:{name}]")
            continue
        text = _text_of(m).strip().replace("\n", " ")
        if not text and m.get("tool_calls"):
            calls = ", ".join(
                (c.get("function") or {}).get("name", "?")
                for c in m.get("tool_calls") or []
                if isinstance(c, dict)
            )
            text = f"[called: {calls}]"
        if len(text) > cap:
            text = text[:cap] + "…"
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)
