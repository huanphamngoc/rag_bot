"""The gate in front of the model: deciding, for free, whether a question should reach it.

Every question costs a paid embedding call and a paid chat call, and the web page may be reachable by
people who never read the README. So the first node of the chat graph classifies the question from the
rules in ``rules/qualify.yaml`` - substring matching, no model, no network - into one of four outcomes:

* ``pass``      - retrieve and answer;
* ``needs_sql`` - a count, total, average or ranking over the whole dataset. Retrieval reads at most
                  ``RAG_TOP_K`` excerpts, so an answer from them would be a guess; the user gets the SQL
                  shape instead;
* ``clarify``   - there is nothing to search for yet;
* ``reject``    - prompt injection, a request for medical or legal advice, or plainly off topic.

The rules are evaluated in the order the YAML lists them and the first match wins, so that order is
part of the behaviour: an aggregate question that also carries an injection attempt is rejected.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from crawlerrag.rules.models import QualifyRules

log = logging.getLogger(__name__)

PASS = "pass"
CLARIFY = "clarify"
NEEDS_SQL = "needs_sql"
REJECT = "reject"

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class QualifyResult:
    decision: str
    question: str                       # normalised
    rule_id: str | None = None
    message: str | None = None          # what to tell the user when the question is not passed on
    sql_hint: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision != PASS


def normalise(question: str) -> str:
    """Strip control characters and collapse runs of whitespace; nothing else is touched.

    What the user wrote is what gets logged and retrieved with, so this only removes what cannot be
    part of a question. Matching is a separate step (see :func:`fold`).
    """
    return WHITESPACE.sub(" ", CONTROL.sub("", question or "")).strip()


def qualify(rules: QualifyRules, question: str, *, filters: dict[str, Any] | None = None,
            doc_types: Sequence[str] | None = None, is_follow_up: bool = False) -> QualifyResult:
    text = normalise(question)
    limits = rules.limits

    # A rule that matches the WHOLE message has identified it exactly, so it decides before any length
    # limit - length has nothing to add. Without this, "OK." (3 characters) was smalltalk while "ok"
    # (2) fell to min_chars and got "Could you give me a bit more to go on?", which is a strange thing
    # to say to someone who just said ok.
    for rule in rules.rules:
        if rule.enabled and rule.equals_any and rule.matches(text):
            log.info("question not passed to the model", extra={"rule": rule.id, "decision": rule.kind})
            return QualifyResult(rule.kind, text, rule.id, rule.message.strip())

    if len(text) < limits.min_chars:
        return QualifyResult(CLARIFY, text, "limit:min_chars", limits.message_too_short)
    # One word ("insulin") names a subject but asks nothing, and it is the case where showing real
    # records beats any refusal: the clarify step looks that word up and offers questions about what it
    # actually found. Checked before the rules, so "insulin" is a clarify and not an off-topic reject.
    #
    # Not for a follow-up. "why?" is one word and perfectly clear once there is a previous turn - the
    # rewrite step resolves it before retrieval - and asking someone to be more specific about it would
    # be absurd. The existing chat-loop test caught this the moment the rule went in.
    if not is_follow_up and len(text.split()) < limits.min_words:
        return QualifyResult(CLARIFY, text, "limit:min_words", limits.message_too_vague)
    if len(text) > limits.max_chars:
        return QualifyResult(REJECT, text, "limit:max_chars", limits.message_too_long)
    if filters and len(filters) > limits.max_filters:
        return QualifyResult(REJECT, text, "limit:max_filters", limits.message_too_many_filters)
    if doc_types and len(doc_types) > limits.max_doc_types:
        return QualifyResult(REJECT, text, "limit:max_doc_types", limits.message_too_many_filters)

    for rule in rules.rules:
        if not rule.enabled or rule.equals_any or (is_follow_up and rule.skip_for_follow_up):
            continue
        if not rule.matches(text):
            continue
        log.info("question not passed to the model", extra={"rule": rule.id, "decision": rule.kind})
        return QualifyResult(rule.kind, text, rule.id, rule.message.strip(),
                             rule.sql_hint.strip() if rule.sql_hint else None)
    return QualifyResult(PASS, text)


def blocked_answer(result: QualifyResult) -> str:
    """What the user reads when the question did not reach the model."""
    parts = [result.message or "That question was not answered."]
    if result.sql_hint:
        parts.append(result.sql_hint)
    return "\n\n".join(parts)
