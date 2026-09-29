"""Guard on the rendered prompts: their text is tied to PROMPT_VERSION.

A calibration is fitted for one prompt (the model fingerprint includes PROMPT_VERSION), so a
prompt whose text changes without a new version would silently reuse calibrations that were
fitted for a different one. The hashes below are those of the prompts of a few fixed requests,
rendered with a fixed chat template, for the version recorded next to them.

Hashes are taken over the UTF-8 bytes of in-memory strings: no file, locale or line-ending
convention is involved, so they are the same on every operating system.
"""

import hashlib
from typing import Any

import pytest

from rizzo_flow import prompts
from rizzo_flow.decisions import candidates
from rizzo_flow.prompts import PROMPT_VERSION, compile_request
from rizzo_flow.schema import Request

RECORDED_VERSION = "spark-decisions-v3"
RECORDED_HASHES = {
    "text/boolean+abstain": "e6c39e3ca223759ba4b908203d45ccdca02ade4411b011fe49291e27518deb2d",
    "text/choice+abstain": "8f1eb6627dcdb5b13c5cb3656f204dc94b7c35d421302fa88fb66cb10cede171",
    "text/score+abstain": "00c8b0615f9be43a7c0fb7d008a462d40529bdf5f5bc82ec227d3552c135569b",
    "text/numeric+abstain": "b31db523a565e48d3eb76e9af668e0c0480cf3926f925bbb1805d30527e2cca7",
    "text/boolean": "13a91b44faa38190297ffdda8cdf53ae433583f00cab8893648e303af5132f4a",
    "text/choice": "01b21b982dd2975fa7659dc673eefd3460bffaa69e978f67348cd460d5da8c8e",
    "text/score": "f198c53ef0301eb8ae92f33696f6ed4300834942aa28e716b22fb9f9a9b4f438",
    "text/numeric": "cc79933a6ec4085a613ae73db4a90a32a98a6f6fd1aab948025bab1f6aa1a859",
    "json/choice": "72a882de23ed44bb07ca354660a7fcef247421ac0ee3c06775eb432ff50168ed",
    "json/score": "06ddf719ed700432fedcb7d54b8e6ce336a6b6aa969940bd4dbb330f4ba86b2e",
}

TEXT_STATE = (
    "Ticket #4821 from Marta Rossi: the payout of €1,240 failed three times since Monday. "
    "She is furious and asks for a refund of the €49 fee."
)
STRUCTURED_STATE = {
    "ticket": {"id": 4821, "channel": "email", "tags": ["payout", "urgent"]},
    "customer": {"name": "Marta Rossi", "vip": False, "notes": None},
    "messages": ["It failed again.", "Caffè? No: I want my money back."],
}
ANCHORS = [
    {"value": 0, "description": "no fee"},
    {"value": 49, "description": "the fee"},
    {"value": 1240, "description": "the whole payout"},
]
LEVELS = ["calm", "annoyed", "furious"]
OPTIONS = [
    {"id": "billing", "description": "Payment or invoice problem"},
    {"id": "access", "description": "Login or permissions problem"},
    {"id": "other", "description": "Anything else"},
]
KEEP = {"allow_abstain": True}
DROP = {"allow_abstain": False}


def questions(policy: dict[str, Any], suffix: str) -> dict[str, dict[str, Any]]:
    common = {"policy": policy}
    return {
        f"boolean{suffix}": {
            "type": "boolean",
            "instructions": "Does the customer ask for a refund?",
            **common,
        },
        f"choice{suffix}": {
            "type": "choice",
            "instructions": "Which team should handle the ticket?",
            "options": OPTIONS,
            **common,
        },
        f"score{suffix}": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "levels": LEVELS,
            **common,
        },
        f"numeric{suffix}": {
            "type": "numeric",
            "instructions": "How much money does the customer want back?",
            "unit": "EUR",
            "anchors": ANCHORS,
            **common,
        },
    }


class SnapshotTokenizer:
    """A fixed stand-in for the chat template and vocabulary: one token per character."""

    def encode(self, value: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in value]

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, **kwargs: Any
    ) -> str:
        assert tokenize is False
        assert kwargs == {"add_generation_prompt": True, "enable_thinking": False}
        turns = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        return turns + "<|assistant|>\n"


def request_of(state: Any, body: dict[str, dict[str, Any]]) -> Request:
    return Request.model_validate({"state": state, "questions": body})


def prompt_hashes() -> dict[str, str]:
    """The prompt hash of every question of the fixed requests, keyed by request and question."""
    text = request_of(TEXT_STATE, {**questions(KEEP, "+abstain"), **questions(DROP, "")})
    structured = request_of(
        STRUCTURED_STATE,
        {
            "choice": questions(KEEP, "")["choice"],
            "score": questions(DROP, "")["score"],
        },
    )
    hashes = {}
    for label, request in (("text", text), ("json", structured)):
        _, jobs = compile_request(SnapshotTokenizer(), request, 100_000)
        for job in jobs:
            hashes[f"{label}/{job.id}"] = job.prompt_sha256
    return hashes


def diagnose(version: str, actual: dict[str, str]) -> str | None:
    """Why the prompts no longer match the record, or None when they do."""
    if actual == RECORDED_HASHES:
        if version == RECORDED_VERSION:
            return None
        return (
            f"PROMPT_VERSION is {version!r} but the prompts are still those recorded for "
            f"{RECORDED_VERSION!r}: set RECORDED_VERSION in this test to {version!r}."
        )
    changed = sorted(
        name
        for name in RECORDED_HASHES.keys() | actual.keys()
        if actual.get(name) != RECORDED_HASHES.get(name)
    )
    if version == RECORDED_VERSION:
        return (
            f"The prompt text changed ({', '.join(changed)}) but PROMPT_VERSION is still "
            f"{version!r}. Changing the prompt text requires incrementing PROMPT_VERSION in "
            "src/rizzo_flow/prompts.py, because it invalidates the calibrations fitted for the "
            "old prompt; then update RECORDED_VERSION and RECORDED_HASHES in this test."
        )
    return (
        f"PROMPT_VERSION is now {version!r} and the prompt text differs from the record "
        f"({', '.join(changed)}): update RECORDED_VERSION and RECORDED_HASHES in this test."
    )


def test_prompts_match_the_record_for_this_prompt_version():
    problem = diagnose(PROMPT_VERSION, prompt_hashes())
    assert problem is None, problem


def test_the_record_covers_every_question_type_with_and_without_abstention():
    names = set(RECORDED_HASHES)
    for kind in ("boolean", "choice", "score", "numeric"):
        assert f"text/{kind}+abstain" in names
        assert f"text/{kind}" in names
    assert {"json/choice", "json/score"} <= names
    assert all(len(value) == 64 and value == value.lower() for value in RECORDED_HASHES.values())
    assert len(set(RECORDED_HASHES.values())) == len(RECORDED_HASHES)  # no two prompts coincide


def test_a_hash_is_the_sha256_of_the_utf8_prompt():
    request = request_of(TEXT_STATE, {"q": questions(KEEP, "")["boolean"]})
    _, (job,) = compile_request(SnapshotTokenizer(), request, 100_000)
    question = request.questions["q"]
    descriptions = [c.description for c in candidates(question)]
    prompt = (
        f"<|system|>\n{prompts.SYSTEM}\n"
        f"<|user|>\n{prompts.render_state(TEXT_STATE)}"
        f"{prompts.render_question(question.instructions, descriptions)}\n"
        "<|assistant|>\n"
    )
    assert "\u20ac" in prompt  # the non-ASCII part of the evidence is in the hashed text
    assert job.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# the guard itself -------------------------------------------------------------------------------


def test_the_diagnosis_demands_a_new_prompt_version_when_only_the_text_changed():
    tampered = {**RECORDED_HASHES, "text/score": "0" * 64}
    message = diagnose(RECORDED_VERSION, tampered)
    assert message is not None
    assert "PROMPT_VERSION" in message
    assert "requires incrementing PROMPT_VERSION" in message
    assert "invalidates the calibrations" in message
    assert "RECORDED_HASHES" in message
    assert "text/score" in message


def test_the_diagnosis_asks_for_the_record_to_follow_a_version_bump():
    tampered = {**RECORDED_HASHES, "text/score": "0" * 64}
    message = diagnose("spark-decisions-v99", tampered)
    assert message is not None
    assert "spark-decisions-v99" in message
    assert "RECORDED_VERSION" in message
    assert "RECORDED_HASHES" in message
    unchanged = diagnose("spark-decisions-v99", dict(RECORDED_HASHES))
    assert unchanged is not None
    assert "set RECORDED_VERSION" in unchanged


def test_the_diagnosis_reports_added_and_removed_prompts():
    added = {**RECORDED_HASHES, "text/extra": "1" * 64}
    assert "text/extra" in (diagnose(RECORDED_VERSION, added) or "")
    removed = dict(RECORDED_HASHES)
    del removed["text/score"]
    assert "text/score" in (diagnose(RECORDED_VERSION, removed) or "")
    assert diagnose(RECORDED_VERSION, dict(RECORDED_HASHES)) is None


@pytest.mark.parametrize("attribute", ["SYSTEM", "CLOSING"])
def test_editing_a_shared_part_of_the_prompt_changes_every_hash(monkeypatch, attribute):
    monkeypatch.setattr(prompts, attribute, getattr(prompts, attribute) + " ")
    actual = prompt_hashes()
    assert actual.keys() == RECORDED_HASHES.keys()
    assert all(actual[name] != RECORDED_HASHES[name] for name in actual)
    assert "requires incrementing PROMPT_VERSION" in (diagnose(RECORDED_VERSION, actual) or "")


def test_editing_the_numeric_guidance_changes_only_the_numeric_prompts(monkeypatch):
    monkeypatch.setattr(prompts, "NUMERIC_GUIDANCE", prompts.NUMERIC_GUIDANCE + " ")
    actual = prompt_hashes()
    changed = {name for name in actual if actual[name] != RECORDED_HASHES[name]}
    assert changed == {"text/numeric+abstain", "text/numeric"}


def test_the_hash_does_not_depend_on_the_order_of_the_questions():
    forward = request_of(TEXT_STATE, questions(KEEP, ""))
    backward = request_of(TEXT_STATE, dict(reversed(list(questions(KEEP, "").items()))))
    one = {j.id: j.prompt_sha256 for j in compile_request(SnapshotTokenizer(), forward, 10**6)[1]}
    two = {j.id: j.prompt_sha256 for j in compile_request(SnapshotTokenizer(), backward, 10**6)[1]}
    assert one == two
    assert len(one) == 4
