"""Strict public request contract. Model outputs never supply JSON or field names."""

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
Number = Annotated[float, Field(allow_inf_nan=False, ge=-1e100, le=1e100)]
MAX_SLOTS = 26  # every candidate, special ones included, is one uppercase answer letter


def require_unicode(value):
    """Refuse a lone surrogate in any string of a parsed JSON value, dict keys included.

    JSON can spell one (`"\\ud800"`, which `json.loads` accepts) but UTF-8 cannot encode it: it
    would break the tokenizer or a response, or come back as U+FFFD in place of a question ID.
    Iterative, so that deeply nested input cannot exhaust the stack.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode()
            except UnicodeEncodeError:
                raise ValueError(
                    "Strings must be valid Unicode text: a lone surrogate, such as the JSON "
                    "escape \\ud800, cannot be encoded as UTF-8"
                ) from None
        elif isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Policy(Strict):
    allow_abstain: bool = True
    max_unavailable_probability: float = Field(default=0.5, gt=0, le=1)
    min_top_probability: float = Field(default=0, ge=0, le=1)


class BaseQuestion(Strict):
    instructions: Text
    policy: Policy = Field(default_factory=Policy)

    def require_slots(self, entries, reserved=0):
        reserved += self.policy.allow_abstain
        if len(entries) + reserved > MAX_SLOTS:
            raise ValueError(
                f"At most {MAX_SLOTS - reserved} entries fit here: {MAX_SLOTS} answer letters, "
                f"{reserved} reserved for abstention/out-of-range"
            )


class BooleanQuestion(BaseQuestion):
    type: Literal["boolean"]
    true_description: Text = "Yes. The evidence supports an affirmative answer to the question."
    false_description: Text = "No. The evidence supports a negative answer to the question."


class Option(Strict):
    id: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]
    description: Text


class ChoiceQuestion(BaseQuestion):
    type: Literal["choice"]
    options: list[Option] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({o.id for o in self.options}) != len(self.options):
            raise ValueError("Option IDs must be unique")
        self.require_slots(self.options)
        return self


class ScoreQuestion(BaseQuestion):
    type: Literal["score"]
    levels: list[Text] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def distinct_levels(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Levels must have distinct descriptions")
        self.require_slots(self.levels)
        return self


class Anchor(Strict):
    value: Number
    description: Text


class NumericQuestion(BaseQuestion):
    type: Literal["numeric"]
    unit: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    anchors: list[Anchor] = Field(min_length=2, max_length=MAX_SLOTS)

    @model_validator(mode="after")
    def increasing_anchors(self):
        # Successive pairs: the two sequences differ in length by design.
        if any(b.value <= a.value for a, b in zip(self.anchors, self.anchors[1:], strict=False)):
            raise ValueError("Numeric anchors must be strictly increasing")
        self.require_slots(self.anchors, reserved=2)
        return self


Question = Annotated[
    BooleanQuestion | ChoiceQuestion | ScoreQuestion | NumericQuestion,
    Field(discriminator="type"),
]


class Request(Strict):
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    mode: Literal["shared", "direct"] = "shared"

    @field_validator("state", "questions", mode="before")
    @classmethod
    def valid_unicode(cls, value):
        return require_unicode(value)

    @model_validator(mode="after")
    def valid_state(self):
        if not self.state or (isinstance(self.state, str) and not self.state.strip()):
            raise ValueError("State must not be empty")
        rendered = json.dumps(self.state, ensure_ascii=False, allow_nan=False)
        if len(rendered.encode()) > 256_000:
            raise ValueError("State exceeds 256 KB; no silent truncation")
        if any(not key.strip() or len(key) > 128 for key in self.questions):
            raise ValueError("Question IDs must have 1–128 nonblank characters")
        return self
