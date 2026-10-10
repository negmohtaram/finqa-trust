"""One record shape for every question, whichever dataset it comes from."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

Source = Literal["finqa", "tatqa"]
Split = Literal["dev", "test"]
AnswerType = Literal["arithmetic", "yes-no", "span", "multi-span", "count"]
Scale = Literal["", "thousand", "million", "billion", "percent"]


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    source: Source
    split: Split
    question: str
    text: list[str]                 # paragraphs around the table, in order
    table: list[list[str]]          # rows of cells, header row first
    gold_answer: float | str | list[str]
    answer_type: AnswerType
    scale: Scale = ""
    program: str | None = None      # FinQA only, e.g. "subtract(1350, 1200), divide(#0, 1200)"

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty question")
        return v.strip()

    @field_validator("table")
    @classmethod
    def table_is_rectangular_enough(cls, v: list[list[str]]) -> list[list[str]]:
        if v and any(len(row) == 0 for row in v):
            raise ValueError("table has an empty row")
        return v

    @property
    def is_numeric(self) -> bool:
        return self.answer_type in ("arithmetic", "count")
