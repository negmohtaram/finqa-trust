"""Load FinQA and TAT-QA into Records."""
import json
from pathlib import Path

from .checker import agrees_up_to_rounding
from .schema import Record

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
FILES = {
    ("finqa", "dev"): "finqa_dev.json",
    ("finqa", "test"): "finqa_test.json",
    ("tatqa", "dev"): "tatqa_dev.json",
    ("tatqa", "test"): "tatqa_test_gold.json",
}


def _finqa_record(item: dict, split: str) -> Record:
    qa = item["qa"]
    exe = qa["exe_ans"]
    if isinstance(exe, str) and exe.lower() in ("yes", "no"):
        gold, answer_type = exe.lower(), "yes-no"
    else:
        gold, answer_type = float(exe), "arithmetic"
    # FinQA marks percentages only in the text answer ("12.5%"); exe_ans is the fraction (0.125)
    scale = "percent" if str(qa.get("answer", "")).strip().endswith("%") else ""
    record = Record(
        id=f"finqa/{item['id']}",
        source="finqa",
        split=split,
        question=qa["question"],
        text=item["pre_text"] + item["post_text"],
        table=item["table"],
        gold_answer=gold,
        answer_type=answer_type,
        scale=scale,
        program=qa.get("program"),
    )
    # FinQA gives each answer twice: computed (exe_ans, our gold) and as text. When they
    # disagree beyond rounding, nobody knows which is right, so the item is flagged.
    if answer_type == "arithmetic" and not agrees_up_to_rounding(qa.get("answer", ""), record):
        record = record.model_copy(update={"ambiguous_gold": True})
    return record


def _tatqa_records(doc: dict, split: str) -> list[Record]:
    paragraphs = [p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"])]
    table = doc["table"]["table"]
    records = []
    for q in sorted(doc["questions"], key=lambda q: q["order"]):
        answer_type = q["answer_type"]
        answer = q["answer"]
        if answer_type in ("arithmetic", "count"):
            gold = float(answer)
        elif answer_type == "span":
            gold = answer[0] if isinstance(answer, list) else str(answer)
        else:
            gold = [str(a) for a in answer]
        records.append(Record(
            id=f"tatqa/{q['uid']}",
            source="tatqa",
            split=split,
            question=q["question"],
            text=paragraphs,
            table=table,
            gold_answer=gold,
            answer_type=answer_type,
            scale=q.get("scale", ""),
            program=None,
        ))
    return records


def load(source: str, split: str, raw_dir: Path = RAW) -> list[Record]:
    path = raw_dir / FILES[(source, split)]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python scripts/download_data.py")
    items = json.loads(path.read_text())
    if source == "finqa":
        return [_finqa_record(item, split) for item in items]
    return [r for doc in items for r in _tatqa_records(doc, split)]


def load_all(split: str, raw_dir: Path = RAW) -> list[Record]:
    return load("finqa", split, raw_dir) + load("tatqa", split, raw_dir)
