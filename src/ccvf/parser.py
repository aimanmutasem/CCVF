from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ccvf.schemas import ModelOutput, ParseStatus

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_DECISION_ALIASES = {
    "accept": "Accept",
    "accepted": "Accept",
    "reject": "Reject",
    "rejected": "Reject",
    "not sure": "Not Sure",
    "not_sure": "Not Sure",
    "unsure": "Not Sure",
    "na": "NA",
    "n/a": "NA",
}


def _extract_json(text: str) -> tuple[dict | None, str | None]:
    cleaned = _THINK_RE.sub("", text).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as exc:
        first_error = str(exc)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return None, first_error
    snippet = _TRAILING_COMMA_RE.sub(r"\1", cleaned[start : end + 1])
    try:
        return json.loads(snippet), first_error
    except json.JSONDecodeError:
        return None, first_error


def _repair(obj: dict) -> dict:
    out = dict(obj)
    if isinstance(out.get("confidence"), str):
        try:
            out["confidence"] = float(out["confidence"].strip().rstrip("%"))
        except ValueError:
            pass
    for key in ("selected_answer_index", "candidate_index"):
        value = out.get(key)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            out[key] = int(value)
    decision = out.get("verification_decision")
    if isinstance(decision, str):
        normalized = _DECISION_ALIASES.get(decision.strip().lower())
        if normalized:
            out["verification_decision"] = normalized
    for key in ("verification_reasoning", "decision_justification", "alternative_option_analysis"):
        out.setdefault(key, "")
    return out


def parse_output(text: str) -> tuple[ParseStatus, ModelOutput | None, str | None]:
    obj, first_error = _extract_json(text)
    if obj is None:
        return ParseStatus.FAILED, None, f"no JSON object found: {first_error}"
    try:
        return ParseStatus.OK, ModelOutput.model_validate(obj), None
    except ValidationError as exc:
        first_error = str(exc)
    try:
        return ParseStatus.REPAIRED, ModelOutput.model_validate(_repair(obj)), first_error
    except ValidationError as exc:
        return ParseStatus.FAILED, None, f"schema validation failed after repair: {exc}"

