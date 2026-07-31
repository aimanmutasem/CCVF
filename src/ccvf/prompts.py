from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ccvf.schemas import Condition, DatasetSample

TEMPLATES = {
    Condition.N: "independent_solve.jinja2",
    Condition.W: "wrong_candidate.jinja2",
    Condition.W_NAIVE: "wrong_candidate_naive.jinja2",
    Condition.SF: "solve_first.jinja2",
    Condition.CC: "correct_candidate.jinja2",
}


class PromptBuilder:
    def __init__(self, prompt_dir: str | Path):
        self.prompt_dir = Path(prompt_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompt_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._hashes: dict[str, str] = {}

    def hash(self, template_name: str) -> str:
        if template_name not in self._hashes:
            self._hashes[template_name] = hashlib.sha256((self.prompt_dir / template_name).read_bytes()).hexdigest()
        return self._hashes[template_name]

    @staticmethod
    def candidate_index(condition: Condition, sample: DatasetSample) -> int | None:
        if condition == Condition.N:
            return None
        if condition == Condition.CC:
            return sample.label
        return sample.selected_distractor

    def build(
        self,
        condition: Condition,
        sample: DatasetSample,
        *,
        committed_answer_index: int | None = None,
    ) -> tuple[str, str, str]:
        template_name = TEMPLATES[condition]
        candidate_index = self.candidate_index(condition, sample)
        ctx = {
            "sample_id": sample.id,
            "context": sample.context,
            "question": sample.question,
            "answers": sample.answers,
            "candidate_index": candidate_index,
            "committed_answer_index": committed_answer_index,
        }
        return self.env.get_template(template_name).render(**ctx), template_name, self.hash(template_name)
