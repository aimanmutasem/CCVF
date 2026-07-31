from pathlib import Path
import unittest

import yaml

from ccvf.client import _wrap_completion_prompt
from ccvf.prompts import TEMPLATES, PromptBuilder
from ccvf.schemas import Condition, DatasetSample, ExperimentConfig


class PromptBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = DatasetSample(
            id="S1",
            context="Context",
            question="Question?",
            answers=["A", "B", "C"],
            label=0,
            selected_distractor=1,
            dataset="LogiQA",
        )
        self.builder = PromptBuilder(Path("prompts"))

    def test_condition_template_mapping_is_explicit(self) -> None:
        self.assertEqual(TEMPLATES[Condition.N], "independent_solve.jinja2")
        self.assertEqual(TEMPLATES[Condition.W], "wrong_candidate.jinja2")
        self.assertEqual(TEMPLATES[Condition.W_NAIVE], "wrong_candidate_naive.jinja2")
        self.assertEqual(TEMPLATES[Condition.SF], "solve_first.jinja2")
        self.assertEqual(TEMPLATES[Condition.CC], "correct_candidate.jinja2")

    def test_w_naive_prompt_loads_with_w_naive_condition(self) -> None:
        prompt, template_name, _ = self.builder.build(Condition.W_NAIVE, self.sample)

        self.assertEqual(template_name, "wrong_candidate_naive.jinja2")
        self.assertIn('"condition": "W_naive"', prompt)
        self.assertIn('"confidence": <number 0-100>\n}', prompt)
        self.assertNotIn('"condition": "W"', prompt)

    def test_solve_first_prompt_defines_final_selected_answer(self) -> None:
        prompt, _, _ = self.builder.build(
            Condition.SF,
            self.sample,
            committed_answer_index=0,
        )

        self.assertIn("Your committed independent answer before seeing the candidate was Option 0: A", prompt)
        self.assertIn("Candidate answer: Option 1: B", prompt)
        self.assertIn("verification_decision refers specifically to whether the presented candidate is correct.", prompt)
        self.assertIn("For selected_answer_index, report the option you now believe is the correct answer", prompt)

    def test_gemma4_completion_wrapper_is_supported(self) -> None:
        config = yaml.safe_load(Path("configs/gemma_4_E2B_it.yaml").read_text(encoding="utf-8"))
        parsed = ExperimentConfig.model_validate(config)

        self.assertEqual(parsed.models[0].prompt_wrapper, "gemma4")
        wrapped = _wrap_completion_prompt("Return JSON only.", "gemma4")
        self.assertTrue(wrapped.startswith("<start_of_turn>user\nReturn JSON only.<end_of_turn>"))
        self.assertTrue(wrapped.endswith("<start_of_turn>model\n{\n"))

    def test_qwen_chat_config_disables_thinking(self) -> None:
        config = yaml.safe_load(Path("configs/qwen3_8b.yaml").read_text(encoding="utf-8"))
        parsed = ExperimentConfig.model_validate(config)

        self.assertEqual(parsed.models[0].api_endpoint, "chat")
        self.assertEqual(parsed.models[0].prompt_wrapper, "none")
        self.assertFalse(parsed.models[0].extra_body["chat_template_kwargs"]["enable_thinking"])

    def test_aya_cohere_completion_wrapper_is_supported(self) -> None:
        config = yaml.safe_load(Path("configs/aya_expanse_8b.yaml").read_text(encoding="utf-8"))
        parsed = ExperimentConfig.model_validate(config)

        self.assertEqual(parsed.models[0].api_endpoint, "completions")
        self.assertEqual(parsed.models[0].prompt_wrapper, "cohere")
        wrapped = _wrap_completion_prompt("Return JSON only.", "cohere")
        self.assertTrue(
            wrapped.startswith(
                "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Return JSON only.<|END_OF_TURN_TOKEN|>"
            )
        )
        self.assertTrue(wrapped.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{\n"))

    def test_phi4_uses_completion_wrapper(self) -> None:
        config = yaml.safe_load(Path("configs/phi4.yaml").read_text(encoding="utf-8"))
        parsed = ExperimentConfig.model_validate(config)

        self.assertEqual(parsed.models[0].api_endpoint, "completions")
        self.assertEqual(parsed.models[0].prompt_wrapper, "phi4")
        wrapped = _wrap_completion_prompt("Return JSON only.", "phi4")
        self.assertTrue(wrapped.startswith("<|im_start|>user<|im_sep|>Return JSON only.<|im_end|>"))
        self.assertTrue(wrapped.endswith("<|im_start|>assistant<|im_sep|>{\n"))


if __name__ == "__main__":
    unittest.main()
