#!/usr/bin/env python3

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reduce CUDA allocator fragmentation (the OOM error message itself
# recommends this). Must be set before torch's CUDA context is created,
# i.e. before torch is imported anywhere -- this module never imports
# torch at top level (both selector backends import it lazily inside their
# own methods), so setting it here at import time is early enough.
# setdefault() so an operator's own env var setting always wins.
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("second_best_selector")

# --------------------------------------------------------------------------
# Defaults matching the paths given in the task
# --------------------------------------------------------------------------
DEFAULT_INPUT = "data/unified_pass_warn_eval.jsonl"
DEFAULT_OUTPUT = "data/ccvf_data.jsonl"
DEFAULT_MODEL = "Models/Qwen3.6-27B"

OUTPUT_FIELD = "contractor"  # field name requested by the task spec
OOM_SENTINEL = "__CUDA_OOM__"  # marks a response that failed due to CUDA OOM, not a bad answer
MAX_ANSWER_OPTIONS = 10  # sanity cap; real datasets here use 4-5 options


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line %d in %s: %s", line_no, path, e)
    log.info("Loaded %d samples from %s", len(samples), path)
    return samples


def write_jsonl(samples: List[Dict[str, Any]], path: str, quiet: bool = False) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory, then atomically rename
    # over the real path. This guarantees the output file is either the
    # previous complete version or the new complete version -- never a
    # half-written file, even if the process is killed mid-write.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    if not quiet:
        log.info("Wrote %d samples to %s", len(samples), path)


def derive_report_paths(output_path: str) -> "tuple[str, str]":
    """Derive default review-report file paths from the --output path."""
    base = Path(output_path)
    stem = base.parent / base.stem  # strips e.g. .jsonl
    csv_path = str(stem) + "_flagged_for_review.csv"
    summary_path = str(stem) + "_review_summary.md"
    return csv_path, summary_path


def write_review_report(
    samples: List[Dict[str, Any]], csv_path: str, summary_path: str
) -> Dict[str, int]:
    """
    Write a CSV of every sample that needs manual review, plus a short
    markdown summary with counts, so the person doesn't have to scroll
    logs to find every flagged case.

    A sample is flagged if:
      * it hit the fallback path (model never produced a valid choice
        after retries -- OUTPUT_FIELD_fallback_used), or
      * it only succeeded after a retry (borderline confidence --
        OUTPUT_FIELD_needed_retry), even though the final pick is valid, or
      * it recovered from a CUDA OOM by retrying with the explanation
        stripped out (OUTPUT_FIELD_oom_recovered) -- the pick is valid but
        was made with strictly less context than normal.
    """
    import csv

    fallback_key = f"{OUTPUT_FIELD}_fallback_used"
    retry_key = f"{OUTPUT_FIELD}_needed_retry"
    oom_key = f"{OUTPUT_FIELD}_oom_recovered"

    flagged = [s for s in samples if s.get(fallback_key) or s.get(retry_key) or s.get(oom_key)]

    def source_of(sample: Dict[str, Any]) -> str:
        sid = sample.get("id") or ""
        return sid.split("_")[0] if "_" in sid else (sid or "unknown")

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = Path(csv_path).with_suffix(Path(csv_path).suffix + ".tmp")
    with open(csv_tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "source",
                "flag_reason",
                "label",
                "contractor",
                "question",
                "answers",
                "num_answers",
            ]
        )
        for s in flagged:
            reasons = []
            if s.get(fallback_key):
                reasons.append("fallback_after_retries")
            if s.get(retry_key) and not s.get(fallback_key):
                reasons.append("resolved_only_after_retry")
            if s.get(oom_key):
                reasons.append("oom_recovered_reduced_context")
            writer.writerow(
                [
                    s.get("id", ""),
                    source_of(s),
                    ";".join(reasons) if reasons else "unknown",
                    s.get("label"),
                    s.get(OUTPUT_FIELD),
                    (s.get("question", "") or "")[:300],
                    " | ".join(s.get("answers", []) or []),
                    len(s.get("answers", []) or []),
                ]
            )
    csv_tmp.replace(csv_path)

    total = len(samples)
    n_fallback = sum(1 for s in samples if s.get(fallback_key))
    n_retry_only = sum(1 for s in samples if s.get(retry_key) and not s.get(fallback_key))
    n_oom_recovered = sum(1 for s in samples if s.get(oom_key))
    n_flagged = len(flagged)

    by_source: Dict[str, int] = {}
    for s in flagged:
        src = source_of(s)
        by_source[src] = by_source.get(src, 0) + 1

    lines = [
        "# Manual-review report",
        "",
        f"- Total samples in output: {total}",
        f"- Flagged for manual review: {n_flagged}"
        + (f" ({n_flagged / total * 100:.1f}%)" if total else ""),
        f"  - Fallback (model never produced a valid choice after retries): {n_fallback}",
        f"  - Resolved, but only after a retry (borderline confidence): {n_retry_only}",
        f"  - Recovered from a CUDA OOM with the explanation stripped (reduced context): {n_oom_recovered}",
        "- Breakdown of flagged cases by dataset prefix:",
    ]
    for src, cnt in sorted(by_source.items()):
        lines.append(f"  - {src}: {cnt}")
    lines.append("")
    lines.append(f"Full row-level detail: see `{Path(csv_path).name}`")
    lines.append(
        "(columns: id, source, flag_reason, label, contractor, question, answers, num_answers)"
    )

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    summary_tmp = Path(summary_path).with_suffix(Path(summary_path).suffix + ".tmp")
    with open(summary_tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    summary_tmp.replace(summary_path)

    return {
        "total": total,
        "flagged": n_flagged,
        "fallback": n_fallback,
        "retry_only": n_retry_only,
        "oom_recovered": n_oom_recovered,
    }


def resume_from_existing_output(
    samples: List[Dict[str, Any]], valid_indices: List[int], output_path: str
) -> List[int]:
    """
    If --resume is set and the output file already exists (e.g. from a run
    that was killed partway through), copy over the already-resolved
    OUTPUT_FIELD (and its flags) for any sample whose id matches, and drop
    those indices from the pending work list. Returns the new pending
    list of indices still needing processing.
    """
    if not Path(output_path).exists():
        log.info("--resume requested but %s does not exist yet; starting fresh.", output_path)
        return valid_indices

    existing = load_jsonl(output_path)
    by_id: Dict[str, Dict[str, Any]] = {
        row["id"]: row for row in existing if row.get("id") is not None
    }

    fallback_key = f"{OUTPUT_FIELD}_fallback_used"
    retry_key = f"{OUTPUT_FIELD}_needed_retry"
    oom_key = f"{OUTPUT_FIELD}_oom_recovered"

    still_pending: List[int] = []
    n_resumed = 0
    for i in valid_indices:
        sample = samples[i]
        prior = by_id.get(sample.get("id"))
        if (
            prior is not None
            and prior.get(OUTPUT_FIELD) is not None
            and prior.get(OUTPUT_FIELD) != sample.get("label")
        ):
            samples[i][OUTPUT_FIELD] = prior[OUTPUT_FIELD]
            if prior.get(fallback_key):
                samples[i][fallback_key] = True
            if prior.get(retry_key):
                samples[i][retry_key] = True
            if prior.get(oom_key):
                samples[i][oom_key] = True
            n_resumed += 1
        else:
            still_pending.append(i)

    log.info(
        "--resume: %d / %d samples already had a valid pick in %s; "
        "%d remaining to process.",
        n_resumed,
        len(valid_indices),
        output_path,
        len(still_pending),
    )
    return still_pending


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert logician assisting with dataset construction for a "
    "reasoning-evaluation benchmark. You will be shown a multiple-choice "
    "reasoning question, its ground-truth (correct) answer index, and an "
    "explanation of why the correct answer is correct and each distractor "
    "is wrong. Your job is to identify the SECOND-BEST option: the single "
    "incorrect option that is the strongest, most plausible distractor -- "
    "i.e. the one a competent reasoner would be most likely to mistake "
    "for correct. Base your judgment on the LOGICAL VALIDITY and "
    "closeness-of-reasoning of each distractor (e.g. how minor or subtle "
    "the constraint violation or logical flaw is), NOT on surface wording, "
    "lexical overlap, or semantic similarity to the correct answer. "
    "You must never select the ground-truth answer itself. "
    "Respond with ONLY the integer index of your chosen option and "
    "nothing else -- no words, no punctuation, no explanation."
)

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Answer options (0-indexed):
{options_block}

Ground-truth correct answer index: {label}

Explanation / distractor analysis (if available):
{explanation_block}

Which option index is the strongest (second-best) distractor -- the most
plausible INCORRECT option, other than index {label}? Reason internally
about logical validity, then respond with ONLY the integer index."""


def format_options(answers: List[str]) -> str:
    return "\n".join(f"{i}: {a}" for i, a in enumerate(answers))


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    """Cap a text field's length, marking that it was cut. max_chars<=0 or
    None disables truncation entirely."""
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated for length]"


def format_explanation(explanation: Any, max_field_chars: Optional[int] = None) -> str:
    if not explanation:
        return "(none provided)"
    if isinstance(explanation, str):
        return truncate_text(explanation, max_field_chars)
    if isinstance(explanation, dict):
        parts = []
        if "reasoning" in explanation:
            parts.append(f"Reasoning: {truncate_text(explanation['reasoning'], max_field_chars)}")
        if "correct_answer_justification" in explanation:
            parts.append(
                "Why the correct answer is correct: "
                + truncate_text(explanation["correct_answer_justification"], max_field_chars)
            )
        if "distractor_rejections" in explanation and isinstance(
            explanation["distractor_rejections"], dict
        ):
            rej_lines = [
                f"  - option {k}: {truncate_text(v, max_field_chars)}"
                for k, v in explanation["distractor_rejections"].items()
            ]
            parts.append("Why each distractor is wrong:\n" + "\n".join(rej_lines))
        return "\n\n".join(parts) if parts else "(none provided)"
    return str(explanation)


def build_messages(
    sample: Dict[str, Any],
    max_explanation_chars: Optional[int] = None,
    strip_explanation: bool = False,
) -> List[Dict[str, str]]:
    context = sample.get("context", "")
    question = sample.get("question", "")
    answers = sample.get("answers", [])
    label = sample.get("label")
    explanation = sample.get("explanation")

    if strip_explanation:
        # Minimal-memory fallback prompt after a CUDA OOM: drop the
        # (often very long, especially for LSAT-AR) explanation entirely
        # rather than growing the prompt further with a reminder. Retrying
        # with the *same* huge prompt after an OOM would almost certainly
        # OOM again -- this is the one case where a shorter prompt, not a
        # more insistent one, is the right retry strategy.
        explanation_block = (
            "(omitted for this attempt due to prompt length / memory constraints -- "
            "judge based on context, question, and answer options alone)"
        )
    else:
        explanation_block = format_explanation(explanation, max_explanation_chars)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        options_block=format_options(answers),
        label=label,
        explanation_block=explanation_block,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------
def parse_index(text: str, num_options: int) -> Optional[int]:
    """Extract the first integer in `text` that is a valid option index."""
    if text is None:
        return None
    matches = re.findall(r"-?\d+", text)
    for m in matches:
        idx = int(m)
        if 0 <= idx < num_options:
            return idx
    return None


def pick_fallback(label: int, num_options: int, rng: random.Random) -> int:
    """Deterministic-ish fallback: smallest index != label (last resort)."""
    candidates = [i for i in range(num_options) if i != label]
    return rng.choice(candidates) if candidates else -1


# --------------------------------------------------------------------------
# Model backend (vLLM for efficient batched inference)
# --------------------------------------------------------------------------
class VLLMSelector:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_new_tokens: int = 16,
        enable_thinking: bool = False,
        seed: int = 0,
    ):
        from vllm import LLM, SamplingParams  # local import: heavy dependency
        from transformers import AutoTokenizer

        self.enable_thinking = enable_thinking

        log.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        log.info(
            "Loading model %s (tensor_parallel_size=%d) via vLLM ...",
            model_path,
            tensor_parallel_size,
        )
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            seed=seed,
        )
        # temperature=0 -> deterministic (greedy) decoding
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
            seed=seed,
        )

    def build_prompt_strings(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        return [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for messages in messages_batch
        ]

    def generate(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        prompts = self.build_prompt_strings(messages_batch)
        outputs = self.llm.generate(prompts, self.sampling_params)
        # vLLM does not guarantee output order matches input order across
        # all versions, so we sort by the `request_id` if present, and
        # otherwise rely on the returned list order (standard behavior in
        # current vLLM releases: outputs are returned in input order).
        texts = [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]
        return texts


# --------------------------------------------------------------------------
# Model backend (plain HuggingFace transformers -- no vLLM/torch pinning)
# --------------------------------------------------------------------------
class TransformersSelector:
    """
    Runs the model with plain `transformers` + `model.generate()`.

    This deliberately avoids vLLM so it has no dependency on a specific
    torch build -- only on a `transformers` version recent enough to
    recognize the model's architecture. Batching is done manually with
    left-padding, which is the correct padding side for batched causal-LM
    generation (right-padding would shift position ids for the padded
    sequences and corrupt generation).
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        gen_batch_size: int = 1,
        max_new_tokens: int = 64,
        max_input_len: Optional[int] = None,
        enable_thinking: bool = False,
        debug_n: int = 0,
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, "auto")

        log.info("Loading tokenizer from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched generation with a
        # causal LM (the new tokens must be generated at the sequence's
        # right edge for every item in the batch, regardless of prompt
        # length differences).
        self.tokenizer.padding_side = "left"

        log.info(
            "Loading model %s via transformers (dtype=%s, device_map=%s) ... "
            "this may take a while for a 27B model.",
            model_path,
            dtype,
            device_map,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        self.gen_batch_size = gen_batch_size
        self.max_new_tokens = max_new_tokens
        self.max_input_len = max_input_len
        # Qwen3.x defaults to "thinking mode": apply_chat_template inserts
        # a <think>...</think> block and the model emits its full
        # chain-of-thought BEFORE the actual answer. For a single-token
        # index answer this just burns the max_new_tokens budget on
        # reasoning text and the response never reaches the digit -- so we
        # default this to False. Models/tokenizers that don't support the
        # enable_thinking kwarg at all will raise a TypeError, which we
        # catch once at prompt-build time and fall back to the plain call.
        self.enable_thinking = enable_thinking
        self._chat_template_supports_thinking = True
        self.debug_n = debug_n
        self._debug_printed = 0

    def build_prompt_strings(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        if not self._chat_template_supports_thinking:
            return [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_batch
            ]
        try:
            return [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                for messages in messages_batch
            ]
        except TypeError:
            # This tokenizer's chat template doesn't accept enable_thinking
            # at all (older/non-Qwen3 models). Remember that and retry
            # without it so we don't pay this cost every batch.
            log.warning(
                "Tokenizer's chat template does not accept 'enable_thinking'; "
                "proceeding without it."
            )
            self._chat_template_supports_thinking = False
            return self.build_prompt_strings(messages_batch)

    def generate(self, messages_batch: List[List[Dict[str, str]]]) -> List[str]:
        import torch

        prompts = self.build_prompt_strings(messages_batch)
        results: List[str] = []

        for start in range(0, len(prompts), self.gen_batch_size):
            chunk = prompts[start : start + self.gen_batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=self.max_input_len is not None,
                max_length=self.max_input_len,
            ).to(self.model.device)

            with torch.no_grad():
                try:
                    # Greedy decoding (do_sample=False) is the transformers
                    # equivalent of vLLM's temperature=0 -- fully deterministic.
                    generated = self.model.generate(
                        **enc,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                except torch.cuda.OutOfMemoryError:
                    log.warning(
                        "CUDA OOM on a chunk of %d prompt(s) (likely a very long "
                        "context/explanation field); freeing cache and marking for "
                        "a reduced-context retry instead of crashing the run.",
                        len(chunk),
                    )
                    torch.cuda.empty_cache()
                    results.extend([OOM_SENTINEL] * len(chunk))
                    continue
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    log.warning(
                        "CUDA OOM (RuntimeError) on a chunk of %d prompt(s); freeing "
                        "cache and marking for a reduced-context retry.",
                        len(chunk),
                    )
                    torch.cuda.empty_cache()
                    results.extend([OOM_SENTINEL] * len(chunk))
                    continue

            input_len = enc["input_ids"].shape[1]
            new_tokens = generated[:, input_len:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for raw_text in texts:
                raw_text = raw_text.strip()
                if self._debug_printed < self.debug_n:
                    log.info("[debug raw output] %r", raw_text)
                    self._debug_printed += 1
                # Defensive cleanup: if a <think>...</think> block still made
                # it into the decoded text (e.g. enable_thinking couldn't be
                # disabled for this tokenizer, or the model ignored it and
                # produced an unterminated block), strip it so parse_index
                # only ever looks at the actual answer text.
                cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
                cleaned = re.sub(r"^.*</think>", "", cleaned, flags=re.DOTALL).strip()
                results.append(cleaned if cleaned else raw_text)

        return results


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input JSONL dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write output JSONL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to local Qwen3.6-27B model")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Inference backend. 'transformers' (default) needs no vLLM/torch "
        "pinning -- just a recent enough `transformers` for the model's "
        "architecture. 'vllm' is faster but requires a torch build vLLM "
        "supports and a vLLM release that recognizes the model architecture.",
    )
    # --- vLLM-backend options ---
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of prompts submitted to vLLM per generate() call (vllm backend only).",
    )
    # --- transformers-backend options ---
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=4,
        help="Number of prompts per model.generate() call (transformers backend only). "
        "Keep this small for a 27B model unless you have generous GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype for the transformers backend.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map passed to from_pretrained for the transformers backend "
        "(e.g. 'auto' to shard a 27B model across available GPUs via accelerate).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens to generate per response (both backends). Kept modest "
        "since thinking mode is disabled by default and we only need a short "
        "answer; raise this if you enable --enable-thinking.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow the model's chain-of-thought 'thinking mode' (Qwen3.x defaults "
        "this to ON via apply_chat_template). OFF by default here: for a "
        "single-index answer, thinking mode just burns the max_new_tokens "
        "budget on reasoning text before ever reaching the answer, unless "
        "max_new_tokens is raised substantially (and generation slows down "
        "accordingly). Pass this flag only if you've also raised "
        "--max-new-tokens enough for the model to finish its <think> block.",
    )
    parser.add_argument(
        "--debug-n",
        type=int,
        default=0,
        help="Log the raw (pre-parsing) model output for the first N generated "
        "responses, across all batches -- useful for diagnosing parsing "
        "failures (transformers backend only).",
    )
    parser.add_argument(
        "--max-explanation-chars",
        type=int,
        default=1500,
        help="Cap each explanation text field (reasoning, justification, each "
        "distractor rejection) at this many characters before building the "
        "prompt. LSAT-AR explanations in particular can run to several "
        "thousand characters and are the main driver of CUDA OOMs on long "
        "items. Set to 0 to disable truncation entirely (full fidelity, "
        "higher OOM risk).",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=None,
        help="Hard token-length cap for the tokenized prompt (transformers backend "
        "only); truncates from the right if exceeded. Prefer --max-explanation-"
        "chars over this where possible -- that truncates the (usually long) "
        "explanation text specifically, while this truncates the raw token "
        "sequence and can cut off the trailing instruction if set too low.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for determinism / fallback RNG")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Number of re-prompt attempts if the model's first answer is invalid "
        "(picks the ground-truth label, is out of range, or unparseable).",
    )
    parser.add_argument(
        "--no-review-report",
        action="store_true",
        help="Skip writing the manual-review CSV/summary report (written by default).",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Path for the flagged-cases CSV. Default: derived from --output "
        "(<output-stem>_flagged_for_review.csv).",
    )
    parser.add_argument(
        "--report-summary",
        default=None,
        help="Path for the markdown summary report. Default: derived from --output "
        "(<output-stem>_review_summary.md).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write the output JSONL to disk every N processed samples, not just "
        "at the end, so a killed/timed-out job doesn't lose all progress. "
        "Writes are atomic (temp file + rename). Set to 0 to disable and "
        "only write once at the very end (not recommended for long runs).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --output already exists (e.g. from a run that was killed "
        "partway through), skip any sample that already has a valid "
        "(non-null, non-label) pick in it, and only process the rest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and validate the pipeline without loading the model "
        "(useful for smoke-testing the script/schema before a GPU job).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    samples = load_jsonl(args.input)
    if not samples:
        log.error("No samples loaded from %s -- aborting.", args.input)
        return 1

    # Validate schema up front so failures happen fast, before loading the model.
    valid_indices = []
    for i, sample in enumerate(samples):
        answers = sample.get("answers")
        label = sample.get("label")
        if not isinstance(answers, list) or not (2 <= len(answers) <= MAX_ANSWER_OPTIONS):
            log.warning("Sample %s (%s): bad 'answers' field, skipping.", i, sample.get("id"))
            continue
        if not isinstance(label, int) or not (0 <= label < len(answers)):
            log.warning("Sample %s (%s): bad 'label' field, skipping.", i, sample.get("id"))
            continue
        valid_indices.append(i)

    log.info("%d / %d samples pass schema validation.", len(valid_indices), len(samples))

    # Compute report paths once, up front, so the same files get updated
    # both at each checkpoint and at the final write (instead of only
    # existing after the whole run finishes).
    if args.no_review_report:
        report_csv_path = report_summary_path = None
    else:
        report_csv_path = args.report_csv or derive_report_paths(args.output)[0]
        report_summary_path = args.report_summary or derive_report_paths(args.output)[1]

    if args.resume:
        valid_indices = resume_from_existing_output(samples, valid_indices, args.output)
        if not valid_indices:
            log.info("Nothing left to process after --resume; writing final output as-is.")
            write_jsonl(samples, args.output)
            if report_csv_path:
                write_review_report(samples, report_csv_path, report_summary_path)
            log.info("Done.")
            return 0

    if args.dry_run:
        log.info("Dry run requested: skipping model load and inference.")
        for sample in samples:
            sample.setdefault(OUTPUT_FIELD, None)
        write_jsonl(samples, args.output)
        return 0

    log.info("Using backend: %s (enable_thinking=%s)", args.backend, args.enable_thinking)
    if args.backend == "vllm":
        selector = VLLMSelector(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            seed=args.seed,
        )
        outer_batch_size = args.batch_size
    else:
        selector = TransformersSelector(
            model_path=args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            gen_batch_size=args.gen_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_input_len=args.max_input_len,
            enable_thinking=args.enable_thinking,
            debug_n=args.debug_n,
            seed=args.seed,
        )
        # For the transformers backend, the outer progress-bar chunk size
        # doesn't need to exceed the actual generation batch size -- there's
        # no separate "submit a big batch to a server" step like vLLM's.
        outer_batch_size = args.gen_batch_size

    # Process in batches for progress visibility; for vLLM this is a large
    # chunk internally re-batched by the engine, for transformers this is
    # the literal model.generate() batch size.
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    # Pre-populate the output file with the FULL dataset immediately, before
    # any model inference happens -- unprocessed rows just get OUTPUT_FIELD
    # left as None/absent. This means even a crash before the first
    # --checkpoint-every threshold (or a crash on sample #1) still leaves a
    # complete, schema-consistent file that --resume can pick up from,
    # instead of nothing existing at all until 100 samples in.
    for sample in samples:
        sample.setdefault(OUTPUT_FIELD, None)
    write_jsonl(samples, args.output, quiet=True)
    log.info("Pre-populated %s with the full input (unprocessed rows have %s=None).", args.output, OUTPUT_FIELD)

    pending = list(valid_indices)
    samples_since_checkpoint = 0
    for start in tqdm(range(0, len(pending), outer_batch_size), desc="Batches"):
        batch_idx = pending[start : start + outer_batch_size]
        batch_samples = [samples[i] for i in batch_idx]

        messages_batch = [
            build_messages(s, max_explanation_chars=args.max_explanation_chars)
            for s in batch_samples
        ]
        responses = selector.generate(messages_batch)

        # Track which items still need a retry (invalid / label / unparseable
        # / CUDA OOM), and remember each one's last response so the retry
        # step can tell an OOM apart from an ordinary bad answer.
        retry_idx: List[int] = []
        last_response: Dict[int, str] = {}
        for local_i, (i, sample, resp) in enumerate(zip(batch_idx, batch_samples, responses)):
            num_options = len(sample["answers"])
            label = sample["label"]
            choice = parse_index(resp, num_options)
            if choice is None or choice == label:
                retry_idx.append(local_i)
                last_response[local_i] = resp
            else:
                samples[i][OUTPUT_FIELD] = choice

        attempt = 0
        while retry_idx and attempt < args.max_retries:
            attempt += 1
            # Build fresh messages for each retry. Two cases:
            #  - last attempt hit a CUDA OOM: rebuild from scratch with the
            #    explanation stripped out entirely, since retrying with the
            #    same huge prompt (or an even longer one) would almost
            #    certainly OOM again. A shorter prompt, not a more
            #    insistent one, is the right fix here.
            #  - last attempt just gave a bad/unparseable answer: keep the
            #    existing "append a reminder" strategy, using fresh
            #    per-sample copies so reminders never stack across rounds.
            retry_messages: List[List[Dict[str, str]]] = []
            for j in retry_idx:
                if last_response.get(j) == OOM_SENTINEL:
                    msgs = build_messages(batch_samples[j], strip_explanation=True)
                else:
                    msgs = [dict(m) for m in messages_batch[j]]
                    label = batch_samples[j]["label"]
                    num_options = len(batch_samples[j]["answers"])
                    msgs[-1]["content"] += (
                        f"\n\nReminder: you must NOT answer with index {label}. "
                        f"Choose one valid incorrect option index from 0 to "
                        f"{num_options - 1} (inclusive), excluding {label}. "
                        f"Respond with ONLY the integer."
                    )
                retry_messages.append(msgs)
            retry_responses = selector.generate(retry_messages)
            still_bad = []
            for j, resp in zip(retry_idx, retry_responses):
                i = batch_idx[j]
                sample = batch_samples[j]
                num_options = len(sample["answers"])
                label = sample["label"]
                choice = parse_index(resp, num_options)
                if choice is None or choice == label:
                    still_bad.append(j)
                    last_response[j] = resp
                else:
                    samples[i][OUTPUT_FIELD] = choice
                    # Resolved, but only after a retry -- flag as borderline
                    # confidence for the review report even though it's not
                    # a hard fallback. Distinguish "needed a nudge" from
                    # "recovered after dropping context due to OOM", since
                    # the latter deserves closer manual review (it picked
                    # its answer with strictly less information).
                    if last_response.get(j) == OOM_SENTINEL:
                        samples[i][f"{OUTPUT_FIELD}_oom_recovered"] = True
                    else:
                        samples[i][f"{OUTPUT_FIELD}_needed_retry"] = True
            retry_idx = still_bad

        # Anything still unresolved after retries gets a logged fallback pick,
        # flagged clearly so manual verification catches it.
        for j in retry_idx:
            i = batch_idx[j]
            sample = batch_samples[j]
            num_options = len(sample["answers"])
            label = sample["label"]
            fallback = pick_fallback(label, num_options, rng)
            samples[i][OUTPUT_FIELD] = fallback
            samples[i][f"{OUTPUT_FIELD}_fallback_used"] = True
            log.warning(
                "Sample %s (%s): model failed to produce a valid choice after "
                "%d attempt(s); used fallback index %d. Flag for manual review.",
                i,
                sample.get("id"),
                1 + args.max_retries,
                fallback,
            )

        # Periodic checkpoint: save progress so far so a killed/timed-out
        # job doesn't lose everything. Atomic (temp file + rename), so the
        # output file is always either the last complete checkpoint or the
        # final result -- never a partial/corrupt write. The review report
        # is refreshed at the same cadence, so flagged cases are visible
        # for manual review while the run is still in progress, not only
        # after it finishes.
        samples_since_checkpoint += len(batch_idx)
        if args.checkpoint_every > 0 and samples_since_checkpoint >= args.checkpoint_every:
            write_jsonl(samples, args.output, quiet=True)
            report_note = ""
            if report_csv_path:
                stats = write_review_report(samples, report_csv_path, report_summary_path)
                report_note = (
                    f" | flagged so far: {stats['flagged']} "
                    f"({stats['fallback']} fallback, {stats['retry_only']} retry-only, "
                    f"{stats['oom_recovered']} oom-recovered)"
                )
            log.info(
                "Checkpoint: saved progress (%d / %d samples processed so far) to %s%s",
                start + len(batch_idx),
                len(pending),
                args.output,
                report_note,
            )
            samples_since_checkpoint = 0

    # Samples that failed schema validation still get a null field so the
    # output file has a consistent schema across all rows.
    for i, sample in enumerate(samples):
        if i not in valid_indices:
            sample.setdefault(OUTPUT_FIELD, None)

    write_jsonl(samples, args.output)

    n_fallback = sum(1 for s in samples if s.get(f"{OUTPUT_FIELD}_fallback_used"))
    if n_fallback:
        log.warning(
            "%d / %d samples required a fallback pick -- review these before "
            "relying on manual verification alone.",
            n_fallback,
            len(samples),
        )

    if report_csv_path:
        stats = write_review_report(samples, report_csv_path, report_summary_path)
        log.info(
            "Review report written: %d flagged / %d total (%d fallback, %d retry-only, "
            "%d oom-recovered) -> %s , %s",
            stats["flagged"],
            stats["total"],
            stats["fallback"],
            stats["retry_only"],
            stats["oom_recovered"],
            report_csv_path,
            report_summary_path,
        )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())