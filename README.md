# CCVF Clean Runner

This directory is a clean replacement for the older messy `caver` working copy.
It runs the CAVER/CCVF decision conditions:

```text
N, W_naive, W, SF, CC
```

The standard model path is OpenAI-compatible chat completions:

- `provider: openai` for GPT-4o-mini API.
- `provider: local_openai` for local checkpoints served by vLLM or SGLang.
- `provider: hf_local` is available as a slow fallback when installed vLLM cannot serve a checkpoint.

## Local Cluster Setup

```bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load shared
module load cuda12.4/toolkit/12.4.0
module load anaconda3/anaconda3
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate qwnccr
cd /u/erdos/csga2/aalfatemi/LLM_MSGCNET/NSMA/ccvf
```

## Run GPT-4o-mini

```bash
export OPENAI_API_KEY=...
python scripts/run_ccvf.py --config configs/gpt_4o_mini.yaml --run-id gpt_4o_mini
```

## Run A Local Model

Start vLLM first. Example for the Llama checkpoint currently available on the server:

```bash
python scripts/serve_model.py --config configs/llama3_1_8b_available.yaml --port 8000 --tp 2
```

For large checkpoints such as Phi-4, wait until the server prints
`Application startup complete` before starting `run_ccvf.py`. The runner also
polls server health before preflight, so if the server is still loading weights
the run will wait instead of sending a generation request too early.
Run `serve_model...` only once per model/port. After it stays open in the first
tmux, use a second tmux for `python scripts/run_ccvf.py run ...`; starting
`serve_model...` again on the same node tries to load a second copy of the model
and will correctly fail the GPU-memory preflight.

For newer Qwen3 / Mistral3 checkpoints on the V100 nodes, use the isolated
vLLM 0.8.5 environment:

```bash
scripts/serve_model_vllm085.sh --config configs/qwen3_8b.yaml --port 8000 --tp 1
```

The wrapper uses `.venv_vllm085` and automatically enables the newer
Transformers overlay only when the installed Transformers cannot parse the
checkpoint config. Set `CCVF_DISABLE_TRANSFORMERS_OVERLAY=1` only when you
explicitly want to diagnose behavior without the overlay. The wrapper is
intended for model types that `qwnccr`'s `vllm 0.6.4` cannot execute, such as
`Qwen3ForCausalLM`. The cluster V100 nodes use CUDA 12.2 drivers, so newer vLLM
builds that require CUDA 12.8/13 are not suitable there.
The wrapper writes vLLM stdout/stderr to `logs/vllm_<timestamp>.log` by default.
The serving command uses `--generation-config vllm` so hidden Hugging Face
generation defaults do not override the benchmark's explicit request settings.
The wrapper also checks free GPU memory before launching. If it reports only a
few GiB free, another process is already occupying the selected GPU(s); choose an
idle node/GPU or stop only your own stale vLLM process. The configs use
conservative `gpu_memory_utilization` and `max_num_seqs` values for V100 nodes,
but no config can start a model when the selected GPUs are already full.

Qwen configs use the standard OpenAI-compatible chat-completions path with
`extra_body.chat_template_kwargs.enable_thinking: false`. Aya/Cohere configs use
the completions path with `prompt_wrapper: cohere`, which renders the tokenizer's
Cohere turn markers and prefills `{` so the model returns the expected compact
JSON object.

Gemma4/Gemma4Unified checkpoints need the newer Transformers overlay for
`AutoConfig.from_pretrained()` because `.venv_vllm085` uses Transformers 4.51.3.
The wrapper now detects this and applies the overlay automatically. However,
vLLM 0.8.5 still has no registered execution implementation for
`Gemma4ForConditionalGeneration` or `Gemma4UnifiedForConditionalGeneration`, so
Gemma4 12B/31B cannot be served with this vLLM 0.8.5 environment. Use a newer
vLLM release with explicit Gemma4 support, another backend that supports Gemma4,
or a Gemma3/other supported checkpoint.

Gemma3 multimodal checkpoints such as `gemma-3-12b-it` can load in vLLM 0.8.5,
but this environment can produce empty completions that run until
`max_output_tokens` with `finish_reason=length` when vLLM defaults ignore Gemma's
checkpoint generation config. The Gemma3 config therefore sets
`serve.generation_config: auto` so vLLM uses the checkpoint EOS settings. The
runner sends a small preflight request before the full benchmark and aborts
immediately if the server returns unusable output. Use a newer V100-compatible
vLLM environment for Gemma3 if the old-vLLM preflight still fails.
For vLLM 0.8.5, the Gemma3 config uses `api_endpoint: completions` with
`prompt_wrapper: gemma3` to bypass the older chat-completions frontend when it
hangs during preflight. The wrapper intentionally omits an explicit `<bos>`
because vLLM adds BOS during tokenization, and it prefills the assistant turn
with `{` to discourage empty special-token-only generations.

Phi-4 checkpoints commonly declare `torch_dtype: bfloat16`, but the cluster V100
GPUs have compute capability 7.0 and cannot run BF16. `configs/phi4.yaml` uses
`serve.dtype: half`; the serving wrapper also downgrades BF16 to half on visible
pre-Ampere GPUs before vLLM starts.

A separate Gemma3 test environment is available at `.venv_vllm0191_py310` with
`vllm==0.19.1`, `torch==2.10.0+cu128`, and `transformers==5.14.0`. It leaves
`.venv_vllm085` unchanged. This venv is built from the shared `qwnccr` Python
so GPU nodes do not accidentally run an older `/usr/bin/python3`. The
`serve_model_vllm0191.sh` wrapper skips the old launcher's Transformers overlay
probe and lets vLLM 0.19.1 use its installed Transformers directly. Start Gemma3
with:

```bash
scripts/serve_model_vllm0191.sh --config configs/gemma_3_12b_it_vllm0191.yaml --port 8003 --tp 2
```

If this fails immediately with an NVIDIA driver / CUDA 12.8 error, the node
driver is too old for this newer vLLM wheel; continue using `.venv_vllm085` for
Qwen/Llama and run Gemma3 on a node with a newer driver or another backend.

In a second terminal:

```bash
export LOCAL_OPENAI_API_KEY=EMPTY
python scripts/run_ccvf.py run --config configs/gemma_3_12b_it_vllm0191.yaml --run-id gemma_3_12b_it_vllm0191
```

The same pattern works for:

```text
configs/qwen3_8b.yaml
configs/qwen3_14b.yaml
configs/llama3_2_3b.yaml
configs/llama4_scout_17b_16e.yaml
configs/ministral_3_14b_2512.yaml
configs/mistral_small_4.yaml
configs/aya_expanse_8b.yaml
configs/gpt_4o_mini.yaml
```

Before serving, make sure the `model:` path in the config actually exists.

Local-model configs set `extra_body.chat_template_kwargs.enable_thinking: false`.
GPT-4o-mini has no thinking-mode flag, so it uses JSON mode and the same prompts
that forbid hidden reasoning output.

## Results

Each run writes here:

```text
results/runs/<run_id>/
```

Important files:

```text
raw/*.jsonl
parsed/outputs.csv
metrics/decision_metrics.csv
metrics/decision_metrics_diag.csv
metrics/statistical_tests.csv
metrics/statistical_tests.json
reports/report.md
logs/runner_events.jsonl
```

The runner distinguishes model parse failures from infrastructure failures
such as `APIConnectionError` and `APITimeoutError`. Infrastructure failures are
not considered completed work for resume. If the model server becomes unhealthy,
the circuit breaker aborts the condition before thousands of invalid rows are
written; restart vLLM and rerun the same command with resume enabled.

Every parsed output has a required `confidence` value from 0 to 100. The parsed
CSV stores it per row, and the metrics/statistical tests include the confidence-
weighted rates and Brier scores. Dataset split values are ignored and are not
written to outputs.

The report separates verification quality from verification decisiveness:

- Verification quality: `Naive_WCAR`, `WCAR`, `SF_WCAR`, `SFPR_abs`, `SFPR_rel`, and `CCAR`.
- Verification decisiveness: `NotSureRate_*` and `DecisionRate_*`.

`Not Sure` is treated as an explicit verification outcome. It is not merged with
`Reject` and is not discarded. The benchmark reports:

```text
NSR = NotSure / ICS
DecisionRate = (Accept + Reject) / ICS
```

`W_naive` uses the same wording as `W` except for the serialized condition label.
It is operationally identical as a prompt, but it is run over the full dataset so
`Naive_WCAR` can use the conventional full-dataset denominator. `WCAR`,
`SF_WCAR`, `CCAR`, `NSR`, and `DecisionRate` all use the model-specific
independent-correct set (ICS) denominator. If old run artifacts do not contain
`W_naive`, `Naive_WCAR` falls back to the older W-based calculation. The
diagnostic `Coverage_*` columns are retained as a backward-compatible alias for
`DecisionRate_*`.

`Candidate_Alignment_Rate` is:

```text
P(selected_answer_index == wrong candidate_index | sample in ICS, W condition)
```

It measures whether the model's selected answer aligns with the injected wrong
candidate; it does not require the verification decision to be `Accept`.

For `SF`, `committed_answer_index` is the earlier N-condition answer. The SF
`selected_answer_index` is the model's final answer after seeing the candidate.
The analysis keeps behavioral inconsistencies as valid outputs and reports:

```text
SF_Answer_Change_Rate
SF_Candidate_Alignment_Rate
SF_Accept_Candidate_Mismatch_Rate
SF_Reject_Candidate_Match_Rate
```

`decision_metrics.csv` and `statistical_tests.csv` report each model at three
aggregation levels:

```text
Overall
LogiQA
AR-LSAT
```

This is dataset-level reporting only. The benchmark remains split-free
internally and does not use train/dev/test split fields.

## Why Llama Was Hanging Before

The old `caver/scripts/run_caver.py` loaded Llama directly with Transformers and
generated one request at a time. It also passed sampling parameters while
`do_sample=False`, producing warnings and slow behavior. This clean runner uses
vLLM/OpenAI-compatible serving by default, which is the standard and much faster
way to run Llama/Qwen/Mistral/Gemma models for the 10k-sample benchmark.
