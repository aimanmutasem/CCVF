# CCVF Runner

A benchmark tool for evaluating LLM verification decisions across five conditions: **N, W_naive, W, SF, CC**.
Models are queried through an OpenAI-compatible API — either a remote service (GPT-4o-mini) or a local vLLM server.

---

## 1. Environment Setup

Load cluster modules and activate the base conda environment:

```bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load shared cuda12.4/toolkit/12.4.0 anaconda3/anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ccvf
cd project_directory
```

Create the vLLM 0.19.1 virtual environment (**first time only**):

```bash
/.../.conda/envs/ccvf/bin/python -m venv .venv_vllm0191_py310
.venv_vllm0191_py310/bin/pip install vllm==0.19.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Make scripts executable (**first time only**):

```bash
chmod +x scripts/*.sh
```

---

## 2. Install Requirements

```bash
.venv_vllm0191_py310/bin/pip install -r requirements.txt
```

---

## 3. Generate 2nd-Best Candidates

Run this before serving the model (**first time only or use the uploaded dataset**):

```bash
python scripts/2nd_best_candidate.py 
```

---

## 4. Start the Model Server

Open a **tmux** session and run:

```bash
scripts/serve_model_vllm0191.sh --config configs/qwen3_8b.yaml --port 8000 --tp 2
```

Wait until the terminal prints `Application startup complete` before moving on.
Run the server **once per model/port** — starting it again on the same node will fail due to GPU memory.

---

## 5. Run the Benchmark

In a **second tmux** session:

```bash
export LOCAL_OPENAI_API_KEY=EMPTY
python scripts/run_ccvf.py run --config configs/qwen3_8b.yaml --run-id qwen3_8b
```

For GPT-4o-mini (no local server needed):

```bash
export OPENAI_API_KEY=<your_key>
python scripts/run_ccvf.py run --config configs/gpt_4o_mini.yaml --run-id gpt_4o_mini
```

---

## Supported Configs

```
configs/qwen3_8b.yaml
configs/qwen3_14b.yaml
configs/llama3_2_3b.yaml
configs/llama4_scout_17b_16e.yaml
configs/ministral_3_14b_2512.yaml
configs/mistral_small_4.yaml
configs/aya_expanse_8b.yaml
configs/gemma_3_12b_it_vllm0191.yaml
configs/gpt_4o_mini.yaml
```

> Verify that the `model:` path in each config file exists on the cluster before serving.

---

## Results

Each run saves output under:

```
results/runs/<run_id>/
├── raw/                        # raw JSONL responses
├── parsed/outputs.csv          # structured outputs
├── metrics/                    # decision metrics and statistical tests
├── reports/report.md           # summary report
└── logs/runner_events.jsonl    # run log
```
