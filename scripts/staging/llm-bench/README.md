# LLM Inference Benchmark

Benchmarking framework that compares LLM inference across three backends:
OpenAI API, vLLM, and SystemDS JMLC with the native `llmPredict` built-in.
Evaluated on 5 workloads (math, reasoning, summarization, JSON extraction,
embeddings) with n=50 per workload.

## Purpose

- How does SystemDS's `llmPredict` built-in compare to dedicated LLM backends
  (OpenAI, vLLM) in terms of accuracy and throughput?
- What is the cost-performance tradeoff across cloud APIs and GPU-accelerated
  backends?

The framework runs standardized workloads against all backends under identical
conditions (same prompts, same evaluation metrics). Default workload configs
use `n_samples: 50` and `temperature: 0.0` for reproducibility; hardware and
cloud settings are up to you when you run the suite locally.

## Quick Start

```bash
cd scripts/staging/llm-bench
pip install -r requirements.txt

# Optional: download HuggingFace datasets into the local cache (needs network once)
python scripts/prefetch_datasets.py

# Set OpenAI API key (required for openai backend)
export OPENAI_API_KEY="sk-..."

# Run a single benchmark
python runner.py \
  --backend openai \
  --workload workloads/math/config.yaml \
  --out results/openai_math

# Run all workloads for a backend (with hardware cost flags for GPU)
./scripts/run_all_benchmarks.sh vllm Qwen/Qwen2.5-3B-Instruct \
  --power-draw-w 350 --hardware-cost 30000

# Run vLLM + SystemDS back-to-back (GPU comparison mode)
./scripts/run_all_benchmarks.sh gpu Qwen/Qwen2.5-3B-Instruct \
  --power-draw-w 350 --hardware-cost 30000

# Run all backends at once
./scripts/run_all_benchmarks.sh all

# Generate report
python scripts/report.py --results-dir results/ --out results/report.html
```

## Project Structure

```
scripts/staging/llm-bench/
├── runner.py                  # Main benchmark runner (CLI entry point)
├── backends/
│   ├── openai_backend.py      # OpenAI API (gpt-4.1-mini)
│   ├── vllm_backend.py        # vLLM serving engine (non-streaming HTTP)
│   └── systemds_backend.py    # SystemDS JMLC via Py4J + llmPredict DML
├── workloads/
│   ├── math/                  # GSM8K dataset, numerical accuracy
│   ├── reasoning/             # BoolQ dataset, logical accuracy
│   ├── summarization/         # XSum dataset, ROUGE-1 scoring
│   ├── json_extraction/       # CoNLL-2003, structured extraction
│   └── embeddings/            # STS-Benchmark, similarity scoring
├── evaluation/
│   └── perf.py                # Latency, throughput metrics
├── scripts/
│   ├── report.py              # HTML report generator
│   ├── aggregate.py           # Cross-run aggregation
│   ├── prefetch_datasets.py   # HuggingFace dataset prefetch (offline-friendly)
│   └── run_all_benchmarks.sh  # Batch automation
├── results/                   # Created locally when you run benchmarks (gitignored)
└── tests/                     # Unit tests for accuracy checks + runner
```

## Backends

| Backend | Type | Model | Hardware | Inference Path |
|---------|------|-------|----------|----------------|
| OpenAI | Cloud API | gpt-4.1-mini | MacBook (API call) | Python HTTP to OpenAI servers |
| vLLM | GPU server | Qwen2.5-3B-Instruct | NVIDIA H100 | Python HTTP to vLLM engine |
| SystemDS | JMLC API | Qwen2.5-3B-Instruct | NVIDIA H100 | Py4J -> JMLC -> DML llmPredict -> Java HTTP -> vLLM |

All backends implement the same interface (`generate(prompts, config) -> List[Result]`),
producing identical output format: text, latency_ms, token counts. SystemDS and
vLLM use the same model on the same vLLM inference server with identical
parameters (temperature=0.0, top_p=0.9, max_tokens).

## Workloads

| Workload | Dataset | Evaluation |
|----------|---------|------------|
| `math` | GSM8K (HuggingFace) | Exact numerical match |
| `reasoning` | BoolQ (HuggingFace) | Extracted yes/no match |
| `summarization` | XSum (HuggingFace) | ROUGE-1 F1 >= 0.2 |
| `json_extraction` | CoNLL-2003 (HuggingFace) | Entity-level F1 >= 0.5 |
| `embeddings` | STS-B (HuggingFace) | Score within +/-1.0 of reference |

## SystemDS Backend

The SystemDS backend uses Py4J to bridge Python and Java, running the
`llmPredict` DML built-in through JMLC:

```
Python -> Py4J -> JMLC -> DML compilation -> llmPredict instruction -> Java HTTP -> vLLM server
```

```bash
# Build SystemDS
mvn package -DskipTests

# Start inference server
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --port 8080

# Run benchmark
export LLM_INFERENCE_URL="http://localhost:8080/v1/completions"
python runner.py --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml --out results/systemds_math
```

Environment variables:
- `SYSTEMDS_JAR` -- path to SystemDS.jar (default: auto-detected)
- `LLM_INFERENCE_URL` -- inference server endpoint (default: `http://localhost:8080/v1/completions`)
- `CUBLAS_WORKSPACE_CONFIG` -- set to `:4096:8` for deterministic cuBLAS

## Reproducibility and reports

Per-run metrics, prompts, and predictions are **not** committed to the
repository. After you run `runner.py` (or `scripts/run_all_benchmarks.sh`),
outputs land under `results/` (gitignored). Aggregate them and build an HTML
summary with:

```bash
python scripts/report.py --results-dir results/ --out results/report.html
```

Use `scripts/aggregate.py` when you want cross-run summaries from multiple
backend directories.

## `runner.py` CLI: cost, concurrency, logging

| Flag | Meaning |
|------|---------|
| `--backend` | `openai`, `vllm`, or `systemds` (required) |
| `--workload` | Path to a workload `config.yaml` (required) |
| `--model` | Backend-specific model id or path |
| `--out` | Output directory for `samples.jsonl`, `metrics.json`, etc. (required) |
| `--concurrency` | Parallel requests (default `1`; SystemDS uses Java-side concurrency when greater than 1) |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default `INFO`) |

**Cloud GPU rental (mutually exclusive with owned-hardware electricity/amortization below):**

| Flag | Meaning |
|------|---------|
| `--gpu-hour-cost` | USD per GPU-hour (e.g. cloud H100 rate). Rental already bundles energy and depreciation, so do not combine with `--power-draw-w` / `--hardware-cost`. |
| `--gpu-count` | GPUs used for wall-clock → GPU-hour conversion (default `1`). |

If both rental and owned flags are set, `runner.py` logs a warning and **uses
only the rental path** (`--gpu-hour-cost`) to avoid double-counting.

**Owned hardware (electricity + amortization):**

| Flag | Meaning |
|------|---------|
| `--power-draw-w` | Average power draw in watts (for electricity cost). |
| `--electricity-rate` | USD per kWh (default `0.30`). |
| `--hardware-cost` | Purchase price in USD (for straight-line amortization over lifetime). |
| `--hardware-lifetime-hours` | Useful life in hours (default `15000`). |

**How costs are computed** (see `runner.py` after the benchmark wall time `wall_s`):

- **Rental:** `gpu_hours = (wall_s / 3600) * gpu_count`, then
  `compute_cost_usd = gpu_hours * gpu_hour_cost` (stored in `metrics.json`).
- **Electricity (owned):** `kwh = (power_draw_w / 1000) * (wall_s / 3600)`,
  `electricity_cost_usd = kwh * electricity_rate`.
- **Amortization (owned):** `hourly_depreciation = hardware_cost / hardware_lifetime_hours`,
  `hardware_amortization_usd = hourly_depreciation * (wall_s / 3600)`.
- **Total compute:** `total_compute_cost_usd` is the sum of whichever of the
  above components apply; OpenAI API usage may also set `api_cost_usd` from
  per-request billing metadata.

## Output

Each run produces:
- `samples.jsonl` -- per-sample predictions, references, correctness, latency
- `metrics.json` -- aggregate accuracy, latency stats (mean/p50/p95), throughput, cost
- `manifest.json` -- git hash, timestamp, GPU info, config SHA256
- `run_config.json` -- backend and workload configuration

## Tests

```bash
# Python tests (accuracy checkers, workload loaders)
python -m pytest tests/ -v

# Java tests (JMLCLLMInferenceTest)
# 7 mock-based negative tests run without a server
# 3 live tests skip gracefully when no server is available
```

