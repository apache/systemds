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

Machine-readable run outputs (metrics, per-prompt predictions in `samples.jsonl`,
and similar files under `results/`) are **not** committed. After you run
`runner.py` (or `scripts/run_all_benchmarks.sh`), those artifacts are written
locally (gitignored). The **tables in [Reference benchmark results](#reference-benchmark-results-documentation)** below are fixed documentation summarizing one completed evaluation so reviewers can read the findings without running the suite. Aggregate local runs and build an HTML summary with:

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

## Reference benchmark results (documentation)

Summary from a full run: GPU backends (vLLM, SystemDS) on NVIDIA H100 PCIe
(81 GB), OpenAI on cloud API, `n=50` per workload, `temperature=0.0`. (Raw
`metrics.json` / `samples.jsonl` for this run are not in the repo; see above.)

### Accuracy (% correct, n=50 per workload)

| Workload | OpenAI gpt-4.1-mini | vLLM Qwen 3B | SystemDS Qwen 3B |
|----------|---------------------|--------------|------------------|
| math | **96%** (48/50) | 68% (34/50) | 68% (34/50) |
| reasoning | **88%** (44/50) | 58% (29/50) | 58% (29/50) |
| summarization | **86%** (43/50) | 50% (25/50) | 62% (31/50) |
| json_extraction | **61%** (28/46) | **66%** (33/50) | **66%** (33/50) |
| embeddings | 88% (44/50) | **90%** (45/50) | **90%** (45/50) |

SystemDS matches vLLM on 4/5 workloads. The summarization gap (25 vs 31) is
caused by vLLM Automatic Prefix Caching (APC), not the SystemDS pipeline. A
reverse-order experiment confirmed this: the first-run backend scores 25/50
and the second-run backend scores 31/50, regardless of which backend runs
first.

### Text identity (vLLM vs SystemDS)

| Workload | Identical | % Identical |
|----------|-----------|-------------|
| math | 50/50 | **100%** |
| json_extraction | 50/50 | **100%** |
| embeddings | 50/50 | **100%** |
| reasoning | 33/50 | 66% |
| summarization | 28/50 | 56% |

On three workloads, predictions are byte-for-byte identical, consistent with
the JMLC path being a lossless pass-through for those tasks. Divergence on
reasoning and summarization aligns with APC cache state (see reverse-order
experiments above).

### Per-prompt latency (mean ms, n=50)

| Workload | OpenAI (Cloud) | vLLM Qwen 3B (H100) | SystemDS Qwen 3B (H100) |
|----------|----------------|----------------------|--------------------------|
| math | 4577 | 1913 | 1917 (+0.2%) |
| reasoning | 1735 | 1109 | 1134 (+2.2%) |
| summarization | 1131 | 364 | 362 (-0.6%) |
| json_extraction | 1498 | 266 | 266 (+0.0%) |
| embeddings | 773 | 47 | 60 (+29.1%) |

SystemDS adds under 3% overhead for generation workloads. The embeddings
overhead is larger in relative terms because the HTTP segment is short, so
fixed JMLC cost dominates.

**SystemDS JMLC pipeline breakdown (ms):**

| Workload | compile | marshal | exec/prompt | unmarshal | overhead |
|----------|---------|---------|-------------|-----------|----------|
| math | 316 | 113 | 1909 | 0.8 | 483 |
| reasoning | 241 | 43 | 1128 | 0.8 | 337 |
| summarization | 305 | 52 | 355 | 0.8 | 412 |
| json_extraction | 299 | 48 | 259 | 0.9 | 403 |
| embeddings | 338 | 166 | 50 | 1.4 | 563 |

### Throughput (requests/second)

| Workload | OpenAI | vLLM Qwen 3B | SystemDS Qwen 3B |
|----------|--------|--------------|------------------|
| math | 0.22 | 0.52 | 0.52 |
| reasoning | 0.58 | 0.90 | 0.88 |
| summarization | 0.88 | 2.74 | 2.76 |
| json_extraction | 0.67 | 3.76 | 3.75 |
| embeddings | 1.29 | 21.30 | 15.88 |

### Cost

| Workload | OpenAI API Cost | vLLM Compute Cost | SystemDS Compute Cost |
|----------|----------------|-------------------|----------------------|
| math | $0.0223 | $0.0560 | $0.0561 |
| reasoning | $0.0100 | $0.0324 | $0.0332 |
| summarization | $0.0075 | $0.0107 | $0.0106 |
| json_extraction | $0.0056 | $0.0078 | $0.0078 |
| embeddings | $0.0019 | $0.0014 | $0.0018 |
| **Total** | **$0.047** | **$0.108** | **$0.109** |

At low sequential utilization, OpenAI can be cheaper than attributing full
GPU CapEx; with continuous batching on the GPU, per-query economics shift toward
local inference. Interpret costs using the flags documented above when you
reproduce runs.

## Conclusions

1. **SystemDS `llmPredict` matches vLLM on constrained workloads** (math,
   json_extraction, embeddings): predictions are identical for the evaluated
   samples. Remaining divergence on open-ended tasks is consistent with vLLM
   APC, not the SystemDS bridge alone.

2. **JMLC overhead is small** for generation-heavy workloads (on the order of
   a few percent in the table above).

3. **Cost tradeoffs depend on utilization and pricing model** (API vs owned
   hardware vs rental); use `runner.py` cost flags for your environment.

4. **Gap between OpenAI and Qwen 3B reflects model capability**; vLLM vs
   SystemDS on the same model shows parity in the tables above.

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

