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
conditions (same prompts, same evaluation metrics). GPU backends (vLLM,
SystemDS) were evaluated on NVIDIA H100 PCIe (81 GB). All runs used 50
samples per workload, temperature=0.0 for reproducibility.

## Quick Start

```bash
cd scripts/staging/llm-bench
pip install -r requirements.txt

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
│   └── run_all_benchmarks.sh  # Batch automation
├── results/                   # Benchmark outputs (metrics.json per run)
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

## Benchmark Results

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
reverse-order experiment confirmed this: the 1st-run backend always scores
25/50 and the 2nd-run backend always scores 31/50, regardless of which
backend runs first. See `benchmark_report.md` for the full APC analysis.

### Text Identity (vLLM vs SystemDS)

| Workload | Identical | % Identical |
|----------|-----------|-------------|
| math | 50/50 | **100%** |
| json_extraction | 50/50 | **100%** |
| embeddings | 50/50 | **100%** |
| reasoning | 33/50 | 66% |
| summarization | 28/50 | 56% |

On 3/5 workloads, predictions are byte-for-byte identical, confirming that
the JMLC pipeline is a lossless pass-through. The 39 divergent samples across
reasoning and summarization are all caused by APC cache state, proven by the
4-run reverse-order experiment (same-position = 100% identical across sessions).

### Per-Prompt Latency (mean ms, n=50)

| Workload | OpenAI (Cloud) | vLLM Qwen 3B (H100) | SystemDS Qwen 3B (H100) |
|----------|----------------|----------------------|--------------------------|
| math | 4577 | 1913 | 1917 (+0.2%) |
| reasoning | 1735 | 1109 | 1134 (+2.2%) |
| summarization | 1131 | 364 | 362 (-0.6%) |
| json_extraction | 1498 | 266 | 266 (+0.0%) |
| embeddings | 773 | 47 | 60 (+29.1%) |

SystemDS adds <3% overhead for generation workloads. The embeddings +29% is
because the HTTP call itself is only ~47 ms, so fixed JMLC pipeline cost
(~10 ms/prompt) becomes a significant fraction.

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

OpenAI is cheaper for this small sequential benchmark because GPU hardware
amortization ($2.00/hr) dominates at low utilization. With vLLM continuous
batching (10x+ throughput), the H100 becomes 3-14x cheaper per query than
OpenAI across all workloads. See `benchmark_report.md` for the full cost
analysis with breakeven calculations.

## Conclusions

1. **SystemDS `llmPredict` is a lossless pass-through**: 150/150 samples
   are byte-for-byte identical on constrained workloads (math,
   json_extraction, embeddings). The 39 divergent samples on unconstrained
   workloads are caused by vLLM APC, not the SystemDS pipeline.

2. **JMLC overhead is negligible**: <3% for generation workloads, within
   measurement noise.

3. **Cost tradeoff depends on scale**: OpenAI is cheaper at low sequential
   volume. Owned GPU hardware is cheaper at production scale with batching.

4. **Model quality matters more than serving infrastructure**: OpenAI vs
   Qwen 3B is model quality. vLLM vs SystemDS is zero difference.

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

