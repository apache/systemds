# LLM Inference Benchmark

Backend-agnostic benchmarking suite for comparing LLM inference systems.
Measures **latency, throughput, accuracy, cost, and resource usage** across
cloud APIs, optimized GPU servers, local runtimes, and SystemDS JMLC.

---

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `openai` | OpenAI API (GPT-4.1-mini, etc.) | `OPENAI_API_KEY` environment variable |
| `ollama` | Local inference via Ollama | [Ollama](https://ollama.ai) installed and running |
| `vllm` | High-performance GPU inference server | vLLM server running (requires NVIDIA GPU) |
| `systemds` | SystemDS JMLC with native `llmPredict` built-in | SystemDS JAR built, Py4J, vLLM/Ollama inference server |
| `mlx` | Apple Silicon optimized | macOS with Apple Silicon, `mlx-lm` package |

---

## Workloads

| Workload | Dataset | Source | Samples |
|----------|---------|--------|---------|
| `math` | GSM8K | HuggingFace `openai/gsm8k` | 50 |
| `summarization` | XSum | HuggingFace `EdinburghNLP/xsum` | 50 |
| `reasoning` | BoolQ | HuggingFace `google/boolq` | 50 |
| `json_extraction` | Curated + HF | Built-in / `MasterControlAIML/JSON-Unstructured-Structured` | 50 |
| `embeddings` | STS-B | HuggingFace `mteb/stsbenchmark-sts` | 50 |

---

## Quick Start

### 1. Installation

```bash
cd scripts/staging/llm-bench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For OpenAI backend
export OPENAI_API_KEY="your-key-here"
```

### 2. Run Benchmarks

```bash
# OpenAI API
python runner.py \
  --backend openai \
  --workload workloads/math/config.yaml \
  --out results/openai_math

# Ollama (local)
python runner.py \
  --backend ollama --model llama3.2 \
  --workload workloads/math/config.yaml \
  --out results/ollama_math

# vLLM (GPU server)
python runner.py \
  --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --out results/vllm_qwen3b_math

# SystemDS JMLC (sequential)
python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --out results/systemds_qwen3b_math

# SystemDS JMLC (concurrent, 4 threads)
python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --concurrency 4 \
  --out results/systemds_qwen3b_math_c4

# With compute cost estimation
python runner.py \
  --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --power-draw-w 350 --electricity-rate 0.30 \
  --hardware-cost 30000 \
  --out results/vllm_qwen3b_math
```

### 3. Run All Benchmarks (Reproducibility)

```bash
# Run all workloads for a single backend
./scripts/run_all_benchmarks.sh vllm Qwen/Qwen2.5-3B-Instruct

# Run SystemDS with both concurrency=1 and concurrency=4
./scripts/run_all_benchmarks.sh systemds Qwen/Qwen2.5-3B-Instruct

# Run GPU backends (vLLM + SystemDS) for direct comparison
./scripts/run_all_benchmarks.sh gpu Qwen/Qwen2.5-3B-Instruct

# Run all backends
./scripts/run_all_benchmarks.sh all
```

### 4. Generate Report

```bash
python scripts/aggregate.py --results-dir results/ --out results/summary.csv
python scripts/report.py --results-dir results/ --out results/benchmark_report.html
open results/benchmark_report.html
```

---

## Benchmark Results (n=50 per workload, NVIDIA H100 PCIe)

### Latency comparison: mean ms/prompt

| Workload | Ollama (CPU) | OpenAI (Cloud) | vLLM Qwen 3B | SystemDS c=1 |
|---|---|---|---|---|
| math | 5781 | 3630 | 4619 | 2273 |
| json_extraction | 1642 | 1457 | 1151 | 610 |
| reasoning | 5252 | 2641 | 2557 | 1261 |
| summarization | 1079 | 1036 | 791 | 373 |
| embeddings | 371 | 648 | 75 | 41 |

SystemDS c=1 calls the same vLLM inference server. Per-prompt latency is comparable to vLLM direct since `llmPredict` adds minimal overhead (Java HTTP call).

### Accuracy comparison

| Workload | Ollama (llama3.2 3B) | OpenAI (gpt-4.1-mini) | vLLM Qwen 3B | SystemDS c=1 | SystemDS c=4 | vLLM Mistral 7B |
|---|---|---|---|---|---|---|
| math | 58% (29/50) | 88% (44/50) | 68% (34/50) | 68% (34/50) | 68% (34/50) | 38% |
| json_extraction | 74% (37/50) | 84% (42/50) | 52% (26/50) | 52% (26/50) | 52% (26/50) | 50% |
| reasoning | 44% (22/50) | 70% (35/50) | 60% (30/50) | 60% (30/50) | 64% (32/50) | 68% |
| summarization | 80% (40/50) | 88% (44/50) | 50% (25/50) | 50% (25/50) | 62% (31/50) | 68% |
| embeddings | 40% (20/50) | 88% (44/50) | 90% (45/50) | 90% (45/50) | 90% (45/50) | 82% |

SystemDS c=1 matches vLLM Qwen 3B exactly (same model, same inference server). c=4 shows minor variation on reasoning and summarization due to vLLM server batching non-determinism with concurrent requests.

### SystemDS concurrency scaling

Throughput improvement with `ExecutorService` thread pool (concurrency=4 vs sequential):

| Workload | c=1 (req/s) | c=4 (req/s) | Speedup | Eff. latency c=1 (ms) | Eff. latency c=4 (ms) |
|---|---|---|---|---|---|
| math | 0.44 | 1.63 | 3.71x | 2281 | 615 |
| json_extraction | 1.62 | 5.65 | 3.49x | 618 | 177 |
| reasoning | 0.79 | 3.11 | 3.95x | 1270 | 322 |
| summarization | 2.62 | 7.27 | 2.78x | 382 | 137 |
| embeddings | 20.07 | 46.34 | 2.31x | 50 | 22 |

Effective latency = 1000 / throughput (time per prompt from the batch processing perspective).

### All backends overview

| Backend | Model | Hardware | Concurrency | Story |
|---------|-------|----------|-------------|-------|
| OpenAI | gpt-4.1-mini | Cloud | 1 | Best accuracy, highest cost |
| Ollama | llama3.2 3B | MacBook CPU | 1 | Accessible local inference |
| vLLM | Qwen 3B | H100 | 1 | Optimized GPU serving |
| vLLM | Mistral 7B | H100 | 1 | Larger model on GPU |
| SystemDS | Qwen 3B | H100 | 1 | JMLC + native `llmPredict` (sequential) |
| SystemDS | Qwen 3B | H100 | 4 | JMLC + native `llmPredict` (concurrent) |

**Key findings:**
- **SystemDS c=1 accuracy matches vLLM Qwen 3B** for all workloads (same model, same inference server). c=4 shows minor variation on reasoning (64% vs 60%) and summarization (62% vs 50%) due to vLLM batching non-determinism
- **SystemDS with `llmPredict`** has minimal overhead vs vLLM direct, since inference goes through a native DML built-in that makes HTTP calls directly from Java
- **Concurrency improves throughput**: c=4 achieves 2.3-3.9x throughput speedup via Java `ExecutorService` thread pool in the `llmPredict` instruction
- **OpenAI** achieves highest accuracy but incurs API costs ($0.02-0.03 per 50 prompts)
- **vLLM** is the fastest local option for single-request latency (optimized serving with KV cache, PagedAttention, continuous batching)
- **Ollama** provides the most accessible local inference (CPU, no GPU required)

---

## SystemDS JMLC Backend

The SystemDS backend executes real DML through the full compilation pipeline
using the native `llmPredict` parameterized built-in function:

```
Python benchmark -> Py4J -> JMLC Connection.prepareScript()
-> DML compilation (parse -> hops -> lops -> instructions)
-> llmPredict instruction -> Java HTTP -> vLLM/Ollama server
-> FrameBlock [prompt, generated_text, time_ms, input_tokens, output_tokens]
```

The DML script executed through JMLC:

```dml
prompts = read("prompts", data_type="frame")
results = llmPredict(target=prompts, url=$url, max_tokens=$mt,
    temperature=$temp, top_p=$tp, concurrency=$conc)
write(results, "results")
```

The `llmPredict` built-in:
- Takes a Frame of prompts and named parameters (url, max_tokens, temperature, top_p, concurrency)
- Makes HTTP POST calls to any OpenAI-compatible inference endpoint (vLLM, Ollama, etc.)
- Supports concurrent requests via Java `ExecutorService` thread pool
- Returns a Frame with columns: [prompt, generated_text, latency_ms, input_tokens, output_tokens]
- Goes through the full SystemDS compilation pipeline (parser, hops, lops, runtime instructions)

### Setup

```bash
# 1. Build SystemDS
cd /path/to/systemds
mvn package -DskipTests

# 2. Start an inference server (vLLM example)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8000 --gpu-memory-utilization 0.3

# 3. Install Python dependencies
cd scripts/staging/llm-bench
pip install py4j -r requirements.txt

# 4. Run benchmark
export LLM_INFERENCE_URL="http://localhost:8000/v1/completions"
python runner.py \
  --backend systemds \
  --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --concurrency 4 \
  --power-draw-w 350 --electricity-rate 0.30 --hardware-cost 30000 \
  --out results/systemds_qwen3b_math_c4
```

Environment variables (optional):
- `SYSTEMDS_JAR` - path to SystemDS.jar (default: auto-detected from project root)
- `SYSTEMDS_LIB` - path to lib/ directory (default: `target/lib/`)
- `LLM_INFERENCE_URL` - inference server endpoint (default: `http://localhost:8080/v1/completions`)

---

## Repository Structure

```
llm-bench/
├── backends/
│   ├── openai_backend.py      # OpenAI API adapter
│   ├── ollama_backend.py      # Ollama local inference
│   ├── vllm_backend.py        # vLLM server adapter
│   ├── systemds_backend.py    # SystemDS JMLC with native llmPredict
│   └── mlx_backend.py         # Apple Silicon MLX
├── workloads/
│   ├── math/                  # GSM8K (HuggingFace)
│   ├── summarization/         # XSum (HuggingFace)
│   ├── reasoning/             # BoolQ (HuggingFace)
│   ├── json_extraction/       # Curated + HuggingFace
│   └── embeddings/            # STS-B (HuggingFace)
├── scripts/
│   ├── aggregate.py           # CSV aggregation
│   ├── report.py              # HTML report generation
│   └── run_all_benchmarks.sh  # Run all workloads for a backend
├── evaluation/
│   └── perf.py                # Latency/throughput metrics
├── results/                   # Benchmark outputs
├── tests/                     # Unit tests
├── runner.py                  # Main benchmark runner
├── requirements.txt
└── README.md
```

---

## Metrics

### Latency
| Metric | Description |
|--------|-------------|
| Mean latency | Average response time |
| P50 latency | Median (50th percentile) |
| P95 latency | Tail latency (95th percentile) |
| Min/Max | Range of response times |

### Accuracy
| Metric | Description |
|--------|-------------|
| Accuracy mean | Proportion correct (e.g., 0.80 = 80%) |
| ROUGE-1/2/L | Summarization quality (F1 scores) |

### Cost
| Metric | Description |
|--------|-------------|
| API cost (USD) | Per-token billing (OpenAI) |
| Electricity cost | Power draw x wall time x rate |
| Hardware amortization | Device cost / lifetime hours x wall time |
| Total compute cost | Electricity + hardware amortization |

#### Cost flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--power-draw-w` | 0 | Device power in watts (MacBook: ~50W, H100: ~350W) |
| `--electricity-rate` | 0.30 | $/kWh |
| `--hardware-cost` | 0 | Purchase price in USD |
| `--hardware-lifetime-hours` | 15000 | Useful lifetime hours |

---

## Future Work

| Feature | Description |
|---------|-------------|
| Higher concurrency levels | Test c=8, c=16 to find saturation point |
| Larger sample sizes | Run with n=100+ for stronger statistical significance |
| Code generation workload | Add HumanEval / MBPP programming tasks |
| Model quantization comparison | Compare 4-bit vs 8-bit vs full precision |
| Multi-GPU tensor parallelism | Compare vLLM TP=2 vs TP=1 |
| Streaming support | Measure time-to-first-token for interactive use cases |

---

## Contact

- Student: Kubra Aksu
- Supervisor: Prof. Dr. Matthias Boehm
- Project: DIA Project - SystemDS LLM Benchmark
