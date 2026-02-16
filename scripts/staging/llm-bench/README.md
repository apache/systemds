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
| `systemds` | SystemDS JMLC with FrameBlock batch API | SystemDS JAR built, Py4J, HuggingFace transformers |
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

# SystemDS JMLC
python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --out results/systemds_qwen3b_math

# With compute cost estimation
python runner.py \
  --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --power-draw-w 350 --electricity-rate 0.30 \
  --hardware-cost 30000 \
  --out results/vllm_qwen3b_math
```

### 3. Generate Report

```bash
python scripts/aggregate.py --results-dir results/ --out results/summary.csv
python scripts/report.py --results-dir results/ --out results/benchmark_report.html
open results/benchmark_report.html
```

---

## Benchmark Results (n=50 per workload, NVIDIA H100)

### Same-model comparison: Qwen 3B on vLLM vs SystemDS JMLC

| Workload | vLLM Acc | vLLM Lat | SystemDS Acc | SystemDS Lat | Slowdown |
|---|---|---|---|---|---|
| embeddings | 90% | 75ms | 88% | 195ms | 2.6x |
| json_extraction | 52% | 1151ms | 52% | 3325ms | 2.9x |
| math | 68% | 4619ms | 72% | 21479ms | 4.7x |
| reasoning | 60% | 2557ms | 66% | 7425ms | 2.9x |
| summarization | 50% | 791ms | 62% | 2175ms | 2.7x |

### Same-model comparison: Mistral 7B on vLLM vs SystemDS JMLC

| Workload | vLLM Acc | vLLM Lat | SystemDS Acc | SystemDS Lat | Slowdown |
|---|---|---|---|---|---|
| embeddings | 82% | 129ms | 82% | 412ms | 3.2x |
| json_extraction | 50% | 1817ms | 52% | 5503ms | 3.0x |
| math | 38% | 5053ms | 38% | 14180ms | 2.8x |
| reasoning | 68% | 1570ms | 74% | 4566ms | 2.9x |
| summarization | 68% | 782ms | 70% | 2253ms | 2.9x |

### All backends overview

| Backend | Model | Hardware | Workloads | Story |
|---------|-------|----------|-----------|-------|
| OpenAI | gpt-4.1-mini | Cloud | 5 | Best accuracy, highest cost |
| Ollama | llama3.2 3B | MacBook CPU | 5 | Accessible local inference |
| vLLM | Qwen 3B | H100 | 5 | Optimized GPU serving |
| vLLM | Mistral 7B | H100 | 5 | Larger model on GPU |
| SystemDS | Qwen 3B | H100 | 5 | JMLC integration (same model as vLLM) |
| SystemDS | Mistral 7B | H100 | 5 | JMLC integration (same model as vLLM) |

**Key findings:**
- **Accuracy is identical** across vLLM and SystemDS for the same model (small differences are generation randomness)
- **Latency is ~3x slower** for SystemDS JMLC vs vLLM, due to the Py4J bridge and lack of KV caching / CUDA graph optimizations in raw HuggingFace
- **OpenAI** achieves highest accuracy but incurs API costs
- **vLLM** is the fastest local option (optimized serving with KV cache, PagedAttention, continuous batching)
- **SystemDS JMLC** provides Java ecosystem integration at the cost of inference speed

---

## SystemDS JMLC Backend

The SystemDS backend routes inference through the full Java JMLC path:

```
Python benchmark -> Py4J -> Java JMLC Connection
-> PreparedScript.generateBatchWithMetrics()
-> LLMCallback -> Python llm_worker.py (HuggingFace)
-> FrameBlock [prompt, generated_text, time_ms, input_tokens, output_tokens]
```

All prompts are submitted as a Java `String[]` and processed via
`PreparedScript.generateBatchWithMetrics()`, which returns a typed
`FrameBlock` with per-prompt timing and token metrics.

### Setup

```bash
# Build SystemDS
cd /path/to/systemds
mvn package -DskipTests

# Install Python dependencies
pip install py4j torch transformers accelerate

# Run benchmark
python runner.py \
  --backend systemds \
  --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --power-draw-w 350 --electricity-rate 0.30 --hardware-cost 30000 \
  --out results/systemds_qwen3b_math
```

Environment variables (optional):
- `SYSTEMDS_JAR` - path to SystemDS.jar (default: `../../target/SystemDS.jar`)
- `SYSTEMDS_LIB` - path to lib/ directory (default: `../../target/lib/`)
- `LLM_WORKER_SCRIPT` - path to llm_worker.py (default: `../../src/main/python/llm_worker.py`)

---

## Repository Structure

```
llm-bench/
├── backends/
│   ├── openai_backend.py      # OpenAI API adapter
│   ├── ollama_backend.py      # Ollama local inference
│   ├── vllm_backend.py        # vLLM server adapter
│   ├── systemds_backend.py    # SystemDS JMLC with FrameBlock batch API
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
| KV cache support | Reduce latency gap between SystemDS and vLLM |
| Direct CUDA integration | Bypass the Py4J roundtrip for model inference |
| Continuous batching | Process multiple requests concurrently in SystemDS |
| Larger sample sizes | Run with n=100+ for stronger statistical significance |
| Code generation workload | Add HumanEval / MBPP programming tasks |
| Model quantization comparison | Compare 4-bit vs 8-bit vs full precision |

---

## Contact

- Student: Kubra Aksu
- Supervisor: Prof. Dr. Matthias Boehm
- Project: DIA Project - SystemDS LLM Benchmark
