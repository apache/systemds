# LLM Inference Benchmark

Backend-agnostic benchmarking suite for comparing LLM inference systems.
Measures **latency, throughput, accuracy, and resource usage** across
cloud APIs, optimized GPU servers, local runtimes, and SystemDS JMLC.

---

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `openai` | OpenAI API (GPT-4.1-mini, etc.) | `OPENAI_API_KEY` environment variable |
| `ollama` | Local inference via Ollama | [Ollama](https://ollama.ai) installed and running |
| `vllm` | High-performance GPU inference server | vLLM server running (requires NVIDIA GPU) |
| `systemds` | SystemDS JMLC with native `llmPredict` built-in | SystemDS JAR built, Py4J, vLLM/Ollama inference server |

---

## Workloads

| Workload | Dataset | Source | Samples | Task | Evaluation |
|----------|---------|--------|---------|------|------------|
| `math` | GSM8K | HuggingFace `openai/gsm8k` | 50 | Grade-school math | Exact numerical match |
| `reasoning` | BoolQ | HuggingFace `google/boolq` | 50 | Yes/no comprehension | Extracted answer match |
| `summarization` | XSum | HuggingFace `EdinburghNLP/xsum` | 50 | Article summary | ROUGE-1 F1 >= 0.2 |
| `json_extraction` | Built-in | 10 templates x 5 samples | 50 | Structured extraction | Valid JSON + >= 90% field match |
| `embeddings` | STS-B | HuggingFace `mteb/stsbenchmark-sts` | 50 | Semantic similarity | Within 1.0 of reference |

All workloads use temperature=0.0 for reproducibility. Each run processes 50 samples.

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
```

### 3. Run All Benchmarks

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

## SystemDS `llmPredict` Built-in

The `llmPredict` function is a native **parameterized built-in** (following the `tokenize` pattern) added in PR #2430. It goes through the full SystemDS compilation pipeline:

```
DML script --> Parser --> Hops --> Lops --> CP Instructions --> Execution
```

### Architecture

```
Python benchmark --> Py4J --> JMLC Connection.prepareScript()
  --> DML compilation (parse --> hops --> lops --> instructions)
  --> ParameterizedBuiltinCPInstruction (opcode: llmpredict)
    --> Java HTTP POST to OpenAI-compatible endpoint (vLLM, Ollama, etc.)
    --> Concurrent dispatch via ExecutorService (concurrency parameter)
  --> FrameBlock output [prompt, generated_text, latency_ms, input_tokens, output_tokens]
```

### DML usage

```dml
prompts = read("prompts", data_type="frame")
results = llmPredict(target=prompts, url=$url, max_tokens=$mt,
    temperature=$temp, top_p=$tp, concurrency=$conc)
write(results, "results")
```

### How it works

- Takes a Frame of prompts and named parameters (url, max_tokens, temperature, top_p, concurrency)
- Makes HTTP POST calls to any OpenAI-compatible endpoint using `java.net.HttpURLConnection`
- Parses JSON responses with `org.apache.wink.json4j` (existing SystemDS dependency)
- Supports concurrent requests via Java `ExecutorService` thread pool
- Returns a 5-column FrameBlock: `[prompt, generated_text, latency_ms, input_tokens, output_tokens]`
- No Python dependency in Java -- all inference is done via HTTP from the CP instruction

### JMLC integration

```java
Connection conn = new Connection();
HashMap<String, String> args = new HashMap<>();
args.put("$url", "http://localhost:8000/v1/completions");
args.put("$mt", "512"); args.put("$temp", "0.0");
args.put("$tp", "0.9"); args.put("$conc", "4");

PreparedScript ps = conn.prepareScript(dml, args,
    new String[]{"prompts"}, new String[]{"results"});
ps.setFrame("prompts", promptData);
ResultVariables rv = ps.executeScript();
FrameBlock results = rv.getFrameBlock("results");
```

### SystemDS compilation pipeline files

```
src/main/java/org/apache/sysds/
├── common/Builtins.java              # LLMPREDICT enum entry
├── common/Types.java                 # ParamBuiltinOp.LLMPREDICT
├── common/Opcodes.java               # llmpredict opcode
├── parser/ParameterizedBuiltinFunctionExpression.java  # Validation
├── parser/DMLTranslator.java         # Hop construction
├── hops/ParameterizedBuiltinOp.java  # CP-only exec type, lop construction
├── lops/ParameterizedBuiltin.java    # Instruction generation
└── runtime/instructions/cp/ParameterizedBuiltinCPInstruction.java  # HTTP execution
```

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
  --out results/systemds_qwen3b_math_c4
```

Environment variables (optional):
- `SYSTEMDS_JAR` - path to SystemDS.jar (default: auto-detected from project root)
- `SYSTEMDS_LIB` - path to lib/ directory (default: `target/lib/`)
- `LLM_INFERENCE_URL` - inference server endpoint (default: `http://localhost:8080/v1/completions`)

---

## Benchmark Results (n=50 per workload, NVIDIA H100 PCIe)

### Accuracy comparison

| Workload | Ollama (llama3.2 3B) | OpenAI (gpt-4.1-mini) | vLLM Qwen 3B | SystemDS c=1 | SystemDS c=4 | vLLM Mistral 7B |
|---|---|---|---|---|---|---|
| math | 58% (29/50) | 88% (44/50) | 68% (34/50) | 68% (34/50) | 68% (34/50) | 38% |
| json_extraction | 74% (37/50) | 84% (42/50) | 52% (26/50) | 52% (26/50) | 52% (26/50) | 50% |
| reasoning | 44% (22/50) | 70% (35/50) | 60% (30/50) | 60% (30/50) | 64% (32/50) | 68% |
| summarization | 80% (40/50) | 88% (44/50) | 50% (25/50) | 50% (25/50) | 62% (31/50) | 68% |
| embeddings | 40% (20/50) | 88% (44/50) | 90% (45/50) | 90% (45/50) | 90% (45/50) | 82% |

SystemDS c=1 matches vLLM Qwen 3B exactly on all workloads (same model, same vLLM inference server). c=4 shows minor variation on reasoning and summarization due to vLLM server-side batching behavior with concurrent requests.

### Per-prompt latency (mean ms/prompt)

| Workload | Ollama (CPU) | OpenAI (Cloud) | vLLM Qwen 3B | SystemDS c=1 |
|---|---|---|---|---|
| math | 5781 | 3630 | 4619 | 2273 |
| json_extraction | 1642 | 1457 | 1151 | 610 |
| reasoning | 5252 | 2641 | 2557 | 1261 |
| summarization | 1079 | 1036 | 791 | 373 |
| embeddings | 371 | 648 | 75 | 41 |

**Note on measurement methodology**: Each backend measures per-prompt latency differently. The vLLM backend uses Python `requests` with streaming (SSE parsing overhead), while SystemDS measures Java-side `HttpURLConnection` time (non-streaming). Both call the same vLLM server, but the numbers reflect client-side differences (Python streaming vs Java non-streaming HTTP) rather than inference speed differences. The accuracy comparison above is the apples-to-apples metric since all backends process the same prompts with the same model.

### SystemDS concurrency scaling

Throughput improvement with `ExecutorService` thread pool (concurrency=4 vs sequential):

| Workload | c=1 (req/s) | c=4 (req/s) | Speedup |
|---|---|---|---|
| math | 0.44 | 1.63 | 3.71x |
| json_extraction | 1.62 | 5.65 | 3.49x |
| reasoning | 0.79 | 3.11 | 3.95x |
| summarization | 2.62 | 7.27 | 2.78x |
| embeddings | 20.07 | 46.34 | 2.31x |

Throughput measured as n / total_wall_clock_seconds (Python-side end-to-end).

---

## Conclusions

1. **`llmPredict` produces correct results**: SystemDS c=1 matches vLLM Qwen 3B accuracy exactly on all 5 workloads (68%, 52%, 60%, 50%, 90%). The `llmPredict` instruction goes through the full DML compilation pipeline (parser, hops, lops, CP instruction) and produces the same outputs as calling the vLLM server directly.

2. **Concurrency improves throughput 2.3-3.9x**: The `ExecutorService` thread pool in the `llmPredict` instruction dispatches up to 4 requests concurrently to the inference server. Math and reasoning workloads (longer generation) benefit most (3.7-3.9x). Embeddings (short responses, already fast at 41ms/prompt) show the least benefit (2.3x) due to fixed overhead dominating.

3. **OpenAI leads on accuracy**: gpt-4.1-mini achieves the highest accuracy on 4/5 workloads (88% math, 84% json, 70% reasoning, 88% summarization). The local 3B models (Qwen, llama3.2) trade accuracy for cost and privacy.

4. **Model size matters more than backend**: Accuracy differences between backends come from the model, not the serving framework. Qwen 3B outperforms Mistral 7B on math (68% vs 38%) and embeddings (90% vs 82%), while Mistral 7B is stronger on reasoning (68% vs 60%) and summarization (68% vs 50%).

5. **Latency comparison requires careful interpretation**: Per-prompt latency numbers are not directly comparable across backends because each backend uses a different HTTP client and protocol (Python streaming vs Java non-streaming vs cloud API). The key takeaway is that `llmPredict` adds no significant overhead on top of the HTTP call to the inference server.

---

## Repository Structure

```
llm-bench/
├── backends/
│   ├── openai_backend.py      # OpenAI API adapter
│   ├── ollama_backend.py      # Ollama local inference
│   ├── vllm_backend.py        # vLLM server adapter (streaming)
│   ├── systemds_backend.py    # SystemDS JMLC with native llmPredict
│   └── mlx_backend.py         # Apple Silicon MLX (not benchmarked)
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
| Mean latency | Average response time per prompt |
| P50 latency | Median (50th percentile) |
| P95 latency | Tail latency (95th percentile) |
| Min/Max | Range of response times |

### Throughput
| Metric | Description |
|--------|-------------|
| Throughput (req/s) | n / total_wall_clock_seconds |

### Accuracy
| Metric | Description |
|--------|-------------|
| Accuracy mean | Proportion correct (e.g., 0.80 = 80%) |
| ROUGE-1/2/L | Summarization quality (F1 scores) |

### Cost (optional flags)

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
| Non-streaming vLLM baseline | Add non-streaming vLLM measurement for direct latency comparison with SystemDS |
| Multi-GPU tensor parallelism | Compare vLLM TP=2 vs TP=1 |
| Streaming support in llmPredict | Measure time-to-first-token for interactive use cases |

---

## Contact

- Student: Kubra Aksu
- Supervisor: Prof. Dr. Matthias Boehm
- Project: DIA Project - SystemDS LLM Benchmark
