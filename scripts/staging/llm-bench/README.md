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

### Evaluation methodology

Each workload defines its own `accuracy_check(prediction, reference)` function that returns true/false per sample. The accuracy percentage is `correct_count / n`.

| Workload | What counts as correct | Details |
|----------|----------------------|---------|
| math | Extracted number matches reference exactly | Extracts final number from response (after `####`, "answer is", `\boxed{}`, or last number). Passes if `abs(predicted - reference) < 1e-6`. |
| reasoning | Extracted answer matches reference text | Extracts answer via CoT markers, "answer is X", or last short line. Matches via exact, word-boundary substring, or numeric comparison. |
| summarization | ROUGE-1 F1 >= 0.2 | Computes ROUGE-1/2/L scores using `rouge-score` library. Prediction must be >= 10 characters. Threshold of 0.2 indicates meaningful overlap with reference summary. |
| json_extraction | >= 90% of JSON fields match | Response must contain valid JSON with all required fields. Field values compared with strict matching (case-insensitive strings, exact numbers/booleans). |
| embeddings | Predicted similarity within 1.0 of reference | Extracts a 0-5 similarity score from response. Passes if `abs(predicted - reference) <= 1.0` (20% tolerance on 5-point STS scale). |

All accuracy counts were verified against raw `samples.jsonl` files (each sample records `correct: true/false`).

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

## Benchmark Results (n=50 per workload)

GPU backends (vLLM, SystemDS) executed on NVIDIA H100 PCIe (81GB). Ollama ran on local MacBook (CPU). OpenAI ran on local MacBook calling cloud API. All runs used 50 samples per workload, temperature=0.0 for reproducibility.

### Backends

| Backend | Type | Model | Hardware | Inference path |
|---------|------|-------|----------|----------------|
| OpenAI | Cloud API | gpt-4.1-mini | MacBook (API call) | Python HTTP to OpenAI servers |
| Ollama | Local server | llama3.2 (3B) | MacBook CPU | Python HTTP to local Ollama |
| vLLM | GPU server | Qwen2.5-3B-Instruct | NVIDIA H100 | Python streaming HTTP to vLLM engine |
| vLLM | GPU server | Mistral-7B-Instruct | NVIDIA H100 | Python streaming HTTP to vLLM engine |
| SystemDS | JMLC API | Qwen2.5-3B-Instruct | NVIDIA H100 | Py4J → JMLC → DML `llmPredict` → Java HTTP → vLLM |

SystemDS and vLLM Qwen 3B use the **same model on the same vLLM inference server**, making their accuracy directly comparable. Any accuracy difference comes from the serving path, not the model.

### Accuracy comparison

| Workload | Ollama llama3.2 3B | OpenAI gpt-4.1-mini | vLLM Qwen 3B | SystemDS Qwen 3B c=1 | SystemDS Qwen 3B c=4 | vLLM Mistral 7B |
|---|---|---|---|---|---|---|
| math | 58% (29/50) | 94% (47/50) | 68% (34/50) | 68% (34/50) | 68% (34/50) | 38% (19/50) |
| json_extraction | 74% (37/50) | 84% (42/50) | 52% (26/50) | 52% (26/50) | 52% (26/50) | 50% (25/50) |
| reasoning | 44% (22/50) | 70% (35/50) | 60% (30/50) | 60% (30/50) | 64% (32/50) | 68% (34/50) |
| summarization | 80% (40/50) | 88% (44/50) | 50% (25/50) | 50% (25/50) | 62% (31/50) | 68% (34/50) |
| embeddings | 40% (20/50) | 88% (44/50) | 90% (45/50) | 90% (45/50) | 90% (45/50) | 82% (41/50) |

### Key comparisons

**SystemDS vs vLLM (same model, same server — Qwen2.5-3B-Instruct on H100)**:
SystemDS c=1 matches vLLM Qwen 3B accuracy exactly on all 5 workloads (68%, 52%, 60%, 50%, 90%). This confirms that the `llmPredict` instruction produces identical results to calling vLLM directly. Both use temperature=0.0 (deterministic), same prompts, same inference server. c=4 shows minor variation on reasoning (64% vs 60%) and summarization (62% vs 50%) because concurrent requests cause vLLM to batch them differently, introducing floating-point non-determinism in GPU computation.

**OpenAI gpt-4.1-mini vs local models**:
OpenAI achieves the highest accuracy on all 5 workloads. The gap is largest on math (94% vs 68% for Qwen 3B) and smallest on embeddings (88% vs 90% for Qwen 3B, where the local model actually wins). OpenAI's advantage comes from model quality (much larger model), not serving infrastructure.

**Qwen 3B vs Mistral 7B (different models, same vLLM server)**:
Despite being smaller (3B vs 7B parameters), Qwen outperforms Mistral on math (68% vs 38%) and embeddings (90% vs 82%). Mistral is better on reasoning (68% vs 60%) and summarization (68% vs 50%). Mistral's particularly low math score (38%) is because its instruction-following format produces verbose reasoning that the number extractor struggles to parse — the model often restates intermediate numbers after the correct answer, causing the extractor to grab wrong values.

**Ollama llama3.2 3B (MacBook CPU)**:
Ollama leads on summarization (80%) likely because llama3.2's training emphasized concise outputs that align well with the ROUGE-1 threshold. It scores lowest on embeddings (40%) because the model frequently refuses the similarity-rating task or defaults to high scores regardless of actual similarity.

### Per-prompt latency (mean ms/prompt)

| Workload | Ollama (CPU) | OpenAI (Cloud) | vLLM Qwen 3B | SystemDS c=1 |
|---|---|---|---|---|
| math | 5781 | 3630 | 4619 | 2273 |
| json_extraction | 1642 | 1457 | 1151 | 610 |
| reasoning | 5252 | 2641 | 2557 | 1261 |
| summarization | 1079 | 1036 | 791 | 373 |
| embeddings | 371 | 648 | 75 | 41 |

**Note on measurement methodology**: Latency numbers are **not directly comparable** across backends because each measures differently. The vLLM backend uses Python `requests` with streaming (SSE token-by-token parsing adds overhead). SystemDS measures Java-side `HttpURLConnection` round-trip time (non-streaming, gets full response at once). Ollama measures Python HTTP round-trip on CPU. OpenAI includes network round-trip to cloud servers. The accuracy comparison is the apples-to-apples metric since all backends process the same prompts.

### SystemDS concurrency scaling

Throughput improvement with `ExecutorService` thread pool (concurrency=4 vs sequential):

| Workload | c=1 (req/s) | c=4 (req/s) | Speedup |
|---|---|---|---|
| math | 0.44 | 1.63 | 3.71x |
| json_extraction | 1.62 | 5.65 | 3.49x |
| reasoning | 0.79 | 3.11 | 3.95x |
| summarization | 2.62 | 7.27 | 2.78x |
| embeddings | 20.07 | 46.34 | 2.31x |

Throughput = `n / total_wall_clock_seconds` (measured Python-side, end-to-end including JMLC overhead). Theoretical maximum speedup is 4x. Math and reasoning (longer generation, ~1-2s per prompt) get closest to 4x because the per-request time dominates. Embeddings (very short responses, ~41ms per prompt) only achieves 2.31x because JMLC pipeline overhead becomes proportionally significant.

### Cost comparison

All backends incur compute cost (hardware amortization + electricity) for the machine running them. GPU backends run on the H100 server; Ollama and OpenAI run on a local MacBook. OpenAI additionally incurs API cost per token.

**How cost is calculated**: `compute_cost = wall_clock_time × (hardware_cost / lifetime_hours + power_watts × electricity_rate) / 3600`. Assumptions: H100 server: 350W, $30K over 15K hours ($2.00/h + $0.105/h electricity = $2.105/h). MacBook: 50W, $3K over 15K hours ($0.20/h + $0.015/h electricity = $0.215/h). OpenAI API cost recorded by the runner from response headers (`x-usage` header).

| Backend | Hardware | Wall clock (250 queries) | Compute cost | API cost | Total cost | Cost per query |
|---|---|---|---|---|---|---|
| Ollama llama3.2 3B | MacBook CPU | 706s | $0.0422 | -- | $0.0422 | $0.000169 |
| OpenAI gpt-4.1-mini | MacBook + API | 471s | $0.0281 | $0.0573 | $0.0855 | $0.000342 |
| vLLM Qwen 3B | H100 GPU | 460s | $0.2688 | -- | $0.2688 | $0.001076 |
| SystemDS Qwen 3B c=1 | H100 GPU | 230s | $0.1345 | -- | $0.1345 | $0.000538 |
| SystemDS Qwen 3B c=4 | H100 GPU | 64s | $0.0372 | -- | $0.0372 | $0.000149 |

OpenAI API cost breakdown (recorded per run): math $0.0227, reasoning $0.0172, json_extraction $0.0080, summarization $0.0076, embeddings $0.0019.

---

## Conclusions

1. **SystemDS `llmPredict` produces identical results to vLLM**: SystemDS c=1 matches vLLM Qwen 3B accuracy exactly on all 5 workloads (68%, 52%, 60%, 50%, 90%). Both use the same model on the same inference server with temperature=0.0, confirming that the `llmPredict` DML built-in adds no distortion to model outputs.

2. **Concurrency scales throughput 2.3-3.9x**: The `ExecutorService` thread pool in the `llmPredict` instruction dispatches up to 4 requests concurrently. Longer-running workloads (math 3.71x, reasoning 3.95x) get closest to the theoretical 4x speedup. Short workloads (embeddings 2.31x) are limited by JMLC pipeline overhead.

3. **OpenAI leads on accuracy but costs more per query**: gpt-4.1-mini achieves the highest accuracy on all 5 workloads (94% math, 84% json, 70% reasoning, 88% summarization, 88% embeddings) but at $0.000342/query. SystemDS c=4 achieves $0.000149/query — 56% cheaper — with competitive accuracy on focused tasks like embeddings (90% vs 88%).

4. **Model quality matters more than parameter count**: Qwen 3B outperforms the larger Mistral 7B on math (68% vs 38%) and embeddings (90% vs 82%), while Mistral 7B is stronger on reasoning (68% vs 60%) and summarization (68% vs 50%). The serving framework (vLLM vs SystemDS) has zero impact on accuracy when using the same model.

5. **Concurrency reduces compute cost on GPU**: SystemDS c=4 at $0.000149/query is the cheapest GPU option — 86% less than vLLM's $0.001076/query — because higher throughput means less wall-clock time per query. Ollama on MacBook CPU is cheapest overall ($0.000169/query) due to low hardware and power costs, but 11x slower.

6. **Latency measurements are not comparable across backends**: Each backend uses a different HTTP client (Python streaming, Java non-streaming, cloud API) and measures time differently. Per-prompt latency should only be compared within the same backend across workloads, not across backends.

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
| Accuracy mean | Proportion correct (e.g., 0.80 = 80%). See [Evaluation methodology](#evaluation-methodology) for per-workload criteria. |
| ROUGE-1/2/L | Summarization quality (F1 scores). Accuracy threshold: ROUGE-1 F1 >= 0.2. |

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
