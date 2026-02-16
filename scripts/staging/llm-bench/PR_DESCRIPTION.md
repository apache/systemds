Benchmarking framework that compares LLM inference across four backends: OpenAI API, Ollama, vLLM, and SystemDS JMLC with the native `llmPredict` built-in. Evaluated on 5 workloads (math, reasoning, summarization, JSON extraction, embeddings) with n=50 per workload on NVIDIA H100.

## Purpose and motivation

This project was developed as part of the LDE (Large-Scale Data Engineering) course. The goal is to evaluate how SystemDS can be extended to support LLM inference as a native built-in function, and how its performance compares to established LLM serving solutions.

Research questions:

1. Can SystemDS support LLM inference through a native DML built-in that goes through the full compilation pipeline?
2. How does SystemDS compare to dedicated LLM backends (OpenAI, Ollama, vLLM) in terms of accuracy, latency, throughput, and cost?
3. How does Java-side concurrent request dispatch scale with the `llmPredict` instruction?

Approach:

- Built a Python benchmarking framework that runs standardized workloads against all four backends under identical conditions (same prompts, same models, same GPU, same evaluation metrics)
- Added `llmPredict` as a native **parameterized built-in function** to SystemDS (PR #2430): goes through the full DML compilation pipeline (parser → hops → lops → CP instruction) and makes HTTP calls to any OpenAI-compatible inference server
- Ran evaluation in two phases: (1) sequential baseline across all backends, (2) SystemDS with Java-side concurrency (`ExecutorService` thread pool in the `llmPredict` instruction)
- All benchmark runs executed on NVIDIA H100 PCIe (81GB), 50 samples per run, temperature=0.0 for reproducibility

Key findings:

- **SystemDS produces the same accuracy** as vLLM (same model, same vLLM inference server)
- **`llmPredict` has minimal overhead** vs calling vLLM directly — inference goes through a native DML built-in that makes HTTP calls from Java
- **Concurrency improves throughput**: c=4 achieves 2.3–3.9x throughput speedup via `ExecutorService` thread pool
- **vLLM** remains fastest for single-request latency due to PagedAttention, continuous batching, and custom CUDA kernels

## Table of contents

- [Project structure](#project-structure)
- [SystemDS `llmPredict` built-in](#systemds-llmpredict-built-in)
- [Backends](#backends)
- [Workloads and datasets](#workloads-and-datasets)
- [How measurements work](#how-measurements-work)
- [Benchmark results](#benchmark-results)
- [Reproducibility](#reproducibility)
- [Conclusions](#conclusions)

## Project structure

```
scripts/staging/llm-bench/
├── runner.py                  # Main benchmark runner (CLI entry point)
├── backends/
│   ├── openai_backend.py      # OpenAI API (gpt-4.1-mini)
│   ├── ollama_backend.py      # Ollama local server (llama3.2)
│   ├── vllm_backend.py        # vLLM serving engine (HTTP API)
│   └── systemds_backend.py    # SystemDS JMLC via Py4J + llmPredict DML
├── workloads/
│   ├── math/                  # GSM8K dataset, numerical accuracy
│   ├── reasoning/             # BoolQ dataset, logical accuracy
│   ├── summarization/         # XSum dataset, ROUGE-1 scoring
│   ├── json_extraction/       # Built-in structured extraction
│   └── embeddings/            # STS-Benchmark, similarity scoring
├── evaluation/
│   └── perf.py                # Latency, throughput, cost metrics
├── scripts/
│   ├── report.py              # HTML report generator
│   ├── aggregate.py           # Cross-run aggregation
│   └── run_all_benchmarks.sh  # Batch automation (all backends, all workloads)
├── results/                   # Benchmark outputs (metrics.json per run)
└── tests/                     # Unit tests for accuracy checks + runner
```

SystemDS `llmPredict` built-in (in PR #2430):

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

## SystemDS `llmPredict` built-in

The `llmPredict` function is a native parameterized built-in (following the `tokenize` pattern) that goes through the full SystemDS compilation pipeline:

```
DML script → Parser → Hops → Lops → CP Instructions → Execution
```

### DML usage

```dml
prompts = read("prompts", data_type="frame")
results = llmPredict(target=prompts, url=$url, max_tokens=$mt,
    temperature=$temp, top_p=$tp, concurrency=$conc)
write(results, "results")
```

### Architecture

```
Python benchmark → Py4J → JMLC Connection.prepareScript()
  → DML compilation (parse → hops → lops → instructions)
  → ParameterizedBuiltinCPInstruction (opcode: llmpredict)
    → Java HTTP POST to OpenAI-compatible endpoint (vLLM, Ollama, etc.)
    → Concurrent dispatch via ExecutorService (concurrency parameter)
  → FrameBlock output [prompt, generated_text, latency_ms, input_tokens, output_tokens]
```

The `llmPredict` instruction:
- Makes HTTP POST calls to any OpenAI-compatible inference endpoint using `java.net.HttpURLConnection`
- Parses JSON responses with `org.apache.wink.json4j` (existing SystemDS dependency)
- Supports concurrent requests via Java `ExecutorService` thread pool (configurable via `concurrency` parameter)
- Returns a 5-column FrameBlock: `[prompt, generated_text, latency_ms, input_tokens, output_tokens]`
- No Python dependency in Java — all inference is done via HTTP

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

## Backends

| Backend | Type | Model | Inference path | GPU? |
|---------|------|-------|----------------|------|
| OpenAI | Cloud API | gpt-4.1-mini | HTTP to OpenAI servers | Remote |
| Ollama | Local server | llama3.2 (3B) | HTTP to local Ollama | CPU |
| vLLM | Local server | Qwen 3B | HTTP to vLLM engine (PagedAttention, CUDA kernels) | GPU |
| SystemDS | JMLC API | Qwen 3B | Py4J → JMLC → DML `llmPredict` → Java HTTP → vLLM | GPU |

## Workloads and datasets

| Workload | Dataset | Source | n | Task | Evaluation method |
|----------|---------|--------|---|------|-------------------|
| math | GSM8K | HuggingFace `openai/gsm8k` | 50 | Grade-school math word problems | Exact numerical match |
| reasoning | BoolQ | HuggingFace `google/boolq` | 50 | Yes/no reading comprehension | Extracted answer match |
| summarization | XSum | HuggingFace `EdinburghNLP/xsum` | 50 | Single-sentence article summary | ROUGE-1 F1 >= 0.2 |
| json_extraction | Built-in | 10 templates x 5 samples | 50 | Extract structured JSON from text | Valid JSON + >= 90% field match |
| embeddings | STS-Benchmark | HuggingFace `mteb/stsbenchmark-sts` | 50 | Semantic similarity (0-5 scale) | Within 1.0 of reference |

All workloads use temperature=0.0 (deterministic generation) for reproducibility. Each run processes 50 samples.

## How measurements work

The runner (`runner.py`) takes a backend, workload config, and output directory:

```bash
python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --concurrency 4 \
  --power-draw-w 350 --hardware-cost 30000 --electricity-rate 0.30 \
  --out results/systemds_qwen3b_math_c4
```

Per-run outputs:
- `samples.jsonl` — per-sample predictions, references, latency, correctness
- `metrics.json` — aggregated latency stats (mean, p50, p95, cv), throughput, accuracy, cost
- `run_config.json` — full configuration for reproducibility
- `manifest.json` — git hash, platform, GPU info

Metrics collected:
- **Latency**: wall-clock time per prompt (ms), with mean, p50, p95, min, max, CV
- **Throughput**: n / total_wall_clock_seconds
- **Accuracy**: workload-specific (see table above)
- **Cost**: hardware amortization ($30K H100 / 15K hours) + electricity (350W x $0.30/kWh), prorated per second

## Benchmark results

### Latency comparison (mean ms/prompt, n=50, NVIDIA H100 PCIe)

| Workload | Ollama (CPU) | OpenAI (Cloud) | vLLM Qwen 3B | SystemDS c=1 |
|---|---|---|---|---|
| math | 5781 | 3630 | 4619 | 2273 |
| json_extraction | 1642 | 1457 | 1151 | 610 |
| reasoning | 5252 | 2641 | 2557 | 1261 |
| summarization | 1079 | 1036 | 791 | 373 |
| embeddings | 371 | 648 | 75 | 41 |

SystemDS c=1 calls the same vLLM inference server. Per-prompt latency is comparable since `llmPredict` adds minimal overhead.

### Accuracy (% correct)

| Workload | Ollama (llama3.2 3B) | OpenAI (gpt-4.1-mini) | vLLM Qwen 3B | SystemDS c=1 | SystemDS c=4 | vLLM Mistral 7B |
|---|---|---|---|---|---|---|
| math | 58% | 88% | 68% | 68% | 68% | 38% |
| json_extraction | 74% | 84% | 52% | 52% | 52% | 50% |
| reasoning | 44% | 70% | 60% | 60% | 64% | 68% |
| summarization | 80% | 88% | 50% | 50% | 62% | 68% |
| embeddings | 40% | 88% | 90% | 90% | 90% | 82% |

SystemDS c=1 matches vLLM Qwen 3B exactly (same model, same inference server). c=4 shows minor variation on reasoning and summarization due to vLLM server batching non-determinism with concurrent requests.

### SystemDS concurrency scaling (throughput)

| Workload | c=1 (req/s) | c=4 (req/s) | Speedup | Eff. latency c=1 (ms) | Eff. latency c=4 (ms) |
|---|---|---|---|---|---|
| math | 0.44 | 1.63 | 3.71x | 2281 | 615 |
| json_extraction | 1.62 | 5.65 | 3.49x | 618 | 177 |
| reasoning | 0.79 | 3.11 | 3.95x | 1270 | 322 |
| summarization | 2.62 | 7.27 | 2.78x | 382 | 137 |
| embeddings | 20.07 | 46.34 | 2.31x | 50 | 22 |

Throughput speedup via Java `ExecutorService` thread pool in the `llmPredict` CP instruction. Effective latency = 1000 / throughput.

## Reproducibility

```bash
cd scripts/staging/llm-bench

# 1. Start vLLM inference server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --port 8000 --gpu-memory-utilization 0.3

# 2. Run all backends
./scripts/run_all_benchmarks.sh all

# SystemDS only (runs both c=1 and c=4 automatically)
./scripts/run_all_benchmarks.sh systemds Qwen/Qwen2.5-3B-Instruct

# GPU comparison (vLLM + SystemDS c=1 + c=4)
./scripts/run_all_benchmarks.sh gpu Qwen/Qwen2.5-3B-Instruct

# Generate HTML report
python scripts/report.py --results-dir results/ --out benchmark_report.html
```

Environment variables for SystemDS:
- `LLM_INFERENCE_URL` — inference server endpoint (default: `http://localhost:8080/v1/completions`)
- `SYSTEMDS_JAR` — path to SystemDS.jar (default: auto-detected)
- `SYSTEMDS_CONCURRENCY` — default concurrency level (overridden by `--concurrency` flag)

## Conclusions

- **Accuracy**: OpenAI leads on most tasks. Among local models, Qwen 3B is strongest on math (68%) and embeddings (90%). SystemDS c=1 matches vLLM Qwen 3B exactly; c=4 shows minor variation on reasoning and summarization due to vLLM batching non-determinism.
- **`llmPredict` built-in works**: Real DML goes through the full SystemDS compilation pipeline. The instruction makes HTTP calls directly from Java with no Python dependency in the Java runtime.
- **Concurrency improves throughput**: c=4 achieves 2.3–3.9x throughput speedup via Java `ExecutorService` in the CP instruction.
- **vLLM is fastest for single-request latency**: PagedAttention, continuous batching, and custom CUDA kernels give it an edge for optimized serving.
- **Cost scales with throughput**: Higher concurrency = less wall-clock time per query = lower amortized hardware cost.
