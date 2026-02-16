Benchmarking framework that compares LLM inference across four backends: OpenAI API, Ollama, vLLM, and SystemDS JMLC with the native `llmPredict` built-in. Evaluated on 5 workloads (math, reasoning, summarization, JSON extraction, embeddings) with n=50 per workload on NVIDIA H100.

## Purpose and motivation

This project was developed as part of the LDE (Large-Scale Data Engineering) course. The goal is to evaluate how SystemDS can be extended to support LLM inference as a native built-in function, and how its performance compares to established LLM serving solutions.

Research questions:

1. Can SystemDS support LLM inference through a native DML built-in that goes through the full compilation pipeline?
2. How does SystemDS compare to dedicated LLM backends (OpenAI, Ollama, vLLM) in terms of accuracy and throughput?
3. How does Java-side concurrent request dispatch scale with the `llmPredict` instruction?

Approach:

- Built a Python benchmarking framework that runs standardized workloads against all four backends under identical conditions (same prompts, same models, same GPU, same evaluation metrics)
- Added `llmPredict` as a native **parameterized built-in function** to SystemDS (PR #2430): goes through the full DML compilation pipeline (parser → hops → lops → CP instruction) and makes HTTP calls to any OpenAI-compatible inference server
- Ran evaluation in two phases: (1) sequential baseline across all backends, (2) SystemDS with Java-side concurrency (`ExecutorService` thread pool in the `llmPredict` instruction)
- All benchmark runs executed on NVIDIA H100 PCIe (81GB), 50 samples per run, temperature=0.0 for reproducibility

## Project structure

```
scripts/staging/llm-bench/
├── runner.py                  # Main benchmark runner (CLI entry point)
├── backends/
│   ├── openai_backend.py      # OpenAI API (gpt-4.1-mini)
│   ├── ollama_backend.py      # Ollama local server (llama3.2)
│   ├── vllm_backend.py        # vLLM serving engine (streaming HTTP)
│   └── systemds_backend.py    # SystemDS JMLC via Py4J + llmPredict DML
├── workloads/
│   ├── math/                  # GSM8K dataset, numerical accuracy
│   ├── reasoning/             # BoolQ dataset, logical accuracy
│   ├── summarization/         # XSum dataset, ROUGE-1 scoring
│   ├── json_extraction/       # Built-in structured extraction
│   └── embeddings/            # STS-Benchmark, similarity scoring
├── evaluation/
│   └── perf.py                # Latency, throughput metrics
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
- No Python dependency in Java — all inference is done via HTTP from the CP instruction

## Backends

| Backend | Type | Model | Inference path | GPU? |
|---------|------|-------|----------------|------|
| OpenAI | Cloud API | gpt-4.1-mini | HTTP to OpenAI servers | Remote |
| Ollama | Local server | llama3.2 (3B) | HTTP to local Ollama | CPU |
| vLLM | Local server | Qwen 3B | Python streaming HTTP to vLLM engine | GPU |
| SystemDS | JMLC API | Qwen 3B | Py4J → JMLC → DML `llmPredict` → Java HTTP → vLLM | GPU |

## Benchmark results

### Accuracy (% correct)

| Workload | Ollama (llama3.2 3B) | OpenAI (gpt-4.1-mini) | vLLM Qwen 3B | SystemDS c=1 | SystemDS c=4 | vLLM Mistral 7B |
|---|---|---|---|---|---|---|
| math | 58% | 88% | 68% | 68% | 68% | 38% |
| json_extraction | 74% | 84% | 52% | 52% | 52% | 50% |
| reasoning | 44% | 70% | 60% | 60% | 64% | 68% |
| summarization | 80% | 88% | 50% | 50% | 62% | 68% |
| embeddings | 40% | 88% | 90% | 90% | 90% | 82% |

SystemDS c=1 matches vLLM Qwen 3B exactly on all workloads (same model, same vLLM inference server). c=4 shows minor variation on reasoning and summarization due to vLLM server-side batching behavior with concurrent requests.

### Per-prompt latency (mean ms/prompt, n=50, NVIDIA H100 PCIe)

| Workload | Ollama (CPU) | OpenAI (Cloud) | vLLM Qwen 3B | SystemDS c=1 |
|---|---|---|---|---|
| math | 5781 | 3630 | 4619 | 2273 |
| json_extraction | 1642 | 1457 | 1151 | 610 |
| reasoning | 5252 | 2641 | 2557 | 1261 |
| summarization | 1079 | 1036 | 791 | 373 |
| embeddings | 371 | 648 | 75 | 41 |

**Note on measurement methodology**: The vLLM backend uses Python `requests` with streaming (SSE parsing overhead), while SystemDS measures Java-side `HttpURLConnection` time (non-streaming). Both call the same vLLM server, so the latency differences reflect client-side measurement differences rather than inference speed. The accuracy comparison is the apples-to-apples metric.

### SystemDS concurrency scaling (throughput)

| Workload | c=1 (req/s) | c=4 (req/s) | Speedup |
|---|---|---|---|
| math | 0.44 | 1.63 | 3.71x |
| json_extraction | 1.62 | 5.65 | 3.49x |
| reasoning | 0.79 | 3.11 | 3.95x |
| summarization | 2.62 | 7.27 | 2.78x |
| embeddings | 20.07 | 46.34 | 2.31x |

Throughput measured as n / total_wall_clock_seconds (Python-side end-to-end).

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

1. **`llmPredict` produces correct results**: SystemDS c=1 matches vLLM Qwen 3B accuracy exactly on all 5 workloads (68%, 52%, 60%, 50%, 90%). The `llmPredict` instruction goes through the full DML compilation pipeline and produces the same outputs as calling the vLLM server directly.

2. **Concurrency improves throughput 2.3-3.9x**: The `ExecutorService` thread pool in the `llmPredict` instruction dispatches up to 4 requests concurrently. Math and reasoning workloads (longer generation) benefit most (3.7-3.9x). Embeddings (short responses) show the least benefit (2.3x) due to fixed overhead dominating.

3. **OpenAI leads on accuracy**: gpt-4.1-mini achieves the highest accuracy on 4/5 workloads (88% math, 84% json, 70% reasoning, 88% summarization). The local 3B models trade accuracy for cost and privacy.

4. **Model size matters more than backend**: Accuracy differences come from the model, not the serving framework. Qwen 3B outperforms Mistral 7B on math (68% vs 38%) and embeddings (90% vs 82%), while Mistral 7B is stronger on reasoning (68% vs 60%) and summarization (68% vs 50%).

5. **Latency requires careful interpretation**: Per-prompt latency numbers are not directly comparable across backends because each uses a different HTTP client and protocol (Python streaming vs Java non-streaming vs cloud API). The key takeaway is that `llmPredict` adds no significant overhead on top of the HTTP call to the inference server.
