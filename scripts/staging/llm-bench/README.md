# LLM Inference Benchmark

Backend-agnostic benchmark for comparing LLM inference across cloud APIs,
GPU servers, and SystemDS JMLC. Measures accuracy, latency, throughput, and cost.

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

# Run all workloads for a backend
./scripts/run_all_benchmarks.sh vllm Qwen/Qwen2.5-3B-Instruct

# Run all backends at once
./scripts/run_all_benchmarks.sh all

# Generate report
python scripts/report.py --results-dir results/ --out results/report.html
```

## Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `openai` | OpenAI cloud API | `OPENAI_API_KEY` env var |
| `vllm` | GPU inference server | vLLM running, NVIDIA GPU |
| `systemds` | SystemDS JMLC with `llmPredict` built-in | SystemDS JAR, Py4J, inference server |
| `ollama` | Local inference via Ollama | Ollama installed |

All backends implement the same interface (`generate(prompts, config) → List[GenerationResult]`),
producing identical output format: text, latency_ms, token counts, ttft_ms.

## Workloads

| Workload | Dataset | Evaluation |
|----------|---------|------------|
| `math` | GSM8K (HuggingFace) | Exact numerical match |
| `reasoning` | BoolQ (HuggingFace) | Extracted yes/no match |
| `summarization` | XSum (HuggingFace) | ROUGE-1 F1 >= 0.2 |
| `json_extraction` | CoNLL-2003 (HuggingFace) | Entity-level F1 >= 0.5 |
| `embeddings` | STS-B (HuggingFace) | Score within ±1.0 of reference |

All workloads use temperature=0.0 for deterministic, reproducible results.
Datasets are loaded from HuggingFace at runtime (strict loader — raises `RuntimeError` on failure).

## SystemDS Backend

The SystemDS backend uses Py4J to bridge Python and Java, running the `llmPredict`
DML built-in through JMLC:

```
Python → Py4J → JMLC → DML compilation → llmPredict instruction → Java HTTP → inference server
```

```bash
# Build SystemDS
mvn package -DskipTests

# Start inference server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --port 8000

# Run benchmark
export LLM_INFERENCE_URL="http://localhost:8000/v1/completions"
python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --concurrency 4 --out results/systemds_math
```

Environment variables:
- `SYSTEMDS_JAR` — path to SystemDS.jar (default: auto-detected)
- `SYSTEMDS_LIB` — path to lib/ directory (default: `target/lib/`)
- `LLM_INFERENCE_URL` — inference server endpoint (default: `http://localhost:8080/v1/completions`)

## Cost Flags

```bash
# Cloud GPU rental
python runner.py --backend vllm ... --gpu-hour-cost 2.50

# Owned hardware (electricity + depreciation)
python runner.py --backend systemds ... \
  --power-draw-w 350 --hardware-cost 30000 --hardware-lifetime-hours 15000
```

## Output

Each run produces:
- `samples.jsonl` — per-sample predictions, references, correctness, latency
- `metrics.json` — aggregate accuracy, latency stats (mean/p50/p95), throughput, cost
- `manifest.json` — git hash, timestamp, GPU info, config SHA256
- `run_config.json` — backend and workload configuration

## Tests

```bash
python -m pytest tests/ -v
```
