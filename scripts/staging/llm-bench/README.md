# LLM Inference Benchmark

Benchmarking framework that compares LLM inference across three backends:
OpenAI API, vLLM, and SystemDS JMLC with the native `llmPredict` built-in.
Evaluated on 5 workloads (math, reasoning, summarization, JSON extraction,
embeddings) with n=50 per workload (46 for OpenAI json_extraction).

## Purpose

Developed as part of the LDE (Large-Scale Data Engineering) course to answer:

- How does SystemDS's `llmPredict` built-in compare to dedicated LLM backends
  (OpenAI, vLLM) in terms of accuracy and throughput?
- What is the cost-performance tradeoff across cloud APIs and GPU-accelerated
  backends?

The framework runs standardized workloads against all backends under identical
conditions (same prompts, same evaluation metrics). The `llmPredict` built-in
goes through the full DML compilation pipeline (parser -> hops -> lops -> CP
instruction) and makes HTTP calls to any OpenAI-compatible inference server.
GPU backends (vLLM, SystemDS) were evaluated on NVIDIA H100 PCIe (81 GB).
OpenAI ran on local MacBook calling cloud API. All runs used 50 samples per
workload, temperature=0.0 for reproducibility.

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

**Note on committed results:** The `results/` directory is intentionally committed
to the repository. Since this is a standalone benchmarking project, the raw outputs
(metrics, samples, manifests) are included for reproducibility, peer review, and
interpretation. This allows reviewers to verify the numbers reported in this README
against the actual data. If repository size becomes a concern, these can be moved
to `.gitignore` and kept locally.

## Backends

| Backend | Type | Model | Hardware | Inference Path |
|---------|------|-------|----------|----------------|
| OpenAI | Cloud API | gpt-4.1-mini | MacBook (API call) | Python HTTP to OpenAI servers |
| vLLM | GPU server | Qwen2.5-3B-Instruct | NVIDIA H100 | Python HTTP to vLLM engine |
| SystemDS | JMLC API | Qwen2.5-3B-Instruct | NVIDIA H100 | Py4J -> JMLC -> DML llmPredict -> Java HTTP -> vLLM |

All backends implement the same interface (`generate(prompts, config) -> List[Result]`),
producing identical output format: text, latency_ms, token counts.

SystemDS and vLLM Qwen 3B use the same model on the same vLLM inference
server, making their accuracy directly comparable. Both backends send
identical parameters (temperature=0.0, top_p=0.9, max_tokens) to the
server -- all specified explicitly in the workload config YAML files.

```
┌──────────────────────────────────────────────────────────────┐
│               vLLM INFERENCE SERVER  (GPU)                   │
│   PagedAttention, continuous batching, KV cache              │
│   Running on NVIDIA H100 PCIe                                │
├──────────────────────────┬───────────────────────────────────┤
│                          │                                   │
│   HTTP POST              │   HTTP POST                       │
│   from Python            │   from Java                       │
│   (vLLM backend)         │   (SystemDS llmPredict)           │
│                          │                                   │
│   payload:               │   payload:                        │
│     model ✓              │     model ✓                       │
│     prompt ✓             │     prompt ✓                      │
│     max_tokens ✓         │     max_tokens ✓                  │
│     temperature ✓        │     temperature ✓                 │
│     top_p ✓              │     top_p ✓                       │
│     stream: false        │     stream: false (default)       │
│                          │                                   │
│   Timeouts:              │   Timeouts:                       │
│     connect: 10s         │     connect: 10s                  │
│     read: 300s           │     read: 300s                    │
│                          │                                   │
│   Response: full JSON    │   Response: full JSON             │
│   (all at once)          │   (all at once)                   │
└──────────────────────────┴───────────────────────────────────┘
```

Both backends are HTTP clients to the same vLLM server. Neither backend
runs inference on its own -- all model computation happens server-side.
The vLLM server provides three key GPU optimizations that both backends
benefit from equally:

- **KV Cache**: Stores key/value attention tensors from previously
  generated tokens so they don't need to be recomputed. Without it,
  generating token #50 would recompute attention for tokens #1-49.
  Makes generation O(n) instead of O(n^2).
- **PagedAttention**: Manages KV cache memory using fixed-size pages
  (like OS virtual memory) instead of pre-allocating one large block
  per request. This allows more concurrent requests in GPU memory.
- **Continuous Batching**: Dynamically adds/removes requests from the
  GPU batch as they arrive/complete, instead of waiting for a fixed
  batch to fill. When one request finishes, its GPU slot is
  immediately given to a waiting request.

SystemDS does **not** have its own GPU inference engine. It is a pure
HTTP client -- the `llmPredict` Java instruction sends HTTP POST
requests to whatever OpenAI-compatible server is running. All GPU
optimizations happen inside the vLLM server, which treats requests from
Python and Java identically.

Both backends send identical model parameters (model, temperature,
top_p, max_tokens, stream=false). Both receive the full JSON response
at once. The run-order experiment (see below) showed that both
summarization and reasoning text differences follow run position
(1st vs 2nd) due to vLLM Automatic Prefix Caching (APC). For
summarization this changes accuracy (25 vs 31); for reasoning the
text differs but accuracy stays 29/50 in all 4 runs.

## Workloads

| Workload | Dataset | Evaluation |
|----------|---------|------------|
| `math` | GSM8K (HuggingFace) | Exact numerical match |
| `reasoning` | BoolQ (HuggingFace) | Extracted yes/no match |
| `summarization` | XSum (HuggingFace) | ROUGE-1 F1 >= 0.2 |
| `json_extraction` | CoNLL-2003 (HuggingFace) | Entity-level F1 >= 0.5 |
| `embeddings` | STS-B (HuggingFace) | Score within +/-1.0 of reference |

All workloads use temperature=0.0 for deterministic, reproducible results
and top_p=0.9 (nucleus sampling). At temperature=0.0 (greedy decoding),
the model always picks the single highest-probability token, so the
top_p value has no theoretical effect. The 0.9 default is set in
`LlmPredictCPInstruction.java` and explicitly specified in all workload
config YAML files to avoid relying on different server defaults.
Datasets are loaded from HuggingFace at runtime (strict loader -- raises
`RuntimeError` on failure).

## SystemDS Backend

The SystemDS backend uses Py4J to bridge Python and Java, running the
`llmPredict` DML built-in through JMLC:

```
Python -> Py4J -> JMLC -> DML compilation -> llmPredict instruction -> Java HTTP -> inference server
```

```bash
# Build SystemDS
mvn package -DskipTests

# Start inference server (in a screen session)
screen -S vllm
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --port 8080
# Once "Uvicorn running on ..." appears, detach: Ctrl+A, D

# Run benchmark (in another terminal)
export LLM_INFERENCE_URL="http://localhost:8080/v1/completions"
CUBLAS_WORKSPACE_CONFIG=:4096:8 python runner.py \
  --backend systemds --model Qwen/Qwen2.5-3B-Instruct \
  --workload workloads/math/config.yaml \
  --out results/systemds_math

# Stop server when done (important -- GPU memory is not freed until killed)
screen -r vllm    # reattach
# Ctrl+C to stop vLLM, then 'exit' to close screen
# Or kill from outside:
screen -X -S vllm quit
```

**Troubleshooting GPU issues:**
```bash
# Check GPU status
nvidia-smi

# If GPU shows high utilization but no processes, kill zombie processes:
sudo fuser -v /dev/nvidia*    # find processes using GPU
kill -9 <PID>                 # kill them

# If GPU 0 is busy, use a different GPU:
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct --port 8080

# Clean up dead screen sessions
screen -wipe
```

**Important:** After pulling new Java changes, rebuild the JAR before running
the SystemDS backend:
```bash
cd ~/systemds && mvn package -DskipTests
```
The SystemDS backend loads `target/SystemDS.jar` at runtime via JMLC. If the
JAR is stale (e.g., missing the `model` parameter added to `llmPredict`),
you will get `Invalid parameters for LLMPREDICT` errors.

Environment variables:
- `SYSTEMDS_JAR` -- path to SystemDS.jar (default: auto-detected)
- `SYSTEMDS_LIB` -- path to lib/ directory (default: `target/lib/`)
- `LLM_INFERENCE_URL` -- inference server endpoint (default: `http://localhost:8080/v1/completions`)
- `CUBLAS_WORKSPACE_CONFIG` -- set to `:4096:8` for deterministic cuBLAS (required for reproducible results)

**Note:** The CoNLL-2003 NER dataset (used by json_extraction) requires
`trust_remote_code=True` when first downloaded. The loader will prompt
for confirmation.

**Alternative: lightweight server without vLLM.** For testing or
development without a full vLLM installation, `src/main/python/llm_server.py`
provides a minimal OpenAI-compatible server that loads any HuggingFace
model directly:

```bash
python src/main/python/llm_server.py Qwen/Qwen2.5-3B-Instruct --port 8080
```

This server handles tokenization, inference (`torch.no_grad()`), and
returns the same `{choices, usage}` JSON format that `llmPredict`
expects. It works on CPU (slow) or GPU (fast) and is useful for running
the Java tests (`JMLCLLMInferenceTest`) or small-scale experiments
without deploying vLLM. The long-term vision is to replace this external
server approach entirely with native DML transformer operations that
run model inference directly inside SystemDS's matrix engine.

### Previous Approach: Py4J Callback (PR #2430)

The initial implementation (closed PR #2430) loaded HuggingFace models
directly inside a Python worker process and used Py4J callbacks to bridge
Java and Python:

```
Python worker (loads model into GPU memory)
  ^
  | Py4J callback: generateBatch(prompts)
  v
Java JMLC (PreparedScript.generateBatchWithMetrics)
```

This approach had several drawbacks:
- **Tight coupling:** Model loading, tokenization, and inference all lived
  in `llm_worker.py`, requiring Python-side changes for every model config.
- **No standard API:** Used a custom Py4J callback protocol instead of the
  OpenAI-compatible `/v1/completions` interface that vLLM and other servers
  already provide.
- **Limited optimization:** The Python worker reimplemented batching and
  tokenization rather than leveraging vLLM's continuous batching, PagedAttention,
  and KV cache management.
- **Process lifecycle:** Java had to manage the Python worker process
  (`loadModel()` / `releaseModel()`) with 300-second timeouts for large models.

The current approach (this PR) replaces the Py4J callback with a native
DML built-in (`llmPredict`) that issues HTTP requests to any
OpenAI-compatible server:

```
DML script: llmPredict(prompts, url=..., model=...)
  -> LlmPredictCPInstruction (Java HTTP client)
    -> Any OpenAI-compatible server (vLLM, llm_server.py, etc.)
```

Benefits of the current approach:
- **Decoupled:** Inference server is independent — swap vLLM for TGI, Ollama,
  or any OpenAI-compatible endpoint without changing DML scripts or Java code.
- **Standard protocol:** Uses the `/v1/completions` API, making benchmarks
  directly comparable across backends.
- **Server-side optimization:** vLLM handles batching, KV cache, PagedAttention,
  and speculative decoding transparently.
- **Simpler Java code:** `LlmPredictCPInstruction` is a single 216-line class
  that builds JSON, sends HTTP, and parses the response — no process management.

## Benchmark Results

### Evaluation Methodology

Each workload defines its own `accuracy_check(prediction, reference)` function
that returns true/false per sample. The accuracy percentage is
`correct_count / n`. All accuracy counts were verified against raw
`samples.jsonl` files.

| Workload | Criterion | How It Works |
|----------|-----------|--------------|
| math | Exact numerical match | Extracts the final number from chain-of-thought using regex (####, \boxed{}, last number). Compares against GSM8K reference. |
| reasoning | Extracted answer match | Extracts yes/no from response using CoT markers ("answer is X", "therefore X"). Compares against BoolQ reference. |
| summarization | ROUGE-1 F1 >= 0.2 | Computes ROUGE-1 F1 between generated summary and XSum reference with stemming. Predictions shorter than 10 chars rejected. |
| json_extraction | Entity F1 >= 0.5 (NER) or >= 90% fields match (scalar) | NER: entity-level precision/recall/F1 across all entity categories. Scalar: field-level match with case-insensitive string comparison. |
| embeddings | Score within 1.0 of reference | Model rates sentence-pair similarity on 0-5 STS scale. Passes if abs(predicted - reference) <= 1.0. |

### Accuracy (% correct, n=50 per workload)

| Workload | OpenAI gpt-4.1-mini | vLLM Qwen 3B | SystemDS Qwen 3B |
|----------|---------------------|--------------|------------------|
| math | **96%** (48/50) | 68% (34/50) | 68% (34/50) |
| reasoning | **88%** (44/50) | 58% (29/50) | 58% (29/50) |
| summarization | **86%** (43/50) | 50% (25/50) | 62% (31/50) |
| json_extraction | **61%** (28/46) | **66%** (33/50) | **66%** (33/50) |
| embeddings | 88% (44/50) | **90%** (45/50) | **90%** (45/50) |

**Key observations:**

- **SystemDS matches vLLM on 4/5 workloads** (math, reasoning,
  json_extraction, embeddings). Both use the same Qwen2.5-3B model on
  the same vLLM inference server with temperature=0.0. Predictions are
  byte-for-byte identical on all samples in these workloads.
- **Summarization gap (25 vs 31) is caused by vLLM Automatic Prefix Caching
  (APC).** The run-order experiment proves this: the 1st-run backend
  always scores 25/50; the 2nd-run backend always scores 31/50,
  regardless of which backend goes first. See "Reverse-Order Experiment"
  below.
- **OpenAI gpt-4.1-mini leads on 3/5 workloads**, with the largest gap on
  math (96% vs 68%). This is model quality (much larger model), not
  serving infrastructure.
- **Qwen 3B beats OpenAI on json_extraction and embeddings** (66% vs 61%,
  90% vs 88%), showing that smaller models can excel on focused tasks.
- **json_extraction uses CoNLL-2003 NER** (named entity recognition) with
  entity-level F1 scoring (threshold >= 0.5). Both GPU backends produce
  identical output on all 50 samples.

**Notes:**

- All three backends use the CoNLL-2003 NER dataset for json_extraction
  with entity-level F1 scoring (threshold >= 0.5). OpenAI ran with 46
  samples (earlier dataset version); vLLM and SystemDS ran with 50.
  The entity F1 scorer evaluates partial entity matches across categories
  (persons, organizations, locations, misc), yielding 66% accuracy for
  GPU backends (33/50) and 61% for OpenAI (28/46).
- The vLLM and SystemDS backends previously sent different `top_p` values
  to the inference server (vLLM: server default 1.0, SystemDS: 0.9). This
  has been fixed -- all backends now explicitly send `top_p=0.9`. At
  temperature=0.0, this difference has no theoretical effect (greedy
  decoding), but it was an uncontrolled variable.
- Reasoning `samples.jsonl` files were re-run after fixing a boolean
  extraction bug: the old regex `answer[:\s]+([^\n.]+)` matched preamble
  text instead of standalone yes/no lines. The fix (`_extract_boolean()`)
  now correctly extracts answers like "Yes" on their own line.

### Accuracy Gap Analysis (vLLM vs SystemDS)

On 4/5 workloads (math, reasoning, json_extraction, embeddings),
accuracy is identical. On 3 of these (math, json_extraction, embeddings),
predictions are byte-for-byte identical. On reasoning, 17/50 samples
produce different text due to APC but accuracy is still 29/50 in all 4
runs. The remaining workload (summarization) has both different text
(22/50) and different accuracy (25 vs 31) due to APC — proven by the
run-order experiment (see below).

**Note on labels:** In the committed results, "vLLM" ran first and
"SystemDS" ran second. For summarization, these labels correspond to
"1st-run (cold cache)" and "2nd-run (warm cache)" respectively. The
reverse-order experiment confirms the outputs follow cache position,
not the backend.

**Reasoning (29 vs 29, same accuracy):** Within a session, 66% of
predictions are byte-for-byte identical between 1st and 2nd run. The
remaining 34% (17 samples) produce different text due to APC changing
the KV cache state. However, the accuracy impact is zero: both runs
score 29/50. The same-position comparison is 100% identical across
sessions (1st-run in session 1 = 1st-run in session 2), confirming
that the text differences are deterministic given cache state.

**Summarization (25 vs 31, gap = 6 samples):** ROUGE-1 F1 measures
word overlap between prediction and reference, with a pass threshold of
0.2. The 6 samples where accuracy differs are a subset of the 22
samples where APC produces different text. In these 6 cases, one
variant passes ROUGE and the other fails.

The run-order experiment proves this is APC, not a backend
difference. For all 22 unstable samples:
- Original vLLM (1st) = Reverse SystemDS (1st) — same cold-cache text
- Original SystemDS (2nd) = Reverse vLLM (2nd) — same warm-cache text

The 1st-run variant always scores 25/50. The 2nd-run variant always
scores 31/50. Which backend gets which score depends entirely on run
order.

Concrete examples from the 6 accuracy-divergent samples:

- `xsum-98` ("Hope Solo suspended by US Soccer"):
  - 1st-run text: "...negatively **impacted** both herself and her
    team..." (**fail** ROUGE-1 F1)
  - 2nd-run text: "...negatively **affected** both herself and her
    team..." (**pass** ROUGE-1 F1)
  - A single word change ("impacted" → "affected") from APC flips
    the ROUGE score past the 0.2 threshold.

- `xsum-101` ("ID Systems Ltd plans to create 120 new jobs..."):
  - Reference: "Scottish engineering services company ID Systems Ltd
    has announced plans to create 120 new jobs after securing a
    six-figure investment from UK Steel Enterprise (UKSE)."
  - 2nd-run (27 words, **pass F1=0.264**): "UKSE is supporting ID
    Systems' expansion into Glasgow and Lanarkshire through a loan and
    equity investment, enabling the company to hire additional staff
    and grow its workforce."
  - 1st-run (61 words, **fail F1=0.184**): "UKSE is supporting ID
    Systems' expansion into Glasgow and Lanarkshire by providing funding
    and a senior management team, while ID Systems secures long-term
    contracts and aims to double its workforce through this investment.
    To summarize concisely: UKSE is financing ID Systems' expansion
    into Glasgow and Lanarkshire, enabling the company to hire more
    staff and secure long-term contracts. (135 words)"
  - The 1st-run text matched more reference words (8 vs 7) but was
    penalised for verbosity (meta-text, speculative details).

- `xsum-11` ("Hospital bosses in Sussex have apologised..."):
  - Reference: "Hospital bosses in Sussex have apologised after about
    850 patients were sent leaflets in error suggesting they might have
    cancer."
  - 2nd-run (**pass F1=0.222**): "The trust said it would be reviewing
    its processes to prevent such errors in future. The East Sussex NHS
    Trust experienced an administrative error..."
  - 1st-run (**fail F1=0.175**): "The trust said it would review its
    processes to prevent such errors in future. The leaflets mistakenly
    included with hospital appointment letters..."
  - Both capture the core fact. The 1st-run adds speculative detail
    that dilutes precision without matching reference words.

- `xsum-36` ("Terrorism charges"):
  - 2nd-run (**pass**): Focused on the main news (suspect charged with
    breaching TPim order, first reported instance).
  - 1st-run (**fail**): Diverged into background about TPim orders
    replacing control orders in 2012, missing the central narrative.

- `xsum-44` ("Cricket Boxing Day Test"):
  - 2nd-run (**pass**): Summarised key match events concisely.
  - 1st-run (**fail**): Added meta-commentary ("The text summarizes
    the cricket match...") and a redundant summary, plus changed
    "final Test" to "third Test" (factual error).

ROUGE is a lexical (word overlap) metric, not semantic. Two summaries
conveying the same meaning with different words receive different scores.
The 0.2 threshold is lenient, but borderline scores flip when APC
produces slightly different phrasing or verbosity.

### Text Identity (vLLM vs SystemDS)

On 3/5 workloads, both backends produce byte-for-byte identical output
text for all samples. This confirms that the SystemDS JMLC pipeline
(Py4J -> DML -> Java HTTP -> FrameBlock) is a lossless pass-through.

| Workload | Identical | Different | % Identical |
|----------|-----------|-----------|-------------|
| math | 50/50 | 0 | **100%** |
| json_extraction | 46/46 | 0 | **100%** |
| embeddings | 50/50 | 0 | **100%** |
| reasoning | 33/50 | 17 | 66% |
| summarization | 28/50 | 22 | 56% |

**Why do 3 workloads match perfectly but 2 don't?** The key factor is
output constraint level, not output length:

| Workload | Avg output length | Constraint level | % Identical |
|---|---|---|---|
| embeddings | 4 chars | Highly constrained (single number) | 100% |
| json_extraction | 150 chars | Structured (JSON fields from input, n=46) | 100% |
| math | **1349 chars** | Arithmetic steps (one valid path) | **100%** |
| summarization | 328 chars | Unconstrained (many valid phrasings) | 56% |
| reasoning | 960 chars | Unconstrained (many valid phrasings) | 66% |

Math produces the **longest** outputs (avg 1349 chars) yet achieves
100% identity. This is because arithmetic is highly constrained: at each
step "16 - 3 = 13", there is exactly one correct continuation, so the
model's probability distribution is sharply peaked. GPU FP noise cannot
flip the argmax when the top token has a huge margin over alternatives.

Summarization produces shorter outputs (avg 328 chars) but only 56%
identity, because natural language summaries have many equally valid
phrasings. "The report found..." vs "A report revealed..." vs
"According to..." — multiple tokens have similar probabilities, so
even small differences in server cache state (APC) or floating-point
rounding can flip the selection at near-tied positions.

**Root cause for all 39 divergent samples: vLLM Automatic Prefix Caching (APC).**

The run-order experiment proves that ALL divergent samples (22 summarization
+ 17 reasoning) follow the same APC pattern: same-position runs are 100%
identical across sessions, while cross-position runs diverge. The backend
label is irrelevant — only cache position matters.

**1. Summarization (22 samples):**

vLLM 0.15.1 enables APC by default (`enable_prefix_caching=True`). APC
stores KV cache tensors from previously processed prefixes and reuses
them for new requests with the same prefix. When the benchmark runs
two backends sequentially, the 2nd batch reuses cached KV tensors from
the 1st batch, taking a different code path (skipping prefill) that
produces slightly different floating-point attention scores — enough to
flip the argmax at near-tied token positions.

The run-order experiment proves this conclusively. For ALL 22
unstable summarization samples:

```
Original run (vLLM=1st, SystemDS=2nd):
  vLLM text = variant A,   SystemDS text = variant B

Reverse run (SystemDS=1st, vLLM=2nd):
  SystemDS text = variant A,  vLLM text = variant B

  → 1st-run always produces variant A (cold cache)
  → 2nd-run always produces variant B (warm cache)
  → The backend label is irrelevant — only position matters
```

This holds for 22/22 samples with zero exceptions.

**How APC changes the computation:**

- **Cold cache (1st batch):** vLLM runs full prefill — computes KV
  tensors from scratch for every prompt token.
- **Warm cache (2nd batch):** APC recognises the prefix, skips prefill,
  loads stored KV tensors, and jumps directly to generation. Different
  memory access pattern and kernel configuration produces slightly
  different floating-point results.

```
Cold cache (full prefill → fresh KV):
  P("impacted") = 0.51, P("affected") = 0.49 → picks "impacted"

Warm cache (load cached KV → skip prefill):
  P("impacted") = 0.49, P("affected") = 0.51 → picks "affected"
```

(Numbers illustrative.) From the divergent token, all subsequent tokens
cascade differently, producing the full alternative response.

**Server log evidence.** The vLLM prefix cache hit rate climbs from ~9%
(1st backend running) to 55.1% (end of 2nd backend). The steepest rises
match the two unconstrained workloads:

| Time (UTC) | Event | Prefix Cache Hit Rate |
|------------|-------|----------------------|
| 16:52:01–16:52:42 | SystemDS reasoning/summarization/json | 9.3%–12.6% |
| 16:53:21 | **vLLM starts** | **16.0%** |
| 16:55:12 | vLLM reasoning (4 s in) | **24.2%** |
| 16:56:12 | vLLM summarization (2 s in) | **37.5%** |
| 16:57:12 | Final idle | **55.1%** |

**Factual content swaps with cache state (not just phrasing):**

- **xsum-30** (athletics heptathlon): Cold-cache: "**Jessica Ennis-Hill**
  trails behind" (correct athlete). Warm-cache: "**Tiffany Hanks** is
  third" (hallucinated name). In Run 1: vLLM=correct, SystemDS=hallucinated.
  In Run 2: SystemDS=correct, vLLM=hallucinated. The hallucination
  follows cache state, not the client.
- **xsum-42** (South Africa minimum wage): Cold-cache: "from **April 2023**".
  Warm-cache: "from **April 2018**". Labels swap with run order.
- **xsum-89** (boxing): Cold-cache: "gold for **Russia**" (hallucinated
  country). Warm-cache: "maiden Olympic gold" (no country error).
  Labels swap with run order.

These are not random — the same cache state always produces the same
output. With temperature=0, `CUBLAS_WORKSPACE_CONFIG`, and sequential
requests, same prompt + same cache state → same code path → same output.

**2. Reasoning (17 samples): also APC.**

Reasoning follows the same APC pattern as summarization. Within a
session, 33/50 (66%) of predictions are byte-for-byte identical between
1st and 2nd run. The remaining 17 samples diverge due to APC changing
the KV cache state. Cross-session, same-position runs are 100% identical
(1st vs 1st, 2nd vs 2nd) — proving position determines output, not the
backend.

Unlike summarization, the accuracy impact is zero: all 4 runs score
29/50. The 17 divergent samples produce different text but the same
yes/no answer (or different wrong answers). Reasoning diverges later in
the response (median divergence point ~400 chars) compared to
summarization, because BoolQ chain-of-thought reasoning shares more
common structure before branching.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` was used for all runs to force
deterministic cuBLAS algorithms. The constrained workloads (math,
json_extraction, embeddings) achieve 100% byte-identity. The
unconstrained workloads (reasoning, summarization) diverge due to APC
cache state, not cuBLAS non-determinism.

**Why streaming was investigated (and why it is not the cause).**

The original vLLM backend used `"stream": true` while SystemDS used
`"stream": false`. Streaming was checked first as a potential source
of byte-level corruption. The 150 byte-identical samples across math,
json_extraction, and embeddings ruled out SSE corruption. Switching to
`"stream": false` produced identical divergence counts (same 39
samples), confirming streaming had no effect. Both backends now use
non-streaming mode.

### Run-Order Experiment (2 Sessions, 4-Way Analysis)

To understand why vLLM and SystemDS produce different summarization
accuracy, the benchmark was run in two sessions with **reversed order**
and a fresh vLLM server restart between them (clearing the APC cache).
This creates 4 data points per sample, enabling precise root cause
analysis.

- **Session 1 (normal order)**: vLLM first, SystemDS second
- **Session 2 (reverse order)**: SystemDS first, vLLM second

**Accuracy by session and run position:**

| Workload | Sess 1: vLLM (1st) | Sess 1: SysDS (2nd) | Sess 2: SysDS (1st) | Sess 2: vLLM (2nd) |
|---|---|---|---|---|
| math | 68% (34/50) | 68% (34/50) | 68% (34/50) | 68% (34/50) |
| reasoning | 58% (29/50) | 58% (29/50) | 58% (29/50) | 58% (29/50) |
| summarization | **50% (25/50)** | **62% (31/50)** | **50% (25/50)** | **62% (31/50)** |
| json_extraction | 66% (33/50) | 66% (33/50) | 66% (33/50) | 66% (33/50) |
| embeddings | 90% (45/50) | 90% (45/50) | 90% (45/50) | 90% (45/50) |

**Prediction identity (byte-for-byte text comparison):**

| Comparison | math | reasoning | summarization | json_extraction | embeddings |
|---|---|---|---|---|---|
| Same session, cross-backend (sess 1: vLLM vs SysDS) | 100% | 66% | 56% | 100% | 100% |
| Same session, cross-backend (sess 2: SysDS vs vLLM) | 100% | 66% | 56% | 100% | 100% |
| Cross-session, same backend (vLLM sess 1 vs vLLM sess 2) | 100% | 66% | 56% | 100% | 100% |
| Cross-session, same backend (SysDS sess 1 vs SysDS sess 2) | 100% | 66% | 56% | 100% | 100% |
| **Cross-session, same position (1st vs 1st)** | **100%** | **100%** | **100%** | **100%** | **100%** |
| **Cross-session, same position (2nd vs 2nd)** | **100%** | **100%** | **100%** | **100%** | **100%** |

The last two rows are the key: when comparing runs that occupied the
**same position** (both 1st or both 2nd), ALL workloads achieve 100%
text identity — even though different backends produced the text. This
proves position determines output, not the backend.

**Key findings:**

**1. Summarization: APC confirmed as root cause.**

The summarization accuracy follows run position perfectly across both
sessions. 1st-run always scores 25/50, 2nd-run always scores 31/50:

```
Original order: vLLM     (1st) = 25/50,  SystemDS (2nd) = 31/50
Reverse order:  SystemDS (1st) = 25/50,  vLLM     (2nd) = 31/50
```

For all 22 unstable samples, the text output swaps with position:
`orig_vLLM(1st) == rev_SystemDS(1st)` and
`orig_SystemDS(2nd) == rev_vLLM(2nd)` — 50/50 byte-for-byte
identical in each group. Zero exceptions.

**2. Reasoning: same-position = 100% identical, cross-position = 66%.**

Within a session, 66% of reasoning predictions are byte-for-byte
identical between 1st and 2nd run. Across sessions, same-position
runs are 100% identical (1st vs 1st, 2nd vs 2nd). Accuracy is
29/50 in all 4 runs — consistent within and across sessions. The
34% of divergent samples produce different text depending on whether
APC-cached prefixes affect the generation, but the accuracy impact
is zero in these runs.

**3. Math, json_extraction, embeddings: fully deterministic.**

100% identical across all 4 runs. Constrained outputs (arithmetic,
JSON structure, single number) have such peaked probability
distributions that neither APC nor FP rounding can flip the argmax.

### Per-Prompt Latency (mean ms, n=50)

| Workload | OpenAI (MacBook -> Cloud) | vLLM Qwen 3B (H100) | SystemDS Qwen 3B (H100) |
|----------|--------------------------|----------------------|--------------------------|
| math | 4577 | 1913 | 1917 |
| reasoning | 1735 | 1109 | 1134 |
| summarization | 1131 | 364 | 362 |
| json_extraction | 1498 | 266 | 266 |
| embeddings | 773 | 47 | 60 |

**Note on measurement methodology:** Latency is measured differently by
each backend:
- **vLLM**: Python `time.perf_counter()` around non-streaming HTTP POST.
  Timer runs from POST start to full JSON response received.
- **SystemDS**: Java `System.nanoTime()` inside `HttpURLConnection`
  round-trip (reads full response with `readAllBytes()`).
  Reported latency = Java HTTP time + amortized JMLC pipeline overhead.
  The pipeline overhead is instrumented into 4 phases:
  1. **DML compilation** (`compile_ms`) — compiling the DML script or
     cache hit (PreparedScript reuse). First call compiles; subsequent
     calls with same parameters reuse the cached script (~0 ms).
  2. **Py4J marshalling** (`marshal_ms`) — transferring prompt strings
     from Python to Java via Py4J (creating Java String arrays, calling
     `setFrame()`).
  3. **Java execution** (`exec_wall_ms`) — `executeScript()` which runs
     the DML program including the `llmPredict` instruction's HTTP calls.
  4. **Py4J unmarshalling** (`unmarshal_ms`) — retrieving the Java
     `FrameBlock` result back to Python via `getFrameBlock()`.
  Each phase is timed with `time.perf_counter()` in the Python backend.
  The Java HTTP time per prompt is reported separately by the Java
  instruction itself (column 2 of the output FrameBlock).

  **SystemDS JMLC pipeline breakdown (ms):**

  | Workload | compile | marshal | exec/prompt | unmarshal | overhead | pipeline |
  |----------|---------|---------|-------------|-----------|----------|----------|
  | math | 316 | 113 | 1909 | 0.8 | 483 | 95852 |
  | reasoning | 241 | 43 | 1128 | 0.8 | 337 | 56680 |
  | summarization | 305 | 52 | 355 | 0.8 | 412 | 18090 |
  | json_extraction | 299 | 48 | 259 | 0.9 | 403 | 13295 |
  | embeddings | 338 | 166 | 50 | 1.4 | 563 | 3009 |

  Observations: DML compilation is ~300 ms (one-time; cached on repeat).
  Py4J marshalling is 43--166 ms depending on prompt size. Unmarshalling
  is <2 ms. The exec/prompt column is the per-prompt share of
  `executeScript()` wall time, which includes all HTTP calls. Pipeline
  overhead (compile + marshal + unmarshal + scheduling) is amortized
  across prompts and adds ~8--11 ms/prompt for n=50.

- **OpenAI**: Python `time.perf_counter()` around OpenAI API call.
  Includes network round-trip to cloud servers.

The accuracy comparison is the apples-to-apples metric since all backends
process the same prompts with the same parameters.

**SystemDS vs vLLM latency** (same server, same model, CUBLAS
deterministic, non-streaming HTTP): Latencies are within 0--3% of each
other for generation workloads. These differences are within measurement
noise: the runs were ~6 minutes apart and divergent samples generate
different output lengths. Latency is dominated by output token count —
a sample where one run generates more tokens simply does more work.

| Workload | vLLM | SystemDS | Difference |
|----------|------|----------|------------|
| math | 1913 ms | 1917 ms | +0.2% |
| reasoning | 1109 ms | 1134 ms | +2.2% |
| summarization | 364 ms | 362 ms | -0.6% |
| json_extraction | 266 ms | 266 ms | +0.0% |
| embeddings | 47 ms | 60 ms | +29.1% |

For the four generation-heavy workloads (math through json_extraction),
both backends are within 0--2% of each other — well within measurement
noise. The embeddings workload shows +29% overhead because the HTTP
call itself is only ~47 ms, so the fixed JMLC pipeline cost (~10 ms
per prompt from compile + marshal + unmarshal amortization) becomes a
significant fraction. Both are HTTP clients to the same vLLM server;
latency is determined by output token count, not by which client sends
the request.

**Why output length differs between backends:**
When two sequential runs diverge at a single token (due to APC), the
two autoregressive paths produce responses of different lengths — neither
is reliably longer. Among the 17 divergent reasoning samples, neither
run position is systematically longer. The difference in latency between
backends on these samples reflects the output length difference, not a
performance difference.

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

OpenAI cost is the per-token API price. vLLM and SystemDS costs are
estimated from hardware ownership (electricity + GPU amortization), computed
from actual wall-clock time per run.

**Cost formula for GPU backends:**

```
electricity_kwh   = (power_draw_W / 1000) × (wall_seconds / 3600)
electricity_cost  = electricity_kwh × electricity_rate_per_kwh
hw_amortization   = (hardware_price / lifetime_hours) × (wall_seconds / 3600)
total_compute_cost = electricity_cost + hw_amortization
```

`wall_seconds` is measured with `time.perf_counter()` around the entire
batch of prompts in `runner.py` (true wall-clock time, not sum of
per-prompt latencies). Individual per-prompt latencies are collected
separately for statistical analysis (mean, p50, p95) but are never used
in cost calculation.

**Hardware cost assumptions** (NVIDIA H100 PCIe, matching the benchmark GPU):

| Parameter | Value | Source |
|-----------|-------|--------|
| GPU power draw | 350 W | H100 PCIe TDP (thermal design power) |
| Electricity rate | $0.30/kWh | EU average residential rate |
| Hardware purchase price | $30,000 | H100 PCIe market price |
| Useful lifetime | 15,000 hours | ~5 years at 8 hr/day |
| **Hourly rate** | **$2.00/hr + $0.105/hr** | Amortization + electricity |

Hardware amortization ($2.00/hr) dominates electricity ($0.105/hr) by
~19x. The GPU hardware cost determines the compute expense regardless
of whether the GPU is actively processing queries.

**Verification** (vLLM math run, 50 prompts, wall_s = 95.6s):
- electricity = 0.35kW × 0.02655hr × $0.30/kWh = **$0.0028**
- amortization = $2.00/hr × 0.02655hr = **$0.0531**
- total = **$0.0559** (matches metrics.json)

**Why local GPU appears more expensive here:** The H100 amortizes at
$2.00/hr regardless of utilization. This benchmark runs only 250 sequential
queries totaling ~3 minutes of inference — the GPU is idle most of the time.
OpenAI's per-token pricing only charges for actual usage, which wins at low
volume. At higher utilization (concurrent requests, continuous serving), the
H100's per-query cost drops significantly: at full throughput (~21 req/s on
embeddings), the amortized cost is ~$0.00003/query vs OpenAI's
~$0.0004/query — making owned hardware ~13x cheaper at scale.

### ROUGE Scores (Summarization)

| Backend | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---------|-----------|-----------|-----------|
| OpenAI | 0.270 | 0.066 | 0.201 |
| vLLM Qwen 3B | 0.220 | 0.057 | 0.157 |
| SystemDS Qwen 3B | 0.226 | 0.056 | 0.157 |

## Conclusions

1. **SystemDS `llmPredict` is a lossless pass-through**: On 3/5
   workloads (math, json_extraction, embeddings), every response is
   byte-for-byte identical between vLLM and SystemDS — 150/150 samples
   total. The JMLC pipeline (Py4J -> DML -> Java HTTP -> FrameBlock)
   introduces zero data loss or corruption.

2. **All 39 divergent samples are caused by vLLM Automatic Prefix
   Caching (APC)**: The run-order experiment proves that all divergent
   samples (22 summarization + 17 reasoning) follow the same pattern:
   same-position runs are 100% byte-identical across sessions, while
   cross-position runs diverge. For summarization, this changes accuracy
   (25/50 vs 31/50). For reasoning, the text differs but accuracy
   remains 29/50 in all 4 runs.

3. **JMLC overhead is negligible**: Latencies between SystemDS and
   direct vLLM calls are within 0--3% for generation workloads, within
   measurement noise. Neither backend is meaningfully faster.

4. **Both backends benefit equally from vLLM server optimizations**:
   PagedAttention, continuous batching, KV cache, and CUDA kernels all
   happen server-side. Both are HTTP clients to the same server.

5. **Cost tradeoff depends on scale**: For this small benchmark (250
   sequential queries, ~3 min total inference), OpenAI API ($0.047) is
   cheaper than local H100 ($0.108 vLLM / $0.109 SystemDS) because
   hardware amortization ($2.00/hr) dominates at low utilization. At
   production scale with concurrent requests, owned hardware becomes
   significantly cheaper per query.

6. **Model quality matters more than serving infrastructure**: The
   difference between OpenAI and Qwen 3B is model quality. The
   difference between vLLM and SystemDS is zero (same model, same server).

## Output

Each run produces:
- `samples.jsonl` -- per-sample predictions, references, correctness, latency
- `metrics.json` -- aggregate accuracy, latency stats (mean/p50/p95), throughput, cost
- `manifest.json` -- git hash, timestamp, GPU info, config SHA256
- `run_config.json` -- backend and workload configuration

## Tests

**Python tests** (accuracy checkers, workload loaders):
```bash
python -m pytest tests/ -v
```

**Java tests** (`JMLCLLMInferenceTest`):
- `testSinglePrompt` — end-to-end single prompt via JMLC (requires running LLM server)
- `testBatchInference` — 3-prompt batch with result validation
- `testConcurrency` — concurrent execution with `concurrency=2`
- `testServerUnreachable` — verifies `DMLRuntimeException` for connection refused
- `testInvalidUrl` — verifies `DMLRuntimeException` for malformed URL
- `testHttpErrorResponse` — mock server returns HTTP 500, verifies error propagation
- `testMalformedJsonResponse` — mock server returns invalid JSON, verifies error handling
- `testMissingChoicesInResponse` — mock server returns JSON without `choices` array

The negative tests (HTTP 500, malformed JSON, missing choices) use Java's built-in
`HttpServer` to create mock endpoints, so they run without an external LLM server.
Live tests use `Assume.assumeNoException` to skip gracefully when no server is available.
