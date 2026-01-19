# SYSTEMDS-BENCH-GPT

Backend-agnostic benchmarking suite for Large Language Model (LLM) inference systems.

SYSTEMDS-BENCH-GPT is a systems-oriented evaluation harness for comparing local LLM inference runtimes and hosted LLM APIs under controlled workloads, with a focus on **latency, throughput, accuracy, cost, and resource usage**.

---

## Features

- **Multiple Backends**: OpenAI API, Ollama (local), vLLM (GPU server), MLX (Apple Silicon)
- **Real Datasets**: GSM8K (math), XSum (summarization), BoolQ (reasoning), CoNLL-2003 NER (JSON extraction)
- **Comprehensive Metrics**: Latency (mean, p50, p95), throughput, accuracy, cost, tokens, TTFT
- **HTML Reports**: Auto-generated reports with charts and visualizations
- **Extensible**: Easy to add new backends and workloads
- **Reproducible**: Shell scripts for easy benchmarking

---

## Supported Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `openai` | OpenAI API (GPT-4, etc.) | `OPENAI_API_KEY` environment variable |
| `ollama` | Local inference via Ollama | [Ollama](https://ollama.ai) installed and running |
| `vllm` | High-performance inference server | vLLM server running (requires GPU) |
| `mlx` | Apple Silicon optimized | macOS with Apple Silicon, `mlx-lm` package |

---

## Workloads

| Workload | Dataset | Description |
|----------|---------|-------------|
| `math` | GSM8K | Grade school math word problems |
| `summarization` | XSum, CNN/DM | Text summarization |
| `reasoning` | BoolQ, LogiQA | Logical reasoning / QA |
| `json_extraction` | Curated toy | Structured JSON extraction |

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kubraaksux/systemds-bench-gpt.git
cd systemds-bench-gpt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For OpenAI backend
export OPENAI_API_KEY="your-key-here"
```

### 2. Run Benchmarks

**Using shell scripts (recommended):**

```bash
# Run all workloads for a backend
./scripts/run_all_benchmarks.sh openai   # OpenAI API
./scripts/run_all_benchmarks.sh ollama   # Local Ollama
./scripts/run_all_benchmarks.sh mlx      # Apple Silicon
./scripts/run_all_benchmarks.sh all      # All backends
./scripts/run_all_benchmarks.sh          # Local only (ollama + mlx)

# For vLLM (requires GPU): Use Google Colab notebook
# Open notebooks/vllm_colab.ipynb in Google Colab
```

**Using Python directly:**

```bash
# OpenAI API
python runner.py \
  --backend openai \
  --workload workloads/math/config.yaml \
  --out results/openai_math

# Ollama (local)
ollama pull llama3.2
python runner.py \
  --backend ollama \
  --model llama3.2 \
  --workload workloads/math/config.yaml \
  --out results/ollama_math

# MLX (Apple Silicon)
python runner.py \
  --backend mlx \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --workload workloads/summarization/config.yaml \
  --out results/mlx_summarization

# vLLM (requires GPU server)
python runner.py \
  --backend vllm \
  --model microsoft/phi-2 \
  --workload workloads/reasoning/config.yaml \
  --out results/vllm_reasoning
```

### 3. Generate Report

```bash
python scripts/report.py --out benchmark_report.html
open benchmark_report.html
```

---

## Repository Structure

```
systemds-bench-gpt/
├── backends/
│   ├── openai_backend.py   # OpenAI API adapter
│   ├── ollama_backend.py   # Ollama local inference
│   ├── vllm_backend.py     # vLLM server adapter
│   └── mlx_backend.py      # Apple Silicon MLX
├── workloads/
│   ├── math/               # GSM8K dataset (HuggingFace)
│   ├── summarization/      # XSum dataset (HuggingFace)
│   ├── reasoning/          # BoolQ dataset (HuggingFace)
│   └── json_extraction/    # Curated toy dataset (reliable ground truth)
├── scripts/
│   ├── aggregate.py        # CSV aggregation
│   └── report.py           # HTML report generation
├── notebooks/
│   └── vllm_colab.ipynb    # Google Colab for vLLM (GPU)
├── results/                # Benchmark outputs (gitignored)
├── runner.py               # Main benchmark runner
├── requirements.txt        # Python dependencies
├── meeting_notes.md        # Project requirements from Matthias
└── README.md
```


### Latency Metrics
| Metric | Description |
|--------|-------------|
| **Mean latency** | Average response time across all requests |
| **P50 latency** | Median response time (50th percentile) |
| **P95 latency** | Tail latency (95th percentile) |
| **Min/Max** | Range of response times |

### Latency Breakdown (Prefill vs Decode)
| Metric | Description |
|--------|-------------|
| **TTFT** | Time-To-First-Token (prompt processing / prefill phase) |
| **Generation time** | Token decoding time after first token |
| **TTFT %** | Proportion of latency spent in prefill |

### Consistency Metrics
| Metric | Description |
|--------|-------------|
| **Latency std** | Standard deviation of response times |
| **CV (Coefficient of Variation)** | std/mean × 100% - lower = more consistent |

### Throughput
| Metric | Description |
|--------|-------------|
| **Requests/sec** | How many requests can be handled per second |
| **Tokens/sec** | Generation speed (output tokens per second) |
| **ms/token** | Time per output token |

### Accuracy
| Metric | Description |
|--------|-------------|
| **Accuracy mean** | Proportion correct (e.g., 0.80 = 80%) |
| **Accuracy count** | e.g., "8/10" correct |

### Cost Analysis
| Metric | Description |
|--------|-------------|
| **Total cost (USD)** | For API-based backends |
| **Cost per query** | Average cost per inference request |
| **Cost per 1M tokens** | Normalized cost comparison |
| **Cost per correct answer** | Cost efficiency metric |
| **Local backends** | API cost = $0 (hardware costs not estimated) |

### Resource Utilization
| Metric | Description |
|--------|-------------|
| **Memory peak (MB)** | Peak memory usage during inference |
| **CPU usage (%)** | Average CPU utilization |

### Token Accounting
| Metric | Description |
|--------|-------------|
| **Input tokens** | Prompt tokens sent |
| **Output tokens** | Tokens generated |
| **Total tokens** | Sum of input + output |

---

## Datasets

| Workload | Dataset | Source | Samples |
|----------|---------|--------|---------|
| **Math** | GSM8K | HuggingFace `openai/gsm8k` | 10 (configurable) |
| **Reasoning** | BoolQ | HuggingFace `google/boolq` | 10 (configurable) |
| **Summarization** | XSum | HuggingFace `EdinburghNLP/xsum` | 10 (configurable) |
| **JSON Extraction** | Curated toy | Built-in | 10 |

**Why JSON extraction uses a toy dataset:**
- Real JSON datasets (CoNLL-2003 NER, etc.) have inconsistent ground truth
- Toy dataset has clean, verifiable field values for exact accuracy checking
- Enables meaningful accuracy comparison between backends (OpenAI: 90%, local: 60-80%)
- HuggingFace alternatives available via config: `source: ner` or `source: json_struct`

**Fallback behavior:** All loaders include toy datasets as fallback if HuggingFace download fails.

---

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `samples.jsonl` | Per-request outputs with predictions, latencies, tokens |
| `metrics.json` | Aggregated performance metrics |
| `run_config.json` | Exact configuration used |
| `manifest.json` | Timestamp, environment, git hash |

---

## Backend Setup

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
python runner.py --backend openai --workload workloads/math/config.yaml --out results/test
```

### Ollama
```bash
# Install from https://ollama.ai
ollama pull llama3.2
python runner.py --backend ollama --model llama3.2 --workload workloads/math/config.yaml --out results/test
```

### vLLM (requires GPU)

vLLM is the industry-standard for LLM inference serving. Since it requires an NVIDIA GPU, here are your options:

#### Option 1: Google Colab (FREE - Recommended)
The easiest option for students. We provide a ready-to-use notebook:

```bash
# Open in Google Colab:
# notebooks/vllm_colab.ipynb

# Steps:
# 1. Open notebook in Colab
# 2. Runtime → Change runtime type → T4 GPU
# 3. Run all cells
# 4. Download results.zip
# 5. Extract to results/ folder locally
```

#### Option 2: RunPod (~$0.20/hour)
Cheap GPU cloud with easy vLLM setup:

```bash
# 1. Create account at https://runpod.io
# 2. Deploy a GPU pod (RTX 3090 is cheap and good)
# 3. SSH into pod and run:
pip install vllm
python -m vllm.entrypoints.openai.api_server --model microsoft/phi-2 --host 0.0.0.0 --port 8000

# 4. Use ngrok or pod's public URL to connect:
export VLLM_BASE_URL="https://your-pod-url:8000"
python runner.py --backend vllm --model microsoft/phi-2 --workload workloads/math/config.yaml
```

#### Option 3: Lambda Labs (~$0.50/hour)
Professional GPU cloud with better GPUs:

```bash
# 1. Create account at https://lambdalabs.com/cloud
# 2. Launch an A10 or A100 instance
# 3. SSH and run same vLLM commands as above
```

#### Option 4: Local GPU
If you have access to an NVIDIA GPU:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model microsoft/phi-2 --port 8000

# In another terminal:
python runner.py --backend vllm --model microsoft/phi-2 --workload workloads/math/config.yaml --out results/test
```

#### Option 5: University Server
Ask your supervisor for access to university GPU resources.

### MLX (Apple Silicon only)
```bash
pip install mlx mlx-lm
python runner.py --backend mlx --model mlx-community/Phi-3-mini-4k-instruct-4bit --workload workloads/math/config.yaml --out results/test
```

---

## Sample Results

*Latest benchmark results (n=10 samples per workload):*

| Backend | Model | Workload | Accuracy | Latency (p50) | Cost |
|---------|-------|----------|----------|---------------|------|
| OpenAI | gpt-4.1-mini | math | 100% (10/10) | 4.5s | $0.0044 |
| OpenAI | gpt-4.1-mini | reasoning | 60% (6/10) | 4.0s | $0.0043 |
| OpenAI | gpt-4.1-mini | summarization | 100% (10/10) | 1.3s | $0.0015 |
| OpenAI | gpt-4.1-mini | json_extraction | 100% (10/10) | 1.6s | $0.0014 |
| Ollama | llama3.2 | math | 50% (5/10) | 5.9s | $0 |
| Ollama | llama3.2 | reasoning | 50% (5/10) | 4.9s | $0 |
| Ollama | llama3.2 | summarization | 100% (10/10) | 1.0s | $0 |
| Ollama | llama3.2 | json_extraction | 100% (10/10) | 1.5s | $0 |
| vLLM | microsoft/phi-2 | math | 10% (1/10) | 14.8s | $0 |
| vLLM | microsoft/phi-2 | reasoning | 70% (7/10) | 10.4s | $0 |
| vLLM | microsoft/phi-2 | summarization | 90% (9/10) | 2.4s | $0 |
| vLLM | microsoft/phi-2 | json_extraction | 90% (9/10) | 2.1s | $0 |
| MLX | Phi-3-mini-4bit | math | 30% (3/10) | 10.0s | $0 |
| MLX | Phi-3-mini-4bit | reasoning | 50% (5/10) | 10.7s | $0 |
| MLX | Phi-3-mini-4bit | summarization | 100% (10/10) | 2.1s | $0 |
| MLX | Phi-3-mini-4bit | json_extraction | 40% (4/10) | 5.5s | $0 |

**Key Observations:**
- **OpenAI** achieves highest accuracy but incurs API costs
- **Local backends** (Ollama, MLX, vLLM) are free but have lower accuracy on complex tasks
- **Math** is the hardest task for small models (requires multi-step reasoning)
- **Summarization** is easiest (all backends achieve 90-100%)

---

## Extending the Framework

### Adding a New Backend

Create `backends/mybackend_backend.py`:

```python
class MyBackend:
    def __init__(self, model: str):
        self.model = model
    
    def generate(self, prompts: list, config: dict) -> list:
        results = []
        for prompt in prompts:
            # Your inference logic here
            results.append({
                "text": "generated text",
                "latency_ms": 100.0,
                "ttft_ms": 10.0,
                "extra": {
                    "usage": {"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
                    "cost_usd": 0.0
                }
            })
        return results
```

### Adding a New Workload

Create `workloads/myworkload/`:
- `config.yaml` - Configuration
- `loader.py` - `load_samples()` and `accuracy_check()` functions
- `prompt.py` - `make_prompt()` function
- `__init__.py`

---

## Intended Use

This benchmark is intended for:
- Systems research and evaluation
- Inference runtime comparison
- Performance profiling under controlled workloads
- Cost-benefit analysis of local vs. hosted inference

---

## Key Design Decisions

### Why These Backends?
- **OpenAI API**: Cloud-based baseline with state-of-the-art accuracy
- **vLLM**: Industry-standard GPU inference server (as recommended by Prof. Matthias)
- **MLX**: Apple Silicon local inference (for Macs without NVIDIA GPU)
- **Ollama**: Easy-to-use local inference for quick testing

### Why These Datasets?
All datasets are from HuggingFace for reproducibility:
- **GSM8K**: Standard math reasoning benchmark (openai/gsm8k)
- **BoolQ**: Binary reading comprehension (google/boolq)
- **XSum**: News summarization benchmark (EdinburghNLP/xsum)
- **JSON Extraction**: Toy dataset with clean ground truth

### Metrics Philosophy
Following the approach of existing benchmarks (MLPerf, etc.):
- Measure both **accuracy** and **runtime** under controlled workloads
- Report **multiple latency percentiles** (mean, p50, p95, min, max)
- Track **resource usage** (memory, CPU) for local backends
- Calculate **cost efficiency** for cloud APIs

---

## SystemDS Integration (Planned)

This benchmarking framework is designed to eventually evaluate **SystemDS LLM inference** capabilities when they become available. The current implementation uses existing inference systems (vLLM, Ollama, etc.) as baselines.

### Integration Plan

When SystemDS adds LLM inference support, integration will require:

1. **Create `backends/systemds_backend.py`** implementing the standard interface:
   ```python
   class SystemDSBackend:
       def generate(self, prompts: list, config: dict) -> list:
           # Connect to SystemDS inference API
           # Return results with latency, tokens, etc.
   ```

2. **Run comparative benchmarks** against existing baselines (OpenAI, vLLM)

3. **Analyze performance trade-offs** in terms of:
   - Inference latency vs. accuracy
   - Memory efficiency
   - Integration with SystemDS data pipelines

This design ensures the benchmark is ready for SystemDS evaluation while providing immediate value through existing system comparisons.

---

## Future Work

### Planned Enhancements

| Feature | Description | Priority |
|---------|-------------|----------|
| **Concurrent Testing** | Test throughput under load with multiple simultaneous requests | High |
| **SystemDS Backend** | Integrate when SystemDS LLM inference is available | High |
| **Real TTFT for All Backends** | Implement streaming mode for MLX/vLLM to measure actual TTFT | High |
| **GPU Profiling** | GPU memory and utilization via `nvidia-smi` or `pynvml` | High |
| **Larger Models for vLLM** | Test Llama-2-7B or Llama-3-8B for better accuracy (phi-2 is 2.7B) | High |
| **Embeddings Workload** | Add similarity/clustering tasks using embedding APIs | Medium |
| **Hardware Cost Analysis** | Estimate $/query for local backends (electricity, GPU rental) | Medium |
| **Larger Sample Sizes** | Run benchmarks with n=100+ for statistical significance | Medium |
| **HuggingFace JSON Datasets** | Switch JSON extraction from toy to CoNLL-2003 NER or larger datasets | Medium |
| **More Backends** | Hugging Face TGI, llama.cpp, Anthropic Claude | Medium |
| **Code Generation** | Add programming task benchmark (HumanEval, MBPP) | Medium |
| **Model Quantization** | Compare 4-bit vs 8-bit vs full precision performance/accuracy | Medium |
| **Accurate Token Counting** | Use actual tokenizer for Ollama/MLX instead of ~4 chars/token | Medium |
| **Batch Processing** | Compare batch vs. single request performance | Low |
| **Prompt Optimization** | Test different prompt strategies for each workload | Low |

### Metrics Coverage by Backend

Some metrics are estimated rather than precisely measured:

| Metric | OpenAI | Ollama | MLX | vLLM |
|--------|--------|--------|-----|------|
| Latency | ✅ Real | ✅ Real | ✅ Real | ✅ Real |
| TTFT | ✅ Streaming | ✅ Streaming | ⚠️ ~10% est. | ⚠️ ~10% est. |
| Token counts | ✅ API | ⚠️ ~4 chars/tok | ⚠️ ~4 chars/tok | ✅ Real |
| Cost | ✅ API pricing | ⚠️ $0.30/hr est. | ❌ None | ❌ None |
| Memory/CPU | ✅ Local | ✅ Local | ✅ Local | ⚠️ Remote |
| GPU metrics | ❌ N/A | ❌ None | ❌ None | ❌ None |

### Known Limitations

1. **Sequential Requests Only**: Current implementation processes one request at a time. Real production systems handle concurrent requests.

2. **Small Sample Sizes**: Default n=10 for quick testing. Production benchmarks should use n=100+ for reliable statistics.

3. **Limited Model Variety**: Each backend tested with one model. More comprehensive would test multiple model sizes.

4. **No Quantization Comparison**: Could compare 4-bit vs 8-bit vs full precision models.

5. **No Hardware Cost Estimation**: Local backends show $0 or estimated cost. Real hardware has costs (electricity, depreciation, GPU rental).

6. **No GPU Profiling**: GPU memory and utilization not tracked for any backend. Would require `nvidia-smi` or `pynvml` integration.

7. **TTFT Estimation for Non-Streaming**: MLX and vLLM (non-streaming) estimate TTFT as ~10% of total latency rather than measuring actual first-token time.

8. **Token Estimation for Local Backends**: Ollama and MLX estimate token counts (~4 characters per token) rather than using actual tokenizer.

---

## Contact

- Student: Kübra Aksu
- Supervisor: Prof. Dr. Matthias Boehm
- Project: DIA Project - SystemDS Benchmark 
