#!/bin/bash
# =============================================================================
# LLM Benchmark Runner
# =============================================================================
# Usage: ./scripts/run_all_benchmarks.sh [backend] [model] [options]
#
#   backend: openai, vllm, systemds, gpu, or all (default: gpu)
#   model:   model name/path (required for vllm, systemds)
#
# Options (passed after backend and model):
#   --concurrency N        parallel requests (default: 1)
#   --power-draw-w W       device watts for cost calc (e.g. 350 for H100)
#   --hardware-cost USD    hardware price for amortization (e.g. 30000)
#
# Examples:
#   ./scripts/run_all_benchmarks.sh openai
#   ./scripts/run_all_benchmarks.sh vllm Qwen/Qwen2.5-3B-Instruct
#   ./scripts/run_all_benchmarks.sh systemds Qwen/Qwen2.5-3B-Instruct
#   ./scripts/run_all_benchmarks.sh gpu                    # vllm + systemds
#   ./scripts/run_all_benchmarks.sh all                    # every backend
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        echo -e "${RED}Error: Python not found. Install Python 3.8+${NC}"
        exit 1
    fi
    echo -e "${GREEN}Using: $($PYTHON --version)${NC}"
}

check_dependencies() {
    echo -n "Checking dependencies... "
    if ! $PYTHON -c "import yaml, numpy, psutil, datasets" 2>/dev/null; then
        echo -e "${RED}MISSING${NC}"
        echo -e "${YELLOW}Run: pip install -r requirements.txt${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK${NC}"
}

check_runner() {
    if [ ! -f "runner.py" ]; then
        echo -e "${RED}Error: runner.py not found in $PROJECT_DIR${NC}"
        exit 1
    fi
}

check_python
check_dependencies
check_runner

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKLOADS=("math" "reasoning" "summarization" "json_extraction" "embeddings")

# Default models per backend
default_model_for() {
    case "$1" in
        vllm)      echo "Qwen/Qwen2.5-3B-Instruct" ;;
        systemds)  echo "Qwen/Qwen2.5-3B-Instruct" ;;
        *)         echo "" ;;
    esac
}

# Short name for output directory (e.g. "Qwen/Qwen2.5-3B-Instruct" -> "qwen3b")
short_model_name() {
    local model="$1"
    case "$model" in
        *Qwen2.5-3B*)           echo "qwen3b" ;;
        *Mistral-7B*)           echo "mistral7b" ;;
        *llama3.2*)             echo "llama3.2" ;;
        *Phi-3*)                echo "phi3" ;;
        *phi-2*)                echo "phi2" ;;
        *)                      echo "$(echo "$model" | sed 's|.*/||; s|-Instruct.*||' | tr '[:upper:]' '[:lower:]')" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

BACKEND_ARG="${1:-gpu}"
MODEL_ARG=""
EXTRA_FLAGS=""

# If first arg is a backend, shift it
if [[ -n "$1" ]]; then
    shift
fi

# If next arg is not a flag, it's the model
if [[ "$1" != --* ]] && [[ -n "$1" ]]; then
    MODEL_ARG="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --concurrency)   EXTRA_FLAGS="$EXTRA_FLAGS --concurrency $2"; shift 2 ;;
        --power-draw-w)  EXTRA_FLAGS="$EXTRA_FLAGS --power-draw-w $2"; shift 2 ;;
        --hardware-cost) EXTRA_FLAGS="$EXTRA_FLAGS --hardware-cost $2"; shift 2 ;;
        --electricity-rate) EXTRA_FLAGS="$EXTRA_FLAGS --electricity-rate $2"; shift 2 ;;
        *)               shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Run logic
# ---------------------------------------------------------------------------

FAILED_RUNS=0
TOTAL_RUNS=0
FAILED_LIST=""

run_benchmark() {
    local backend=$1
    local workload=$2
    local model=$3
    local suffix="${4:-}"        # optional dir suffix (e.g. "_c4")
    local extra_run_flags="${5:-}" # optional extra flags for this run

    # Build output directory name: backend_model_workload[_suffix] or backend_workload[_suffix]
    local model_short=""
    if [ -n "$model" ] && [ "$backend" != "openai" ]; then
        model_short="_$(short_model_name "$model")"
    fi
    local output_dir="results/${backend}${model_short}_${workload}${suffix}"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo -e "${YELLOW}  ${backend} / ${workload}${suffix}${model:+ ($model)}${NC}"

    local model_flag=""
    if [ -n "$model" ]; then
        model_flag="--model $model"
    fi

    if $PYTHON runner.py \
        --backend "$backend" \
        --workload "workloads/${workload}/config.yaml" \
        $model_flag \
        $EXTRA_FLAGS $extra_run_flags \
        --out "$output_dir" 2>&1; then
        echo -e "${GREEN}    -> ${output_dir}${NC}"
        return 0
    else
        echo -e "${RED}    FAILED${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${backend}/${workload}${suffix}"
        return 1
    fi
}

run_backend() {
    local backend=$1
    local model=$2
    local suffix="${3:-}"
    local extra_run_flags="${4:-}"
    echo ""
    echo -e "${BLUE}--- ${backend}${suffix} (${model:-default model}) ---${NC}"
    for workload in "${WORKLOADS[@]}"; do
        run_benchmark "$backend" "$workload" "$model" "$suffix" "$extra_run_flags" || true
    done
}

resolve_model() {
    local backend=$1
    local model=$2
    if [ -n "$model" ]; then
        echo "$model"
    else
        default_model_for "$backend"
    fi
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLUE}LLM Benchmark Runner${NC}"
echo -e "${BLUE}=====================${NC}"

case "$BACKEND_ARG" in
    openai)
        run_backend "openai" "$MODEL_ARG"
        ;;
    vllm)
        run_backend "vllm" "$(resolve_model vllm "$MODEL_ARG")"
        ;;
    systemds)
        local_model="$(resolve_model systemds "$MODEL_ARG")"
        run_backend "systemds" "$local_model"
        ;;
    all)
        run_backend "openai" "$MODEL_ARG"
        run_backend "vllm" "$(resolve_model vllm "$MODEL_ARG")"
        local_model="$(resolve_model systemds "$MODEL_ARG")"
        run_backend "systemds" "$local_model"
        ;;
    gpu|*)
        # GPU backends: vLLM + SystemDS with same model for comparison
        local_model="$(resolve_model vllm "$MODEL_ARG")"
        echo -e "${YELLOW}GPU comparison mode: vLLM + SystemDS with ${local_model}${NC}"
        run_backend "vllm" "$local_model"
        run_backend "systemds" "$local_model"
        ;;
esac

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo -e "${BLUE}=====================${NC}"
if [ "$FAILED_RUNS" -eq 0 ]; then
    echo -e "${GREEN}Done: $TOTAL_RUNS/$TOTAL_RUNS passed${NC}"
else
    echo -e "${RED}Done: $FAILED_RUNS/$TOTAL_RUNS failed${NC}"
    echo -e "${RED}Failed:${FAILED_LIST}${NC}"
fi
echo ""
echo "Generate report:"
echo "  $PYTHON scripts/report.py --results-dir results/ --out benchmark_report.html"
echo ""
echo -e "${YELLOW}Reminder: If you're done benchmarking, stop the vLLM server to free GPU memory:${NC}"
echo "  screen -X -S vllm quit"

[ "$FAILED_RUNS" -eq 0 ]
