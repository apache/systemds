#!/bin/bash
# =============================================================================
# SYSTEMDS-BENCH-GPT: Run All Benchmarks
# =============================================================================
# Usage: ./scripts/run_all_benchmarks.sh [backend] [model] [--concurrency N]
#   backend: openai, ollama, mlx, vllm, all, or local (default: local)
#   model:   model name (required for ollama, mlx, vllm)
#
# Examples:
#   ./scripts/run_all_benchmarks.sh openai                          # OpenAI (model from config)
#   ./scripts/run_all_benchmarks.sh ollama llama3.2                 # Ollama with llama3.2
#   ./scripts/run_all_benchmarks.sh mlx mlx-community/Phi-3-mini-4k-instruct-4bit
#   ./scripts/run_all_benchmarks.sh vllm microsoft/phi-2            # vLLM with phi-2
#   ./scripts/run_all_benchmarks.sh                                 # Local backends (ollama, mlx)
#   ./scripts/run_all_benchmarks.sh openai "" --concurrency 4       # Concurrent requests
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

check_python() {
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python is not installed or not in PATH.${NC}"
        echo "Install Python 3.8+ from https://www.python.org/"
        exit 1
    fi
    # Prefer python3 if available
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    else
        PYTHON=python
    fi
    echo -e "${GREEN}Using: $($PYTHON --version)${NC}"
}

check_dependencies() {
    echo -n "Checking dependencies... "
    if ! $PYTHON -c "import yaml, numpy, psutil, datasets" 2>/dev/null; then
        echo -e "${RED}MISSING${NC}"
        echo -e "${YELLOW}Install dependencies: pip install -r requirements.txt${NC}"
        exit 1
    fi
    echo -e "${GREEN}OK${NC}"
}

check_runner() {
    if [ ! -f "runner.py" ]; then
        echo -e "${RED}Error: runner.py not found in $PROJECT_DIR${NC}"
        echo "Make sure you are running this script from the llm-bench directory."
        exit 1
    fi
}

check_python
check_dependencies
check_runner

# Workloads
WORKLOADS=("math" "reasoning" "summarization" "json_extraction" "embeddings")

# Parse arguments
BACKEND_ARG="${1:-local}"
MODEL_ARG="${2:-}"
CONCURRENCY_FLAG=""

# Parse --concurrency flag
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --concurrency)
            CONCURRENCY_FLAG="--concurrency $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Default models per backend (used when no model is specified)
default_model_for() {
    case "$1" in
        ollama) echo "llama3.2" ;;
        mlx)    echo "mlx-community/Phi-3-mini-4k-instruct-4bit" ;;
        vllm)   echo "microsoft/phi-2" ;;
        *)      echo "" ;;
    esac
}

FAILED_RUNS=0
TOTAL_RUNS=0
FAILED_LIST=""

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}SYSTEMDS-BENCH-GPT Benchmark Runner${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

run_benchmark() {
    local backend=$1
    local workload=$2
    local model=$3
    local output_dir="results/${backend}_${workload}_$(date +%Y%m%d_%H%M%S)"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo -e "${YELLOW}Running: ${backend} / ${workload} (model: ${model:-default})${NC}"

    local model_flag=""
    if [ -n "$model" ]; then
        model_flag="--model $model"
    fi

    if $PYTHON runner.py \
        --backend "$backend" \
        --workload "workloads/${workload}/config.yaml" \
        $model_flag \
        $CONCURRENCY_FLAG \
        --out "$output_dir"; then
        echo -e "${GREEN}  Complete: ${output_dir}${NC}"
        return 0
    else
        echo -e "${RED}  Failed: ${backend} / ${workload}${NC}"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_LIST="${FAILED_LIST}\n  - ${backend}/${workload}"
        return 1
    fi
}

run_backend() {
    local backend=$1
    local model=$2
    echo ""
    echo -e "${BLUE}=== Running ${backend} benchmarks (model: ${model:-default}) ===${NC}"

    for workload in "${WORKLOADS[@]}"; do
        run_benchmark "$backend" "$workload" "$model" || true
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

# Determine which backends to run
case "$BACKEND_ARG" in
    openai)
        run_backend "openai" "$MODEL_ARG"
        ;;
    ollama)
        run_backend "ollama" "$(resolve_model ollama "$MODEL_ARG")"
        ;;
    mlx)
        run_backend "mlx" "$(resolve_model mlx "$MODEL_ARG")"
        ;;
    vllm)
        echo -e "${YELLOW}Note: vLLM requires a running server.${NC}"
        run_backend "vllm" "$(resolve_model vllm "$MODEL_ARG")"
        ;;
    all)
        run_backend "openai" "$MODEL_ARG"
        run_backend "ollama" "$(resolve_model ollama "$MODEL_ARG")"
        run_backend "mlx" "$(resolve_model mlx "$MODEL_ARG")"
        ;;
    local|*)
        echo -e "${YELLOW}Running local backends only (ollama, mlx)${NC}"
        echo -e "${YELLOW}Use './scripts/run_all_benchmarks.sh openai' for OpenAI${NC}"
        echo ""
        run_backend "ollama" "$(resolve_model ollama "$MODEL_ARG")"
        run_backend "mlx" "$(resolve_model mlx "$MODEL_ARG")"
        ;;
esac

echo ""
echo -e "${BLUE}=============================================${NC}"
if [ "$FAILED_RUNS" -eq 0 ]; then
    echo -e "${GREEN}ALL $TOTAL_RUNS BENCHMARKS COMPLETE!${NC}"
else
    echo -e "${RED}$FAILED_RUNS/$TOTAL_RUNS BENCHMARKS FAILED${NC}"
    echo -e "${RED}Failed runs:${FAILED_LIST}${NC}"
fi
echo -e "${BLUE}=============================================${NC}"
echo ""
echo "Generate report:"
echo "  python scripts/report.py --out benchmark_report.html"
echo "  open benchmark_report.html"

# Exit with failure if any runs failed
[ "$FAILED_RUNS" -eq 0 ]
