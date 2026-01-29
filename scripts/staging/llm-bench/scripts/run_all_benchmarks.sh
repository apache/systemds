#!/bin/bash
# =============================================================================
# SYSTEMDS-BENCH-GPT: Run All Benchmarks
# =============================================================================
# Usage: ./scripts/run_all_benchmarks.sh [backend]
#   backend: openai, ollama, mlx, or all (default: all local backends)
#
# Examples:
#   ./scripts/run_all_benchmarks.sh openai    # Run only OpenAI
#   ./scripts/run_all_benchmarks.sh ollama    # Run only Ollama
#   ./scripts/run_all_benchmarks.sh mlx       # Run only MLX
#   ./scripts/run_all_benchmarks.sh all       # Run all backends
#   ./scripts/run_all_benchmarks.sh           # Run local backends (ollama, mlx)
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

# Workloads
WORKLOADS=("math" "reasoning" "summarization" "json_extraction")

# Parse argument
BACKEND_ARG="${1:-local}"

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}SYSTEMDS-BENCH-GPT Benchmark Runner${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

run_benchmark() {
    local backend=$1
    local workload=$2
    local output_dir="results/${backend}_${workload}_$(date +%Y%m%d_%H%M%S)"
    
    echo -e "${YELLOW}Running: ${backend} / ${workload}${NC}"
    
    if python runner.py \
        --backend "$backend" \
        --workload "workloads/${workload}/config.yaml" \
        --out "$output_dir"; then
        echo -e "${GREEN}✓ Complete: ${output_dir}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: ${backend} / ${workload}${NC}"
        return 1
    fi
}

run_backend() {
    local backend=$1
    echo ""
    echo -e "${BLUE}=== Running ${backend} benchmarks ===${NC}"
    
    for workload in "${WORKLOADS[@]}"; do
        run_benchmark "$backend" "$workload" || true
    done
}

# Determine which backends to run
case "$BACKEND_ARG" in
    openai)
        run_backend "openai"
        ;;
    ollama)
        run_backend "ollama"
        ;;
    mlx)
        run_backend "mlx"
        ;;
    vllm)
        echo -e "${YELLOW}Note: vLLM requires a running server. Use Google Colab notebook instead.${NC}"
        run_backend "vllm"
        ;;
    all)
        run_backend "openai"
        run_backend "ollama"
        run_backend "mlx"
        ;;
    local|*)
        echo -e "${YELLOW}Running local backends only (ollama, mlx)${NC}"
        echo -e "${YELLOW}Use './scripts/run_all_benchmarks.sh openai' for OpenAI${NC}"
        echo ""
        run_backend "ollama"
        run_backend "mlx"
        ;;
esac

echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}BENCHMARKS COMPLETE!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo "Generate report:"
echo "  python scripts/report.py --out benchmark_report.html"
echo "  open benchmark_report.html"
