#!/bin/bash
# Run all experiments for the mechanistic interpretability paper

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running in conda/venv environment
if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -z "$VIRTUAL_ENV" ]]; then
    print_error "No virtual environment detected. Please activate your environment first."
    exit 1
fi

# Parse command line arguments
USE_WANDB=false
DEBUG_MODE=false
MODEL="gpt2"
GPUS="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wandb] [--debug] [--model MODEL] [--gpus GPU_IDS]"
            exit 1
            ;;
    esac
done

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$GPUS
print_status "Using GPUs: $GPUS"

# Create results directory
RESULTS_DIR="results/full_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
print_status "Results will be saved to: $RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/experiment_log.txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

print_status "Starting full experimental pipeline"
print_status "Model: $MODEL"
print_status "Debug mode: $DEBUG_MODE"
print_status "Using W&B: $USE_WANDB"

# Common arguments
COMMON_ARGS="--model $MODEL"
if [ "$USE_WANDB" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --use_wandb"
fi
if [ "$DEBUG_MODE" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --debug"
fi

# Experiment 1: SAE Architecture Comparison
print_status "Running Experiment 1: SAE Architecture Comparison"
python experiments/01_sae_comparison/run_comparison.py \
    $COMMON_ARGS \
    --layer 6 \
    --n_features 32768 \
    --epochs 10 \
    --batch_size 32

# Check if experiment succeeded
if [ $? -eq 0 ]; then
    print_status "Experiment 1 completed successfully"
else
    print_error "Experiment 1 failed"
    exit 1
fi

# Experiment 2: Scaling Laws Derivation
print_status "Running Experiment 2: Scaling Laws Derivation"
python experiments/02_scaling_laws/derive_scaling_laws.py \
    $COMMON_ARGS \
    --feature_sizes "8192,16384,32768,65536" \
    --layers "4,6,8,10"

if [ $? -eq 0 ]; then
    print_status "Experiment 2 completed successfully"
else
    print_error "Experiment 2 failed"
    exit 1
fi

# Experiment 3: Hybrid Architecture Ablation
print_status "Running Experiment 3: Hybrid Architecture Ablation Study"
python experiments/03_hybrid_architecture/ablation_study.py \
    $COMMON_ARGS \
    --n_features 32768 \
    --ablations "router,utility,projector,dropout"

if [ $? -eq 0 ]; then
    print_status "Experiment 3 completed successfully"
else
    print_error "Experiment 3 failed"
    exit 1
fi

# Experiment 4: Adaptive Sparsity Optimization
print_status "Running Experiment 4: Adaptive Sparsity Optimization"
python experiments/04_optimization/adaptive_sparsity_exp.py \
    $COMMON_ARGS \
    --sparsity_levels "64,128,256,512" \
    --adaptation_strategies "utility,frequency,magnitude"

if [ $? -eq 0 ]; then
    print_status "Experiment 4 completed successfully"
else
    print_error "Experiment 4 failed"
    exit 1
fi

# Generate paper figures
print_status "Generating paper figures"
python scripts/generate_figures.py \
    --results_dir $RESULTS_DIR \
    --output_dir $RESULTS_DIR/paper_figures

# Generate LaTeX tables
print_status "Generating LaTeX tables"
python scripts/generate_tables.py \
    --results_dir $RESULTS_DIR \
    --output_dir $RESULTS_DIR/paper_tables

# Create summary report
print_status "Creating summary report"
python scripts/create_summary_report.py \
    --results_dir $RESULTS_DIR \
    --output_file $RESULTS_DIR/summary_report.pdf

# Compress results
print_status "Compressing results"
tar -czf "$RESULTS_DIR.tar.gz" "$RESULTS_DIR"

print_status "All experiments completed successfully!"
print_status "Results saved to: $RESULTS_DIR"
print_status "Compressed archive: $RESULTS_DIR.tar.gz"

# Print summary statistics
echo ""
echo "=== Experiment Summary ==="
echo "Total runtime: $SECONDS seconds"
echo "Results directory size: $(du -sh $RESULTS_DIR | cut -f1)"
echo "Number of checkpoints: $(find $RESULTS_DIR -name "*.pt" | wc -l)"
echo "Number of figures: $(find $RESULTS_DIR -name "*.png" | wc -l)"
echo "========================="