#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-classification-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyConceptCLIP/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyConceptCLIP/logs/%x-%j.err

echo "=========================================="
echo "MILK10k Classification Pipeline Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# ==================== PROJECT SETUP ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR/OnlyConceptCLIP"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$SCRIPT_DIR/Classification.py"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Navigate to script directory
cd "$SCRIPT_DIR" || {
    echo "ERROR: Failed to change directory to $SCRIPT_DIR. Exiting."
    exit 1
}

# ==================== MODULE SETUP ====================
echo "Loading modules for Narval..."

module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gcc/12.3
module load cuda/12.6
module load cudnn/9.3
module load opencv/4.12.0

echo "Modules loaded successfully."

# ==================== VIRTUAL ENVIRONMENT ====================
echo ""
echo "Activating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment directory not found: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate" || {
    echo "ERROR: Failed to activate virtual environment at $VENV_DIR. Exiting."
    exit 1
}
echo "Virtual environment activated."

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export OUTPUT_PATH="$OUTPUT_DIR"
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"

# Narval optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Ensure offline mode for Hugging Face
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$HUGGINGFACE_CACHE_PATH"
export TRANSFORMERS_CACHE="$HUGGINGFACE_CACHE_PATH"
export HF_HUB_OFFLINE=1

echo "Environment variables set:"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  DATASET_PATH=$DATASET_PATH"
echo "  GROUNDTRUTH_PATH=$GROUNDTRUTH_PATH"
echo "  OUTPUT_PATH=$OUTPUT_PATH"
echo "  CONCEPTCLIP_MODEL_PATH=$CONCEPTCLIP_MODEL_PATH"
echo "  HUGGINGFACE_CACHE_PATH=$HUGGINGFACE_CACHE_PATH"

# ==================== PRE-EXECUTION CHECKS ====================
echo ""
echo "Pre-execution checks:"
echo "===================="

# Check required paths
for path in "$DATASET_PATH" "$GROUNDTRUTH_PATH" "$PYTHON_SCRIPT" "$CONCEPTCLIP_MODEL_PATH" "$HUGGINGFACE_CACHE_PATH"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: Path does not exist: $path"
        exit 1
    fi
done

echo "All required paths verified."

# Create output and log directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" || {
    echo "ERROR: Failed to create directories: $OUTPUT_DIR or $LOG_DIR"
    exit 1
}

# ==================== GPU TEST ====================
echo ""
echo "Testing GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}')
else:
    print('WARNING: CUDA not available, job may fail or run on CPU')
"

# ==================== RUN CLASSIFICATION ====================
echo ""
echo "Starting MILK10k Classification Pipeline..."
echo "=========================================="

# Use srun to ensure GPU access and capture output
srun python "$PYTHON_SCRIPT" 2>&1 | tee "${OUTPUT_DIR}/pipeline_log.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "Post-execution analysis:"
echo "======================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Classification pipeline completed successfully!"
    
    # Check output files
    if [ -f "${OUTPUT_DIR}/reports/classification_results.csv" ]; then
        CLASSIFIED_COUNT=$(wc -l < "${OUTPUT_DIR}/reports/classification_results.csv")
        echo "Classification results: $((CLASSIFIED_COUNT-1)) images processed"
    else
        echo "WARNING: Classification results file not found: ${OUTPUT_DIR}/reports/classification_results.csv"
    fi
    
    if [ -f "${OUTPUT_DIR}/reports/classification_report.json" ]; then
        echo "Classification report generated"
    else
        echo "WARNING: Classification report file not found: ${OUTPUT_DIR}/reports/classification_report.json"
    fi
    
    if [ -d "${OUTPUT_DIR}/processed_images" ]; then
        PROCESSED_COUNT=$(find "${OUTPUT_DIR}/processed_images" -name "*_processed.png" 2>/dev/null | wc -l)
        echo "Processed images saved: $PROCESSED_COUNT files"
    else
        echo "WARNING: Processed images directory not found: ${OUTPUT_DIR}/processed_images"
    fi
    
    if [ -f "${OUTPUT_DIR}/visualizations/classification_summary_plots.png" ]; then
        echo "Summary visualizations created"
    else
        echo "WARNING: Summary visualizations not found: ${OUTPUT_DIR}/visualizations/classification_summary_plots.png"
    fi
    
else
    echo "Classification pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log (${LOG_DIR}/%x-${SLURM_JOB_ID}.err) and pipeline_log.txt (${OUTPUT_DIR}/pipeline_log.txt) for details"
    
    # Basic error checking
    if [ -f "${OUTPUT_DIR}/pipeline_log.txt" ]; then
        echo "Checking log for common issues..."
        if grep -q "CUDA out of memory" "${OUTPUT_DIR}/pipeline_log.txt"; then
            echo "  - GPU memory issue detected. Consider reducing batch size or increasing memory allocation."
        fi
        if grep -q "No module named" "${OUTPUT_DIR}/pipeline_log.txt"; then
            echo "  - Missing Python module detected. Check virtual environment dependencies."
        fi
        if grep -q "FileNotFoundError" "${OUTPUT_DIR}/pipeline_log.txt"; then
            echo "  - File/directory access issue detected. Verify all input paths."
        fi
        if grep -q "NameError" "${OUTPUT_DIR}/pipeline_log.txt"; then
            echo "  - NameError detected. Check variable definitions in $PYTHON_SCRIPT."
        fi
    fi
fi

# ==================== CLEANUP ====================
echo ""
echo "Cleaning up..."
deactivate 2>/dev/null || true
module --force purge

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "Output location: $OUTPUT_DIR"
echo "=========================================="

exit $EXIT_CODE
