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
PYTHON_SCRIPT="Classification.py"

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
module load opencv/4.12.0

echo "Modules loaded successfully."

# ==================== VIRTUAL ENVIRONMENT ====================
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
}
echo "Virtual environment activated."

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/groundtruth.csv"
export OUTPUT_PATH="$SCRIPT_DIR/outputs"

# Narval optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Environment variables set."

# ==================== PRE-EXECUTION CHECKS ====================
echo ""
echo "Pre-execution checks:"
echo "===================="

# Check required paths
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/ConceptModel" ]; then
    echo "ERROR: ConceptCLIP model path not found: $PROJECT_DIR/ConceptModel"
    exit 1
fi

echo "All required paths verified."

# Create output directories
mkdir -p "$OUTPUT_PATH"
mkdir -p "$SCRIPT_DIR/logs"

# ==================== GPU TEST ====================
echo ""
echo "Testing GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}')
"

# ==================== RUN CLASSIFICATION ====================
echo ""
echo "Starting MILK10k Classification Pipeline..."
echo "=========================================="

# Use srun to ensure GPU access
srun python "$PYTHON_SCRIPT" 2>&1 | tee "${OUTPUT_PATH}/pipeline_log.txt"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "Post-execution analysis:"
echo "======================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Classification pipeline completed successfully!"
    
    # Check output files
    if [ -f "${OUTPUT_PATH}/reports/classification_results.csv" ]; then
        CLASSIFIED_COUNT=$(wc -l < "${OUTPUT_PATH}/reports/classification_results.csv")
        echo "Classification results: $((CLASSIFIED_COUNT-1)) images processed"
    fi
    
    if [ -f "${OUTPUT_PATH}/reports/classification_report.json" ]; then
        echo "Classification report generated"
    fi
    
    if [ -d "${OUTPUT_PATH}/processed_images" ]; then
        PROCESSED_COUNT=$(find "${OUTPUT_PATH}/processed_images" -name "*_processed.png" 2>/dev/null | wc -l)
        echo "Processed images saved: $PROCESSED_COUNT files"
    fi
    
    if [ -f "${OUTPUT_PATH}/visualizations/classification_summary_plots.png" ]; then
        echo "Summary visualizations created"
    fi
    
else
    echo "Classification pipeline failed with exit code: $EXIT_CODE"
    echo "Check the error log and pipeline_log.txt for details"
    
    # Basic error checking
    if [ -f "${OUTPUT_PATH}/pipeline_log.txt" ]; then
        echo "Checking log for common issues..."
        if grep -q "CUDA out of memory" "${OUTPUT_PATH}/pipeline_log.txt"; then
            echo "  - GPU memory issue detected"
        fi
        if grep -q "No module named" "${OUTPUT_PATH}/pipeline_log.txt"; then
            echo "  - Missing Python module detected"
        fi
        if grep -q "FileNotFoundError" "${OUTPUT_PATH}/pipeline_log.txt"; then
            echo "  - File/directory access issue detected"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Job End Time: $(date)"
echo "Final Exit Code: $EXIT_CODE"
echo "Output location: $OUTPUT_PATH"
echo "=========================================="

exit $EXIT_CODE