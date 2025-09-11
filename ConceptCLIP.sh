#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=MILK10k-ConceptCLIP-Enhanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/logs/%x-%j.err

echo "=========================================="
echo "MILK10k ConceptCLIP Enhanced Classification Pipeline Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# ==================== PROJECT SETUP ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$SCRIPT_DIR/Classification.py"  # Your actual Python filename
LOG_DIR="$PROJECT_DIR/logs"
OUTPUT_BASE_DIR="$PROJECT_DIR"  # Where timestamped folders will be created

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
module load cudnn/9.5.1.17
module load opencv/4.12.0

echo "Modules loaded successfully."

# ==================== VIRTUAL ENVIRONMENT ====================
echo ""
echo "Setting up virtual environment..."

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    echo "Installing required packages..."
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support for A100
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install required packages for MILK10k pipeline
    pip install numpy pandas matplotlib seaborn tqdm
    pip install opencv-python-headless
    pip install pydicom nibabel
    pip install scikit-learn
    pip install transformers datasets accelerate
    pip install Pillow
    
    echo "Packages installed successfully."
else
    echo "Activating existing virtual environment..."
    source "$VENV_DIR/bin/activate" || {
        echo "ERROR: Failed to activate virtual environment at $VENV_DIR. Exiting."
        exit 1
    }
fi

echo "Virtual environment activated."

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "Setting environment variables..."

# Python path for ConceptModel imports
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# These paths match what's in your Python code
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export BASE_OUTPUT_PATH="$PROJECT_DIR"  # Your code creates timestamped folders here
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"

# Narval optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Ensure offline mode for Hugging Face (as your code expects)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$HUGGINGFACE_CACHE_PATH"
export TRANSFORMERS_CACHE="$HUGGINGFACE_CACHE_PATH"
export HF_HUB_OFFLINE=1

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # For better error messages

# OpenCV settings to avoid GUI issues
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Environment variables set:"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  DATASET_PATH=$DATASET_PATH"
echo "  GROUNDTRUTH_PATH=$GROUNDTRUTH_PATH"
echo "  BASE_OUTPUT_PATH=$BASE_OUTPUT_PATH"
echo "  CONCEPTCLIP_MODEL_PATH=$CONCEPTCLIP_MODEL_PATH"
echo "  HUGGINGFACE_CACHE_PATH=$HUGGINGFACE_CACHE_PATH"

# ==================== PRE-EXECUTION CHECKS ====================
echo ""
echo "Pre-execution checks:"
echo "===================="

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    echo "Please ensure Classification.py exists in $SCRIPT_DIR"
    exit 1
else
    echo "✓ Python script found: Classification.py"
fi

# Check dataset directory
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset directory not found: $DATASET_PATH"
    exit 1
else
    echo "✓ Dataset directory found"
    IMAGE_COUNT=$(find "$DATASET_PATH" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.dcm" -o -name "*.tiff" \) 2>/dev/null | wc -l)
    echo "  Medical images found: $IMAGE_COUNT"
fi

# Check ground truth file
if [ ! -f "$GROUNDTRUTH_PATH" ]; then
    echo "WARNING: Ground truth file not found: $GROUNDTRUTH_PATH"
    echo "  Pipeline will run but evaluation metrics will be limited"
else
    echo "✓ Ground truth file found"
    SAMPLES=$(wc -l < "$GROUNDTRUTH_PATH")
    echo "  Ground truth samples: $((SAMPLES-1))"
fi

# Check ConceptCLIP model directory
if [ ! -d "$CONCEPTCLIP_MODEL_PATH" ]; then
    echo "ERROR: ConceptCLIP model directory not found: $CONCEPTCLIP_MODEL_PATH"
    exit 1
else
    echo "✓ ConceptCLIP model directory found"
    if [ -f "$CONCEPTCLIP_MODEL_PATH/config.json" ]; then
        echo "  Model config present"
    fi
fi

# Check/create cache directory
if [ ! -d "$HUGGINGFACE_CACHE_PATH" ]; then
    echo "Creating cache directory..."
    mkdir -p "$HUGGINGFACE_CACHE_PATH"
else
    echo "✓ Cache directory exists"
    CACHE_SIZE=$(du -sh "$HUGGINGFACE_CACHE_PATH" 2>/dev/null | cut -f1)
    echo "  Cache size: $CACHE_SIZE"
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR" || {
    echo "ERROR: Failed to create log directory: $LOG_DIR"
    exit 1
}

echo "All required paths verified."

# ==================== GPU TEST ====================
echo ""
echo "Testing GPU availability..."
python -c "
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Memory: {props.total_memory / 1e9:.1f} GB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
    
    # Test GPU allocation
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print('✓ GPU allocation test successful')
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'✗ GPU allocation test failed: {e}')
        sys.exit(1)
else:
    print('ERROR: CUDA not available, job will fail')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU test failed. Exiting."
    exit 1
fi

# ==================== RUN CLASSIFICATION ====================
echo ""
echo "Starting MILK10k ConceptCLIP Enhanced Classification Pipeline..."
echo "=========================================="

# Note what output folder pattern to expect (created by your Python code)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Your code will create folder: MILK10k_ConceptCLIP_Classification_${TIMESTAMP}"
echo ""

# Use srun to ensure GPU access and capture output
srun python "$PYTHON_SCRIPT" 2>&1 | tee "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"
EXIT_CODE=${PIPESTATUS[0]}

# ==================== POST-EXECUTION ANALYSIS ====================
echo ""
echo "Post-execution analysis:"
echo "======================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Classification pipeline completed successfully!"
    
    # Find the latest output folder created by your Python script
    LATEST_OUTPUT=$(ls -td ${OUTPUT_BASE_DIR}/MILK10k_ConceptCLIP_Classification_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_OUTPUT" ]; then
        echo "Output folder created: $LATEST_OUTPUT"
        echo ""
        echo "Checking output structure:"
        
        # Check classification results
        if [ -f "${LATEST_OUTPUT}/reports/classification_results.csv" ]; then
            CLASSIFIED_COUNT=$(wc -l < "${LATEST_OUTPUT}/reports/classification_results.csv")
            echo "✓ Classification results: $((CLASSIFIED_COUNT-1)) images processed"
        else
            echo "✗ Classification results file not found"
        fi
        
        # Check comprehensive report
        if [ -f "${LATEST_OUTPUT}/reports/comprehensive_report.json" ]; then
            echo "✓ Comprehensive report generated"
            
            # Extract and display key metrics
            echo ""
            echo "==== KEY PERFORMANCE METRICS ===="
            python -c "
import json
with open('${LATEST_OUTPUT}/reports/comprehensive_report.json', 'r') as f:
    report = json.load(f)
    
    # Paper comparison metrics
    metrics = report.get('paper_comparison_metrics', {})
    print('Accuracy:             {:.4f}'.format(metrics.get('accuracy', 0)))
    print('Precision (Macro):    {:.4f}'.format(metrics.get('precision_macro', 0)))
    print('Recall (Macro):       {:.4f}'.format(metrics.get('recall_macro', 0)))
    print('F1-Score (Macro):     {:.4f}'.format(metrics.get('f1_score_macro', 0)))
    print('F1-Score (Weighted):  {:.4f}'.format(metrics.get('f1_score_weighted', 0)))
    
    # Dataset info
    dataset = report.get('dataset_info', {})
    print('')
    print('==== DATASET STATISTICS ====')
    print('Total images:         {}'.format(dataset.get('total_images_found', 0)))
    print('With ground truth:    {}'.format(dataset.get('total_with_ground_truth', 0)))
    
    # System info
    system = report.get('system_info', {})
    print('')
    print('==== SYSTEM INFO ====')
    print('Device used:          {}'.format(system.get('device_used', 'unknown')))
    print('Offline mode:         {}'.format(system.get('offline_mode', False)))
"
        fi
        
        # Check evaluation metrics
        echo ""
        echo "Checking evaluation outputs:"
        
        if [ -f "${LATEST_OUTPUT}/evaluation_metrics/detailed_metrics.json" ]; then
            echo "✓ Detailed evaluation metrics saved"
        fi
        
        if [ -f "${LATEST_OUTPUT}/evaluation_metrics/paper_comparison_metrics.csv" ]; then
            echo "✓ Paper comparison metrics CSV generated"
            echo "  You can use this for paper tables!"
        fi
        
        if [ -f "${LATEST_OUTPUT}/evaluation_metrics/classification_report.txt" ]; then
            echo "✓ Classification report text file created"
        fi
        
        # Check processed images
        if [ -d "${LATEST_OUTPUT}/processed_images" ]; then
            PROCESSED_COUNT=$(find "${LATEST_OUTPUT}/processed_images" -name "*_processed.png" 2>/dev/null | wc -l)
            echo "✓ Processed images saved: $PROCESSED_COUNT files"
        fi
        
        # Check visualizations
        echo ""
        echo "Checking visualizations:"
        if [ -f "${LATEST_OUTPUT}/visualizations/comprehensive_evaluation.png" ]; then
            echo "✓ Comprehensive evaluation visualization (9 subplots)"
        fi
        
        if [ -f "${LATEST_OUTPUT}/visualizations/detailed_confusion_matrix.png" ]; then
            echo "✓ Detailed confusion matrix with percentages"
        fi
        
        # List all folders created
        echo ""
        echo "Complete folder structure created:"
        ls -la "$LATEST_OUTPUT/" | grep "^d" | awk '{print "  📁 " $NF}'
        
        # Create job summary
        echo ""
        echo "Creating job summary..."
        cat > "${LATEST_OUTPUT}/slurm_job_${SLURM_JOB_ID}.txt" <<EOF
SLURM Job Summary - MILK10k ConceptCLIP Enhanced Classification
================================================================
Job ID: $SLURM_JOB_ID
Job Name: $SLURM_JOB_NAME
Node: $SLURMD_NODENAME
Script: Classification.py
Start Time: $(date)
Exit Code: $EXIT_CODE

Resource Allocation:
  CPUs: $SLURM_CPUS_PER_TASK
  Memory: 64GB
  GPU: A100

Paths:
  Python Script: $PYTHON_SCRIPT
  Dataset: $DATASET_PATH
  Ground Truth: $GROUNDTRUTH_PATH
  ConceptCLIP Model: $CONCEPTCLIP_MODEL_PATH
  Output Directory: $LATEST_OUTPUT

Pipeline Version: Enhanced v2.0 with Comprehensive Evaluation
Model Type: ConceptCLIP
Classification Mode: Medical Image Classification with Full Metrics
EOF
        
        echo "✓ Job summary saved"
        
        echo ""
        echo "=========================================="
        echo "✅ SUCCESS! All outputs saved to:"
        echo "📁 $LATEST_OUTPUT"
        echo "=========================================="
        
    else
        echo "WARNING: No output folder found matching expected pattern"
        echo "Expected pattern: MILK10k_ConceptCLIP_Classification_*"
        echo "Check if your Python script ran correctly"
    fi
    
else
    echo "✗ Classification pipeline failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check the following logs for details:"
    echo "  1. Error log: ${LOG_DIR}/MILK10k-ConceptCLIP-Enhanced-${SLURM_JOB_ID}.err"
    echo "  2. Execution log: ${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"
    
    # Basic error checking
    if [ -f "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log" ]; then
        echo ""
        echo "Checking for common issues..."
        
        if grep -q "CUDA out of memory" "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"; then
            echo "  ✗ GPU memory issue detected"
            echo "    Solution: Reduce batch size or request more GPU memory"
        fi
        
        if grep -q "No module named" "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"; then
            echo "  ✗ Missing Python module detected"
            MISSING=$(grep "No module named" "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log" | head -1)
            echo "    $MISSING"
            echo "    Solution: Install missing package in virtual environment"
        fi
        
        if grep -q "FileNotFoundError" "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"; then
            echo "  ✗ File not found error"
            echo "    Solution: Check all input paths are correct"
        fi
        
        if grep -q "KeyError" "${LOG_DIR}/pipeline_execution_${SLURM_JOB_ID}.log"; then
            echo "  ✗ KeyError in data processing"
            echo "    Solution: Check ground truth CSV format"
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
if [ -n "$LATEST_OUTPUT" ]; then
    echo "📁 Results saved in: $(basename $LATEST_OUTPUT)"
fi
echo "=========================================="

exit $EXIT_CODE
