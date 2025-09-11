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

# ==================== PROJECT SETUP & VALIDATION ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
SCRIPT_DIR="$PROJECT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$SCRIPT_DIR/Classification.py"
LOG_DIR="$PROJECT_DIR/logs"

# Ensure all essential directories exist before proceeding.
# The 'set -e' command ensures that the script will exit immediately if any command fails.
set -e

echo "Validating project directory existence..."
# Change directory and check for success
cd "$SCRIPT_DIR" || {
    echo "ERROR: Failed to change directory to $SCRIPT_DIR. Exiting."
    exit 1
}
echo "Project directory validated and entered."

# Create log directory if it doesn't exist.
mkdir -p "$LOG_DIR"

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

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

# This will fail and exit if the venv cannot be activated, thanks to 'set -e'.
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

echo "Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib seaborn tqdm opencv-python-headless pydicom nibabel scikit-learn transformers datasets accelerate Pillow

echo "Packages installed successfully."

# ==================== ENVIRONMENT VARIABLES ====================
echo ""
echo "Setting environment variables..."
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export DATASET_PATH="$PROJECT_DIR/MILK10k_Training_Input"
export GROUNDTRUTH_PATH="$PROJECT_DIR/MILK10k_Training_GroundTruth.csv"
export BASE_OUTPUT_PATH="$PROJECT_DIR"
export CONCEPTCLIP_MODEL_PATH="$PROJECT_DIR/ConceptModel"
export HUGGINGFACE_CACHE_PATH="$PROJECT_DIR/huggingface_cache"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$HUGGINGFACE_CACHE_PATH"
export TRANSFORMERS_CACHE="$HUGGINGFACE_CACHE_PATH"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export OPENCV_IO_ENABLE_OPENEXR=1
echo "Environment variables set."

# ==================== PRE-EXECUTION CHECKS (Critical) ====================
echo "Running pre-execution checks..."
# The 'set -e' command above will handle this, but an explicit check is clearer.
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi
echo "All required paths verified."

# ==================== GPU TEST ====================
echo "Testing GPU availability..."
python -c "
import torch
import sys

if not torch.cuda.is_available():
    print('ERROR: CUDA not available, job will fail')
    sys.exit(1)
else:
    print('✓ GPU available and ready.')
    # A simple tensor operation to verify functionality
    try:
        _ = torch.randn(1, 1).cuda()
    except Exception as e:
        print(f'✗ GPU test failed: {e}')
        sys.exit(1)
"
echo "GPU test passed."

# ==================== RUN CLASSIFICATION ====================
echo ""
echo "Starting classification pipeline..."
echo "=========================================="
# srun ensures the command runs on the allocated compute node and handles process binding.
srun python "$PYTHON_SCRIPT"

# ==================== POST-EXECUTION ====================
# The script will only reach this point if the 'srun' command completes without an error.
echo "=========================================="
echo "Pipeline execution finished."

# Deactivate the virtual environment to clean up the shell.
deactivate

echo "Job completed successfully!"
echo "Job End Time: $(date)"
echo "=========================================="
exit 0
