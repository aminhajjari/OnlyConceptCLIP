#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=OnlyConceptCLIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G  # Sufficient for ConceptCLIP and large dataset
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=06:00:00  # Increased to 6 hours for safety
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyResualts/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyResualts/logs/%x-%j.err

# ==================== PATHS ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$PROJECT_DIR/OnlyConceptCLIP/Classification.py"  # Adjust if file name/location differs
OUTPUT_PATH="$PROJECT_DIR/OnlyResualts"  # Matches your specified output path

# ==================== LOAD MODULES ====================
module load python/3.11.5
module load cuda/12.6
module load opencv/4.12.0

# ==================== VENV ACTIVATION CHECK ====================
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR. Please set it up with required packages."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ==================== ENV VARIABLES ====================
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ==================== RUN ====================
echo "Starting job at $(date)"
srun python "$PYTHON_SCRIPT"
EXIT_CODE=$?

# ==================== CLEANUP ====================
deactivate
echo "Job finished at $(date) with exit code $EXIT_CODE"

exit $EXIT_CODE
