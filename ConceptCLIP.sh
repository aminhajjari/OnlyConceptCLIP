#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=OnlyConceptCLIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G  # Increased for ConceptCLIP model
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=aminhjjr@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=03:00:00  # Increased time
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyResualts/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyResualts/logs/%x-%j.err

# ==================== PATHS ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$PROJECT_DIR/OnlyConceptCLIP/Classification.py"  # This path is correct
OUTPUT_PATH="$PROJECT_DIR/OnlyResualts"

# ==================== LOAD MODULES ====================
module load python/3.11.5
module load cuda/12.6
module load opencv/4.12.0

# ==================== VENV ====================
source "$VENV_DIR/bin/activate"

# ==================== ENV VARIABLES ====================
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export BASE_OUTPUT_PATH="$OUTPUT_PATH"
export CUDA_VISIBLE_DEVICES=0

# ConceptCLIP specific variables (matching the corrected code)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ==================== RUN ====================
srun python "$PYTHON_SCRIPT"

# ==================== CLEANUP ====================
deactivate
