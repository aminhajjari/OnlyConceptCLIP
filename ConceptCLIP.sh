#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --job-name=OnlyConceptCLIP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=<aminhjjr@gmail@gmail.com>
#SBATCH --time=06:00:00
#SBATCH --output=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/logs/%x-%j.out
#SBATCH --error=/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/logs/%x-%j.err

# ==================== PATHS ====================
PROJECT_DIR="/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_SCRIPT="$PROJECT_DIR/OnlyConceptCLIP/Classification.py"
OUTPUT_PATH="$PROJECT_DIR/OnlyResualts"

# ==================== LOAD MODULES ====================
module load python/3.11.5
module load cuda/12.6

# ==================== VENV ====================
source "$VENV_DIR/bin/activate"

# ==================== ENV VARIABLES ====================
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export BASE_OUTPUT_PATH="$OUTPUT_PATH"
export CUDA_VISIBLE_DEVICES=0

# ==================== RUN ====================
srun python "$PYTHON_SCRIPT"

# ==================== CLEANUP ====================
deactivate
