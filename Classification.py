import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
import nibabel as nib
from collections import Counter, defaultdict
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

print("="*60)
print("✓ All standard imports loaded successfully")
print("="*60)

# Set up Python path for ConceptModel imports
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')

# Import local ConceptCLIP modules directly with error checking
try:
    from ConceptModel.modeling_conceptclip import ConceptCLIP
    from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor
    print("✓ Real ConceptCLIP modules imported successfully")
except ImportError as e:
    raise ImportError(f"❌ Failed to import real ConceptCLIP modules: {e}. Ensure the files modeling_conceptclip.py and preprocessor_conceptclip.py are in /project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel and the path is correct.")

print("✓ SECTION: Module imports completed successfully")
print("-"*60)

# ==================== CONFIGURATION ====================

# DEBUG MODE - Process 50 folders instead of all
DEBUG_MODE = False
MAX_FOLDERS = 50  # Limit to 50 folders as requested

# Dataset paths
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"

# Output path
BASE_OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptCLIP_Only_Results"

# Local model paths
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

print("✓ Configuration paths set")
print(f"✓ DEBUG MODE: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
print(f"✓ Max folders to process: {MAX_FOLDERS}")
print("✓ SECTION: Configuration completed successfully")
print("-"*60)

# ==================== OUTPUT FOLDER SETUP ====================

def setup_output_folder():
    """Setup the output folder structure for ConceptCLIP-only results"""
    print("\n=== SETTING UP OUTPUT FOLDERS ===")
    output_path = Path(BASE_OUTPUT_PATH)
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Main output directory created: {output_path}")
    
    subdirs = [
        "classifications",
        "visualizations",
        "reports",
        "processed_images",
        "evaluation_metrics",
        "debug_logs",
        "comparison_data"
    ]
    
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)
        print(f"✓ Subdirectory created: {subdir}")
    
    debug_log_path = output_path / "debug_logs" / "conceptclip_pipeline_debug.log"
    with open(debug_log_path, 'w') as f:
        f.write(f"ConceptCLIP-only pipeline debug log started at {datetime.now()}\n")
        f.write(f"Debug mode: {DEBUG_MODE}\n")
        f.write(f"Max folders: {MAX_FOLDERS}\n\n")
    
    print(f"✓ Debug log created: {debug_log_path}")
    print("✓ All output folders setup complete")
    print("✓ SECTION: Output folder setup completed successfully")
    print("-"*60)
    return output_path

# ==================== GPU DETECTION AND SETUP ====================

def setup_gpu_environment():
    """Setup GPU environment with proper error handling"""
    print("\n=== GPU ENVIRONMENT SETUP ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
        
        device = f"cuda:{torch.cuda.current_device()}"
        print(f"Using device: {device}")
        
        try:
            test_tensor = torch.randn(10, 10).to(device)
            print("✓ GPU allocation test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            raise RuntimeError(f"❌ GPU allocation test failed: {e}. Check GPU availability or configuration.")
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"
        
        slurm_gpus = os.environ.get('SLURM_GPUS_ON_NODE', 'Not set')
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        print(f"SLURM_GPUS_ON_NODE: {slurm_gpus}")
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    print("✓ GPU environment setup complete")
    print(f"✓ Final device selected: {device}")
    print("✓ SECTION: GPU environment setup completed successfully")
    print("-"*60)
    return device

# ==================== CACHE AND OFFLINE SETUP ====================

def setup_offline_environment(cache_path: str):
    """Setup offline environment for Hugging Face models"""
    print("\n=== OFFLINE ENVIRONMENT SETUP ===")
    
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"✓ Offline mode enabled")
    print(f"✓ Cache directory set to: {cache_path}")
    
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        print(f"✓ Cache directory exists")
        cached_models = list(cache_path_obj.glob("models--*"))
        print(f"✓ Found {len(cached_models)} cached models:")
        for model in cached_models[:5]:
            print(f"   - {model.name}")
        if len(cached_models) > 5:
            print(f"   ... and {len(cached_models) - 5} more")
    else:
        raise FileNotFoundError(f"❌ Cache directory does not exist: {cache_path}")
        
    print("✓ Offline environment setup complete")
    print("✓ SECTION: Offline environment setup completed successfully")
    print("-"*60)

# ==================== LOCAL MODEL LOADING ====================

def load_local_conceptclip_models(model_path: str, cache_path: str, device: str):
    """Load local ConceptCLIP models with offline support"""
    print("\n=== LOADING CONCEPTCLIP MODELS ===")
    
    setup_offline_environment(cache_path)
    
    print(f"Loading ConceptCLIP from local path: {model_path}")
    print(f"Using cache directory: {cache_path}")
    
    # Check if model path exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"❌ Model path does not exist: {model_path}. Ensure the ready model files are downloaded to this location.")
    
    # Load the model
    model = ConceptCLIP.from_pretrained(
        model_path,
        local_files_only=True,
        cache_dir=cache_path,
        trust_remote_code=True
    )
    print("✓ ConceptCLIP model loaded successfully")
    
    processor = ConceptCLIPProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        cache_dir=cache_path,
        trust_remote_code=True
    )
    print("✓ ConceptCLIP processor loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    # Test model with a local MILK10k image to ensure it works
    try:
        sample_image_path = next(Path(DATASET_PATH).rglob("*.jpg"), None)  # Find first .jpg in dataset
        if sample_image_path is None:
            raise FileNotFoundError("❌ No sample .jpg image found in dataset path for testing. Check if dataset contains images.")
        print(f"Testing with local sample image: {sample_image_path}")
        
        image = Image.open(sample_image_path).convert('RGB')
        text = ["a dermatoscopic image showing actinic keratosis", "a dermatoscopic image showing basal cell carcinoma"]  # Sample texts from your prompts
        
        inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        print("✓ Model inference test successful. Sample probabilities:", probs.cpu().numpy().tolist())
    except Exception as e:
        raise RuntimeError(f"❌ Model inference test failed: {e}. The model may not be working correctly with local data. Check if weights are properly loaded.")
    
    print(f"✓ ConceptCLIP loaded and tested successfully on {device}")
    print("✓ SECTION: ConceptCLIP model loading completed successfully")
    print("-"*60)
    return model, processor

# ==================== MILK10k DOMAIN CONFIGURATION ====================

@dataclass
class MedicalDomain:
    """Configuration for MILK10k medical imaging domain"""
    name: str
    image_extensions: List[str]
    text_prompts: List[str]
    label_mappings: Dict[str, str]
    preprocessing_params: Dict
    class_names: List[str]

MILK10K_DOMAIN = MedicalDomain(
    name="milk10k",
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom'],
    text_prompts=[
        'a dermatoscopic image showing actinic keratosis',
        'a dermatoscopic image showing basal cell carcinoma',
        'a dermatoscopic image showing benign proliferation',
        'a dermatoscopic image showing benign keratinocytic lesion',
        'a dermatoscopic image showing dermatofibroma',
        'a dermatoscopic image showing inflammatory condition',
        'a dermatoscopic image showing malignant proliferation',
        'a dermatoscopic image showing melanoma',
        'a dermatoscopic image showing melanocytic nevus',
        'a dermatoscopic image showing squamous cell carcinoma',
        'a dermatoscopic image showing vascular lesion'
    ],
    label_mappings={
        'AKIEC': 'actinic keratosis',
        'BCC': 'basal cell carcinoma',
        'BEN_OTH': 'benign proliferation',
        'BKL': 'benign keratinocytic lesion',
        'DF': 'dermatofibroma',
        'INF': 'inflammatory condition',
        'MAL_OTH': 'malignant proliferation',
        'MEL': 'melanoma',
        'NV': 'melanocytic nevus',
        'SCCKA': 'squamous cell carcinoma',
        'VASC': 'vascular lesion'
    },
    preprocessing_params={'normalize': True, 'enhance_contrast': True},
    class_names=[
        'actinic keratosis',
        'basal cell carcinoma',
        'benign proliferation',
        'benign keratinocytic lesion',
        'dermatofibroma',
        'inflammatory condition',
        'malignant proliferation',
        'melanoma',
        'melanocytic nevus',
        'squamous cell carcinoma',
        'vascular lesion'
    ]
)

print(f"✓ MILK10k domain configured with {len(MILK10K_DOMAIN.class_names)} classes")
print("✓ SECTION: Domain configuration completed successfully")
print("-"*60)

# ==================== ENHANCED EVALUATION METRICS ====================

class ComprehensiveEvaluator:
    """Comprehensive evaluation with all required metrics including ROC-AUC"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        print(f"✓ Evaluator initialized with {len(class_names)} classes")
        
    def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str], 
                                      y_pred_proba: Optional[List[List[float]]] = None) -> Dict:
        print("\n=== CALCULATING COMPREHENSIVE METRICS ===")
        
        print(f"Evaluating {len(y_true)} true labels and {len(y_pred)} predictions")
        
        # Clean and validate input data
        y_true_cleaned = []
        y_pred_cleaned = []
        y_pred_proba_cleaned = []
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            # Skip None values
            if true_label is None or pred_label is None:
                continue
            
            # Check if labels are valid
            if true_label in self.class_names and pred_label in self.class_names:
                y_true_cleaned.append(true_label)
                y_pred_cleaned.append(pred_label)
                
                if y_pred_proba and i < len(y_pred_proba):
                    proba = y_pred_proba[i]
                    # Check for NaN or infinite values in probabilities
                    if proba and len(proba) == len(self.class_names):
                        proba_array = np.array(proba)
                        if np.all(np.isfinite(proba_array)) and not np.any(np.isnan(proba_array)):
                            # Normalize probabilities to sum to 1
                            if np.sum(proba_array) > 0:
                                proba_normalized = proba_array / np.sum(proba_array)
                                y_pred_proba_cleaned.append(proba_normalized.tolist())
                            else:
                                # If all probabilities are 0, create uniform distribution
                                uniform_proba = [1.0 / len(self.class_names)] * len(self.class_names)
                                y_pred_proba_cleaned.append(uniform_proba)
                        else:
                            # Replace NaN/inf with uniform distribution
                            uniform_proba = [1.0 / len(self.class_names)] * len(self.class_names)
                            y_pred_proba_cleaned.append(uniform_proba)
                    else:
                        # Create uniform distribution if probabilities are missing or invalid
                        uniform_proba = [1.0 / len(self.class_names)] * len(self.class_names)
                        y_pred_proba_cleaned.append(uniform_proba)
        
        print(f"Valid samples after cleaning: {len(y_true_cleaned)}")
        
        if len(y_true_cleaned) == 0:
            print("❌ No valid samples for evaluation after cleaning")
            return self._empty_metrics()
        
        # Convert labels to indices
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        y_true_idx = [label_to_idx[label] for label in y_true_cleaned]
        y_pred_idx = [label_to_idx[label] for label in y_pred_cleaned]
        
        # Find unique classes present in the data
        unique_classes = sorted(list(set(y_true_idx + y_pred_idx)))
        print(f"Unique classes found in data: {len(unique_classes)} out of {len(self.class_names)}")
        print(f"Present classes: {[self.class_names[i] for i in unique_classes]}")
        
        if y_pred_proba_cleaned:
            print(f"✓ Clean probability data available for ROC-AUC calculation")
        else:
            y_pred_proba_cleaned = None
            print("⚠️ No valid probability data available - ROC-AUC will be 0")
        
        print("Calculating basic metrics...")
        accuracy = accuracy_score(y_true_idx, y_pred_idx)
        
        precision_macro = precision_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
        
        print(f"✓ Basic metrics calculated - Accuracy: {accuracy:.4f}")
        
        print("Calculating per-class metrics...")
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true_idx, y_pred_idx, labels=unique_classes, zero_division=0
        )
        
        print("Calculating ROC-AUC metrics...")
        roc_auc_metrics = self._calculate_roc_auc(y_true_idx, y_pred_proba_cleaned, unique_classes)
        
        print("Creating confusion matrix...")
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=unique_classes)
        
        print("Generating classification report...")
        # Only use class names for classes that are actually present
        present_class_names = [self.class_names[i] for i in unique_classes]
        class_report = classification_report(
            y_true_idx, y_pred_idx,
            labels=unique_classes,  # Specify the actual labels present
            target_names=present_class_names,  # Only names for present classes
            output_dict=True, zero_division=0
        )
        
        # Create per-class metrics dictionary for ALL classes (including absent ones)
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i in unique_classes:
                # Class is present in data
                class_idx_in_unique = unique_classes.index(i)
                per_class_metrics[class_name] = {
                    'precision': float(precision_per_class[class_idx_in_unique]),
                    'recall': float(recall_per_class[class_idx_in_unique]),
                    'f1_score': float(f1_per_class[class_idx_in_unique]),
                    'support': int(support_per_class[class_idx_in_unique]),
                    'roc_auc': roc_auc_metrics['per_class'].get(i, 0.0)
                }
            else:
                # Class is absent from data
                per_class_metrics[class_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': 0,
                    'roc_auc': 0.0
                }
        
        metrics = {
            'overview': {
                'total_samples': len(y_true),
                'valid_samples': len(y_true_cleaned),
                'unique_classes_present': len(unique_classes),
                'total_classes_expected': len(self.class_names),
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'roc_auc_ovr_macro': roc_auc_metrics['ovr_macro'],
                'roc_auc_ovr_weighted': roc_auc_metrics['ovr_weighted'],
                'roc_auc_ovo_macro': roc_auc_metrics['ovo_macro'],
                'roc_auc_ovo_weighted': roc_auc_metrics['ovo_weighted']
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_names': self.class_names,
            'present_classes': present_class_names,
            'unique_class_indices': unique_classes,
            'roc_curves': roc_auc_metrics.get('curves', {})
        }
        
        print("✓ Comprehensive metrics calculation complete")
        print("✓ SECTION: Metrics calculation completed successfully")
        print("-"*60)
        return metrics
        
    def _calculate_roc_auc(self, y_true: List[int], y_pred_proba: Optional[List[List[float]]], 
                          unique_classes: List[int]) -> Dict:
        if y_pred_proba is None or len(y_pred_proba) == 0:
            print("⚠️ No probability data - returning zero ROC-AUC scores")
            return {
                'ovr_macro': 0.0,
                'ovr_weighted': 0.0,
                'ovo_macro': 0.0,
                'ovo_weighted': 0.0,
                'per_class': {i: 0.0 for i in range(len(self.class_names))},
                'curves': {}
            }
        
        try:
            print("Calculating ROC-AUC scores...")
            y_true_array = np.array(y_true)
            y_pred_proba_array = np.array(y_pred_proba)
            
            # Validate probability array
            if np.any(np.isnan(y_pred_proba_array)) or np.any(np.isinf(y_pred_proba_array)):
                print("⚠️ Found NaN or Inf in probability array - cleaning...")
                # Replace NaN/Inf with uniform distribution
                for i in range(y_pred_proba_array.shape[0]):
                    if np.any(np.isnan(y_pred_proba_array[i])) or np.any(np.isinf(y_pred_proba_array[i])):
                        y_pred_proba_array[i] = np.ones(y_pred_proba_array.shape[1]) / y_pred_proba_array.shape[1]
            
            # For multiclass ROC-AUC with missing classes, we need to be careful
            if len(unique_classes) < 2:
                print("⚠️ Need at least 2 classes for ROC-AUC calculation")
                return {
                    'ovr_macro': 0.0,
                    'ovr_weighted': 0.0,
                    'ovo_macro': 0.0,
                    'ovo_weighted': 0.0,
                    'per_class': {i: 0.0 for i in range(len(self.class_names))},
                    'curves': {}
                }
            
            # Binarize for One-vs-Rest (OvR) calculation
            y_true_binarized = label_binarize(y_true_array, classes=unique_classes)
            
            # Handle single class case
            if y_true_binarized.shape[1] == 1:
                print("⚠️ Only one class present - cannot calculate ROC-AUC")
                return {
                    'ovr_macro': 0.0,
                    'ovr_weighted': 0.0,
                    'ovo_macro': 0.0,
                    'ovo_weighted': 0.0,
                    'per_class': {i: 0.0 for i in range(len(self.class_names))},
                    'curves': {}
                }
            
            # Use only columns corresponding to unique classes
            if y_pred_proba_array.shape[1] == len(self.class_names):
                y_pred_proba_filtered = y_pred_proba_array[:, unique_classes]
            else:
                y_pred_proba_filtered = y_pred_proba_array
            
            # Calculate OvR ROC-AUC
            try:
                roc_auc_ovr_macro = roc_auc_score(y_true_binarized, y_pred_proba_filtered, 
                                                average='macro', multi_class='ovr')
                roc_auc_ovr_weighted = roc_auc_score(y_true_binarized, y_pred_proba_filtered, 
                                                   average='weighted', multi_class='ovr')
            except ValueError as e:
                print(f"⚠️ OvR ROC-AUC calculation failed: {e}")
                roc_auc_ovr_macro = 0.0
                roc_auc_ovr_weighted = 0.0
            
            # Calculate OvO ROC-AUC using the original arrays
            try:
                roc_auc_ovo_macro = roc_auc_score(y_true_array, y_pred_proba_filtered, 
                                                average='macro', multi_class='ovo')
                roc_auc_ovo_weighted = roc_auc_score(y_true_array, y_pred_proba_filtered, 
                                                   average='weighted', multi_class='ovo')
            except ValueError as e:
                print(f"⚠️ OvO ROC-AUC calculation failed: {e}")
                roc_auc_ovo_macro = 0.0
                roc_auc_ovo_weighted = 0.0
            
            print(f"✓ ROC-AUC OvR Macro: {roc_auc_ovr_macro:.4f}")
            
            # Calculate per-class AUC
            per_class_auc = {i: 0.0 for i in range(len(self.class_names))}
            curves = {}
            
            for idx, class_idx in enumerate(unique_classes):
                if class_idx < len(self.class_names):
                    class_name = self.class_names[class_idx]
                    try:
                        if idx < y_true_binarized.shape[1] and idx < y_pred_proba_filtered.shape[1]:
                            if np.sum(y_true_binarized[:, idx]) > 0:  # Check if class has positive samples
                                y_true_class = y_true_binarized[:, idx]
                                y_score_class = y_pred_proba_filtered[:, idx]
                                
                                # Check for NaN in class data
                                if not (np.any(np.isnan(y_true_class)) or np.any(np.isnan(y_score_class))):
                                    fpr, tpr, thresholds = roc_curve(y_true_class, y_score_class)
                                    auc_score = auc(fpr, tpr)
                                    per_class_auc[class_idx] = float(auc_score)
                                    curves[class_name] = {
                                        'fpr': fpr.tolist(),
                                        'tpr': tpr.tolist(),
                                        'auc': float(auc_score)
                                    }
                                else:
                                    print(f"⚠️ NaN found in class data for {class_name}")
                                    per_class_auc[class_idx] = 0.0
                            else:
                                per_class_auc[class_idx] = 0.0
                        else:
                            per_class_auc[class_idx] = 0.0
                    except Exception as e:
                        print(f"⚠️ Error calculating AUC for class {class_name}: {e}")
                        per_class_auc[class_idx] = 0.0
            
            print("✓ Per-class ROC-AUC calculated")
            print("✓ SECTION: ROC-AUC calculation completed successfully")
            
            return {
                'ovr_macro': float(roc_auc_ovr_macro),
                'ovr_weighted': float(roc_auc_ovr_weighted),
                'ovo_macro': float(roc_auc_ovo_macro),
                'ovo_weighted': float(roc_auc_ovo_weighted),
                'per_class': per_class_auc,
                'curves': curves
            }
            
        except Exception as e:
            print(f"❌ Error calculating ROC-AUC: {e}")
            import traceback
            traceback.print_exc()
            return {
                'ovr_macro': 0.0,
                'ovr_weighted': 0.0,
                'ovo_macro': 0.0,
                'ovo_weighted': 0.0,
                'per_class': {i: 0.0 for i in range(len(self.class_names))},
                'curves': {}
            }
    
    def _empty_metrics(self) -> Dict:
        return {
            'overview': {
                'total_samples': 0,
                'valid_samples': 0,
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'precision_weighted': 0.0,
                'recall_weighted': 0.0,
                'f1_weighted': 0.0,
                'roc_auc_ovr_macro': 0.0,
                'roc_auc_ovr_weighted': 0.0,
                'roc_auc_ovo_macro': 0.0,
                'roc_auc_ovo_weighted': 0.0
            },
            'per_class_metrics': {},
            'confusion_matrix': [],
            'classification_report': {},
            'class_names': self.class_names,
            'roc_curves': {}
        }

print("✓ SECTION: Evaluator class definition completed successfully")
print("-"*60)

# ==================== MAIN PIPELINE CLASS ====================

class MILK10kConceptCLIPPipeline:
    """MILK10k ConceptCLIP-only classification pipeline for comparison"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, 
                 conceptclip_model_path: str = None, cache_path: str = None):
        print("\n" + "="*80)
        print(" MILK10K CONCEPTCLIP-ONLY CLASSIFICATION PIPELINE INITIALIZATION ".center(80))
        print("="*80)
        
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        
        self.debug_mode = DEBUG_MODE
        self.max_folders = MAX_FOLDERS if DEBUG_MODE else None
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Ground truth path: {self.groundtruth_path}")
        print(f"ConceptCLIP model path: {self.conceptclip_model_path}")
        print(f"Cache path: {self.cache_path}")
        print(f"DEBUG MODE: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        print(f"Max folders to process: {self.max_folders if self.max_folders else 'All'}")
        
        self.output_path = setup_output_folder()
        print(f"Output folder: {self.output_path}")
        
        self.device = setup_gpu_environment()
        print(f"Device: {self.device}")
        
        self.evaluator = ComprehensiveEvaluator(self.domain.class_names)
        print("✓ Evaluator initialized")
        
        self._load_models()
        self._load_ground_truth()
        
        print("✓ MILK10k ConceptCLIP-only pipeline initialization complete")
        print("✓ SECTION: Pipeline initialization completed successfully")
        print("-"*60)
        
    def _load_models(self):
        """Load local ConceptCLIP model"""
        print("\n=== LOADING MODELS ===")
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.cache_path, self.device
        )
        print("✓ Models loaded successfully")
        print("✓ SECTION: Model loading completed successfully")
        
    def _load_ground_truth(self):
        """Load ground truth annotations"""
        print("\n=== LOADING GROUND TRUTH ===")
        
        if os.path.exists(self.groundtruth_path):
            try:
                self.ground_truth = pd.read_csv(self.groundtruth_path)
                print(f"✓ Ground truth loaded: {len(self.ground_truth)} samples")
                print(f"Columns: {list(self.ground_truth.columns)}")
                
                if 'lesion_id' in self.ground_truth.columns:
                    print("✓ lesion_id column found for folder matching")
                    print("Sample ground truth data:")
                    print(self.ground_truth.head())
                    
                    expected_cols = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
                    found_cols = [col for col in expected_cols if col in self.ground_truth.columns]
                    print(f"Expected diagnostic columns found: {found_cols}")
                    
                    print("\nOne-hot diagnostic class distribution:")
                    for col in found_cols:
                        if col in self.ground_truth.columns:
                            count = (self.ground_truth[col] == 1.0).sum()
                            mapped_label = self.domain.label_mappings.get(col, col)
                            print(f"  {col} ({mapped_label}): {count} samples")
                else:
                    raise ValueError("❌ lesion_id column not found! Required for folder-level matching.")
                    
                print("✓ SECTION: Ground truth loading completed successfully")
                print("-"*60)
            except Exception as e:
                raise RuntimeError(f"❌ Error loading ground truth: {e}")
        else:
            raise FileNotFoundError(f"❌ Ground truth file not found: {self.groundtruth_path}")
    
    def get_image_files(self) -> List[Path]:
        """Get image files from the first 50 folders"""
        print("\n=== COLLECTING IMAGE FILES FROM FOLDERS ===")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ Dataset path does not exist: {self.dataset_path}")
        
        image_files = []
        lesion_folders = sorted([f for f in self.dataset_path.iterdir() if f.is_dir() and not f.name.startswith('.')])
        print(f"Found {len(lesion_folders)} folders")
        
        if len(lesion_folders) > self.max_folders:
            print(f"⚠️ Limiting to {self.max_folders} folders")
            lesion_folders = lesion_folders[:self.max_folders]
        
        for folder in lesion_folders:
            for ext in self.domain.image_extensions:
                files = list(folder.glob(f"*{ext}"))
                image_files.extend(files)
                if not files:
                    print(f"⚠️ No images with extension {ext} found in {folder.name}")
                else:
                    print(f"Found {len(files)} images with extension {ext} in {folder.name}")
        
        if not image_files:
            raise ValueError("❌ No image files found in any folders. Check dataset path and extensions.")
        
        print(f"Total images found: {len(image_files)} across {len(lesion_folders)} folders")
        print("✓ SECTION: Image file collection completed successfully")
        print("-"*60)
        return image_files
    
    def preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """Preprocess a single image"""
        try:
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                ds = pydicom.dcmread(str(image_path))
                img_array = ds.pixel_array
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                if len(img_array.shape) == 2:
                    image = Image.fromarray(img_array, mode='L').convert('RGB')
                else:
                    image = Image.fromarray(img_array, mode='RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            
            # Ensure image is valid
            if image.size[0] == 0 or image.size[1] == 0:
                print(f"⚠️ Invalid image size: {image.size}")
                return None
            
            if self.domain.preprocessing_params.get('enhance_contrast', False):
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
            
            if self.domain.preprocessing_params.get('normalize', True):
                img_array = np.array(image)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    try:
                        import cv2
                        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        image = Image.fromarray(img_array)
                    except:
                        # If cv2 processing fails, skip normalization
                        print(f"⚠️ Normalization failed for {image_path}")
                        pass
            
            return image
        except Exception as e:
            print(f"❌ Error preprocessing {image_path}: {e}")
            return None
    
    def classify_with_conceptclip(self, image: Image.Image) -> Tuple[str, List[float]]:
        """Classify image using ConceptCLIP with robust error handling"""
        try:
            inputs = self.conceptclip_processor(
                images=image,
                text=self.domain.text_prompts,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.conceptclip_model(**inputs)
                
                image_features = outputs['image_features']
                text_features = outputs['text_features']
                logit_scale = outputs.get('logit_scale', torch.tensor(2.6592).to(self.device))
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate logits and probabilities
                logits = (image_features @ text_features.T) * logit_scale.exp()
                probs = torch.softmax(logits, dim=-1)
                
                pred_idx = probs.argmax(dim=-1).item()
                pred_class = self.domain.class_names[pred_idx]
                pred_probs = probs.squeeze().cpu().numpy()
                
                # Ensure pred_probs is a list and handle different shapes
                if pred_probs.ndim == 0:
                    pred_probs = [float(pred_probs)]
                else:
                    pred_probs = pred_probs.tolist()
                
                # Ensure we have the right number of probabilities
                if len(pred_probs) != len(self.domain.class_names):
                    raise ValueError(f"❌ Unexpected number of probabilities: {len(pred_probs)} (expected {len(self.domain.class_names)})")
                
                # Check for NaN or infinite values and raise error
                pred_probs_array = np.array(pred_probs)
                if np.any(np.isnan(pred_probs_array)) or np.any(np.isinf(pred_probs_array)):
                    raise ValueError("❌ NaN or Inf detected in predictions")
                
                return pred_class, pred_probs
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"❌ Error in ConceptCLIP classification: {e}")
    
    def get_ground_truth_label(self, folder_name: str) -> Optional[str]:
        """Get ground truth label for a folder (lesion_id is folder-level)"""
        if self.ground_truth is None:
            return None
        
        # Use folder name as lesion_id
        row = self.ground_truth[self.ground_truth['lesion_id'] == folder_name]
        if row.empty:
            return None
        
        row = row.iloc[0]
        
        # Find the label with value 1.0
        for col, label in self.domain.label_mappings.items():
            if col in self.ground_truth.columns and float(row[col]) == 1.0:
                return label
        
        return None
    
    def run_classification(self):
        """Run the complete ConceptCLIP-only classification pipeline"""
        print("\n" + "="*80)
        print(" MILK10K CONCEPTCLIP-ONLY CLASSIFICATION PIPELINE ".center(80))
        print(f" PROCESSING {self.max_folders} FOLDERS ".center(80))
        print("="*80)
        
        image_files = self.get_image_files()
        
        if not image_files:
            raise ValueError("❌ No image files found!")
        
        print(f"\n✓ Processing {len(image_files)} images across {len(set(f.parent.name for f in image_files))} folders...")
        
        results = []
        all_true_labels = []
        all_pred_labels = []
        all_pred_probas = []
        
        successful_classifications = 0
        failed_classifications = 0
        matched_ground_truth = 0
        
        progress_bar = tqdm(image_files, desc="Classifying images")
        
        for idx, image_path in enumerate(progress_bar):
            if (idx + 1) % 50 == 0:
                progress_bar.set_postfix({
                    'Success': successful_classifications,
                    'Failed': failed_classifications,
                    'GT_Match': matched_ground_truth
                })
            
            image_id = image_path.stem
            folder_name = image_path.parent.name
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                failed_classifications += 1
                continue
            
            # Classify with ConceptCLIP
            try:
                pred_class, pred_probs = self.classify_with_conceptclip(image)
                successful_classifications += 1
            except Exception as e:
                print(f"❌ Classification failed for {image_id}: {e}")
                failed_classifications += 1
                continue
            
            # Get ground truth label (folder-level)
            true_class = self.get_ground_truth_label(folder_name)
            if true_class:
                matched_ground_truth += 1
            
            result = {
                'image_id': image_id,
                'image_path': str(image_path.relative_to(self.dataset_path)),
                'folder_id': folder_name,
                'predicted_class': pred_class,
                'prediction_probabilities': pred_probs,
                'true_class': true_class,
                'max_probability': max(pred_probs) if pred_probs else 0.0,
                'prediction_confidence': max(pred_probs) if pred_probs else 0.0,
                'method': 'ConceptCLIP-only',
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Collect data for evaluation
            if true_class and pred_class:
                all_true_labels.append(true_class)
                all_pred_labels.append(pred_class)
                all_pred_probas.append(pred_probs)
        
        progress_bar.close()
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Successful classifications: {successful_classifications}")
        print(f"Failed classifications: {failed_classifications}")
        print(f"Images with ground truth: {matched_ground_truth}")
        print(f"Success rate: {successful_classifications/len(image_files)*100:.1f}%" if image_files else "N/A")
        print(f"Ground truth coverage: {matched_ground_truth/len(image_files)*100:.1f}%" if image_files else "N/A")
        
        # Save results
        self.save_results(results)
        
        # Evaluate if we have ground truth data
        if all_true_labels:
            print(f"\n✓ Evaluating {len(all_true_labels)} samples with ground truth...")
            metrics = self.evaluator.calculate_comprehensive_metrics(
                all_true_labels, all_pred_labels, all_pred_probas
            )
            self.save_metrics(metrics)
            self.save_comparison_data(results, metrics)
            self.print_summary_metrics(metrics)
        else:
            print("\n⚠️ No ground truth labels found - skipping evaluation")
            self.save_basic_comparison_data(results)
        
        print("\n" + "="*80)
        print(" CONCEPTCLIP-ONLY PIPELINE EXECUTION COMPLETE ".center(80))
        print("="*80)
    
    def save_results(self, results: List[Dict]):
        """Save classification results"""
        print("\n=== SAVING RESULTS ===")
        
        df = pd.DataFrame(results)
        csv_path = self.output_path / "classifications" / "conceptclip_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to CSV: {csv_path}")
        
        json_path = self.output_path / "classifications" / "conceptclip_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to JSON: {json_path}")
        print("✓ SECTION: Results saving completed successfully")
    
    def save_metrics(self, metrics: Dict):
        """Save evaluation metrics"""
        print("\n=== SAVING METRICS ===")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        metrics_clean = convert_numpy_types(metrics)
        
        metrics_path = self.output_path / "evaluation_metrics" / "conceptclip_comprehensive_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        print(f"✓ Comprehensive metrics saved: {metrics_path}")
        
        summary_path = self.output_path / "evaluation_metrics" / "conceptclip_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("MILK10k ConceptCLIP-only Classification Results Summary\n")
            f.write("="*60 + "\n\n")
            
            overview = metrics['overview']
            f.write(f"Total samples: {overview['total_samples']}\n")
            f.write(f"Valid samples: {overview['valid_samples']}\n")
            f.write(f"Accuracy: {overview['accuracy']:.4f}\n")
            f.write(f"Precision (macro): {overview['precision_macro']:.4f}\n")
            f.write(f"Recall (macro): {overview['recall_macro']:.4f}\n")
            f.write(f"F1-score (macro): {overview['f1_macro']:.4f}\n")
            f.write(f"ROC-AUC OvR (macro): {overview['roc_auc_ovr_macro']:.4f}\n")
            f.write(f"ROC-AUC OvR (weighted): {overview['roc_auc_ovr_weighted']:.4f}\n")
            f.write(f"ROC-AUC OvO (macro): {overview['roc_auc_ovo_macro']:.4f}\n")
            f.write(f"ROC-AUC OvO (weighted): {overview['roc_auc_ovo_weighted']:.4f}\n\n")
            
            f.write("Per-class Metrics:\n")
            f.write("-" * 40 + "\n")
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {class_metrics['f1_score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")
                f.write(f"  ROC-AUC: {class_metrics['roc_auc']:.4f}\n\n")
        
        print(f"✓ Summary saved: {summary_path}")
        print("✓ SECTION: Metrics saving completed successfully")
    
    def save_comparison_data(self, results: List[Dict], metrics: Dict):
        """Save comparison data for analysis"""
        print("\n=== SAVING COMPARISON DATA ===")
        
        comparison_data = {
            'metadata': {
                'method': 'ConceptCLIP-only',
                'total_images': len(results),
                'total_folders': self.max_folders,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'type': 'ConceptCLIP',
                    'model_path': str(self.conceptclip_model_path),
                    'device': str(self.device)
                }
            },
            'performance_summary': metrics['overview'],
            'per_class_performance': metrics['per_class_metrics'],
            'detailed_results': results[:100],  # Save first 100 detailed results
            'class_distribution': self._calculate_class_distribution(results),
            'confidence_statistics': self._calculate_confidence_stats(results)
        }
        
        comparison_path = self.output_path / "comparison_data" / "conceptclip_comparison_data.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        print(f"✓ Comparison data saved: {comparison_path}")
        
        print("✓ SECTION: Comparison data saving completed successfully")
    
    def save_basic_comparison_data(self, results: List[Dict]):
        """Save basic comparison data when no ground truth is available"""
        print("\n=== SAVING BASIC COMPARISON DATA ===")
        
        comparison_data = {
            'metadata': {
                'method': 'ConceptCLIP-only',
                'total_images': len(results),
                'total_folders': self.max_folders,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'type': 'ConceptCLIP',
                    'model_path': str(self.conceptclip_model_path),
                    'device': str(self.device)
                },
                'note': 'No ground truth available - predictions only'
            },
            'prediction_summary': {
                'total_predictions': len(results),
                'successful_predictions': len([r for r in results if r['predicted_class']]),
                'average_confidence': float(np.mean([r['prediction_confidence'] for r in results if r['prediction_confidence']])) if results else 0.0,
                'prediction_distribution': self._calculate_class_distribution(results)
            },
            'detailed_results': results[:100],  # Save first 100 detailed results
            'confidence_statistics': self._calculate_confidence_stats(results)
        }
        
        comparison_path = self.output_path / "comparison_data" / "conceptclip_basic_comparison_data.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        print(f"✓ Basic comparison data saved: {comparison_path}")
        
        print("✓ SECTION: Basic comparison data saving completed successfully")
    
    def _calculate_class_distribution(self, results: List[Dict]) -> Dict:
        """Calculate class distribution from results"""
        pred_counts = Counter([r['predicted_class'] for r in results if r['predicted_class']])
        true_counts = Counter([r['true_class'] for r in results if r['true_class']])
        
        return {
            'predicted_distribution': dict(pred_counts),
            'true_distribution': dict(true_counts) if true_counts else {}
        }
    
    def _calculate_confidence_stats(self, results: List[Dict]) -> Dict:
        """Calculate confidence statistics"""
        confidences = [r['prediction_confidence'] for r in results if r['prediction_confidence'] and not np.isnan(r['prediction_confidence'])]
        
        if not confidences:
            return {
                'mean_confidence': 0.0,
                'median_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'q25_confidence': 0.0,
                'q75_confidence': 0.0
            }
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'q25_confidence': float(np.percentile(confidences, 25)),
            'q75_confidence': float(np.percentile(confidences, 75))
        }
    
    def print_summary_metrics(self, metrics: Dict):
        """Print summary of evaluation metrics"""
        print("\n" + "="*60)
        print(" EVALUATION SUMMARY ".center(60))
        print("="*60)
        
        overview = metrics['overview']
        print(f"Total samples: {overview['total_samples']}")
        print(f"Valid samples for evaluation: {overview['valid_samples']}")
        print(f"Classes present in data: {overview.get('unique_classes_present', 'N/A')}/{overview.get('total_classes_expected', len(self.domain.class_names))}")
        print(f"Accuracy: {overview['accuracy']:.4f}")
        print(f"Precision (macro): {overview['precision_macro']:.4f}")
        print(f"Recall (macro): {overview['recall_macro']:.4f}")
        print(f"F1-score (macro): {overview['f1_macro']:.4f}")
        print(f"F1-score (weighted): {overview['f1_weighted']:.4f}")
        print(f"ROC-AUC OvR (macro): {overview['roc_auc_ovr_macro']:.4f}")
        print(f"ROC-AUC OvR (weighted): {overview['roc_auc_ovr_weighted']:.4f}")
        print(f"ROC-AUC OvO (macro): {overview['roc_auc_ovo_macro']:.4f}")
        print(f"ROC-AUC OvO (weighted): {overview['roc_auc_ovo_weighted']:.4f}")
        
        print("\n" + "-"*50)
        print(" CLASSES PRESENT IN DATA ".center(50))
        print("-"*50)
        
        present_classes = metrics.get('present_classes', [])
        if present_classes:
            for i, class_name in enumerate(present_classes, 1):
                class_metrics = metrics['per_class_metrics'].get(class_name, {})
                support = class_metrics.get('support', 0)
                f1_score = class_metrics.get('f1_score', 0)
                print(f"{i}. {class_name}: F1={f1_score:.4f}, Support={support}")
        else:
            print("No class information available")
        
        print("\n" + "-"*50)
        print(" TOP 5 PERFORMING CLASSES (BY F1-SCORE) ".center(50))
        print("-"*50)
        
        # Get classes with non-zero support (actually present in data)
        class_f1_scores = []
        for name, class_metrics in metrics['per_class_metrics'].items():
            if class_metrics.get('support', 0) > 0:  # Only include classes with data
                class_f1_scores.append((name, class_metrics['f1_score']))
        
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        if class_f1_scores:
            for i, (class_name, f1_score) in enumerate(class_f1_scores[:5], 1):
                support = metrics['per_class_metrics'][class_name]['support']
                print(f"{i}. {class_name}: F1={f1_score:.4f} (n={support})")
        else:
            print("No classes with valid performance metrics found")
        
        print("\n" + "-"*50)
        print(" CLASSES WITH ZERO SUPPORT ".center(50))
        print("-"*50)
        
        zero_support_classes = [name for name, class_metrics in metrics['per_class_metrics'].items() 
                               if class_metrics.get('support', 0) == 0]
        
        if zero_support_classes:
            print(f"Classes not present in evaluation data ({len(zero_support_classes)}):")
            for class_name in zero_support_classes:
                print(f"  - {class_name}")
        else:
            print("All classes have samples in the evaluation data")
        
        print("="*60)


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print(" MILK10K CONCEPTCLIP-ONLY CLASSIFICATION PIPELINE ".center(80))
    print("="*80)
    
    try:
        pipeline = MILK10kConceptCLIPPipeline(
            dataset_path=DATASET_PATH,
            groundtruth_path=GROUNDTRUTH_PATH,
            conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
            cache_path=HUGGINGFACE_CACHE_PATH
        )
        
        print("\n✓ Pipeline initialized successfully")
        print("Starting classification process...")
        
        pipeline.run_classification()
        
        print("\n" + "="*80)
        print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY ".center(80))
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*80)
        print(" PIPELINE EXECUTION FAILED ".center(80))
        print("="*80)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
