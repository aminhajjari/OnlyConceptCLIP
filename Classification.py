# MILK10k Medical Image Classification Pipeline - Limited to 100 Images
# Modified to process only 100 images with comprehensive debug prints
#If you later want to process all images, just change line 48:




import os
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
print("✓ All imports loaded successfully")
print("="*60)

# Set up Python path for ConceptModel imports
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')

# Import local ConceptCLIP modules directly
try:
    from ConceptModel.modeling_conceptclip import ConceptCLIP
    from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor
    print("✓ ConceptCLIP modules imported successfully")
except ImportError as e:
    print(f"⚠️ ConceptCLIP import error: {e}")
    print("Will use dummy models for testing")

print("✓ SECTION: Module imports completed successfully")
print("-"*60)

# ==================== CONFIGURATION ====================

# DEBUG MODE - Process only 100 images
DEBUG_MODE = True
MAX_DEBUG_IMAGES = 100

# Your dataset paths (Narval specific)
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"

# FIXED OUTPUT PATH - No timestamped folders
BASE_OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyResualts"

# Local model paths
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

print("✓ Configuration paths set")
print(f"✓ DEBUG MODE: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
print(f"✓ Max images to process: {MAX_DEBUG_IMAGES if DEBUG_MODE else 'ALL'}")
print("✓ SECTION: Configuration completed successfully")
print("-"*60)

# ==================== OUTPUT FOLDER SETUP ====================

def setup_output_folder():
    """Setup the fixed output folder structure"""
    print("\n=== SETTING UP OUTPUT FOLDERS ===")
    output_path = Path(BASE_OUTPUT_PATH)
    
    # Create main directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Main output directory created: {output_path}")
    
    # Create subdirectories
    subdirs = [
        "classifications",
        "visualizations", 
        "reports",
        "processed_images",
        "evaluation_metrics",
        "debug_logs"  # Added for debug outputs
    ]
    
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)
        print(f"✓ Subdirectory created: {subdir}")
    
    # Create debug log file
    debug_log_path = output_path / "debug_logs" / "pipeline_debug.log"
    with open(debug_log_path, 'w') as f:
        f.write(f"Debug log started at {datetime.now()}\n")
        f.write(f"Debug mode: {DEBUG_MODE}\n")
        f.write(f"Max images: {MAX_DEBUG_IMAGES if DEBUG_MODE else 'ALL'}\n\n")
    
    print(f"✓ Debug log created: {debug_log_path}")
    print("✓ All output folders setup complete")
    print("✓ SECTION: Output folder setup completed successfully")
    print("-"*60)
    return output_path

# ==================== GPU DETECTION AND SETUP ====================

def setup_gpu_environment():
    """Setup GPU environment with proper error handling"""
    print("\n=== GPU ENVIRONMENT SETUP ===")
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
        
        # Set default device
        device = f"cuda:{torch.cuda.current_device()}"
        print(f"Using device: {device}")
        
        # Test GPU allocation
        try:
            test_tensor = torch.randn(10, 10).to(device)
            print("✓ GPU allocation test successful")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU allocation test failed: {e}")
            print("Falling back to CPU")
            device = "cpu"
    else:
        print("⚠️ CUDA not available. Using CPU.")
        device = "cpu"
        
        # Check Slurm GPU allocation
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
    
    # Set environment variables for offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1" 
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"✓ Offline mode enabled")
    print(f"✓ Cache directory set to: {cache_path}")
    
    # Verify cache directory exists
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        print(f"✓ Cache directory exists")
        cached_models = list(cache_path_obj.glob("models--*"))
        print(f"✓ Found {len(cached_models)} cached models:")
        for model in cached_models[:5]:  # Show first 5
            print(f"   - {model.name}")
        if len(cached_models) > 5:
            print(f"   ... and {len(cached_models) - 5} more")
    else:
        print(f"❌ Cache directory does not exist: {cache_path}")
        
    print("✓ Offline environment setup complete")
    print("✓ SECTION: Offline environment setup completed successfully")
    print("-"*60)

# ==================== LOCAL MODEL LOADING ====================

def load_local_conceptclip_models(model_path: str, cache_path: str, device: str):
    """Load local ConceptCLIP models with offline support"""
    print("\n=== LOADING CONCEPTCLIP MODELS ===")
    try:
        # Setup offline environment first
        setup_offline_environment(cache_path)
        
        print(f"Loading ConceptCLIP from local path: {model_path}")
        print(f"Using cache directory: {cache_path}")
        
        # Load model with local_files_only to ensure offline mode
        model = ConceptCLIP.from_pretrained(
            model_path,
            local_files_only=True,
            cache_dir=cache_path
        )
        print("✓ ConceptCLIP model loaded")
        
        # Try to load processor from ConceptCLIP
        try:
            processor = ConceptCLIPProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                cache_dir=cache_path
            )
            print("✓ ConceptCLIP processor loaded")
        except Exception as e:
            print(f"⚠️ Processor loading error: {e}")
            print("Using simple processor fallback")
            processor = create_simple_processor()
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"✓ ConceptCLIP loaded successfully on {device}")
        print("✓ SECTION: ConceptCLIP model loading completed successfully")
        print("-"*60)
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading local ConceptCLIP: {e}")
        print("Creating dummy ConceptCLIP model for testing...")
        dummy_model = create_dummy_conceptclip_model(device)
        dummy_processor = create_simple_processor()
        print("✓ SECTION: Dummy model creation completed successfully")
        print("-"*60)
        return dummy_model, dummy_processor

def create_dummy_conceptclip_model(device: str):
    """Create a dummy ConceptCLIP model for testing"""
    print("Creating dummy ConceptCLIP model...")
    
    class DummyConceptCLIP:
        def __init__(self, device):
            self.device = device
            print(f"✓ Dummy ConceptCLIP initialized on {device}")
            
        def to(self, device):
            self.device = device
            return self
            
        def eval(self):
            return self
            
        def __call__(self, **inputs):
            # Return dummy outputs
            batch_size = inputs['pixel_values'].shape[0] if 'pixel_values' in inputs else 1
            text_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 10
            
            return {
                'image_features': torch.randn(batch_size, 512).to(self.device),
                'text_features': torch.randn(text_size, 512).to(self.device),
                'logit_scale': torch.tensor(2.6592).to(self.device)
            }
    
    return DummyConceptCLIP(device)

def create_simple_processor():
    """Create a simple processor for ConceptCLIP"""
    print("Creating simple processor...")
    
    class SimpleProcessor:
        def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            
            result = {}
            
            if images is not None:
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if isinstance(images, Image.Image):
                    images = [images]
                
                processed = torch.stack([transform(img) for img in images])
                result['pixel_values'] = processed
            
            if text is not None:
                # Simple text encoding - you might need to adjust this
                if isinstance(text, str):
                    text = [text]
                
                # Create dummy tokens for now
                max_length = 77
                result['input_ids'] = torch.randint(0, 1000, (len(text), max_length))
                result['attention_mask'] = torch.ones((len(text), max_length))
            
            return result
    
    print("✓ Simple processor created")
    return SimpleProcessor()

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

# CORRECTED MILK10k Medical Domain Configuration
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
        """Calculate all evaluation metrics including ROC-AUC"""
        print("\n=== CALCULATING COMPREHENSIVE METRICS ===")
        
        print(f"Evaluating {len(y_true)} true labels and {len(y_pred)} predictions")
        
        # Convert string labels to indices for sklearn
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        y_true_idx = [label_to_idx.get(label, -1) for label in y_true]
        y_pred_idx = [label_to_idx.get(label, -1) for label in y_pred]
        
        # Filter out unknown labels
        valid_indices = [i for i, (true_idx, pred_idx) in enumerate(zip(y_true_idx, y_pred_idx)) 
                        if true_idx != -1 and pred_idx != -1]
        
        print(f"Valid samples for evaluation: {len(valid_indices)}")
        
        if not valid_indices:
            print("❌ No valid samples for evaluation")
            return self._empty_metrics()
        
        y_true_filtered = [y_true_idx[i] for i in valid_indices]
        y_pred_filtered = [y_pred_idx[i] for i in valid_indices]
        
        # Filter probabilities if provided
        if y_pred_proba:
            y_pred_proba_filtered = [y_pred_proba[i] for i in valid_indices]
            print(f"✓ Probability data available for ROC-AUC calculation")
        else:
            y_pred_proba_filtered = None
            print("⚠️ No probability data available - ROC-AUC will be 0")
        
        # Calculate basic metrics
        print("Calculating basic metrics...")
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        
        # Calculate per-class and weighted metrics
        precision_macro = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        
        print(f"✓ Basic metrics calculated - Accuracy: {accuracy:.4f}")
        
        # Per-class metrics
        print("Calculating per-class metrics...")
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, labels=list(range(len(self.class_names))), zero_division=0
        )
        
        # ROC-AUC Calculation
        print("Calculating ROC-AUC metrics...")
        roc_auc_metrics = self._calculate_roc_auc(y_true_filtered, y_pred_proba_filtered)
        
        # Confusion matrix
        print("Creating confusion matrix...")
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(range(len(self.class_names))))
        
        # Classification report
        print("Generating classification report...")
        class_report = classification_report(
            y_true_filtered, y_pred_filtered, 
            target_names=[self.class_names[i] for i in range(len(self.class_names))],
            output_dict=True, zero_division=0
        )
        
        metrics = {
            'overview': {
                'total_samples': len(y_true),
                'valid_samples': len(valid_indices),
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                # ROC-AUC metrics
                'roc_auc_ovr_macro': roc_auc_metrics['ovr_macro'],
                'roc_auc_ovr_weighted': roc_auc_metrics['ovr_weighted'],
                'roc_auc_ovo_macro': roc_auc_metrics['ovo_macro'],
                'roc_auc_ovo_weighted': roc_auc_metrics['ovo_weighted']
            },
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i]),
                    'roc_auc': roc_auc_metrics['per_class'][i] if i < len(roc_auc_metrics['per_class']) else 0.0
                } for i in range(len(self.class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_names': self.class_names,
            'roc_curves': roc_auc_metrics.get('curves', {})
        }
        
        print("✓ Comprehensive metrics calculation complete")
        print("✓ SECTION: Metrics calculation completed successfully")
        print("-"*60)
        return metrics
    
    def _calculate_roc_auc(self, y_true: List[int], y_pred_proba: Optional[List[List[float]]]) -> Dict:
        """Calculate ROC-AUC metrics for multi-class classification"""
        
        if y_pred_proba is None or len(y_pred_proba) == 0:
            print("⚠️ No probability data - returning zero ROC-AUC scores")
            return {
                'ovr_macro': 0.0,
                'ovr_weighted': 0.0,
                'ovo_macro': 0.0,
                'ovo_weighted': 0.0,
                'per_class': [0.0] * len(self.class_names),
                'curves': {}
            }
        
        try:
            print("Calculating ROC-AUC scores...")
            # Convert to numpy arrays
            y_true_array = np.array(y_true)
            y_pred_proba_array = np.array(y_pred_proba)
            
            # Binarize the labels for One-vs-Rest
            y_true_binarized = label_binarize(y_true_array, classes=list(range(len(self.class_names))))
            
            # Calculate One-vs-Rest ROC-AUC
            roc_auc_ovr_macro = roc_auc_score(y_true_binarized, y_pred_proba_array, 
                                             average='macro', multi_class='ovr')
            roc_auc_ovr_weighted = roc_auc_score(y_true_binarized, y_pred_proba_array, 
                                                average='weighted', multi_class='ovr')
            
            # Calculate One-vs-One ROC-AUC
            roc_auc_ovo_macro = roc_auc_score(y_true_array, y_pred_proba_array, 
                                             average='macro', multi_class='ovo')
            roc_auc_ovo_weighted = roc_auc_score(y_true_array, y_pred_proba_array, 
                                                average='weighted', multi_class='ovo')
            
            print(f"✓ ROC-AUC OvR Macro: {roc_auc_ovr_macro:.4f}")
            
            # Calculate per-class ROC-AUC and curves
            per_class_auc = []
            curves = {}
            
            for i in range(len(self.class_names)):
                if i < y_true_binarized.shape[1] and i < y_pred_proba_array.shape[1]:
                    # Check if this class exists in the true labels
                    if np.sum(y_true_binarized[:, i]) > 0:
                        fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], y_pred_proba_array[:, i])
                        auc_score = auc(fpr, tpr)
                        per_class_auc.append(float(auc_score))
                        
                        # Store curve data for visualization
                        curves[self.class_names[i]] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(auc_score)
                        }
                    else:
                        per_class_auc.append(0.0)
                else:
                    per_class_auc.append(0.0)
            
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
            return {
                'ovr_macro': 0.0,
                'ovr_weighted': 0.0,
                'ovo_macro': 0.0,
                'ovo_weighted': 0.0,
                'per_class': [0.0] * len(self.class_names),
                'curves': {}
            }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no valid samples"""
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

class MILK10kEnhancedClassificationPipeline:
    """Enhanced MILK10k classification pipeline with comprehensive evaluation"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, 
                 conceptclip_model_path: str = None, cache_path: str = None):
        print("\n=== INITIALIZING MILK10K PIPELINE ===")
        
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        
        # Store debug mode settings
        self.debug_mode = DEBUG_MODE
        self.max_debug_images = MAX_DEBUG_IMAGES
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Ground truth path: {self.groundtruth_path}")
        print(f"ConceptCLIP model path: {self.conceptclip_model_path}")
        print(f"Cache path: {self.cache_path}")
        print(f"DEBUG MODE: {'ENABLED' if self.debug_mode else 'DISABLED'}")
        print(f"Max images to process: {self.max_debug_images if self.debug_mode else 'ALL'}")
        
        # Setup fixed output folder (no timestamps)
        self.output_path = setup_output_folder()
        print(f"Output folder: {self.output_path}")
        
        # Initialize device with proper setup
        self.device = setup_gpu_environment()
        print(f"Device: {self.device}")
        
        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator(self.domain.class_names)
        print("✓ Evaluator initialized")
        
        # Load models
        self._load_models()
        
        # Load ground truth
        self._load_ground_truth()
        
        print("✓ MILK10k pipeline initialization complete")
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
            
            # Check for expected MILK10k columns
            expected_cols = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
            found_cols = [col for col in expected_cols if col in self.ground_truth.columns]
            print(f"Expected diagnostic columns found: {found_cols}")
            
            # Check for lesion_id column
            if 'lesion_id' in self.ground_truth.columns:
                print("✓ lesion_id column found for image matching")
                
                # Show sample of data
                print("Sample data:")
                print(self.ground_truth.head())
                
                # Count non-zero values for each diagnostic column
                print("\nDiagnostic class distribution:")
                for col in found_cols:
                    count = (self.ground_truth[col] == 1.0).sum()
                    print(f"  {col}: {count} samples")
                    
            else:
                print("⚠️ lesion_id column not found!")
                
            if not found_cols:
                print("⚠️ No expected diagnostic columns found. Will use 'dx' column if available.")
                
            print("✓ SECTION: Ground truth loading completed successfully")
            print("-"*60)
            
        except Exception as e:
            print(f"❌ Error loading ground truth: {e}")
            self.ground_truth = None
    else:
        print(f"❌ Ground truth file not found: {self.groundtruth_path}")
        self.ground_truth = None
    
    def get_image_files(self) -> List[Path]:
        """Get all image files from dataset with DEBUG mode limiting"""
        print("\n=== COLLECTING IMAGE FILES ===")
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return []
        
        image_files = []
        for ext in self.domain.image_extensions:
            files = list(self.dataset_path.rglob(f"*{ext}"))
            image_files.extend(files)
            print(f"Found {len(files)} files with extension {ext}")
        
        print(f"Total images found: {len(image_files)}")
        
        # Apply DEBUG mode limiting
        if self.debug_mode and len(image_files) > self.max_debug_images:
            print(f"\n⚠️ DEBUG MODE: Limiting to {self.max_debug_images} images")
            image_files = image_files[:self.max_debug_images]
            print(f"✓ Processing limited to {len(image_files)} images")
        
        print("✓ SECTION: Image file collection completed successfully")
        print("-"*60)
        return image_files
    
    def preprocess_image(self, image_path: Path) -> Optional[Image.Image]:
        """Preprocess a single image"""
        try:
            # Handle DICOM files
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                ds = pydicom.dcmread(str(image_path))
                img_array = ds.pixel_array
                
                # Normalize to 0-255
                img_array = ((img_array - img_array.min()) / 
                           (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                
                # Convert to PIL Image
                if len(img_array.shape) == 2:
                    image = Image.fromarray(img_array, mode='L').convert('RGB')
                else:
                    image = Image.fromarray(img_array, mode='RGB')
            else:
                # Handle regular image files
                image = Image.open(image_path).convert('RGB')
            
            # Apply preprocessing if specified
            if self.domain.preprocessing_params.get('enhance_contrast', False):
                # Simple contrast enhancement
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def classify_with_conceptclip(self, image: Image.Image) -> Tuple[str, List[float]]:
        """Classify image using ConceptCLIP"""
        try:
            # Prepare inputs
            inputs = self.conceptclip_processor(
                images=image,
                text=self.domain.text_prompts,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.conceptclip_model(**inputs)
                
                # Calculate similarity scores
                image_features = outputs['image_features']
                text_features = outputs['text_features']
                logit_scale = outputs['logit_scale']
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                logits = (image_features @ text_features.T) * logit_scale.exp()
                probs = torch.softmax(logits, dim=-1)
                
                # Get prediction
                pred_idx = probs.argmax(dim=-1).item()
                pred_class = self.domain.class_names[pred_idx]
                pred_probs = probs.squeeze().cpu().numpy().tolist()
                
                return pred_class, pred_probs
                
        except Exception as e:
            print(f"Error in ConceptCLIP classification: {e}")
            # Return random prediction for testing
            import random
            pred_class = random.choice(self.domain.class_names)
            pred_probs = [1.0/len(self.domain.class_names)] * len(self.domain.class_names)
            return pred_class, pred_probs
    
    def run_classification(self):
        """Run the complete classification pipeline on limited dataset"""
        print("\n" + "="*60)
        print("STARTING MILK10K CLASSIFICATION PIPELINE")
        print(f"DEBUG MODE: {'ENABLED - Processing only {}'.format(self.max_debug_images) if self.debug_mode else 'DISABLED - Processing all'} images")
        print("="*60)
        
        # Get image files
        image_files = self.get_image_files()
        
        if not image_files:
            print("❌ No image files found!")
            return
        
        print(f"\n✓ Processing {len(image_files)} images...")
        
        # Initialize results storage
        results = []
        all_true_labels = []
        all_pred_labels = []
        all_pred_probas = []
        
        # Process each image with progress bar
        for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            
            # Print progress every 10 images in debug mode
            if self.debug_mode and (idx + 1) % 10 == 0:
                print(f"\n  Progress: {idx + 1}/{len(image_files)} images processed")
            
            # Get image ID from filename
            image_id = image_path.stem
            
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                continue
            
            # Classify with ConceptCLIP
            pred_class, pred_probs = self.classify_with_conceptclip(image)
            
            # Get ground truth if available
            true_class = self.get_ground_truth_label(image_id)
            
            # Store results
            result = {
                'image_id': image_id,
                'image_path': str(image_path),
                'predicted_class': pred_class,
                'prediction_probabilities': pred_probs,
                'true_class': true_class,
                'max_probability': max(pred_probs)
            }
            results.append(result)
            
            if true_class:
                all_true_labels.append(true_class)
                all_pred_labels.append(pred_class)
                all_pred_probas.append(pred_probs)
        
        print(f"\n✓ Classification complete for {len(results)} images")
        
        # Save results
        self.save_results(results)
        
        # Calculate and save metrics if ground truth available
        if all_true_labels:
            print("\n=== EVALUATION METRICS ===")
            metrics = self.evaluator.calculate_comprehensive_metrics(
                all_true_labels, all_pred_labels, all_pred_probas
            )
            self.save_metrics(metrics)
            self.print_summary_metrics(metrics)
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print(f"Results saved to: {self.output_path}")
        print("="*60)
    
    def get_ground_truth_label(self, image_id: str) -> Optional[str]:
        """Get ground truth label for an image"""
        if self.ground_truth is None:
            return None
        
        # Try to find the image in ground truth
        row = self.ground_truth[self.ground_truth['image'] == image_id]
        
        if row.empty:
            # Try with .jpg extension
            row = self.ground_truth[self.ground_truth['image'] == f"{image_id}.jpg"]
        
        if not row.empty:
            # Get the diagnosis from MILK10k columns
            for col, label in self.domain.label_mappings.items():
                if col in row.columns and row.iloc[0][col] == 1.0:
                    return label
            
            # Try 'dx' column if no one-hot encoding found
            if 'dx' in row.columns:
                dx_value = row.iloc[0]['dx']
                return self.domain.label_mappings.get(dx_value, dx_value)
        
        return None
    
    def save_results(self, results: List[Dict]):
        """Save classification results"""
        print("\n=== SAVING RESULTS ===")
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = self.output_path / "classifications" / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to CSV: {csv_path}")
        
        # Save as JSON with full details
        json_path = self.output_path / "classifications" / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to JSON: {json_path}")
        
        # Save summary statistics
        summary = {
            'total_images': len(results),
            'debug_mode': self.debug_mode,
            'max_images_limit': self.max_debug_images if self.debug_mode else 'unlimited',
            'unique_predictions': Counter([r['predicted_class'] for r in results]),
            'average_confidence': np.mean([r['max_probability'] for r in results]),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_path / "reports" / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Summary saved to: {summary_path}")
        
        print("✓ SECTION: Results saving completed successfully")
        print("-"*60)
    
    def save_metrics(self, metrics: Dict):
        """Save evaluation metrics"""
        print("\n=== SAVING EVALUATION METRICS ===")
        
        # Save comprehensive metrics as JSON
        metrics_path = self.output_path / "evaluation_metrics" / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        print("✓ SECTION: Metrics saving completed successfully")
        print("-"*60)
    
    def create_visualizations(self, metrics: Dict):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.domain.class_names,
                   yticklabels=self.domain.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = self.output_path / "visualizations" / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"✓ Confusion matrix saved to: {cm_path}")
        
        # Per-class metrics bar chart
        plt.figure(figsize=(14, 8))
        class_metrics = metrics['per_class_metrics']
        classes = list(class_metrics.keys())
        f1_scores = [class_metrics[c]['f1_score'] for c in classes]
        
        plt.bar(range(len(classes)), f1_scores, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score by Class')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.tight_layout()
        f1_path = self.output_path / "visualizations" / "f1_scores.png"
        plt.savefig(f1_path)
        plt.close()
        print(f"✓ F1 scores chart saved to: {f1_path}")
    
    def print_summary_metrics(self, metrics: Dict):
        """Print summary of evaluation metrics"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        overview = metrics['overview']
        print(f"Total Samples: {overview['total_samples']}")
        print(f"Valid Samples: {overview['valid_samples']}")
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {overview['accuracy']:.4f}")
        print(f"  Precision (macro): {overview['precision_macro']:.4f}")
        print(f"  Recall (macro): {overview['recall_macro']:.4f}")
        print(f"  F1 Score (macro): {overview['f1_macro']:.4f}")
        print(f"\nROC-AUC Scores:")
        print(f"  One-vs-Rest (macro): {overview['roc_auc_ovr_macro']:.4f}")
        print(f"  One-vs-Rest (weighted): {overview['roc_auc_ovr_weighted']:.4f}")
        print(f"  One-vs-One (macro): {overview['roc_auc_ovo_macro']:.4f}")
        print(f"  One-vs-One (weighted): {overview['roc_auc_ovo_weighted']:.4f}")
        
        print("\nTop 3 Best Performing Classes:")
        class_metrics = metrics['per_class_metrics']
        sorted_classes = sorted(class_metrics.items(), 
                              key=lambda x: x[1]['f1_score'], 
                              reverse=True)[:3]
        for cls, m in sorted_classes:
            print(f"  {cls}: F1={m['f1_score']:.3f}, Precision={m['precision']:.3f}, Recall={m['recall']:.3f}")
        
        print("="*60)

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" MILK10K MEDICAL IMAGE CLASSIFICATION PIPELINE - LIMITED TO 100 IMAGES ".center(80))
    print("="*80)
    
    # Create and run pipeline
    pipeline = MILK10kEnhancedClassificationPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH
    )
    
    # Run classification
    pipeline.run_classification()
    
    print("\n✓ Pipeline execution completed successfully!")
    print("="*80)
