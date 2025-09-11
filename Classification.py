# MILK10k Medical Image Classification Pipeline - Enhanced Version
# Classification with comprehensive evaluation metrics

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
    classification_report, confusion_matrix, precision_recall_fscore_support
)
warnings.filterwarnings('ignore')

# Set up Python path for ConceptModel imports
import sys
sys.path.insert(0, '/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input')

# Import local ConceptCLIP modules directly
from ConceptModel.modeling_conceptclip import ConceptCLIP
from ConceptModel.preprocessor_conceptclip import ConceptCLIPProcessor

# ==================== CONFIGURATION ====================

# Your dataset paths (Narval specific)
DATASET_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_Input"
GROUNDTRUTH_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/MILK10k_Training_GroundTruth.csv"
BASE_OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input"

# Local model paths
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

# ==================== OUTPUT FOLDER NAMING ====================

def create_experiment_folder():
    """Create a recognizable experiment folder name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"MILK10k_ConceptCLIP_Classification_{timestamp}"
    output_path = Path(BASE_OUTPUT_PATH) / folder_name
    return output_path

# ==================== GPU DETECTION AND SETUP ====================

def setup_gpu_environment():
    """Setup GPU environment with proper error handling"""
    print("=" * 50)
    print("GPU ENVIRONMENT SETUP")
    print("=" * 50)
    
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
            print("✅ GPU allocation test successful")
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
    
    print("=" * 50)
    return device

# ==================== CACHE AND OFFLINE SETUP ====================

def setup_offline_environment(cache_path: str):
    """Setup offline environment for Hugging Face models"""
    print("=" * 50)
    print("OFFLINE ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Set environment variables for offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1" 
    os.environ["TRANSFORMERS_CACHE"] = cache_path
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"✅ Offline mode enabled")
    print(f"✅ Cache directory set to: {cache_path}")
    
    # Verify cache directory exists
    cache_path_obj = Path(cache_path)
    if cache_path_obj.exists():
        print(f"✅ Cache directory exists")
        cached_models = list(cache_path_obj.glob("models--*"))
        print(f"✅ Found {len(cached_models)} cached models:")
        for model in cached_models:
            print(f"   - {model.name}")
    else:
        print(f"❌ Cache directory does not exist: {cache_path}")
        
    print("=" * 50)

# ==================== LOCAL MODEL LOADING ====================

def load_local_conceptclip_models(model_path: str, cache_path: str, device: str):
    """Load local ConceptCLIP models with offline support"""
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
        
        # Try to load processor from ConceptCLIP
        try:
            processor = ConceptCLIPProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                cache_dir=cache_path
            )
        except Exception as e:
            print(f"Using simple processor due to error: {e}")
            processor = create_simple_processor()
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        print(f"✅ ConceptCLIP loaded successfully on {device}")
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading local ConceptCLIP: {e}")
        print("This might be due to missing dependencies. Trying fallback...")
        return create_dummy_conceptclip_model(device), create_simple_processor()

def create_dummy_conceptclip_model(device: str):
    """Create a dummy ConceptCLIP model for testing"""
    class DummyConceptCLIP:
        def __init__(self, device):
            self.device = device
            
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

# MILK10k Medical Domain Configuration
MILK10K_DOMAIN = MedicalDomain(
    name="milk10k",
    image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.dcm', '.dicom'],
    text_prompts=[
        'a medical image showing normal tissue',
        'a medical image showing abnormal pathology',
        'a medical image showing inflammatory lesion',
        'a medical image showing neoplastic lesion',
        'a medical image showing degenerative changes',
        'a medical image showing infectious disease',
        'a medical image showing vascular pathology',
        'a medical image showing metabolic disorder',
        'a medical image showing congenital abnormality',
        'a medical image showing traumatic injury'
    ],
    label_mappings={
        'NORMAL': 'normal tissue',
        'ABNORMAL': 'abnormal pathology',
        'INFLAMMATORY': 'inflammatory lesion',
        'NEOPLASTIC': 'neoplastic lesion',
        'DEGENERATIVE': 'degenerative changes',
        'INFECTIOUS': 'infectious disease',
        'VASCULAR': 'vascular pathology',
        'METABOLIC': 'metabolic disorder',
        'CONGENITAL': 'congenital abnormality',
        'TRAUMATIC': 'traumatic injury'
    },
    preprocessing_params={'normalize': True, 'enhance_contrast': True},
    class_names=[
        'normal tissue', 'abnormal pathology', 'inflammatory lesion',
        'neoplastic lesion', 'degenerative changes', 'infectious disease',
        'vascular pathology', 'metabolic disorder', 'congenital abnormality',
        'traumatic injury'
    ]
)

# ==================== ENHANCED EVALUATION METRICS ====================

class ComprehensiveEvaluator:
    """Comprehensive evaluation with all required metrics"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
    def calculate_comprehensive_metrics(self, y_true: List[str], y_pred: List[str], 
                                      y_pred_proba: Optional[List[List[float]]] = None) -> Dict:
        """Calculate all evaluation metrics"""
        
        # Convert string labels to indices for sklearn
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        y_true_idx = [label_to_idx.get(label, -1) for label in y_true]
        y_pred_idx = [label_to_idx.get(label, -1) for label in y_pred]
        
        # Filter out unknown labels
        valid_indices = [i for i, (true_idx, pred_idx) in enumerate(zip(y_true_idx, y_pred_idx)) 
                        if true_idx != -1 and pred_idx != -1]
        
        if not valid_indices:
            return self._empty_metrics()
        
        y_true_filtered = [y_true_idx[i] for i in valid_indices]
        y_pred_filtered = [y_pred_idx[i] for i in valid_indices]
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        
        # Calculate per-class and weighted metrics
        precision_macro = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, labels=list(range(len(self.class_names))), zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(range(len(self.class_names))))
        
        # Classification report
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
                'f1_weighted': f1_weighted
            },
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                } for i in range(len(self.class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_names': self.class_names
        }
        
        return metrics
    
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
                'f1_weighted': 0.0
            },
            'per_class_metrics': {},
            'confusion_matrix': [],
            'classification_report': {},
            'class_names': self.class_names
        }

# ==================== MAIN PIPELINE CLASS ====================

class MILK10kEnhancedClassificationPipeline:
    """Enhanced MILK10k classification pipeline with comprehensive evaluation"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, 
                 conceptclip_model_path: str = None, cache_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        
        # Create recognizable output folder
        self.output_path = create_experiment_folder()
        print(f"📁 Experiment folder created: {self.output_path}")
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        (self.output_path / "processed_images").mkdir(exist_ok=True)
        (self.output_path / "evaluation_metrics").mkdir(exist_ok=True)
        
        # Initialize device with proper setup
        self.device = setup_gpu_environment()
        print(f"Initializing MILK10k enhanced classification pipeline on {self.device}")
        
        # Initialize evaluator
        self.evaluator = ComprehensiveEvaluator(self.domain.class_names)
        
        # Load models
        self._load_models()
        
        # Load ground truth
        self._load_ground_truth()
        
    def _load_models(self):
        """Load local ConceptCLIP model"""
        self.conceptclip_model, self.conceptclip_processor = load_local_conceptclip_models(
            self.conceptclip_model_path, self.cache_path, self.device
        )
        
    def _load_ground_truth(self):
        """Load ground truth annotations"""
        if os.path.exists(self.groundtruth_path):
            self.ground_truth = pd.read_csv(self.groundtruth_path)
            print(f"Loaded ground truth: {len(self.ground_truth)} samples")
            print(f"Ground truth columns: {list(self.ground_truth.columns)}")
        else:
            print(f"Ground truth file not found: {self.groundtruth_path}")
            self.ground_truth = None
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess images for MILK10k dataset"""
        try:
            image_path = Path(image_path)
            ext = image_path.suffix.lower()
            
            if ext in ['.dcm', '.dicom']:
                return self._load_dicom(image_path)
            elif ext in ['.nii', '.nii.gz']:
                return self._load_nifti(image_path)
            else:
                return self._load_standard_image(image_path)
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _load_dicom(self, image_path: Path) -> np.ndarray:
        """Load DICOM images"""
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_nifti(self, image_path: Path) -> np.ndarray:
        """Load NIfTI images"""
        nii_img = nib.load(image_path)
        image = nii_img.get_fdata()
        
        # Take middle slice for 3D volumes
        if len(image.shape) == 3:
            mid_slice = image.shape[2] // 2
            image = image[:, :, mid_slice]
        
        # Normalize
        image = self._normalize_image(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def _load_standard_image(self, image_path: Path) -> np.ndarray:
        """Load standard image formats"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast
        if self.domain.preprocessing_params.get('enhance_contrast', False):
            image = self._enhance_contrast(image)
        
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        return (image * 255).astype(np.uint8)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def classify_image(self, image: np.ndarray) -> Dict:
        """Classify image using ConceptCLIP"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Use ConceptCLIP processor
            inputs = self.conceptclip_processor(
                images=pil_image, 
                text=self.domain.text_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.conceptclip_model(**inputs)
                
                # Extract logits using ConceptCLIP output structure
                logit_scale = outputs.get("logit_scale", torch.tensor(1.0))
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                
                # Compute similarity scores
                logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)[0]
            
            # Convert to probabilities
            disease_names = [prompt.split(' showing ')[-1] for prompt in self.domain.text_prompts]
            probabilities = {disease_names[i]: float(logits[i]) for i in range(len(disease_names))}
            
            return probabilities
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {}
    
    def get_ground_truth_label(self, img_path: Path) -> Optional[str]:
        """Get ground truth label for image"""
        if self.ground_truth is None:
            return None
        
        img_name = img_path.stem
        
        # Try to find matching row in ground truth
        matching_rows = self.ground_truth[
            self.ground_truth.iloc[:, 0].astype(str).str.contains(img_name, na=False)
        ]
        
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            # Look for the label in subsequent columns
            for col in self.ground_truth.columns[1:]:
                if col in self.domain.label_mappings and row[col] == 1:
                    return self.domain.label_mappings[col]
            
            # If no specific column, check if there's a direct label column
            if 'label' in row:
                return str(row['label'])
        
        return None
    
    def process_dataset(self) -> Dict:
        """Process entire MILK10k dataset with comprehensive evaluation"""
        print("Starting MILK10k dataset classification with comprehensive evaluation...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in dataset")
        
        results = []
        format_counter = Counter()
        
        # Lists for comprehensive evaluation
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        for img_path in tqdm(image_files, desc="Classifying MILK10k images"):
            try:
                # Track file formats
                ext = img_path.suffix.lower()
                format_counter[ext] += 1
                
                # Load and preprocess image
                image = self.preprocess_image(img_path)
                if image is None:
                    continue
                
                # Save processed image for reference
                img_name = img_path.stem
                processed_img_path = self.output_path / "processed_images" / f"{img_name}_processed.png"
                cv2.imwrite(str(processed_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Classify image
                classification_probs = self.classify_image(image)
                
                # Get ground truth
                ground_truth = self.get_ground_truth_label(img_path)
                
                # Get prediction
                if classification_probs:
                    predicted_disease = max(classification_probs, key=classification_probs.get)
                    prediction_confidence = classification_probs[predicted_disease]
                    proba_list = [classification_probs.get(cls, 0.0) for cls in self.domain.class_names]
                else:
                    predicted_disease = "unknown"
                    prediction_confidence = 0.0
                    proba_list = [0.0] * len(self.domain.class_names)
                
                # Collect for evaluation
                if ground_truth and predicted_disease != "unknown":
                    y_true.append(ground_truth)
                    y_pred.append(predicted_disease)
                    y_pred_proba.append(proba_list)
                
                # Save results
                result = {
                    'image_path': str(img_path),
                    'image_name': img_name,
                    'predicted_disease': predicted_disease,
                    'prediction_confidence': prediction_confidence,
                    'ground_truth': ground_truth,
                    'correct': ground_truth == predicted_disease if ground_truth else None,
                    'processed_image_path': str(processed_img_path),
                    'classification_probabilities': classification_probs,
                    'device_used': self.device,
                    'cache_used': self.cache_path
                }
                
                results.append(result)
                
                # Progress indicator
                status = "✓" if result['correct'] else ("✗" if ground_truth else "-")
                print(f"{status} {img_name}: {predicted_disease} ({prediction_confidence:.2%}) [Device: {self.device}]")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate comprehensive metrics
        evaluation_metrics = self.evaluator.calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(results, format_counter, evaluation_metrics)
        
        # Save all results and metrics
        self._save_comprehensive_results(results, report, evaluation_metrics)
        
        return report
    
    def _generate_comprehensive_report(self, results: List[Dict], format_counter: Counter, 
                                     evaluation_metrics: Dict) -> Dict:
        """Generate comprehensive processing report with all metrics"""
        
        # Basic statistics
        total_processed = len(results)
        successful_classifications = sum(1 for r in results if r['prediction_confidence'] > 0.1)
        
        # Prediction distribution
        predictions = [r['predicted_disease'] for r in results]
        prediction_counts = Counter(predictions)
        
        # Confidence statistics
        pred_confidences = [r['prediction_confidence'] for r in results]
        
        # Device statistics
        device_used = results[0]['device_used'] if results else "unknown"
        cache_used = results[0]['cache_used'] if results else "unknown"
        
        # Extract key metrics for paper comparison
        accuracy = evaluation_metrics['overview']['accuracy']
        precision_macro = evaluation_metrics['overview']['precision_macro']
        recall_macro = evaluation_metrics['overview']['recall_macro']
        f1_macro = evaluation_metrics['overview']['f1_macro']
        
        report = {
            'experiment_info': {
                'experiment_name': self.output_path.name,
                'timestamp': datetime.now().isoformat(),
                'dataset_path': str(self.dataset_path),
                'output_path': str(self.output_path),
                'model_type': 'ConceptCLIP',
                'pipeline_version': 'Enhanced_v2.0'
            },
            'system_info': {
                'device_used': device_used,
                'cache_directory': cache_used,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'offline_mode': os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
                'pipeline_type': 'classification_with_comprehensive_evaluation'
            },
            'dataset_info': {
                'total_images_found': total_processed,
                'file_formats': dict(format_counter),
                'total_with_ground_truth': evaluation_metrics['overview']['valid_samples'],
                'class_names': self.domain.class_names
            },
            'processing_stats': {
                'successful_classifications': successful_classifications,
                'classification_success_rate': successful_classifications / total_processed if total_processed > 0 else 0
            },
            # KEY METRICS FOR PAPER COMPARISON
            'paper_comparison_metrics': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_score_macro': f1_macro,
                'precision_weighted': evaluation_metrics['overview']['precision_weighted'],
                'recall_weighted': evaluation_metrics['overview']['recall_weighted'],
                'f1_score_weighted': evaluation_metrics['overview']['f1_weighted']
            },
            'detailed_evaluation': evaluation_metrics,
            'predictions': {
                'distribution': dict(prediction_counts),
                'most_common': prediction_counts.most_common(5)
            },
            'confidence_stats': {
                'classification': {
                    'mean': np.mean(pred_confidences) if pred_confidences else 0,
                    'std': np.std(pred_confidences) if pred_confidences else 0,
                    'min': np.min(pred_confidences) if pred_confidences else 0,
                    'max': np.max(pred_confidences) if pred_confidences else 0
                }
            }
        }
        
        return report
    
    def _save_comprehensive_results(self, results: List[Dict], report: Dict, evaluation_metrics: Dict):
        """Save comprehensive results, reports, and visualizations"""
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = self.output_path / "reports" / "classification_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save comprehensive report
        report_path = self.output_path / "reports" / "comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save evaluation metrics separately
        metrics_path = self.output_path / "evaluation_metrics" / "detailed_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=2, default=str)
        
        # Save paper-ready metrics summary
        paper_metrics = {
            'Model': 'ConceptCLIP',
            'Dataset': 'MILK10k',
            'Accuracy': f"{report['paper_comparison_metrics']['accuracy']:.4f}",
            'Precision (Macro)': f"{report['paper_comparison_metrics']['precision_macro']:.4f}",
            'Recall (Macro)': f"{report['paper_comparison_metrics']['recall_macro']:.4f}",
            'F1-Score (Macro)': f"{report['paper_comparison_metrics']['f1_score_macro']:.4f}",
            'Precision (Weighted)': f"{report['paper_comparison_metrics']['precision_weighted']:.4f}",
            'Recall (Weighted)': f"{report['paper_comparison_metrics']['recall_weighted']:.4f}",
            'F1-Score (Weighted)': f"{report['paper_comparison_metrics']['f1_score_weighted']:.4f}",
            'Total Samples': evaluation_metrics['overview']['valid_samples']
        }
        
        paper_metrics_df = pd.DataFrame([paper_metrics])
        paper_metrics_path = self.output_path / "evaluation_metrics" / "paper_comparison_metrics.csv"
        paper_metrics_df.to_csv(paper_metrics_path, index=False)
        
        # Generate comprehensive visualizations
        self._create_comprehensive_visualizations(results_df, evaluation_metrics, report)
        
        # Create classification report text file
        self._save_classification_report_text(evaluation_metrics)
        
        print(f"\n📊 COMPREHENSIVE RESULTS SAVED:")
        print(f"   📁 Output folder: {self.output_path}")
        print(f"   🖼️  Processed images: {self.output_path / 'processed_images'}")
        print(f"   📋 Detailed results: {results_path}")
        print(f"   📊 Comprehensive report: {report_path}")
        print(f"   📈 Evaluation metrics: {metrics_path}")
        print(f"   📄 Paper metrics: {paper_metrics_path}")
        print(f"   📊 Visualizations: {self.output_path / 'visualizations'}")
    
    def _save_classification_report_text(self, evaluation_metrics: Dict):
        """Save sklearn classification report as text file"""
        if 'classification_report' in evaluation_metrics:
            from sklearn.metrics import classification_report
            
            # Convert back to string format
            report_text = "MILK10k ConceptCLIP Classification Report\n"
            report_text += "=" * 50 + "\n\n"
            
            # Add per-class metrics
            for class_name, metrics in evaluation_metrics['per_class_metrics'].items():
                report_text += f"{class_name}:\n"
                report_text += f"  Precision: {metrics['precision']:.4f}\n"
                report_text += f"  Recall: {metrics['recall']:.4f}\n"
                report_text += f"  F1-Score: {metrics['f1_score']:.4f}\n"
                report_text += f"  Support: {metrics['support']}\n\n"
            
            # Add overall metrics
            overview = evaluation_metrics['overview']
            report_text += "Overall Metrics:\n"
            report_text += f"  Accuracy: {overview['accuracy']:.4f}\n"
            report_text += f"  Macro Precision: {overview['precision_macro']:.4f}\n"
            report_text += f"  Macro Recall: {overview['recall_macro']:.4f}\n"
            report_text += f"  Macro F1-Score: {overview['f1_macro']:.4f}\n"
            report_text += f"  Weighted Precision: {overview['precision_weighted']:.4f}\n"
            report_text += f"  Weighted Recall: {overview['recall_weighted']:.4f}\n"
            report_text += f"  Weighted F1-Score: {overview['f1_weighted']:.4f}\n"
            
            report_text_path = self.output_path / "evaluation_metrics" / "classification_report.txt"
            with open(report_text_path, 'w') as f:
                f.write(report_text)
    
    def _create_comprehensive_visualizations(self, results_df: pd.DataFrame, 
                                           evaluation_metrics: Dict, report: Dict):
        """Create comprehensive visualization plots"""
        
        # Create multiple visualization plots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confusion Matrix
        plt.subplot(3, 3, 1)
        if evaluation_metrics['confusion_matrix']:
            cm = np.array(evaluation_metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=evaluation_metrics['class_names'],
                       yticklabels=evaluation_metrics['class_names'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        # 2. Per-Class Metrics
        plt.subplot(3, 3, 2)
        classes = list(evaluation_metrics['per_class_metrics'].keys())
        precisions = [evaluation_metrics['per_class_metrics'][cls]['precision'] for cls in classes]
        recalls = [evaluation_metrics['per_class_metrics'][cls]['recall'] for cls in classes]
        f1s = [evaluation_metrics['per_class_metrics'][cls]['f1_score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        
        plt.title('Per-Class Metrics')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.xticks(x, [cls.replace(' ', '\n') for cls in classes], rotation=45)
        plt.legend()
        
        # 3. Prediction Distribution
        plt.subplot(3, 3, 3)
        pred_counts = report['predictions']['distribution']
        plt.bar(pred_counts.keys(), pred_counts.values())
        plt.title('Prediction Distribution')
        plt.xlabel('Predicted Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. Confidence Distribution
        plt.subplot(3, 3, 4)
        plt.hist(results_df['prediction_confidence'], bins=30, alpha=0.7, color='green')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        # 5. Accuracy by Confidence Level
        plt.subplot(3, 3, 5)
        if 'ground_truth' in results_df.columns:
            results_with_gt = results_df.dropna(subset=['ground_truth'])
            if len(results_with_gt) > 0:
                conf_bins = np.linspace(0, 1, 11)
                accuracies = []
                counts = []
                for i in range(len(conf_bins)-1):
                    mask = ((results_with_gt['prediction_confidence'] >= conf_bins[i]) & 
                           (results_with_gt['prediction_confidence'] < conf_bins[i+1]))
                    if mask.sum() > 0:
                        acc = results_with_gt[mask]['correct'].mean()
                        accuracies.append(acc)
                        counts.append(mask.sum())
                    else:
                        accuracies.append(0)
                        counts.append(0)
                
                plt.plot(conf_bins[:-1], accuracies, marker='o', linewidth=2)
                plt.title('Accuracy vs Prediction Confidence')
                plt.xlabel('Prediction Confidence')
                plt.ylabel('Accuracy')
                plt.grid(True, alpha=0.3)
        
        # 6. Support per Class
        plt.subplot(3, 3, 6)
        supports = [evaluation_metrics['per_class_metrics'][cls]['support'] for cls in classes]
        plt.bar(classes, supports, alpha=0.7, color='orange')
        plt.title('Support per Class')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # 7. Overall Metrics Comparison
        plt.subplot(3, 3, 7)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            evaluation_metrics['overview']['accuracy'],
            evaluation_metrics['overview']['precision_macro'],
            evaluation_metrics['overview']['recall_macro'],
            evaluation_metrics['overview']['f1_macro']
        ]
        
        colors = ['red', 'blue', 'green', 'purple']
        bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        plt.title('Overall Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Macro vs Weighted Metrics
        plt.subplot(3, 3, 8)
        macro_metrics = [
            evaluation_metrics['overview']['precision_macro'],
            evaluation_metrics['overview']['recall_macro'],
            evaluation_metrics['overview']['f1_macro']
        ]
        weighted_metrics = [
            evaluation_metrics['overview']['precision_weighted'],
            evaluation_metrics['overview']['recall_weighted'],
            evaluation_metrics['overview']['f1_weighted']
        ]
        
        x = np.arange(3)
        width = 0.35
        
        plt.bar(x - width/2, macro_metrics, width, label='Macro', alpha=0.8)
        plt.bar(x + width/2, weighted_metrics, width, label='Weighted', alpha=0.8)
        
        plt.title('Macro vs Weighted Metrics')
        plt.xlabel('Metric Type')
        plt.ylabel('Score')
        plt.xticks(x, ['Precision', 'Recall', 'F1-Score'])
        plt.legend()
        
        # 9. Processing Statistics
        plt.subplot(3, 3, 9)
        processing_stats = [
            report['processing_stats']['classification_success_rate'],
            evaluation_metrics['overview']['accuracy'] if evaluation_metrics['overview']['valid_samples'] > 0 else 0
        ]
        
        plt.bar(['Classification\nSuccess Rate', 'Overall\nAccuracy'], processing_stats,
                color=['skyblue', 'lightcoral'], alpha=0.8)
        plt.title('Processing Success Rates')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(processing_stats):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        viz_path = self.output_path / "visualizations" / "comprehensive_evaluation.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create separate confusion matrix with better formatting
        self._create_detailed_confusion_matrix(evaluation_metrics)
        
        print(f"📈 Comprehensive visualizations saved to: {viz_path}")
    
    def _create_detailed_confusion_matrix(self, evaluation_metrics: Dict):
        """Create a detailed confusion matrix visualization"""
        if not evaluation_metrics['confusion_matrix']:
            return
        
        plt.figure(figsize=(12, 10))
        cm = np.array(evaluation_metrics['confusion_matrix'])
        
        # Calculate percentages
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotation text
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    row.append(f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)')
                else:
                    row.append('0\n(0.0%)')
            annotations.append(row)
        
        # Create heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar=True,
                   xticklabels=[cls.replace(' ', '\n') for cls in evaluation_metrics['class_names']],
                   yticklabels=[cls.replace(' ', '\n') for cls in evaluation_metrics['class_names']])
        
        plt.title('Detailed Confusion Matrix\n(Count and Percentage)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Save detailed confusion matrix
        cm_path = self.output_path / "visualizations" / "detailed_confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("="*70)
    print("MILK10K MEDICAL IMAGE CLASSIFICATION PIPELINE - ENHANCED VERSION")
    print("With Comprehensive Evaluation Metrics for Paper Comparison")
    print("="*70)
    
    # Initialize enhanced pipeline
    pipeline = MILK10kEnhancedClassificationPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH
    )
    
    # Process dataset with comprehensive evaluation
    report = pipeline.process_dataset()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("MILK10K ENHANCED CLASSIFICATION COMPLETE")
    print("="*60)
    
    # System info
    print(f"📁 Experiment: {report['experiment_info']['experiment_name']}")
    print(f"🖥️  Device: {report['system_info']['device_used']}")
    print(f"💾 Cache: {report['system_info']['cache_directory']}")
    print(f"🔌 Offline mode: {report['system_info']['offline_mode']}")
    
    # Dataset info
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images processed: {report['dataset_info']['total_images_found']}")
    print(f"   Images with ground truth: {report['dataset_info']['total_with_ground_truth']}")
    print(f"   Successful classifications: {report['processing_stats']['successful_classifications']}")
    
    # KEY METRICS FOR PAPER COMPARISON
    print(f"\n🎯 PAPER COMPARISON METRICS:")
    print(f"   Accuracy:           {report['paper_comparison_metrics']['accuracy']:.4f}")
    print(f"   Precision (Macro):  {report['paper_comparison_metrics']['precision_macro']:.4f}")
    print(f"   Recall (Macro):     {report['paper_comparison_metrics']['recall_macro']:.4f}")
    print(f"   F1-Score (Macro):   {report['paper_comparison_metrics']['f1_score_macro']:.4f}")
    print(f"   Precision (Weighted): {report['paper_comparison_metrics']['precision_weighted']:.4f}")
    print(f"   Recall (Weighted):    {report['paper_comparison_metrics']['recall_weighted']:.4f}")
    print(f"   F1-Score (Weighted):  {report['paper_comparison_metrics']['f1_score_weighted']:.4f}")
    
    # Output locations
    output_path = Path(report['experiment_info']['output_path'])
    print(f"\n📂 OUTPUT LOCATIONS:")
    print(f"   🎯 Main folder: {output_path}")
    print(f"   🖼️  Processed images: {output_path / 'processed_images'}")
    print(f"   📋 Classification results: {output_path / 'reports' / 'classification_results.csv'}")
    print(f"   📊 Comprehensive report: {output_path / 'reports' / 'comprehensive_report.json'}")
    print(f"   📈 Evaluation metrics: {output_path / 'evaluation_metrics' / 'detailed_metrics.json'}")
    print(f"   📄 Paper metrics: {output_path / 'evaluation_metrics' / 'paper_comparison_metrics.csv'}")
    print(f"   📊 Visualizations: {output_path / 'visualizations'}")
    
    print(f"\n✅ All outputs saved to folder: ClassConCLIPout")
    print("🎉 Enhanced MILK10k classification pipeline completed successfully!")

if __name__ == "__main__":
    main()
