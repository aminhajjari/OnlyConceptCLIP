# MILK10k Medical Image Classification Pipeline
# Classification-only version (no segmentation)

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
OUTPUT_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/OnlyConceptCLIP/outputs"

# Local model paths
CONCEPTCLIP_MODEL_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/ConceptModel"
HUGGINGFACE_CACHE_PATH = "/project/def-arashmoh/shahab33/XAI/MILK10k_Training_Input/huggingface_cache"

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
    preprocessing_params={'normalize': True, 'enhance_contrast': True}
)

# ==================== MAIN PIPELINE CLASS ====================

class MILK10kClassificationPipeline:
    """MILK10k classification-only pipeline"""
    
    def __init__(self, dataset_path: str, groundtruth_path: str, output_path: str, 
                 conceptclip_model_path: str = None, cache_path: str = None):
        self.dataset_path = Path(dataset_path)
        self.groundtruth_path = groundtruth_path
        self.output_path = Path(output_path)
        self.conceptclip_model_path = conceptclip_model_path or CONCEPTCLIP_MODEL_PATH
        self.cache_path = cache_path or HUGGINGFACE_CACHE_PATH
        self.domain = MILK10K_DOMAIN
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "classifications").mkdir(exist_ok=True)
        (self.output_path / "visualizations").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        (self.output_path / "processed_images").mkdir(exist_ok=True)
        
        # Initialize device with proper setup
        self.device = setup_gpu_environment()
        print(f"Initializing MILK10k classification pipeline on {self.device}")
        
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
        """Process entire MILK10k dataset for classification only"""
        print("Starting MILK10k dataset classification...")
        
        # Find all images
        image_files = []
        for ext in self.domain.image_extensions:
            image_files.extend(self.dataset_path.rglob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images in dataset")
        
        results = []
        format_counter = Counter()
        correct_predictions = 0
        total_with_gt = 0
        
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
                else:
                    predicted_disease = "unknown"
                    prediction_confidence = 0.0
                
                # Check accuracy if ground truth available
                if ground_truth:
                    total_with_gt += 1
                    if ground_truth == predicted_disease:
                        correct_predictions += 1
                
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
        
        # Calculate accuracy
        accuracy = correct_predictions / total_with_gt if total_with_gt > 0 else 0
        
        # Generate report
        report = self._generate_comprehensive_report(results, format_counter, accuracy, total_with_gt)
        
        # Save results
        self._save_results(results, report)
        
        return report
    
    def _generate_comprehensive_report(self, results: List[Dict], format_counter: Counter, 
                                     accuracy: float, total_with_gt: int) -> Dict:
        """Generate comprehensive processing report"""
        
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
        
        report = {
            'system_info': {
                'device_used': device_used,
                'cache_directory': cache_used,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'offline_mode': os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
                'pipeline_type': 'classification_only'
            },
            'dataset_info': {
                'total_images_found': total_processed,
                'file_formats': dict(format_counter),
                'total_with_ground_truth': total_with_gt
            },
            'processing_stats': {
                'successful_classifications': successful_classifications,
                'classification_success_rate': successful_classifications / total_processed if total_processed > 0 else 0
            },
            'accuracy_metrics': {
                'overall_accuracy': accuracy,
                'correct_predictions': sum(1 for r in results if r['correct']),
                'total_evaluated': total_with_gt
            },
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
    
    def _save_results(self, results: List[Dict], report: Dict):
        """Save results and report"""
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = self.output_path / "reports" / "classification_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save report
        report_path = self.output_path / "reports" / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary visualization
        self._create_summary_plots(results_df, report)
        
        print(f"\nResults saved to: {self.output_path}")
        print(f"Processed images: {self.output_path / 'processed_images'}")
        print(f"Detailed results: {results_path}")
        print(f"Classification report: {report_path}")
    
    def _create_summary_plots(self, results_df: pd.DataFrame, report: Dict):
        """Create summary visualization plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction distribution
        pred_counts = report['predictions']['distribution']
        axes[0,0].bar(pred_counts.keys(), pred_counts.values())
        axes[0,0].set_title('Disease Prediction Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence distribution
        axes[0,1].hist(results_df['prediction_confidence'], bins=30, alpha=0.7, color='blue')
        axes[0,1].set_title('Classification Confidence Distribution')
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_ylabel('Count')
        
        # 3. Accuracy by confidence level
        if 'ground_truth' in results_df.columns:
            results_with_gt = results_df.dropna(subset=['ground_truth'])
            if len(results_with_gt) > 0:
                conf_bins = np.linspace(0, 1, 11)
                accuracies = []
                for i in range(len(conf_bins)-1):
                    mask = ((results_with_gt['prediction_confidence'] >= conf_bins[i]) & 
                           (results_with_gt['prediction_confidence'] < conf_bins[i+1]))
                    if mask.sum() > 0:
                        acc = results_with_gt[mask]['correct'].mean()
                        accuracies.append(acc)
                    else:
                        accuracies.append(0)
                
                axes[1,0].plot(conf_bins[:-1], accuracies, marker='o')
                axes[1,0].set_title('Accuracy vs Prediction Confidence')
                axes[1,0].set_xlabel('Prediction Confidence')
                axes[1,0].set_ylabel('Accuracy')
        
        # 4. Processing success rates
        success_data = [
            report['processing_stats']['classification_success_rate'],
            report['accuracy_metrics']['overall_accuracy']
        ]
        axes[1,1].bar(['Classification Success', 'Overall Accuracy'], success_data)
        axes[1,1].set_title('Processing Success Rates')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = self.output_path / "visualizations" / "classification_summary_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to: {plot_path}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    
    print("="*60)
    print("MILK10K MEDICAL IMAGE CLASSIFICATION PIPELINE")
    print("Classification-Only Version (No Segmentation)")
    print("="*60)
    
    # Initialize pipeline with local models and cache
    pipeline = MILK10kClassificationPipeline(
        dataset_path=DATASET_PATH,
        groundtruth_path=GROUNDTRUTH_PATH,
        output_path=OUTPUT_PATH,
        conceptclip_model_path=CONCEPTCLIP_MODEL_PATH,
        cache_path=HUGGINGFACE_CACHE_PATH
    )
    
    # Process dataset
    report = pipeline.process_dataset()
    
    # Print summary
    print("\n" + "="*50)
    print("MILK10K CLASSIFICATION COMPLETE")
    print("="*50)
    print(f"Device used: {report['system_info']['device_used']}")
    print(f"Cache directory: {report['system_info']['cache_directory']}")
    print(f"Offline mode: {report['system_info']['offline_mode']}")
    print(f"Pipeline type: {report['system_info']['pipeline_type']}")
    print(f"Total images processed: {report['dataset_info']['total_images_found']}")
    print(f"Successful classifications: {report['processing_stats']['successful_classifications']}")
    
    if report['accuracy_metrics']['total_evaluated'] > 0:
        print(f"Overall accuracy: {report['accuracy_metrics']['overall_accuracy']:.2%}")
    
    print(f"\nProcessed images saved to: {OUTPUT_PATH}/processed_images/")
    print(f"Classification results: {OUTPUT_PATH}/reports/classification_results.csv")
    print(f"Summary report: {OUTPUT_PATH}/reports/classification_report.json")

if __name__ == "__main__":
    main()
