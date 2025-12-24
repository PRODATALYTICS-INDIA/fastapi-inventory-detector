"""
Detection Model Wrapper
Handles detection/segmentation model initialization and provides a clean interface for inference.
This is a model-agnostic wrapper that can work with any detection/segmentation model.
"""
from typing import Optional, Any
import numpy as np
import logging

# Model-specific imports (can be swapped for different model backends)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)


class DetectionModel:
    """
    Wrapper class for detection/segmentation models.
    Handles model loading, inference, and provides detection results.
    This is a model-agnostic interface that abstracts the underlying model implementation.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize detection model from weights file.
        
        Args:
            model_path: Path to model weights file (.pt, .onnx, etc.)
        """
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.is_segmentation: bool = False
        self.known_skus: set = set()
        self.model_backend: str = "unknown"
        
        self._load_model()
    
    def _load_model(self):
        """Load detection model and detect model type."""
        try:
            # Currently using Ultralytics/YOLO backend
            # This can be extended to support other backends (Facebook SAM, etc.)
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("Ultralytics package not available")
            
            self.model = YOLO(self.model_path)
            self.model_backend = "ultralytics"
            
            # Detect if this is a segmentation model
            self.is_segmentation = (
                hasattr(self.model, 'task') and 
                'seg' in str(self.model.task).lower()
            )
            
            # Get known SKUs from model class names
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.known_skus = set(self.model.model.names.values())
            
            model_type = "SEGMENTATION" if self.is_segmentation else "DETECTION"
            logger.info(f"Loaded {model_type} model from: {self.model_path}")
            logger.info(f"Model backend: {self.model_backend}")
            logger.info(f"Known SKUs: {len(self.known_skus)}")
            
        except Exception as e:
            logger.error(f"Failed to load detection model: {str(e)}")
            raise
    
    def predict(
        self, 
        image: np.ndarray, 
        confidence_threshold: float = 0.5,
        max_det: int = 1000,
        iou: float = 0.4
    ) -> Any:
        """
        Run detection inference on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Confidence threshold (0.0-1.0)
            max_det: Maximum number of detections
            iou: IoU threshold for NMS
            
        Returns:
            Model results object containing detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Currently using Ultralytics interface
        # This can be extended to support other model backends
        if self.model_backend == "ultralytics":
            results = self.model(
                image,
                conf=confidence_threshold,
                verbose=False,
                max_det=max_det,
                iou=iou
            )[0]
            return results
        else:
            raise NotImplementedError(f"Model backend {self.model_backend} not implemented")
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get class name for a given class ID.
        
        Args:
            class_id: Class ID from model
            
        Returns:
            Class name (SKU code)
        """
        if self.model is None:
            return f"class_{class_id}"
        
        # Currently using Ultralytics interface
        if self.model_backend == "ultralytics":
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                return self.model.model.names.get(class_id, f"class_{class_id}")
        
        return f"class_{class_id}"
    
    def get_class_mapping(self) -> dict:
        """
        Get full mapping of class_id to class_name from the model.
        
        Returns:
            Dictionary mapping class_id (int) to class_name (str)
        """
        if self.model is None:
            return {}
        
        if self.model_backend == "ultralytics":
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                return dict(self.model.model.names)
        
        return {}