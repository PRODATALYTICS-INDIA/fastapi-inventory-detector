"""
Image utility functions for crop extraction and preprocessing.
"""
from typing import Optional
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def extract_crop_from_bbox(
    image: np.ndarray, 
    bbox: np.ndarray,
    padding: int = 10
) -> Optional[np.ndarray]:
    """
    Extract a crop from an image using bounding box coordinates.
    
    Args:
        image: Input image as numpy array (BGR format)
        bbox: Bounding box as [x1, y1, x2, y2]
        padding: Additional padding in pixels around the bbox
        
    Returns:
        Cropped image as numpy array, or None if invalid
    """
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Validate dimensions
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = image[y1:y2, x1:x2]
        
        # Ensure minimum size
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        return crop
        
    except Exception as e:
        logger.error(f"Error extracting crop: {str(e)}")
        return None


def extract_crop_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    padding: int = 10
) -> Optional[np.ndarray]:
    """
    Extract a crop from an image using segmentation mask.
    Optionally uses bbox for faster extraction.
    
    Args:
        image: Input image as numpy array (BGR format)
        mask: Binary mask as numpy array (same size as image)
        bbox: Optional bounding box [x1, y1, x2, y2] for faster extraction
        padding: Additional padding in pixels
        
    Returns:
        Cropped image as numpy array, or None if invalid
    """
    try:
        if bbox is not None:
            # Use bbox for faster extraction
            return extract_crop_from_bbox(image, bbox, padding)
        
        # Extract using mask bounds
        h, w = image.shape[:2]
        
        # Find bounding box from mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Validate dimensions
        if x_max <= x_min or y_max <= y_min:
            return None
        
        crop = image[y_min:y_max, x_min:x_max]
        
        # Ensure minimum size
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        
        return crop
        
    except Exception as e:
        logger.error(f"Error extracting crop from mask: {str(e)}")
        return None


def preprocess_image_for_ocr(
    image: np.ndarray,
    enhance_contrast: bool = True,
    denoise: bool = True
) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: Input image as numpy array
        enhance_contrast: Apply CLAHE contrast enhancement
        denoise: Apply denoising filter
        
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    # Convert to grayscale if needed
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed
    
    # Enhance contrast using CLAHE
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Convert back to BGR if original was color
    if len(image.shape) == 3:
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        processed = gray
    
    return processed


def resize_image_for_ocr(
    image: np.ndarray,
    min_size: int = 32,
    max_size: int = 1024
) -> np.ndarray:
    """
    Resize image to optimal size for OCR.
    
    Args:
        image: Input image as numpy array
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit within max_size while maintaining aspect ratio
    scale = min(max_size / max(h, w), 1.0)
    
    # Ensure minimum size
    if min(h, w) * scale < min_size:
        scale = min_size / min(h, w)
    
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return image
