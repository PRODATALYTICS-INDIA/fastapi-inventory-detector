"""
OCR Model Wrapper using PaddleOCR
Handles OCR model initialization and provides text extraction from images.
"""
from typing import List, Tuple, Optional
import numpy as np
import logging
import os

# Set PaddleOCR allocator strategy before any Paddle imports to prevent OOM
# This must be set before importing paddle or paddleocr
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

logger = logging.getLogger(__name__)


class OCRModel:
    """
    Wrapper class for PaddleOCR.
    Handles OCR model loading and text extraction from product images.
    """
    
    def __init__(
        self, 
        use_angle_cls: bool = True,
        lang: str = 'en',
        use_gpu: bool = False,
        enable_mkldnn: bool = False
    ):
        """
        Initialize PaddleOCR model.
        
        Args:
            use_angle_cls: Whether to use angle classifier
            lang: Language code ('en' for English)
            use_gpu: Whether to use GPU acceleration
            enable_mkldnn: Enable MKLDNN acceleration (CPU optimization)
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: pip install paddlepaddle paddleocr"
            )
        
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self.ocr: Optional[PaddleOCR] = None
        
        self._load_model()
    
    def _load_model(self):
        """Lazy load OCR model on first use."""
        # Model will be loaded on first inference call
        logger.info("OCR model will be loaded on first inference")
    
    def _ensure_loaded(self):
        """Ensure OCR model is loaded."""
        if self.ocr is None:
            try:
                logger.info("Loading PaddleOCR model...")
                # Note: Some PaddleOCR versions do not accept newer keyword args
                # like `use_gpu` or `enable_mkldnn`. To keep compatibility across
                # versions, we only pass the arguments that are widely supported.
                self.ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                )
                logger.info("PaddleOCR model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR model: {str(e)}")
                raise
    
    def extract_text(
        self, 
        image: np.ndarray
    ) -> Tuple[str, List[dict]]:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image as numpy array (BGR or RGB format)
            
        Returns:
            Tuple of (concatenated_text, detailed_results)
            - concatenated_text: All extracted text joined with spaces
            - detailed_results: List of dicts with 'text', 'confidence', 'bbox'
        """
        self._ensure_loaded()
        
        if self.ocr is None:
            raise RuntimeError("OCR model not loaded")
        
        try:
            # PaddleOCR expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                image_rgb = image[:, :, ::-1] if image.dtype == np.uint8 else image
            else:
                image_rgb = image
            
            # Run OCR - handle different PaddleOCR API versions
            try:
                # Try with cls parameter (newer versions)
                if self.use_angle_cls:
                    ocr_results = self.ocr.ocr(image_rgb, cls=True)
                else:
                    ocr_results = self.ocr.ocr(image_rgb, cls=False)
            except TypeError as e:
                # Fallback for older versions that don't support cls parameter
                try:
                    ocr_results = self.ocr.ocr(image_rgb)
                except Exception as e2:
                    logger.error(f"OCR call failed with both methods: {str(e2)}")
                    return "", []
            except Exception as e:
                logger.error(f"OCR call failed: {str(e)}")
                return "", []
            
            # Parse results
            if not ocr_results or ocr_results is None:
                return "", []
            
            # Handle case where OCR returns empty list
            if isinstance(ocr_results, list) and len(ocr_results) == 0:
                return "", []
            
            text_lines = []
            detailed_results = []
            
            # Some PaddleOCR versions return [ [ (bbox, (text, conf)), ... ] ]
            # others may return [ (bbox, (text, conf)), ... ]. Handle both.
            lines = ocr_results[0] if isinstance(ocr_results[0], list) else ocr_results
            
            for line in lines:
                if not line or len(line) < 2:
                    continue
                
                bbox, (text, confidence) = line[0], line[1]
                
                # Be very lenient here so we can see what OCR is doing;
                # higher-level code can decide how to filter.
                if text:
                    text_str = str(text).strip()
                    text_lines.append(text_str)
                    detailed_results.append({
                        'text': text_str,
                        'confidence': float(confidence) if confidence is not None else 0.0,
                        'bbox': bbox  # List of 4 points
                    })
            
            concatenated_text = ' '.join(text_lines).lower()
            
            return concatenated_text, detailed_results
            
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"OCR extraction failed: {error_msg}")
            import traceback
            logger.debug(f"OCR error traceback: {traceback.format_exc()}")
            return "", []
    
    def extract_text_batch(
        self, 
        images: List[np.ndarray]
    ) -> List[Tuple[str, List[dict]]]:
        """
        Extract text from multiple images (batch processing).
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of tuples (concatenated_text, detailed_results) for each image
        """
        self._ensure_loaded()
        
        results = []
        for image in images:
            text, details = self.extract_text(image)
            results.append((text, details))
        
        return results
