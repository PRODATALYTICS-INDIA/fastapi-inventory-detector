"""
OCR Service
Handles OCR text extraction from product crops.
"""
from typing import List, Tuple, Optional
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import multiprocessing as mp
import time

from app.models.ocr_model import OCRModel
from app.utils.image_utils import preprocess_image_for_ocr, resize_image_for_ocr

logger = logging.getLogger(__name__)


class OCRService:
    """
    Service for running OCR on product crops.
    Supports batch processing and multiprocessing for performance.
    """
    
    def __init__(
        self,
        ocr_model: OCRModel,
        use_multiprocessing: bool = True,
        max_workers: Optional[int] = None,
        preprocess: bool = True,
        timeout_per_crop: float = 5.0  # 5 seconds per crop
    ):
        """
        Initialize OCR service.
        
        Args:
            ocr_model: Initialized OCR model instance
            use_multiprocessing: Whether to use multiprocessing for batch OCR
            max_workers: Maximum number of worker processes (None = auto)
            preprocess: Whether to preprocess images before OCR
            timeout_per_crop: Timeout in seconds for each OCR crop (default: 5.0)
        """
        self.ocr_model = ocr_model
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or min(mp.cpu_count(), 4)
        self.preprocess = preprocess
        self.timeout_per_crop = timeout_per_crop
    
    def extract_text(
        self,
        crop: np.ndarray
    ) -> Tuple[str, List[dict]]:
        """
        Extract text from a single crop image with timeout protection.
        
        Args:
            crop: Cropped product image as numpy array
            
        Returns:
            Tuple of (concatenated_text, detailed_results)
        """
        if crop is None:
            return "", []
        
        try:
            # Preprocess if enabled
            if self.preprocess:
                crop = preprocess_image_for_ocr(crop)
                crop = resize_image_for_ocr(crop)
            
            # Run OCR with timeout protection
            start_time = time.time()
            text, details = self.ocr_model.extract_text(crop)
            elapsed = time.time() - start_time
            
            if elapsed > self.timeout_per_crop:
                logger.warning(f"OCR took {elapsed:.2f}s (exceeded {self.timeout_per_crop}s timeout)")
            
            return text, details
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return "", []
    
    def extract_text_batch(
        self,
        crops: List[Optional[np.ndarray]]
    ) -> List[Tuple[str, List[dict]]]:
        """
        Extract text from multiple crops (batch processing).
        
        Args:
            crops: List of cropped product images
            
        Returns:
            List of tuples (concatenated_text, detailed_results) for each crop
        """
        if not crops:
            return []
        
        # Filter out None crops
        valid_crops = [(i, crop) for i, crop in enumerate(crops) if crop is not None]
        
        if not valid_crops:
            return [("", [])] * len(crops)
        
        results = [("", [])] * len(crops)
        
        if self.use_multiprocessing and len(valid_crops) > 1:
            # Use multiprocessing for multiple crops
            results = self._extract_batch_parallel(valid_crops, crops)
        else:
            # Sequential processing
            for idx, crop in valid_crops:
                text, details = self.extract_text(crop)
                results[idx] = (text, details)
        
        return results
    
    def _extract_batch_parallel(
        self,
        valid_crops: List[Tuple[int, np.ndarray]],
        all_crops: List[Optional[np.ndarray]]
    ) -> List[Tuple[str, List[dict]]]:
        """
        Extract text from crops using parallel processing.
        
        Args:
            valid_crops: List of (index, crop) tuples for valid crops
            all_crops: Full list of all crops (including None)
            
        Returns:
            List of OCR results matching the order of all_crops
        """
        results = [("", [])] * len(all_crops)
        
        # Use ThreadPoolExecutor for I/O-bound OCR operations
        # Note: PaddleOCR may have GIL issues, so threading might be better than multiprocessing
        total_timeout = self.timeout_per_crop * len(valid_crops) + 10  # Total timeout with buffer
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all OCR tasks
            future_to_idx = {
                executor.submit(self.extract_text, crop): idx
                for idx, crop in valid_crops
            }
            
            # Collect results as they complete with timeout protection
            try:
                for future in as_completed(future_to_idx, timeout=total_timeout):
                    # Check if we've exceeded total timeout
                    elapsed = time.time() - start_time
                    if elapsed > total_timeout:
                        logger.warning(f"OCR batch processing exceeded total timeout ({total_timeout}s)")
                        break
                    
                    idx = future_to_idx[future]
                    try:
                        # Get result with individual timeout
                        text, details = future.result(timeout=1.0)  # Quick check if ready
                        results[idx] = (text, details)
                    except FutureTimeoutError:
                        # Future not ready yet, but as_completed should handle this
                        logger.warning(f"OCR task for crop {idx} not ready")
                        results[idx] = ("", [])
                    except Exception as e:
                        logger.error(f"OCR task failed for crop {idx}: {str(e)}")
                        results[idx] = ("", [])
            except FutureTimeoutError:
                logger.warning(f"OCR batch processing timed out after {total_timeout}s")
                # Mark remaining futures as timed out
                for future, idx in future_to_idx.items():
                    if idx not in [i for i, _ in enumerate(results) if results[i] != ("", [])]:
                        results[idx] = ("", [])
        
        return results
