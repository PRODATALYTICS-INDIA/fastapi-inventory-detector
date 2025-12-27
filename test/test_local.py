#!/usr/bin/env python3
"""
Local Test Suite - Runs detection, OCR, and visualization locally.

This test script:
1. Processes each image from test_images/ directory
2. Detects all objects using the segmentation model
3. Creates visualization with bounding boxes and masks
4. Crops detected objects and saves them
5. Runs OCR on each crop to extract text
6. Matches OCR text to SKU catalog
7. Saves all results to test_output/<image_name>/

Usage:
    python test/test_local.py [--confidence 0.5] [--image path/to/image.jpg]
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import argparse
import cv2
import json
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import project components (after path setup)
from app.config import get_model_path, get_model_metadata, APP_VERSION  # noqa: E402
from app.models.detection_model import DetectionModel  # noqa: E402
from app.models.ocr_model import OCRModel  # noqa: E402
from app.services.detection_service import DetectionService  # noqa: E402
from app.services.ocr_service import OCRService  # noqa: E402
from app.services.sku_matching_service import SKUMatchingService  # noqa: E402
from app.utils.inventory_tracker import InventoryTracker  # noqa: E402


class LocalTestRunner:
    """Runs complete detection pipeline locally without API."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the test runner with models and services.
        
        Args:
            confidence_threshold: Detection confidence threshold (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.output_base = PROJECT_ROOT / "test" / "test_output" / "local_call"
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        print("=" * 80)
        print("Initializing Models and Services")
        print("=" * 80)
        
        try:
            # Detection model
            model_path = get_model_path()
            print(f"📦 Loading detection model: {model_path}")
            self.detection_model = DetectionModel(model_path)
            self.detection_service = DetectionService(self.detection_model)
            print("✅ Detection model loaded")
            
            # OCR model
            print("📦 Initializing OCR model (PaddleOCR)...")
            self.ocr_model = OCRModel(use_gpu=False)
            self.ocr_service = OCRService(self.ocr_model, use_multiprocessing=False)
            print("✅ OCR model initialized")
            
            # SKU matching service
            print("📦 Initializing SKU matching service...")
            self.sku_service = SKUMatchingService(catalog_path=None)
            print("✅ SKU matching service initialized")
            
            # Visualization tracker (for annotated images)
            print("📦 Initializing visualization tracker...")
            self.tracker = InventoryTracker(model_path=model_path, label_mode="item_name")
            print("✅ Visualization tracker initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
    
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"Processing: {image_path.name}")
        print("=" * 80)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
        
        print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Create output directory
        output_dir = self.output_base / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Detection (matches API behavior - only use detection_service)
        print("\n🔍 Step 1: Running object detection...")
        detections, crops = self.detection_service.detect(
            image,
            confidence_threshold=self.confidence_threshold
        )
        print(f"✅ Detected {len(detections)} objects from detection_service")
        
        # Step 2: Create visualization (tracker used only for visualization, not for API response)
        print("\n🎨 Step 2: Creating visualization...")
        live_summary = {}  # Initialize empty summary (API doesn't use tracker summary)
        try:
            annotated_frame, live_summary = self.tracker.track_picture_stream(
                frame=image,
                confidence_threshold=self.confidence_threshold,
                count_all_detections=True
            )
            
            # Save detection response image
            detection_path = output_dir / "detection_response.jpg"
            cv2.imwrite(str(detection_path), annotated_frame)
            print(f"✅ Detection response saved: {detection_path}")
        except Exception as e:
            print(f"⚠️  Failed to create visualization: {e}")
            annotated_frame = image.copy()
            live_summary = {}
        
        if not detections:
            print("⚠️  No detections found")
            return False
        
        # Step 3: OCR on crops (crops are used for OCR but not saved)
        print("\n🔍 Step 3: Running OCR on crops...")
        # Extract crops if needed for OCR processing
        valid_crops = []
        for idx, crop in enumerate(crops):
            if crop is not None:
                valid_crops.append(crop)
            else:
                # If crop is None, try to extract from detection bbox
                if idx < len(detections):
                    det = detections[idx]
                    bbox = det.get('bbox', [])
                    if bbox and len(bbox) == 4:
                        from app.utils.image_utils import extract_crop_from_bbox
                        crop = extract_crop_from_bbox(image, np.array(bbox))
                        if crop is not None:
                            valid_crops.append(crop)
                            crops[idx] = crop  # Update crops list
                        else:
                            valid_crops.append(None)
                    else:
                        valid_crops.append(None)
                else:
                    valid_crops.append(None)
        
        # Run OCR on valid crops
        ocr_results = self.ocr_service.extract_text_batch(valid_crops)
        # Pad OCR results to match detections count
        while len(ocr_results) < len(detections):
            ocr_results.append(("", []))
        print(f"✅ OCR completed on {len(ocr_results)} detections")
        
        # Step 4: Generate OCR response file
        print("\n📝 Step 4: Generating OCR response file...")
        results_file = output_dir / "ocr_response.txt"
        self._write_results_file(
            results_file,
            image_path,
            detections,
            ocr_results,
            live_summary
        )
        print(f"✅ Results saved: {results_file}")
        
        # Step 5: Generate API response JSON (exactly matching FastAPI endpoint)
        print("\n📄 Step 5: Generating API response JSON (matching FastAPI endpoint)...")
        json_output_path = output_dir / "api_response.json"
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Debug: Print paths
        print(f"   Output directory: {output_dir}")
        print(f"   JSON file path: {json_output_path}")
        print(f"   Output directory exists: {output_dir.exists()}")
        print(f"   JSON file parent exists: {json_output_path.parent.exists()}")
        
        try:
            api_response = self._generate_api_response(
                image_path,
                detections,
                ocr_results,
                live_summary
            )
            
            if api_response is None:
                print("❌ ERROR: _generate_api_response returned None")
                return False
                
            print(f"✅ API response generated (type: {type(api_response).__name__})")
            
            # Verify response structure matches FastAPI
            required_keys = ['success', 'status', 'timestamp', 'model_version', 'app_version', 'processing_time_ms', 'input', 'summary', 'details']
            if isinstance(api_response, dict):
                missing_keys = [key for key in required_keys if key not in api_response]
                if missing_keys:
                    print(f"⚠️  Warning: Missing keys in response: {missing_keys}")
                else:
                    print(f"✅ Response structure verified (has all required keys)")
            else:
                print(f"⚠️  Warning: Response is not a dict: {type(api_response)}")
            
            # Save JSON response - use same method as FastAPI
            try:
                # Ensure api_response is a dict (should be from model_dump)
                if not isinstance(api_response, dict):
                    if hasattr(api_response, 'model_dump'):
                        api_response = api_response.model_dump(mode='json')
                    else:
                        raise ValueError(f"API response is not a dict or Pydantic model: {type(api_response)}")
                
                # Use orjson if available (matches FastAPI), otherwise use standard json
                try:
                    import orjson
                    # Use orjson to match FastAPI's serialization exactly
                    json_bytes = orjson.dumps(
                        api_response,
                        option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
                    )
                    with open(json_output_path, 'wb') as f:
                        f.write(json_bytes)
                    print(f"✅ API response JSON saved (using orjson, matches FastAPI): {json_output_path}")
                except ImportError:
                    # Fallback to standard json (matches FastAPI fallback)
                    with open(json_output_path, 'w', encoding='utf-8') as f:
                        json.dump(api_response, f, indent=2, ensure_ascii=False)
                    print(f"✅ API response JSON saved (using json, matches FastAPI): {json_output_path}")
                
                # Verify file was created
                if json_output_path.exists():
                    file_size = json_output_path.stat().st_size
                    print(f"✅ Verified: File exists ({file_size} bytes)")
                    
                    # Read back and verify it's valid JSON
                    try:
                        with open(json_output_path, 'r', encoding='utf-8') as f:
                            test_data = json.load(f)
                        print(f"✅ Verified: Valid JSON with {len(test_data.get('details', []))} detection details")
                    except Exception as e:
                        print(f"⚠️  Warning: File created but JSON validation failed: {e}")
                else:
                    print(f"❌ ERROR: File was not created at {json_output_path}")
                    return False
            except Exception as e:
                print(f"❌ Failed to save API JSON response: {e}")
                import traceback
                traceback.print_exc()
                # Don't return False here - continue to next step
                print("⚠️  Continuing despite JSON save error...")
        except Exception as e:
            print(f"❌ Failed to generate API JSON response: {e}")
            import traceback
            traceback.print_exc()
            # Try to create a minimal error response file
            try:
                error_response = {
                    "success": False,
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "input": {
                        "type": "image",
                        "filename": image_path.name,
                        "frame_count": 1
                    },
                    "summary": {
                        "sku_counts": {},
                        "total_detections": len(detections),
                        "validated_detections": 0
                    },
                "model_version": "unknown",
                "app_version": APP_VERSION,
                "processing_time_ms": 0,
                    "details": []
                }
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_response, f, indent=2, ensure_ascii=False)
                print(f"⚠️  Created error response file: {json_output_path}")
            except Exception as e2:
                print(f"❌ Failed to create error response file: {e2}")
        
        # Final verification
        if json_output_path.exists():
            print(f"✅ Final verification: api_response.json exists at {json_output_path}")
            return True
        else:
            print(f"❌ Final verification FAILED: api_response.json does not exist at {json_output_path}")
            return False
        
        # Print summary
        print("\n" + "-" * 80)
        print("Summary:")
        print(f"  Detections: {len(detections)}")
        print(f"  OCR results: {len(ocr_results)}")
        if live_summary:
            for item, info in live_summary.items():
                if isinstance(info, dict):
                    print(f"  {item}: {info.get('count', 0)} items")
        print("-" * 80)
        
        return True
    
    def _bbox_to_key(self, bbox: List[float]) -> str:
        """
        Convert bbox to a string key for matching.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            String key for bbox matching
        """
        if not bbox or len(bbox) < 4:
            return ""
        # Round to 1 decimal place for matching (handles floating point differences)
        return f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
    
    def _extract_all_detections_from_tracker(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> tuple:
        """
        Extract all detections and crops directly from the tracker.
        This ensures we get ALL objects that appear in the annotated image.
        
        Returns:
            Tuple of (detections_list, crops_list)
        """
        import supervision as sv
        from app.utils.image_utils import extract_crop_from_bbox, extract_crop_from_mask
        
        # Run tracker's model directly to get all detections
        results = self.tracker.model(
            image,
            conf=confidence_threshold,
            verbose=False,
            max_det=1000,
            iou=0.4
        )[0]
        
        # Convert to Supervision Detections
        detections_sv = sv.Detections.from_ultralytics(results)
        
        detections_list = []
        crops_list = []
        
        if len(detections_sv) == 0:
            return detections_list, crops_list
        
        for i in range(len(detections_sv)):
            # Get bounding box
            bbox = detections_sv.xyxy[i].tolist() if hasattr(detections_sv, 'xyxy') else []
            class_id = int(detections_sv.class_id[i]) if hasattr(detections_sv, 'class_id') else 0
            confidence = float(detections_sv.confidence[i]) if hasattr(detections_sv, 'confidence') else 0.0
            
            # Get SKU name
            sku = self.tracker.model.model.names.get(class_id, f"class_{class_id}")
            
            # Get mask if available
            mask = None
            if hasattr(detections_sv, 'mask') and detections_sv.mask is not None:
                if len(detections_sv.mask) > i:
                    mask = detections_sv.mask[i]
            
            # Extract crop
            crop = None
            if mask is not None:
                crop = extract_crop_from_mask(image, mask, np.array(bbox))
            elif bbox:
                crop = extract_crop_from_bbox(image, np.array(bbox))
            
            # Store detection info
            detection_info = {
                'sku': sku,
                'confidence': confidence,
                'bbox': bbox,
                'class_id': class_id,
                'tracker_id': i + 1,  # Use index as tracker ID
                'mask': mask
            }
            detections_list.append(detection_info)
            crops_list.append(crop)
        
        return detections_list, crops_list
    
    def _calculate_keyword_similarity(
        self,
        ocr_text: str,
        detection_sku: str
    ) -> Dict[str, Any]:
        """
        Calculate similarity between OCR text and detection_keywords from sku_master.
        
        Returns:
            Dictionary with best_match_keyword, similarity_score, and all_matches
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            # Fallback to basic string matching
            return {
                'best_match_keyword': None,
                'similarity_score': 0.0,
                'all_matches': []
            }
        
        # Get SKU info and detection_keywords
        sku_info = self.sku_service.get_sku_info(detection_sku)
        if not sku_info:
            return {
                'best_match_keyword': None,
                'similarity_score': 0.0,
                'all_matches': []
            }
        
        detection_keywords = sku_info.get('detection_keywords', [])
        if not detection_keywords or not ocr_text:
            return {
                'best_match_keyword': None,
                'similarity_score': 0.0,
                'all_matches': []
            }
        
        # Calculate similarity with each keyword
        matches = []
        ocr_lower = ocr_text.lower().strip()
        
        for keyword in detection_keywords:
            keyword_lower = keyword.lower().strip()
            
            # Use multiple fuzzy matching methods
            partial_ratio = fuzz.partial_ratio(ocr_lower, keyword_lower) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(ocr_lower, keyword_lower) / 100.0
            token_set_ratio = fuzz.token_set_ratio(ocr_lower, keyword_lower) / 100.0
            
            # Weighted average (favor partial_ratio for substring matches)
            similarity = (partial_ratio * 0.5 + token_sort_ratio * 0.3 + token_set_ratio * 0.2)
            
            matches.append({
                'keyword': keyword,
                'similarity_score': similarity,
                'partial_ratio': partial_ratio,
                'token_sort_ratio': token_sort_ratio,
                'token_set_ratio': token_set_ratio
            })
        
        # Sort by similarity score
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        best_match = matches[0] if matches else None
        
        return {
            'best_match_keyword': best_match['keyword'] if best_match else None,
            'similarity_score': best_match['similarity_score'] if best_match else 0.0,
            'all_matches': matches[:5]  # Top 5 matches
        }
    
    def _generate_api_response(
        self,
        image_path: Path,
        detections: List[dict],
        ocr_results: List[tuple],
        live_summary: dict
    ) -> Dict[str, Any]:
        """
        Generate API response JSON in the new evidence-based, frame-first format.
        
        Args:
            image_path: Original image path
            detections: List of detection dictionaries
            ocr_results: List of (ocr_text, ocr_details) tuples
            live_summary: Summary dictionary from tracker
            
        Returns:
            Dictionary matching new DetectionResponse format
        """
        import time
        import cv2
        from app.utils.inventory_summary import generate_summary
        from app.schemas.response import (
            DetectionDetail,
            DetectionModelOutput,
            OCRModelOutput,
            FinalPrediction,
            ValidationStatus,
            InputInfo,
            Metadata
        )
        from app.services.decision_engine import DecisionEngine
        
        start_time = time.time()
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        frame_h, frame_w = image.shape[:2] if image is not None else (0, 0)
        
        detection_objects = []
        min_validation_confidence = 0.6
        
        for i, detection in enumerate(detections):
            ocr_text, ocr_details = ocr_results[i] if i < len(ocr_results) else ("", [])
            
            # Get detection info (matches API behavior exactly)
            detection_conf = detection.get('confidence', 0.0)
            bbox = detection.get('bbox', [])
            detection_class = detection.get('detection_class', detection.get('sku', 'unknown'))
            
            # Note: mask is used internally for crop extraction but not included in API response
            # to keep the response clean and avoid large payloads (matches main API behavior)
            
            # ====================================================================
            # DETECTION MODEL OUTPUT (Level 3)
            # ====================================================================
            detection_model_output = DetectionModelOutput(
                matched_sku_id=detection_class,
                match_confidence=detection_conf
            )
            
            # ====================================================================
            # OCR MODEL OUTPUT (Level 3 - combines tokens and SKU matching)
            # ====================================================================
            ocr_confidence_avg = 0.0
            
            if ocr_details:
                confidences = [detail.get('confidence', 0.0) for detail in ocr_details if detail.get('text')]
                if confidences:
                    ocr_confidence_avg = sum(confidences) / len(confidences)
            
            # SKU matching (part of OCR model output)
            matched_sku_id = None
            matched_sku_name = None
            match_confidence = None
            match_method = "none"
            
            if ocr_text:
                matched_sku, match_conf, match_info = self.sku_service.find_best_sku_match(ocr_text)
                if matched_sku:
                    matched_sku_id = matched_sku
                    match_confidence = match_conf
                    match_method = match_info.get('match_method', 'fuzzy')
                    
                    sku_info = self.sku_service.get_sku_info(matched_sku) or {}
                    matched_sku_name = sku_info.get('item_name', matched_sku)
            
            ocr_model_output = OCRModelOutput(
                text_raw=ocr_text or "",
                method=match_method,
                matched_sku_id=matched_sku_id,
                match_confidence=match_confidence
            )
            
            # ====================================================================
            # FINAL PREDICTION (Level 2 - uses decision engine)
            # ====================================================================
            # Use decision engine if available, otherwise use simple logic
            try:
                from app.services.catalog_service import CatalogService
                catalog_service = CatalogService(catalog_path="app/data/sku_master.json", use_database=False)
                decision_engine = DecisionEngine(catalog_service=catalog_service)
                
                final_prediction_base = decision_engine.make_final_decision(
                    detection_class=detection_class,
                    detection_confidence=detection_conf,
                    ocr_text_raw=ocr_text or "",
                    ocr_token_confidence_avg=ocr_confidence_avg,
                    matched_sku_id=matched_sku_id,
                    ocr_match_confidence=match_confidence
                )
                
                # Handle unknown objects: ensure sku_id is "-9999" with proper placeholders (matches API)
                if detection_class == "-9999" or detection_class == "unknown":
                    final_sku_id = "-9999"
                    final_sku_name = "unknown"
                    # Use the confidence from decision engine or fallback to detection confidence
                    final_confidence = final_prediction_base.confidence if final_prediction_base.confidence > 0 else detection_conf
                    final_validation_status = ValidationStatus.NOT_VALIDATED
                    final_decision_reason = final_prediction_base.decision_reason.copy() if final_prediction_base.decision_reason else ["unknown_object"]
                else:
                    final_sku_id = final_prediction_base.sku_id
                    final_sku_name = final_prediction_base.sku_name
                    final_confidence = final_prediction_base.confidence
                    final_validation_status = final_prediction_base.validation_status
                    final_decision_reason = list(final_prediction_base.decision_reason) if final_prediction_base.decision_reason else []
                
                # Ensure we always have valid values (no None)
                if final_sku_id is None:
                    final_sku_id = "-9999"
                    final_sku_name = "unknown"
                    final_validation_status = ValidationStatus.NOT_VALIDATED
                    if not final_decision_reason:
                        final_decision_reason = ["unknown_object", "no_sku_match"]
                
                # Add bbox to final prediction
                final_prediction = FinalPrediction(
                    sku_id=final_sku_id,
                    sku_name=final_sku_name,
                    confidence=final_confidence,
                    bbox=[float(b) for b in bbox] if bbox else [0.0, 0.0, 0.0, 0.0],
                    validation_status=final_validation_status,
                    decision_reason=final_decision_reason
                )
            except Exception as e:
                # Fallback to simple logic if decision engine not available (matches API)
                # Handle unknown objects: ensure sku_id is "-9999" with proper placeholders
                detection_is_valid_sku = detection_class not in ("unknown", "-9999", None)
                
                if detection_class == "-9999" or detection_class == "unknown":
                    final_sku_id = "-9999"
                    final_sku_name = "unknown"
                    final_confidence = detection_conf
                    validation_status = ValidationStatus.NOT_VALIDATED
                    decision_reason = ["unknown_object", "legacy_fallback"]
                else:
                    final_sku_id = matched_sku_id or detection_class
                    final_sku_name = matched_sku_name or detection_class
                    final_confidence = detection_conf
                    if ocr_confidence_avg > 0:
                        final_confidence = (detection_conf * 0.6 + ocr_confidence_avg * 0.3)
                    if match_confidence:
                        final_confidence = (final_confidence * 0.9 + match_confidence * 0.1)
                    
                    # Determine validation status based on new statuses
                    if matched_sku_id and match_confidence and detection_is_valid_sku and matched_sku_id == detection_class:
                        validation_status = ValidationStatus.FULLY_VALIDATED
                    elif matched_sku_id and match_confidence:
                        if detection_is_valid_sku:
                            validation_status = ValidationStatus.OCR_VALIDATED
                        else:
                            validation_status = ValidationStatus.ONLY_OCR_VALIDATED
                    elif detection_is_valid_sku:
                        if ocr_text and ocr_confidence_avg > 0 and not matched_sku_id:
                            validation_status = ValidationStatus.OCR_NOT_VALIDATED
                        elif not ocr_text:
                            validation_status = ValidationStatus.ONLY_DETECTION_VALIDATED
                        else:
                            validation_status = ValidationStatus.DETECTION_VALIDATED
                    else:
                        validation_status = ValidationStatus.DETECTION_NOT_VALIDATED
                    
                    decision_reason = ["legacy_fallback"]
                
                # Ensure we always have valid values (no None)
                if final_sku_id is None:
                    final_sku_id = "-9999"
                    final_sku_name = "unknown"
                    validation_status = ValidationStatus.NOT_VALIDATED
                    decision_reason = ["unknown_object", "no_sku_match", "legacy_fallback"]
                
                final_prediction = FinalPrediction(
                    sku_id=final_sku_id,
                    sku_name=final_sku_name,
                    confidence=final_confidence,
                    bbox=[float(b) for b in bbox] if bbox else [0.0, 0.0, 0.0, 0.0],
                    validation_status=validation_status,
                    decision_reason=decision_reason
                )
            
            # ====================================================================
            # CREATE DETECTION DETAIL (Level 2)
            # ====================================================================
            detection_id = f"f0_d{i}"
            
            detection_detail = DetectionDetail(
                detection_id=detection_id,
                final_prediction=final_prediction,
                detection_model=detection_model_output,
                ocr_model=ocr_model_output
            )
            
            detection_objects.append(detection_detail)
        
        # ====================================================================
        # GENERATE SUMMARY
        # ====================================================================
        summary = generate_summary(detection_objects, min_confidence_threshold=min_validation_confidence)
        
        # ====================================================================
        # BUILD RESPONSE (Level 1)
        # ====================================================================
        processing_time = (time.time() - start_time) * 1000  # milliseconds
        model_metadata = get_model_metadata()
        
        # Build response using Pydantic model serialization
        from app.schemas.response import DetectionResponse, InputInfo
        from datetime import datetime
        
        try:
            response_obj = DetectionResponse(
                success=True,
                status="success",
                timestamp=datetime.now().isoformat(),
                model_version=model_metadata.get("version"),
                app_version=APP_VERSION,
                processing_time_ms=processing_time,
                input=InputInfo(
                    type="image",
                    filename=image_path.name,
                    frame_count=1
                ),
                summary=summary,
                details=detection_objects  # Level 2 - details section
            )
            
            # Use FastAPI's JSON encoder to ensure exact same serialization
            # FastAPI uses orjson if available, otherwise json with model_dump(mode='json')
            try:
                # Try to use orjson (FastAPI's preferred encoder) if available
                import orjson
                # Serialize using orjson (same as FastAPI)
                json_bytes = orjson.dumps(
                    response_obj.model_dump(mode='json'),
                    option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
                )
                # Parse back to dict to match FastAPI's response
                result = orjson.loads(json_bytes)
                print(f"   Generated response with {len(detection_objects)} detection details (using orjson)")
            except ImportError:
                # Fallback to standard json (matches FastAPI fallback)
                # Use model_dump(mode='json') to match FastAPI's default serialization
                result = response_obj.model_dump(mode='json')
                print(f"   Generated response with {len(detection_objects)} detection details (using json)")
            
            return result
        except Exception as e:
            print(f"   ⚠️  Error creating DetectionResponse: {e}")
            import traceback
            traceback.print_exc()
            # Return a minimal valid response structure matching FastAPI error format
            return {
                "success": False,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "input": {
                    "type": "image",
                    "filename": image_path.name,
                    "frame_count": 1
                },
                "summary": {
                    "sku_counts": {},
                    "total_detections": len(detection_objects),
                    "validated_detections": 0
                },
                "metadata": {
                    "model_version": model_metadata.get("version", "unknown"),
                    "app_version": APP_VERSION,
                    "processing_time_ms": processing_time
                },
                "details": []
            }
    
    def _write_results_file(
        self,
        output_path: Path,
        image_path: Path,
        detections: List[dict],
        ocr_results: List[tuple],
        live_summary: dict
    ):
        """
        Write comprehensive results to text file.
        
        Args:
            output_path: Path to output file
            image_path: Original image path
            detections: List of detection dictionaries
            ocr_results: List of (ocr_text, ocr_details) tuples
            live_summary: Summary dictionary from tracker
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DETECTION AND OCR RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Confidence Threshold: {self.confidence_threshold}\n")
            f.write(f"Total Detections: {len(detections)}\n\n")
            
            # Detection summary
            if live_summary:
                f.write("DETECTION SUMMARY:\n")
                f.write("-" * 80 + "\n")
                for item, info in live_summary.items():
                    if isinstance(info, dict):
                        count = info.get('count', 0)
                        conf_avg = info.get('confidence_avg', 0)
                        f.write(f"  {item}: {count} items (avg confidence: {conf_avg}%)\n")
                f.write("\n")
            
            # Detailed results per detection
            f.write("DETAILED RESULTS:\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, (det, (ocr_text, ocr_details)) in enumerate(zip(detections, ocr_results), start=1):
                f.write(f"Object {idx}:\n")
                f.write("-" * 80 + "\n")
                
                # Detection info
                detection_sku = det.get('sku', 'N/A')
                detection_conf = det.get('confidence', 0.0)
                bbox = det.get('bbox', [])
                tracker_id = det.get('tracker_id', 'N/A')
                
                f.write(f"  Detection SKU: {detection_sku}\n")
                f.write(f"  Detection Confidence: {detection_conf:.2%}\n")
                f.write(f"  Detection ID: {tracker_id}\n")
                if bbox:
                    f.write(f"  Bounding Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n")
                
                # OCR info
                f.write("\n  OCR Results:\n")
                if ocr_text:
                    f.write(f"    Extracted Text: '{ocr_text}'\n")
                    f.write(f"    Text Lines: {len(ocr_details)}\n")
                    if ocr_details:
                        for j, detail in enumerate(ocr_details, start=1):
                            text = detail.get('text', '')
                            conf = detail.get('confidence', 0.0)
                            f.write(f"      Line {j}: '{text}' (confidence: {conf:.2%})\n")
                else:
                    f.write("    No text extracted\n")
                
                # SKU matching
                f.write("\n  SKU Matching:\n")
                if ocr_text:
                    matched_sku, match_conf, match_info = self.sku_service.find_best_sku_match(ocr_text)
                    if matched_sku:
                        sku_info = self.sku_service.get_sku_info(matched_sku) or {}
                        sku_name = sku_info.get('item_name', matched_sku)
                        f.write(f"    Matched SKU: {matched_sku}\n")
                        f.write(f"    SKU Name: {sku_name}\n")
                        f.write(f"    Match Confidence: {match_conf:.2%}\n")
                    else:
                        f.write("    No confident match found\n")
                        f.write(f"    Best Match Confidence: {match_conf:.2%}\n")
                else:
                    f.write("    Skipped (no OCR text)\n")
                
                # Keyword similarity matching
                f.write("\n  Keyword Similarity (OCR vs Detection Keywords):\n")
                if ocr_text:
                    keyword_sim = self._calculate_keyword_similarity(ocr_text, detection_sku)
                    if keyword_sim['best_match_keyword']:
                        f.write(f"    Best Match Keyword: '{keyword_sim['best_match_keyword']}'\n")
                        f.write(f"    Similarity Score: {keyword_sim['similarity_score']:.2%}\n")
                        if keyword_sim['all_matches']:
                            f.write("    Top Matches:\n")
                            for match in keyword_sim['all_matches'][:3]:
                                f.write(f"      - '{match['keyword']}': {match['similarity_score']:.2%}\n")
                    else:
                        f.write("    No keyword matches found\n")
                else:
                    f.write("    Skipped (no OCR text)\n")
                
                f.write("\n")
    
    def run_batch(self, image_dir: Optional[Path] = None) -> None:
        """
        Process all images in the test_images directory.
        
        Args:
            image_dir: Directory containing test images (default: test/test_images)
        """
        if image_dir is None:
            image_dir = PROJECT_ROOT / "test" / "test_images"
        
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return
        
        # Find all images
        image_files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(image_dir.glob(pattern))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"\n📁 Found {len(image_files)} images to process")
        print(f"📁 Output directory: {self.output_base}\n")
        
        # Process each image
        success_count = 0
        for image_path in image_files:
            try:
                if self.process_image(image_path):
                    success_count += 1
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Final summary
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(image_files) - success_count}")
        print(f"Results saved to: {self.output_base}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Local test suite for detection, OCR, and visualization"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Process a single image (default: process all images in test_images/)"
    )
    
    args = parser.parse_args()
    
    try:
        runner = LocalTestRunner(confidence_threshold=args.confidence)
        
        if args.image:
            # Process single image - resolve path to test/test_images
            image_path = Path(args.image)
            if not image_path.is_absolute():
                # Extract just the filename (handle cases like "test_images/file.jpg" or just "file.jpg")
                image_name = image_path.name
                image_path = PROJECT_ROOT / "test" / "test_images" / image_name
            runner.process_image(image_path)
        else:
            # Process all images
            runner.run_batch()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

