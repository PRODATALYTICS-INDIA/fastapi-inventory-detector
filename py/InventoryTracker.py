# =============================================================================
# IMPORTS
# =============================================================================
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from collections import defaultdict
import os

# =============================================================================
# LABEL CATALOG - Load product metadata from JSON
# =============================================================================
import json

def load_label_catalog():
    """
    Load sku_master dataframe (only once).
    This JSON file contains product metadata (SKU, name, brand, category, etc.)
    """
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "data", "sku_master.json")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        catalog_data = json.load(f)
    return pd.DataFrame(catalog_data)

# Load catalog once at module level for efficiency
label_catalog = load_label_catalog()

# Build lookup dictionary for fast metadata access by SKU code
# Support both 'code' and 'sku_code' column names for backward compatibility
code_column = "code" if "code" in label_catalog.columns else "sku_code"
# Example: {"Sku1_Dettol Soap": {"item_name": "Product A", "brand": "Brand X", ...}, ...}
sku_lookup = label_catalog.set_index(code_column).to_dict(orient="index") 

# =============================================================================
# YOLO INVENTORY TRACKER CLASS
# =============================================================================
class InventoryTracker:
    def __init__(self, model_path=None, label_mode="item_name"):
        """
        Initializes the tracker with YOLO model (detection or segmentation) and summary stats.
        
        This class supports both model types with automatic detection:
        - YOLOv11-seg segmentation models (masks + bounding boxes) - RECOMMENDED
        - YOLOv11 detection models (bounding boxes only) - Legacy support
        
        The system automatically detects the model type and initializes appropriate annotators.
        Segmentation models provide pixel-accurate masks for better accuracy.
        
        Args:
            model_path (str): Path to YOLO model weights (.pt file).
                             Automatically detects if it's a detection or segmentation model.
                             If None, uses default: models/segmentation-model.pt (with fallback to versioned name)
            label_mode (str): Label to display on frames and aggregate stats.
                             Options: "sku_code", "item_name", "brand", "sub_category", "category"
        """
        # Step 1: Determine model path
        if model_path is None:
            model_path = "models/segmentation-model.pt"
        
        # Step 2: Load YOLO model (automatically detects if it's detection or segmentation)
        self.model = YOLO(model_path)
        
        # Step 2: Check if this is a segmentation model
        # Segmentation models have 'seg' in their task name
        self.is_segmentation = hasattr(self.model, 'task') and 'seg' in str(self.model.task).lower()
        
        # Step 2.5: Get list of known/trained SKUs from model (for validation)
        # This helps filter false positives from unknown objects
        self.known_skus = set(self.model.model.names.values()) if hasattr(self.model, 'model') and hasattr(self.model.model, 'names') else set()
        
        # Step 3: Initialize confidence threshold (can be updated per request)
        self.confidence_threshold = 0.0
        
        # Step 3.5: Initialize false positive filtering settings
        self.min_detection_area_ratio = 0.0001  # Minimum area as ratio of image (0.01% of image)
        self.strict_sku_validation = True  # Filter out detections not matching known SKUs
        
        # Step 4: Initialize ByteTrack tracker for object tracking across frames
        self.tracker = sv.ByteTrack()
        
        # Step 5: Initialize annotators based on model type
        if self.is_segmentation:
            # For segmentation models: use MaskAnnotator to draw filled masks
            self.mask_annotator = sv.MaskAnnotator()
            # Also use BoxAnnotator for bounding box overlay (optional, can be removed if you only want masks)
            self.box_annotator = sv.BoxAnnotator()
        else:
            # For detection models: only use BoxAnnotator for bounding boxes
            self.box_annotator = sv.BoxAnnotator()
        
        # Step 6: Initialize label and trace annotators (common for both types)
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        
        # Step 7: Store label catalog reference
        self.label_catalog = label_catalog
       
        # Step 8: Validate and assign label_mode
        self.valid_label_modes = {"sku_code", "item_name", "brand", "sub_category", "category"}
        if label_mode not in self.valid_label_modes:
            print(f"[WARN] Invalid label_mode '{label_mode}', falling back to 'item_name'")
            self.label_mode = "item_name"
        else:
            self.label_mode = label_mode

        # Step 9: Initialize statistics tracking
        self.reset_output_stats()
        
        # Step 10: Log model type for debugging
        model_type = "SEGMENTATION" if self.is_segmentation else "DETECTION"
        print(f"[INFO] Loaded {model_type} model from: {model_path}")

    def reload_model(self, model_path: str):
        """
        Reload a different model and reinitialize annotators accordingly.
        This is useful when switching between detection and segmentation models.
        
        Args:
            model_path (str): Path to the new YOLO model weights (.pt file).
        """
        # Step 1: Load the new model
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Step 2: Re-detect if this is a segmentation model
        self.is_segmentation = hasattr(self.model, 'task') and 'seg' in str(self.model.task).lower()
        
        # Step 2.5: Update known SKUs list for false positive filtering
        self.known_skus = set(self.model.model.names.values()) if hasattr(self.model, 'model') and hasattr(self.model.model, 'names') else set()
        
        # Step 3: Reinitialize annotators based on the new model type
        if self.is_segmentation:
            # For segmentation models: use MaskAnnotator to draw filled masks
            self.mask_annotator = sv.MaskAnnotator()
            # Also use BoxAnnotator for bounding box overlay
            self.box_annotator = sv.BoxAnnotator()
        else:
            # For detection models: only use BoxAnnotator for bounding boxes
            # Remove mask_annotator if it exists (from previous segmentation model)
            if hasattr(self, 'mask_annotator'):
                delattr(self, 'mask_annotator')
            self.box_annotator = sv.BoxAnnotator()
        
        # Step 4: Reset statistics
        self.reset_output_stats()
        
        # Step 5: Log model type change
        model_type = "SEGMENTATION" if self.is_segmentation else "DETECTION"
        print(f"[INFO] Reloaded {model_type} model from: {model_path}")

    def reset_output_stats(self):
        """
        Resets the statistics for a new video or session.
        
        This method clears all tracking data:
        - Frame counter
        - Class appearance counts
        - Tracked object IDs
        - Confidence scores
        """
        self.frame_count = 0
        self.class_appearances = defaultdict(int)  # counts per sku
        self.overall_tracked_ids = defaultdict(set)  # unique IDs per sku
        self.confidence = defaultdict(list)  # list of confidence scores per sku

    def track_picture_stream(
        self, 
        frame: np.ndarray, 
        confidence_threshold: float, 
        count_all_detections: bool = True,
        min_detection_area_ratio: float = None,
        strict_sku_validation: bool = None
    ):
        """
        Core logic to process a single frame for detection/segmentation, tracking, and stats gathering.
        
        This method works with both detection and segmentation models:
        - Detection models: Returns bounding boxes
        - Segmentation models: Returns bounding boxes + segmentation masks
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV).
            confidence_threshold (float): YOLO confidence threshold (0.0-1.0).
            count_all_detections (bool): If True, count ALL detections (for object counting).
                                        If False, count only unique tracked IDs (for video tracking).
                                        Default True for accurate object counting in images.
            min_detection_area_ratio (float): Minimum detection area as ratio of image size (0.0-1.0).
                                             Filters out very small detections that are likely noise.
                                             Default: 0.0001 (0.01% of image).
            strict_sku_validation (bool): If True, filter out detections not matching known/trained SKUs.
                                         Helps prevent false positives from unknown objects.
                                         Default: True.

        Returns:
            annotated_frame (np.ndarray): Frame with annotations (boxes/masks + labels).
            live_summary (dict): Running summary of detections {label: count}.
        """
        # Step 1: Increment frame counter
        self.frame_count += 1
        
        # Step 1.5: Use provided filtering settings or defaults
        min_area_ratio = min_detection_area_ratio if min_detection_area_ratio is not None else self.min_detection_area_ratio
        strict_validation = strict_sku_validation if strict_sku_validation is not None else self.strict_sku_validation
        
        # Step 2: Run YOLO inference on the frame
        # For segmentation models, results will include masks
        # For detection models, results will only include boxes
        # max_det=None removes limit, or use high number (1000) for dense scenes with many objects
        # iou=0.4 is lower NMS threshold to avoid suppressing close objects
        results = self.model(
            frame, 
            conf=confidence_threshold, 
            verbose=False,
            max_det=1000,  # Very high limit for dense object counting
            iou=0.4  # Lower IoU threshold for NMS (default 0.7) to keep close objects
        )[0]
        
        # Step 3: Convert YOLO results to Supervision Detections format
        # This automatically handles both detection and segmentation results
        detections = sv.Detections.from_ultralytics(results)
        
        # Step 3.5: Filter out false positives and invalid detections
        if len(detections) > 0:
            frame_height, frame_width = frame.shape[:2]
            image_area = frame_height * frame_width
            min_area = image_area * min_area_ratio
            
            # Build mask for valid detections
            valid_mask = np.ones(len(detections), dtype=bool)
            
            for i in range(len(detections)):
                # Check 1: SKU validation - ensure detection matches known training SKU
                if strict_validation and hasattr(detections, 'class_id') and detections.class_id is not None:
                    class_id = int(detections.class_id[i])
                    detected_sku = self.model.model.names.get(class_id, "")
                    if detected_sku not in self.known_skus:
                        valid_mask[i] = False
                        continue
                
                # Check 2: Detection size validation - filter very small detections (likely noise)
                if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                    x1, y1, x2, y2 = detections.xyxy[i]
                    detection_area = (x2 - x1) * (y2 - y1)
                    if detection_area < min_area:
                        valid_mask[i] = False
                        continue
                
                # Check 3: Confidence validation - ensure confidence is reasonable
                if hasattr(detections, 'confidence') and detections.confidence is not None:
                    conf = float(detections.confidence[i])
                    # Additional validation: if confidence is suspiciously high (>0.99) but SKU doesn't match, flag it
                    if conf > 0.99 and strict_validation:
                        if hasattr(detections, 'class_id') and detections.class_id is not None:
                            class_id = int(detections.class_id[i])
                            detected_sku = self.model.model.names.get(class_id, "")
                            if detected_sku not in self.known_skus:
                                valid_mask[i] = False
                                continue
            
            # Apply the filter mask and log filtered detections
            filtered_count = np.sum(~valid_mask)
            if filtered_count > 0:
                # Log which detections were filtered (for debugging)
                filtered_reasons = defaultdict(int)
                for i in range(len(valid_mask)):
                    if not valid_mask[i]:
                        if strict_validation and hasattr(detections, 'class_id') and detections.class_id is not None:
                            class_id = int(detections.class_id[i])
                            detected_sku = self.model.model.names.get(class_id, "unknown")
                            if detected_sku not in self.known_skus:
                                filtered_reasons[f"Unknown SKU: {detected_sku}"] += 1
                        if hasattr(detections, 'xyxy') and detections.xyxy is not None:
                            x1, y1, x2, y2 = detections.xyxy[i]
                            detection_area = (x2 - x1) * (y2 - y1)
                            if detection_area < min_area:
                                filtered_reasons["Too small"] += 1
                
                # Apply the filter (filtered detections are silently removed)
                detections = detections[valid_mask]
        
        # Step 4: Update object tracker with new detections
        # ByteTrack assigns persistent IDs to tracked objects across frames
        # Note: ByteTrack should preserve masks automatically, but we verify this
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Verify and restore masks for segmentation models if needed
        # ByteTrack should preserve masks, but we ensure they're present
        if self.is_segmentation:
            # Check if masks exist in original detections
            if hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask) > 0:
                # If tracked_detections lost masks or has wrong number, restore them
                if not hasattr(tracked_detections, 'mask') or tracked_detections.mask is None:
                    # ByteTrack filtered all detections - use original masks if counts match
                    if len(tracked_detections) == len(detections):
                        tracked_detections.mask = detections.mask.copy()
                    elif len(tracked_detections) < len(detections):
                        # ByteTrack filtered some detections - this is expected, masks should be preserved
                        # If masks are missing, we can't perfectly restore without track IDs
                        # In this case, we'll annotate with boxes only if masks are truly lost
                        pass  # ByteTrack should preserve masks for matched detections
        
        # Step 5: Count ALL detections for accurate object counting (before tracking)
        # This is important for single images where we want to count every object, not unique tracks
        detection_counts = defaultdict(int)  # Count all detections per SKU
        detection_confidences = defaultdict(list)  # Collect all confidence scores
        
        # Count from original detections (before ByteTrack) for accurate object counting
        if count_all_detections:
            # Access class_id from Supervision Detections (it's a numpy array)
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                for i in range(len(detections)):
                    class_id = int(detections.class_id[i])
                    detected_sku = self.model.model.names[class_id]
                    detection_counts[detected_sku] += 1
                    
                    # Get confidence from original detections
                    if hasattr(detections, 'confidence') and detections.confidence is not None:
                        confidence = float(detections.confidence[i])
                    elif hasattr(results, 'boxes') and hasattr(results.boxes, 'conf'):
                        confidence = float(results.boxes.conf[i]) if len(results.boxes.conf) > i else 0.0
                    else:
                        confidence = 0.0
                    detection_confidences[detected_sku].append(confidence)
            else:
                # Fallback: count from tracked detections (should not happen, but safe fallback)
                pass
        
        # Step 6: Process each tracked detection to gather statistics and create labels
        labels = []
        for i, det in enumerate(tracked_detections):
            # Extract detection properties
            class_id = det[3]  # Class ID from model
            tracker_id = det[4]  # Unique tracker ID assigned by ByteTrack
            
            # Step 7: Get the SKU code from model class names
            detected_sku = self.model.model.names[class_id]  # e.g., "sku_1"
            
            # Step 8: Lookup product metadata from catalog
            meta = sku_lookup.get(detected_sku, {})
            
            # Step 9: Extract confidence score from results
            # Try multiple ways to get confidence as the structure may vary
            if hasattr(tracked_detections, 'confidence') and tracked_detections.confidence is not None:
                confidence = float(tracked_detections.confidence[i])
            elif hasattr(results, 'boxes') and hasattr(results.boxes, 'conf'):
                # Get confidence from YOLO results
                confidence = float(results.boxes.conf[i]) if len(results.boxes.conf) > i else 0.0
            else:
                confidence = 0.0

            # Step 10: Generate label text based on label_mode and confidence
            # If confidence < 50%, label as "unsure - low confidence"
            if confidence < 0.5:
                label_text = "unsure - low confidence"
            else:
                # Either show SKU code or a metadata field (item_name, brand, etc.)
                label_text = detected_sku if self.label_mode == "sku_code" else meta.get(self.label_mode, detected_sku)
            labels.append(f"#{tracker_id} {label_text} ({confidence*100:.1f}%)")
            
            # Step 11: Track statistics
            # For object counting: use all detections count
            # For video tracking: use unique tracked IDs
            if count_all_detections:
                # Use detection counts from all detections (accurate counting)
                # Update overall_tracked_ids for consistency, but use detection_counts for summary
                if tracker_id not in self.overall_tracked_ids[detected_sku]:
                    self.overall_tracked_ids[detected_sku].add(tracker_id)
                # Count is already in detection_counts, no need to increment here
            else:
                # Video tracking mode: count unique tracked IDs only
                if tracker_id not in self.overall_tracked_ids[detected_sku]:
                    self.overall_tracked_ids[detected_sku].add(tracker_id)
                    self.class_appearances[detected_sku] += 1
                    self.confidence[detected_sku].append(confidence)

        # Step 11: Determine which detections to annotate
        # For object counting in images: annotate ALL detections (not just tracked ones)
        # For video tracking: annotate tracked detections only
        detections_to_annotate = detections if count_all_detections else tracked_detections
        
        # Create labels for ALL detections if counting all (matching what we count)
        if count_all_detections:
            # Generate labels for all original detections to match the count
            all_labels = []
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                for i in range(len(detections)):
                    class_id = int(detections.class_id[i])
                    detected_sku = self.model.model.names[class_id]
                    meta = sku_lookup.get(detected_sku, {})
                    label_text = detected_sku if self.label_mode == "sku_code" else meta.get(self.label_mode, detected_sku)
                    
                    # Get confidence for label
                    if hasattr(detections, 'confidence') and detections.confidence is not None:
                        conf = float(detections.confidence[i])
                    elif hasattr(results, 'boxes') and hasattr(results.boxes, 'conf'):
                        conf = float(results.boxes.conf[i]) if len(results.boxes.conf) > i else 0.0
                    else:
                        conf = 0.0
                    
                    # If confidence < 50%, label as "unsure - low confidence"
                    if conf < 0.5:
                        display_label = "unsure - low confidence"
                    else:
                        display_label = label_text
                    
                    # Use index as ID for single images (no tracking needed)
                    all_labels.append(f"{display_label} ({conf*100:.1f}%)")
            labels_for_annotation = all_labels
        else:
            # Video tracking: use labels from tracked detections
            labels_for_annotation = labels
        
        # Step 12: Annotate the frame with visual overlays
        annotated_frame = frame.copy()
        
        # For segmentation models: draw masks first (as background layer)
        # Check if masks exist and are valid
        if self.is_segmentation:
            has_masks = (hasattr(detections_to_annotate, 'mask') and 
                        detections_to_annotate.mask is not None and 
                        len(detections_to_annotate.mask) > 0)
            
            if has_masks:
                annotated_frame = self.mask_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections_to_annotate
                )
            # If no masks, log a warning but continue with boxes
            elif len(detections_to_annotate) > 0:
                # Masks not available but we have detections - this can happen with ByteTrack
                pass
        
        # Step 13: Draw bounding boxes (for both detection and segmentation models)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections_to_annotate
        )
        
        # Step 14: Add text labels with tracker IDs and product names
        # Ensure labels list matches detections count
        if len(labels_for_annotation) == len(detections_to_annotate):
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections_to_annotate, 
                labels=labels_for_annotation
            )
        elif len(detections_to_annotate) > 0:
            # Fallback: create generic labels if count mismatch
            generic_labels = [f"Detection {i+1}" for i in range(len(detections_to_annotate))]
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections_to_annotate, 
                labels=generic_labels
            )
        
        # Step 15: Draw tracking traces (only for tracked detections in video mode)
        # For single images with all detections, traces aren't meaningful
        if not count_all_detections and len(tracked_detections) > 0:
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, 
                detections=tracked_detections
            )

        # Step 16: Create live summary of current detections with confidence levels
        # For object counting: use detection_counts (all detections) with confidence info
        # For video tracking: use unique tracked IDs with average confidence
        if count_all_detections:
            # Count all detections for accurate object counting with confidence
            live_summary = {}
            unsure_count = 0  # Count detections with low confidence separately
            unsure_confidences = []
            
            for sku, count in detection_counts.items():
                meta = sku_lookup.get(sku, {})
                label_text = sku if self.label_mode == "sku_code" else meta.get(self.label_mode, sku)
                
                # Calculate confidence statistics for this SKU
                conf_scores = detection_confidences.get(sku, [])
                if conf_scores:
                    avg_conf = float(np.mean(conf_scores))
                    min_conf = float(min(conf_scores))
                    max_conf = float(max(conf_scores))
                    
                    # If average confidence < 50%, group under "unsure - low confidence"
                    if avg_conf < 0.5:
                        unsure_count += count
                        unsure_confidences.extend(conf_scores)
                    else:
                        live_summary[label_text] = {
                            "count": count,
                            "confidence_avg": round(avg_conf * 100, 1),  # Percentage with 1 decimal
                            "confidence_min": round(min_conf * 100, 1),
                            "confidence_max": round(max_conf * 100, 1)
                        }
                else:
                    # No confidence scores - treat as unsure
                    unsure_count += count
            
            # Add unsure detections if any
            if unsure_count > 0:
                if unsure_confidences:
                    avg_unsure_conf = float(np.mean(unsure_confidences))
                    min_unsure_conf = float(min(unsure_confidences))
                    max_unsure_conf = float(max(unsure_confidences))
                else:
                    avg_unsure_conf = min_unsure_conf = max_unsure_conf = 0.0
                
                live_summary["unsure - low confidence"] = {
                    "count": unsure_count,
                    "confidence_avg": round(avg_unsure_conf * 100, 1),
                    "confidence_min": round(min_unsure_conf * 100, 1),
                    "confidence_max": round(max_unsure_conf * 100, 1)
                }
        else:
            # Video tracking mode: count unique tracked IDs with average confidence
            live_summary = {}
            unsure_count = 0
            unsure_confidences = []
            
            for name, ids in self.overall_tracked_ids.items():
                if len(ids) > 0:
                    conf_scores = self.confidence.get(name, [])
                    avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
                    
                    # If average confidence < 50%, group under "unsure - low confidence"
                    if avg_conf < 0.5:
                        unsure_count += len(ids)
                        unsure_confidences.extend(conf_scores)
                    else:
                        live_summary[name] = {
                            "count": len(ids),
                            "confidence_avg": round(avg_conf * 100, 1),
                            "confidence_min": round(min(conf_scores) * 100, 1) if conf_scores else 0.0,
                            "confidence_max": round(max(conf_scores) * 100, 1) if conf_scores else 0.0
                        }
            
            # Add unsure detections if any
            if unsure_count > 0:
                if unsure_confidences:
                    avg_unsure_conf = float(np.mean(unsure_confidences)) * 100
                    min_unsure_conf = float(min(unsure_confidences)) * 100
                    max_unsure_conf = float(max(unsure_confidences)) * 100
                else:
                    avg_unsure_conf = min_unsure_conf = max_unsure_conf = 0.0
                
                live_summary["unsure - low confidence"] = {
                    "count": unsure_count,
                    "confidence_avg": round(avg_unsure_conf, 1),
                    "confidence_min": round(min_unsure_conf, 1),
                    "confidence_max": round(max_unsure_conf, 1)
                }
        
        return annotated_frame, live_summary

    def get_output_stats(self):
        """
        Generates a summary DataFrame, aggregating detection statistics by SKU or another attribute.
        
        This method compiles all tracked data into a pandas DataFrame with:
        - Item counts (unique tracked IDs)
        - Average confidence scores
        - Frame presence percentages
        
        Returns:
            pd.DataFrame: Aggregated summary with columns based on label_mode.
                         Empty DataFrame if no frames have been processed.
        """
        # Step 1: Return empty DataFrame if no frames processed
        if self.frame_count == 0:
            return pd.DataFrame()
            
        summary_data = []
        
        # Step 2: Iterate through all tracked SKUs
        for sku, ids in self.overall_tracked_ids.items():
            # Skip SKUs with no tracked IDs
            if not ids:
                continue
            
            # Step 3: Get metadata for this SKU from catalog
            meta = sku_lookup.get(sku, {})
            
            # Step 4: Determine the display key based on label_mode
            key_value = sku if self.label_mode == "sku_code" else meta.get(self.label_mode, sku)

            # Step 5: Calculate statistics
            total_unique_items = len(ids)  # Number of unique tracked objects
            appearance_frames = self.class_appearances[sku]  # Number of frames where SKU appeared
            presence_percentage = (appearance_frames / self.frame_count) * 100  # % of frames with this SKU
            mean_confidence = np.mean(self.confidence.get(sku, [0])) * 100  # Average confidence score
            
            # Step 6: Append row to summary data
            summary_data.append({
                self.label_mode: key_value,
                "count": total_unique_items,
                "confidence(%)": f"{int(round(mean_confidence))}",
                "frame_presence(%)": f"{int(round(presence_percentage))}"
            })

        # Step 7: Convert to DataFrame
        output = pd.DataFrame(summary_data)

        # Step 8: If not grouping by SKU, aggregate by the chosen label_mode
        # Example: Multiple SKUs with same item_name should be combined
        if self.label_mode != "sku_code" and not output.empty:
            output = output.groupby(self.label_mode, as_index=False).agg({
                "count": "sum",  # Sum all counts for same label
                "confidence(%)": lambda x: f"{int(round(np.mean([float(v) for v in x])))}",  # Average confidence
                "frame_presence(%)": lambda x: f"{int(round(np.mean([float(v) for v in x])))}"  # Average presence
            })

        return output

    def track_video_stream(self, frame_generator, confidence_threshold):
        """
        Processes frames from a video stream and yields annotated results.
        
        This is a generator function that processes video frames one at a time,
        yielding annotated frames and live summaries as they're processed.
        Useful for streaming/real-time applications.
        
        Args:
            frame_generator (iterable): Iterator that yields frames (np.ndarray).
            confidence_threshold (float): YOLO confidence threshold (0.0-1.0).

        Yields:
            tuple: (annotated_frame, live_summary) for each processed frame
        """
        # Step 1: Reset statistics before processing video
        self.reset_output_stats()
        
        # Step 2: Process each frame from the generator
        for frame in frame_generator:
            # Yield the annotated frame and current detection summary
            yield self.track_picture_stream(frame, confidence_threshold, count_all_detections=False)
