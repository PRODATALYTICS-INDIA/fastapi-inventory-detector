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
# LABEL CATALOG - Load product metadata from Excel
# =============================================================================

def load_label_catalog():
    """
    Load labelling-catalog dataframe (only once).
    This Excel file contains product metadata (SKU, name, brand, category, etc.)
    """
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "labelling-catalog.xlsx")
    return pd.read_excel(DATA_PATH)

# Load catalog once at module level for efficiency
label_catalog = load_label_catalog()

# Build lookup dictionary for fast metadata access by SKU code
# Example: {"sku_1": {"item_name": "Product A", "brand": "Brand X", ...}, ...}
sku_lookup = label_catalog.set_index("sku_code").to_dict(orient="index") 

# =============================================================================
# YOLO INVENTORY TRACKER CLASS
# =============================================================================
class InventoryTracker:
    def __init__(self, model_path="models/model-segment_25-10-10.pt", label_mode="item_name"):
        """
        Initializes the tracker with YOLO model (detection or segmentation) and summary stats.
        
        This class now supports both:
        - YOLOv11 detection models (bounding boxes)
        - YOLOv11-seg segmentation models (masks + bounding boxes)
        
        Args:
            model_path (str): Path to YOLO model weights (.pt file).
                             Can be a detection model or segmentation model.
            label_mode (str): Label to display on frames and aggregate stats.
                             Options: "sku_code", "item_name", "brand", "sub_category", "category"
        """
        # Step 1: Load YOLO model (automatically detects if it's detection or segmentation)
        self.model = YOLO(model_path)
        
        # Step 2: Check if this is a segmentation model
        # Segmentation models have 'seg' in their task name
        self.is_segmentation = hasattr(self.model, 'task') and 'seg' in str(self.model.task).lower()
        
        # Step 3: Initialize confidence threshold (can be updated per request)
        self.confidence_threshold = 0.0
        
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

    def track_picture_stream(self, frame: np.ndarray, confidence_threshold: float):
        """
        Core logic to process a single frame for detection/segmentation, tracking, and stats gathering.
        
        This method works with both detection and segmentation models:
        - Detection models: Returns bounding boxes
        - Segmentation models: Returns bounding boxes + segmentation masks
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV).
            confidence_threshold (float): YOLO confidence threshold (0.0-1.0).

        Returns:
            annotated_frame (np.ndarray): Frame with annotations (boxes/masks + labels).
            live_summary (dict): Running summary of detections {label: count}.
        """
        # Step 1: Increment frame counter
        self.frame_count += 1
        
        # Step 2: Run YOLO inference on the frame
        # For segmentation models, results will include masks
        # For detection models, results will only include boxes
        results = self.model(frame, conf=confidence_threshold, verbose=False)[0]
        
        # Step 3: Convert YOLO results to Supervision Detections format
        # This automatically handles both detection and segmentation results
        detections = sv.Detections.from_ultralytics(results)
        
        # Step 4: Update object tracker with new detections
        # ByteTrack assigns persistent IDs to tracked objects across frames
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Step 5: Process each tracked detection to gather statistics and create labels
        labels = []
        for i, det in enumerate(tracked_detections):
            # Extract detection properties
            class_id = det[3]  # Class ID from model
            tracker_id = det[4]  # Unique tracker ID assigned by ByteTrack
            
            # Step 6: Get the SKU code from model class names
            detected_sku = self.model.model.names[class_id]  # e.g., "sku_1"
            
            # Step 7: Lookup product metadata from catalog
            meta = sku_lookup.get(detected_sku, {})
            
            # Step 8: Extract confidence score from results
            # Try multiple ways to get confidence as the structure may vary
            if hasattr(tracked_detections, 'confidence') and tracked_detections.confidence is not None:
                confidence = float(tracked_detections.confidence[i])
            elif hasattr(results, 'boxes') and hasattr(results.boxes, 'conf'):
                # Get confidence from YOLO results
                confidence = float(results.boxes.conf[i]) if len(results.boxes.conf) > i else 0.0
            else:
                confidence = 0.0

            # Step 9: Generate label text based on label_mode
            # Either show SKU code or a metadata field (item_name, brand, etc.)
            label_text = detected_sku if self.label_mode == "sku_code" else meta.get(self.label_mode, detected_sku)
            labels.append(f"#{tracker_id} {label_text}")
            
            # Step 10: Track statistics (deduplication by tracker_id)
            # Only count each unique tracked object once
            if tracker_id not in self.overall_tracked_ids[detected_sku]:
                self.overall_tracked_ids[detected_sku].add(tracker_id)
                self.class_appearances[detected_sku] += 1
                self.confidence[detected_sku].append(confidence)

        # Step 11: Annotate the frame with visual overlays
        annotated_frame = frame.copy()
        
        # For segmentation models: draw masks first (as background layer)
        if self.is_segmentation and hasattr(tracked_detections, 'mask') and tracked_detections.mask is not None:
            annotated_frame = self.mask_annotator.annotate(
                scene=annotated_frame, 
                detections=tracked_detections
            )
        
        # Step 12: Draw bounding boxes (for both detection and segmentation models)
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
            detections=tracked_detections
        )
        
        # Step 13: Add text labels with tracker IDs and product names
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=tracked_detections, 
            labels=labels
        )
        
        # Step 14: Draw tracking traces (shows movement path of tracked objects)
        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, 
            detections=tracked_detections
        )

        # Step 15: Create live summary of current detections
        # Returns dictionary like {"Product A": 3, "Product B": 2}
        live_summary = {name: len(ids) for name, ids in self.overall_tracked_ids.items() if len(ids) > 0}
        
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
            yield self.track_picture_stream(frame, confidence_threshold)