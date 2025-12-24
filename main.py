# uvicorn main:app --reload

"""
FastAPI service for deploying the inventory tracking model.
This service provides REST API endpoints for image and video processing.

Supports both:
- YOLOv11 detection models (bounding boxes)
- YOLOv11-seg segmentation models (masks + bounding boxes)

The service automatically detects the model type and adjusts visualization accordingly.
"""

# =============================================================================
# IMPORTS SECTION
# =============================================================================

# Standard library imports for file operations, encoding, and date handling
import os          # For file path operations and existence checks
import io          # For handling byte streams (in-memory files)
import base64      # For encoding images to base64 strings in JSON responses
from typing import Optional  # Type hints for better code clarity
import logging     # For application logging and debugging
from datetime import datetime  # For timestamps in API responses

# FastAPI and uvicorn for building and running the REST API
import uvicorn  # ASGI server to run the FastAPI application
from fastapi import FastAPI, File, UploadFile, HTTPException, Form  # Core FastAPI components
from fastapi.middleware.cors import CORSMiddleware  # Enable Cross-Origin Resource Sharing

# Data processing and computer vision libraries
import numpy as np   # Numerical operations on arrays
import cv2           # OpenCV for image/video processing
from PIL import Image  # Python Imaging Library for image handling

# Import the custom inventory tracking class
from py.InventoryTracker import InventoryTracker

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Configure logging to display INFO level messages and above
# This helps track API requests, errors, and model loading status
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APPLICATION INITIALIZATION
# =============================================================================
# Create the main FastAPI application instance with metadata
app = FastAPI(
    title="Inventory Tracking API",           # API name shown in documentation
    description="REST API for automated inventory monitoring",  # API description
    version="1.0.0",                           # API version number
    docs_url="/docs",                          # Swagger UI documentation URL
    redoc_url="/redoc"                         # ReDoc documentation URL
)

# =============================================================================
# CORS MIDDLEWARE SETUP
# =============================================================================
# Add Cross-Origin Resource Sharing (CORS) middleware to allow web browsers
# from different domains to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (* means any domain). 
                                # For production, specify exact domains: ["https://example.com"]
    allow_credentials=True,     # Allow cookies and authentication headers
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],        # Allow all HTTP headers
)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
# Global tracker instance that will be initialized on startup
# Using Optional[InventoryTracker] type hint to indicate it may be None initially
tracker: Optional[InventoryTracker] = None

# =============================================================================
# STARTUP EVENT HANDLER
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """
    Initialize the YOLO model when the FastAPI application starts.
    This event runs once before the app begins accepting requests.
    """
    global tracker  # Access the global tracker variable
    try:
        # Step 1: Define the path to the trained segmentation model
        # The system automatically detects segmentation models and initializes appropriate annotators
        model_path = "models/segmentation-model.pt"
        
        # Step 2: Verify the model file exists before attempting to load
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Step 3: Initialize the InventoryTracker with the model
        # label_mode="item_name" means we'll group detections by item name
        tracker = InventoryTracker(model_path=model_path, label_mode="item_name")
        
        # Step 4: Log success message
        logger.info(f"✅ Model loaded successfully: {model_path}")
        
    except Exception as e:
        # If model loading fails, log the error and stop the application
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise  # Re-raise the exception to prevent app from starting with broken model

# =============================================================================
# API ENDPOINTS - HEALTH CHECK
# =============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - Simple health check to verify the API is running.
    Access via: GET http://localhost:8000/
    """
    return {
        "message": "Inventory Tracking API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),  # Current timestamp in ISO format
        "model_loaded": tracker is not None       # Verify model is loaded
    }

@app.get("/health")
async def health_check():
    """
    Detailed health check endpoint with more information.
    Access via: GET http://localhost:8000/health
    """
    return {
        "status": "healthy",
        "model_loaded": tracker is not None,
        "model_path": "models/segmentation-model.pt" if tracker else None,
        "timestamp": datetime.now().isoformat()
    }

# =============================================================================
# API ENDPOINTS - IMAGE PREDICTION
# =============================================================================

@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),                    # File upload from form-data
    confidence_threshold: float = Form(0.5),         # YOLO confidence threshold (default: 0.5)
    label_mode: str = Form("item_name")              # How to group detections
):
    """
    Process a single image and return inventory detection results.
    
    This endpoint accepts an image file, processes it with the YOLO model (detection or segmentation),
    and returns detected items with bounding boxes/masks and statistics.
    
    For segmentation models (YOLOv11-seg):
    - Returns pixel-accurate segmentation masks overlaid on the image
    - Bounding boxes are also drawn around each segmented object
    
    For detection models (YOLOv11):
    - Returns only bounding boxes around detected objects
    
    Access via: POST http://localhost:8000/predict/image
    
    Args:
        file: Image file (jpg, jpeg, png) sent as multipart/form-data
        confidence_threshold: YOLO confidence threshold (0.0-1.0). Lower values detect more items
                            but may include false positives. Default is 0.5 (50% confidence).
        label_mode: How to aggregate detected items. Options:
                   - "item_name": Group by item name (default)
                   - "category": Group by product category
                   - "sub_category": Group by product sub-category
                   - "brand": Group by brand name
                   - "sku_code": Group by SKU code
    
    Returns:
        JSON response with:
        - success: Whether processing was successful
        - filename: Original filename
        - confidence_threshold: Applied threshold
        - label_mode: Applied label mode
        - frame_count: Number of frames processed (1 for images)
        - detections: Dictionary of detected items {item_name: count}
        - statistics: Detailed statistics for each detected item type
        - annotated_image: Base64-encoded image with bounding boxes and labels
        - timestamp: When the processing completed
    """
    
    # Step 1: Validate that the model is loaded
    if tracker is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Step 2: Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Step 3: Read the uploaded image file into memory
        contents = await file.read()  # Read file bytes asynchronously
        
        # Step 4: Convert bytes to PIL Image, then to OpenCV format (BGR)
        image = Image.open(io.BytesIO(contents))  # Create PIL Image from bytes
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        
        # Step 5: Update tracker's label mode if it's different from current
        if tracker.label_mode != label_mode:
            tracker.label_mode = label_mode
        
        # Step 6: Reset statistics before processing new image
        # This ensures we're not mixing results from previous images
        tracker.reset_output_stats()
        
        # Step 7: Process the image with the YOLO model
        # Returns: annotated frame with bounding boxes, and live summary of detections
        annotated_frame, live_summary = tracker.track_picture_stream(frame, confidence_threshold)
        
        # Step 8: Get detailed statistics about all detections
        stats_df = tracker.get_output_stats()  # Returns pandas DataFrame
        
        # Step 9: Convert the annotated frame to base64 for JSON response
        # OpenCV imencode converts image to JPEG bytes
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string
        
        # Step 10: Prepare the response dictionary
        # live_summary now contains confidence info: {"Product A": {"count": 5, "confidence_avg": 85.3, ...}, ...}
        response = {
            "success": True,
            "filename": file.filename,
            "confidence_threshold": confidence_threshold,
            "label_mode": label_mode,
            "frame_count": tracker.frame_count,
            "detections": live_summary,  # Dict with count and confidence: {"Product A": {"count": 5, "confidence_avg": 85.3, ...}}
            "statistics": stats_df.to_dict('records') if not stats_df.empty else [],  # Convert DataFrame to list of dicts
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",  # Data URI for displaying in browser
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 11: Log the processing result
        logger.info(f"Processed image: {file.filename}, found {len(live_summary)} item types")
        
        # Step 12: Return the JSON response
        return response
        
    except Exception as e:
        # Error handling: Log the error and return HTTP 500 error
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# =============================================================================
# API ENDPOINTS - VIDEO PREDICTION
# =============================================================================

@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),                    # Video file upload from form-data
    confidence_threshold: float = Form(0.5),         # YOLO confidence threshold (default: 0.5)
    label_mode: str = Form("item_name"),             # How to group detections
    max_frames: int = Form(100)                      # Maximum frames to process (performance limit)
):
    """
    Process a video file and return aggregated inventory detection results.
    
    This endpoint processes video files frame-by-frame (every 5th frame for performance)
    and aggregates detections across all frames to provide final inventory counts.
    
    For segmentation models (YOLOv11-seg):
    - Each frame is processed with pixel-accurate segmentation masks
    - Objects are tracked across frames using ByteTrack algorithm
    
    For detection models (YOLOv11):
    - Each frame is processed with bounding box detection
    - Objects are tracked across frames using ByteTrack algorithm
    
    Access via: POST http://localhost:8000/predict/video
    
    Args:
        file: Video file (mp4, mov, avi, mkv) sent as multipart/form-data
        confidence_threshold: YOLO confidence threshold (0.0-1.0). Default is 0.5.
        label_mode: How to aggregate detected items (item_name, category, sub_category, brand, sku_code)
        max_frames: Maximum number of frames to process. Helps prevent timeouts on long videos.
                   Default is 100 frames. The system processes every 5th frame, so max_frames=100
                   means ~500 actual frames will be read from the video.
    
    Returns:
        JSON response with:
        - success: Whether processing was successful
        - filename: Original filename
        - confidence_threshold: Applied threshold
        - label_mode: Applied label mode
        - total_frames: Total frames in the video
        - processed_frames: Number of frames actually processed by YOLO
        - detections: Dictionary of unique item IDs tracked {item_name: unique_id_count}
        - statistics: Detailed statistics for each detected item type
        - timestamp: When the processing completed
    """
    
    # Step 1: Validate that the model is loaded
    if tracker is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Step 2: Validate that the uploaded file is a video
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Step 3: Read the uploaded video file into memory
        contents = await file.read()
        
        # Step 4: Save video temporarily to disk (OpenCV VideoCapture requires a file path)
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Step 5: Open the video file with OpenCV
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0        # Total frames read from video
        processed_frames = 0   # Frames actually processed by YOLO
        
        # Step 6: Update tracker's label mode if it's different from current
        if tracker.label_mode != label_mode:
            tracker.label_mode = label_mode
        
        # Step 7: Reset statistics before processing new video
        # This ensures we're not mixing results from previous videos
        tracker.reset_output_stats()
        
        # Step 8: Process video frames in a loop
        while cap.isOpened() and processed_frames < max_frames:
            # Read the next frame from the video
            ret, frame = cap.read()
            
            # If ret is False, we've reached the end of the video
            if not ret:
                break
            
            frame_count += 1
            
            # Step 9: Process every 5th frame for performance optimization
            # This reduces processing time while still capturing most inventory items
            # For a 30fps video, this means processing 6 frames per second
            if frame_count % 5 == 0:
                tracker.track_picture_stream(frame, confidence_threshold)
                processed_frames += 1
        
        # Step 10: Release the video capture object to free resources
        cap.release()
        
        # Step 11: Clean up the temporary video file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Step 12: Get final aggregated statistics across all processed frames
        stats_df = tracker.get_output_stats()
        
        # Step 13: Prepare the response dictionary
        # For videos, we return unique tracked IDs instead of per-frame counts
        response = {
            "success": True,
            "filename": file.filename,
            "confidence_threshold": confidence_threshold,
            "label_mode": label_mode,
            "total_frames": frame_count,  # Total frames in video
            "processed_frames": processed_frames,  # Frames actually processed
            # overall_tracked_ids contains unique item IDs across all frames
            # Example: {"Product A": {1, 3, 5}, "Product B": {2, 4}} -> {"Product A": 3, "Product B": 2}
            "detections": {name: len(ids) for name, ids in tracker.overall_tracked_ids.items() if len(ids) > 0},
            "statistics": stats_df.to_dict('records') if not stats_df.empty else [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 14: Log the processing result
        logger.info(f"Processed video: {file.filename}, {processed_frames} frames, found {len(response['detections'])} item types")
        
        # Step 15: Return the JSON response
        return response
        
    except Exception as e:
        # Error handling: Log the error, clean up temp file, and return HTTP 500 error
        logger.error(f"Error processing video {file.filename}: {str(e)}")
        
        # Clean up temp file if it exists
        temp_path = f"temp_{file.filename}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# =============================================================================
# API ENDPOINTS - MODEL INFORMATION & CONFIGURATION
# =============================================================================

@app.get("/model/info")
async def get_model_info():
    """
    Get information about the currently loaded YOLO model.
    
    This endpoint provides metadata about the model including its configuration
    and available options for label modes.
    
    Access via: GET http://localhost:8000/model/info
    
    Returns:
        JSON response with:
        - model_path: Path to the loaded model file
        - label_mode: Current label aggregation mode
        - valid_label_modes: List of all available label modes
        - confidence_threshold: Current confidence threshold
        - model_loaded: Whether the model is successfully loaded
    """
    # Validate that the model is loaded
    if tracker is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_path": "models/segmentation-model.pt",
        "label_mode": tracker.label_mode,
        "valid_label_modes": list(tracker.valid_label_modes),  # Convert set to list for JSON
        "confidence_threshold": tracker.confidence_threshold,
        "model_loaded": True
    }

@app.post("/model/update-label-mode")
async def update_label_mode(label_mode: str = Form(...)):
    """
    Update the label mode (aggregation method) for the tracker.
    
    This endpoint allows changing how detections are grouped without
    restarting the server or reloading the model.
    
    Access via: POST http://localhost:8000/model/update-label-mode
    
    Args:
        label_mode: New label mode to apply. Must be one of:
                   - "item_name": Group by item name
                   - "category": Group by product category
                   - "sub_category": Group by product sub-category
                   - "brand": Group by brand name
                   - "sku_code": Group by SKU code
    
    Returns:
        JSON response with:
        - success: Whether the update was successful
        - label_mode: The new label mode that was set
        - message: Confirmation message
    """
    # Step 1: Validate that the model is loaded
    if tracker is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Step 2: Validate that the requested label_mode is valid
    if label_mode not in tracker.valid_label_modes:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid label_mode. Valid options: {list(tracker.valid_label_modes)}"
        )
    
    # Step 3: Update the tracker's label mode
    tracker.label_mode = label_mode
    
    # Step 4: Return success response
    return {
        "success": True,
        "label_mode": label_mode,
        "message": f"Label mode updated to: {label_mode}"
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
# This block runs when the script is executed directly (not imported as a module)
if __name__ == "__main__":
    """
    Start the FastAPI server using Uvicorn ASGI server.
    
    Server Configuration:
    - app: The FastAPI application instance to run
    - host: "0.0.0.0" means listen on all network interfaces (allows external access)
    - port: 8000 is the default port for the API
    - reload: True enables auto-reload when code changes (useful for development)
    - log_level: "info" provides informative logging without being too verbose
    
    To run the server, execute: python fastapiservice.py
    Then access the API at: http://localhost:8000
    API documentation available at: http://localhost:8000/docs
    """
    uvicorn.run(
        "main:app",    # Module path to the FastAPI app
        host="0.0.0.0",            # Listen on all network interfaces
        port=8000,                 # Default API port
        reload=True,               # Auto-reload on code changes (development mode)
        log_level="info"           # Logging verbosity level
    )

