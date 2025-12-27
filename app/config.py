"""
Application configuration and version information.
"""
import os

# Application version
APP_VERSION = "2.0.0"

# Model configuration
MODEL_BASE_NAME = "segmentation-model.pt"  # Generic model name (model-agnostic, can be any detection/segmentation model)
MODEL_VERSION = "25.11.0"  # Current model version
MODEL_PATH = os.getenv("MODEL_PATH", f"models/{MODEL_BASE_NAME}")

# Determine which model file to use
def get_model_path() -> str:
    """
    Get the model file path.
    Checks for custom path from environment, then falls back to generic name.
    
    Returns:
        Path to model file
    """
    # Check custom path from environment first
    if MODEL_PATH != f"models/{MODEL_BASE_NAME}" and os.path.exists(MODEL_PATH):
        return MODEL_PATH
    
    # Check if generic model exists
    if os.path.exists(f"models/{MODEL_BASE_NAME}"):
        return f"models/{MODEL_BASE_NAME}"
    
    # Default to generic path (will raise error if not found)
    return f"models/{MODEL_BASE_NAME}"

# Model metadata (lazy evaluation to avoid calling get_model_path() at import time)
def get_model_metadata() -> dict:
    """
    Get model metadata dictionary.
    
    Returns:
        Dictionary with model metadata
    """
    return {
        "name": MODEL_BASE_NAME,
        "version": MODEL_VERSION,
        "type": "segmentation",  # or "detection"
        "path": get_model_path()
    }

# For convenience, create a function that returns metadata
# Use get_model_metadata() instead of MODEL_METADATA

# OCR configuration
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

# SKU matching configuration
# Catalog JSON file (will be converted to DuckDB for large datasets)
CATALOG_PATH = os.getenv("CATALOG_PATH", "app/data/sku_master.json")
SKU_MASTER_PATH = os.getenv("SKU_MASTER_PATH", "app/data/sku_master.json")  # Alias for backward compatibility
SKU_MATCHING_MIN_CONFIDENCE = float(os.getenv("SKU_MATCHING_MIN_CONFIDENCE", "0.65"))
USE_CATALOG_DATABASE = os.getenv("USE_CATALOG_DATABASE", "true").lower() == "true"  # Use DuckDB for large datasets

# Decision Engine Configuration
# Decision thresholds (configurable)
DETECTION_MIN = float(os.getenv("DETECTION_MIN", "0.6"))
OCR_MATCH_MIN = float(os.getenv("OCR_MATCH_MIN", "0.7"))
OCR_TOKEN_MIN = float(os.getenv("OCR_TOKEN_MIN", "0.5"))
CONFLICT_DELTA = float(os.getenv("CONFLICT_DELTA", "0.2"))

# Default weights for evidence fusion
W_DET_DEFAULT = float(os.getenv("W_DET_DEFAULT", "0.6"))
W_OCR_DEFAULT = float(os.getenv("W_OCR_DEFAULT", "0.4"))

# Dynamic weight adjustments
# Case A: OCR weak or missing
W_DET_OCR_WEAK = float(os.getenv("W_DET_OCR_WEAK", "0.85"))
W_OCR_OCR_WEAK = float(os.getenv("W_OCR_OCR_WEAK", "0.15"))

# Case B: OCR strong (brand keyword hit)
W_DET_OCR_STRONG = float(os.getenv("W_DET_OCR_STRONG", "0.5"))
W_OCR_OCR_STRONG = float(os.getenv("W_OCR_OCR_STRONG", "0.5"))

# Logging configuration
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB default
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))  # Keep 5 backup files
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
