#!/usr/bin/env python3
"""
Quick test script to verify the new project structure.
Run this to check if all imports and basic functionality work.
"""
import sys
import os

# Add project root to path (this script is in test/ folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.models.detection_model import DetectionModel
        print("✅ Detection model import successful")
    except Exception as e:
        print(f"❌ Detection model import failed: {e}")
        return False
    
    try:
        from app.models.ocr_model import OCRModel
        print("✅ OCR model import successful")
    except Exception as e:
        print(f"⚠️  OCR model import failed (expected if PaddleOCR not installed): {e}")
    
    try:
        from app.services.detection_service import DetectionService
        print("✅ Detection service import successful")
    except Exception as e:
        print(f"❌ Detection service import failed: {e}")
        return False
    
    try:
        from app.services.ocr_service import OCRService
        print("✅ OCR service import successful")
    except Exception as e:
        print(f"⚠️  OCR service import failed (expected if PaddleOCR not installed): {e}")
    
    try:
        from app.services.sku_matching_service import SKUMatchingService
        print("✅ SKU matching service import successful")
    except Exception as e:
        print(f"❌ SKU matching service import failed: {e}")
        return False
    
    try:
        from app.utils.image_utils import extract_crop_from_bbox
        print("✅ Image utils import successful")
    except Exception as e:
        print(f"❌ Image utils import failed: {e}")
        return False
    
    try:
        from app.schemas.response import (
            DetectionResponse, 
            ProductDetection,  # Legacy
            DetectionDetail,
            DetectionModelOutput,
            OCRModelOutput,
            FinalPrediction,
            ValidationStatus
        )
        print("✅ Response schemas import successful")
    except Exception as e:
        print(f"❌ Response schemas import failed: {e}")
        return False
    
    try:
        from app.utils.inventory_summary import generate_summary
        print("✅ Summary aggregator import successful")
    except Exception as e:
        print(f"❌ Summary aggregator import failed: {e}")
        return False
    
    return True

def test_sku_master():
    """Test that SKU master file exists and is valid."""
    print("\nTesting SKU master data...")
    
    import json
    # Use absolute path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sku_path = os.path.join(project_root, "app", "data", "sku_master.json")
    
    if not os.path.exists(sku_path):
        print(f"❌ SKU master file not found: {sku_path}")
        return False
    
    try:
        with open(sku_path, 'r') as f:
            sku_data = json.load(f)
        print(f"✅ SKU master file loaded: {len(sku_data)} SKUs")
        return True
    except Exception as e:
        print(f"❌ Failed to load SKU master: {e}")
        return False

def test_detection_model_path():
    """Test that detection model file exists."""
    print("\nTesting detection model path...")
    
    # Use absolute path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "segmentation-model.pt")
    if os.path.exists(model_path):
        print(f"✅ Detection model file found: {model_path}")
        return True
    else:
        print(f"⚠️  Detection model file not found: {model_path}")
        print("   (This is OK if you haven't placed the model file yet)")
        return True  # Not a critical error

def main():
    """Run all tests."""
    print("=" * 60)
    print("FastAPI Inventory Detector - Structure Verification")
    print("=" * 60)
    
    results = []
    results.append(test_imports())
    results.append(test_sku_master())
    results.append(test_detection_model_path())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All critical tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the server: uvicorn app.main:app --reload")
        print("3. Test the API: curl http://localhost:8000/health")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
