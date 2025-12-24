#!/usr/bin/env python3
"""
Local OCR testing script.
Tests OCR functionality directly on test images without requiring API server.
Useful for debugging and development.
"""

import sys
import os
from pathlib import Path
import cv2

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.models.ocr_model import OCRModel
# DetectionModel imported later when needed
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.sku_matching_service import SKUMatchingService


def test_ocr_on_image(image_path: str, show_preview: bool = False):
    """Test OCR extraction on a single image."""
    print("=" * 70)
    print(f"Testing OCR on Image: {image_path}")
    print("=" * 70)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        print(f"✅ Image loaded: {image.shape}")
        
        # Initialize OCR model
        print("\n📦 Initializing OCR model...")
        try:
            ocr_model = OCRModel(use_gpu=False)  # Use CPU for testing
            ocr_service = OCRService(ocr_model, use_multiprocessing=False)
            print("✅ OCR model initialized")
        except Exception as e:
            print(f"❌ Failed to initialize OCR model: {e}")
            print("   Make sure PaddleOCR is installed: pip install paddlepaddle paddleocr")
            return None
        
        # Test OCR on full image
        print("\n🔍 Running OCR on full image...")
        text, details = ocr_service.extract_text(image)
        
        print(f"\n📝 OCR Results:")
        print(f"   Extracted Text: '{text}'")
        print(f"   Text Lines: {len(details)}")
        
        if details:
            print(f"\n   Detailed Results:")
            for i, detail in enumerate(details[:5], 1):  # Show first 5
                print(f"      {i}. '{detail.get('text', '')}' "
                      f"(confidence: {detail.get('confidence', 0):.2%})")
        
        # If detection model is available, test on crops
        detection_model_path = "models/segmentation-model.pt"
        if Path(detection_model_path).exists():
            print("\n" + "=" * 70)
            print("Testing OCR on detected product crops")
            print("=" * 70)
            
            try:
                # Initialize detection model
                from app.models.detection_model import DetectionModel
                from app.services.detection_service import DetectionService
                
                print("\n📦 Initializing detection model...")
                detection_model = DetectionModel(detection_model_path)
                detection_service = DetectionService(detection_model)
                print("✅ Detection model initialized")
                
                # Run detection
                print("\n🔍 Running detection...")
                detections, crops = detection_service.detect(image, confidence_threshold=0.5)
                print(f"✅ Detected {len(detections)} products")
                
                # Run OCR on each crop
                if crops:
                    print(f"\n🔍 Running OCR on {len(crops)} product crops...")
                    ocr_results = ocr_service.extract_text_batch(crops)
                    
                    print(f"\n📝 OCR Results per Product:")
                    for i, (detection, (ocr_text, ocr_details)) in enumerate(zip(detections, ocr_results), 1):
                        print(f"\n   Product {i}:")
                        print(f"      Detection Class: {detection.get('sku', 'N/A')}")
                        print(f"      Confidence: {detection.get('confidence', 0):.2%}")
                        print(f"      OCR Text: '{ocr_text}'")
                        if ocr_details:
                            print(f"      OCR Lines: {len(ocr_details)}")
                            for j, detail in enumerate(ocr_details[:3], 1):
                                print(f"         {j}. '{detail.get('text', '')}' "
                                      f"({detail.get('confidence', 0):.2%})")
                    
                    # Test SKU matching if available
                    sku_master_path = "app/data/sku_master.json"
                    if Path(sku_master_path).exists():
                        print("\n" + "=" * 70)
                        print("Testing SKU Matching")
                        print("=" * 70)
                        
                        try:
                            sku_service = SKUMatchingService(sku_master_path)
                            print("✅ SKU matching service initialized")
                            
                            print(f"\n🔍 Matching OCR text to SKUs...")
                            for i, (detection, (ocr_text, _)) in enumerate(zip(detections, ocr_results), 1):
                                if ocr_text:
                                    matched_sku, match_conf, match_info = sku_service.find_best_sku_match(ocr_text)
                                    print(f"\n   Product {i} OCR: '{ocr_text}'")
                                    if matched_sku:
                                        sku_info = sku_service.get_sku_info(matched_sku)
                                        print(f"      ✅ Matched: {matched_sku} "
                                              f"(confidence: {match_conf:.2%})")
                                        if sku_info:
                                            print(f"      SKU Name: {sku_info.get('item_name', 'N/A')}")
                                    else:
                                        print(f"      ⚠️  No SKU match (confidence: {match_conf:.2%})")
                        except Exception as e:
                            print(f"⚠️  SKU matching failed: {e}")
                
            except Exception as e:
                print(f"⚠️  Detection failed: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'image_path': image_path,
            'ocr_text': text,
            'ocr_details': details
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_ocr(image_dir: str = "test_images", max_images: int = 5):
    """Test OCR on multiple images."""
    print("=" * 70)
    print(f"Batch OCR Testing")
    print(f"Directory: {image_dir}")
    print(f"Max Images: {max_images}")
    print("=" * 70)
    
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        print(f"❌ Directory not found: {image_dir}")
        return
    
    # Find image files
    image_files = (
        list(image_dir_path.glob("*.jpeg")) +
        list(image_dir_path.glob("*.jpg")) +
        list(image_dir_path.glob("*.png"))
    )
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    image_files = image_files[:max_images]
    
    results = []
    for image_file in image_files:
        print(f"\n{'='*70}")
        result = test_ocr_on_image(str(image_file))
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("Batch Test Summary")
    print("=" * 70)
    print(f"Processed: {len(results)}/{len(image_files)} images")
    
    ocr_success = sum(1 for r in results if r.get('ocr_text'))
    print(f"OCR Success: {ocr_success}/{len(results)}")
    
    if results:
        print(f"\nSample OCR Texts:")
        for i, result in enumerate(results[:3], 1):
            text = result.get('ocr_text', '')
            if text:
                print(f"   {i}. '{text[:60]}...' if len(text) > 60 else f\"   {i}. '{text}'\"")
            else:
                print(f"   {i}. (no text extracted)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OCR locally on images")
    parser.add_argument(
        "--image",
        help="Path to specific test image"
    )
    parser.add_argument(
        "--dir",
        default="test_images",
        help="Directory containing test images (default: test_images)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Test on all images in directory"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Maximum number of images for batch testing (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Change to test directory if running from project root
    if Path("test").exists():
        os.chdir(Path(__file__).parent.parent)
    
    if args.image:
        test_ocr_on_image(args.image)
    elif args.batch:
        test_batch_ocr(args.dir, args.max_images)
    else:
        # Default: test first available image
        test_images_dir = Path(args.dir)
        if test_images_dir.exists():
            image_files = list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.jpg"))
            if image_files:
                test_ocr_on_image(str(image_files[0]))
            else:
                print(f"❌ No images found in {test_images_dir}")
        else:
            print(f"❌ Test images directory not found: {test_images_dir}")


if __name__ == "__main__":
    main()
