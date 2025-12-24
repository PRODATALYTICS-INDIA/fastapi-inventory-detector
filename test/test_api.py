#!/usr/bin/env python3
"""
Test script for the Inventory Tracking API
Tests the live API hosted at https://inventory-tracker-api-latest.onrender.com/
"""

import requests
import json
from pathlib import Path
import base64

# API base URL
API_BASE_URL = "https://inventory-tracker-api-latest.onrender.com"

def test_health_check():
    """Test the health check endpoint"""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detailed_health():
    """Test the detailed health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Detailed Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        response.raise_for_status()
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_image_prediction(image_path, confidence_threshold=0.5, label_mode="item_name"):
    """Test the image prediction endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing Image Prediction Endpoint")
    print(f"Image: {image_path}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Label Mode: {label_mode}")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        # Prepare the file and form data
        with open(image_path, 'rb') as f:
            files = {
                'file': (Path(image_path).name, f, 'image/jpeg')
            }
            data = {
                'confidence_threshold': confidence_threshold,
                'label_mode': label_mode
            }
            
            print(f"📤 Sending request to {API_BASE_URL}/predict/image...")
            response = requests.post(
                f"{API_BASE_URL}/predict/image",
                files=files,
                data=data,
                timeout=120  # 2 minute timeout for image processing
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"✅ Filename: {result.get('filename', 'N/A')}")
            print(f"✅ Frame Count: {result.get('frame_count', 0)}")
            print(f"✅ Detections: {len(result.get('detections', {}))} item types found")
            
            # Print detections
            if result.get('detections'):
                print("\n📦 Detected Items:")
                for item_name, item_data in result.get('detections', {}).items():
                    if isinstance(item_data, dict):
                        count = item_data.get('count', 0)
                        conf_avg = item_data.get('confidence_avg', 0)
                        print(f"   - {item_name}: {count} items (avg confidence: {conf_avg:.1f}%)")
                    else:
                        print(f"   - {item_name}: {item_data}")
            
            # Print statistics summary
            stats = result.get('statistics', [])
            if stats:
                print(f"\n📊 Statistics: {len(stats)} entries")
                for stat in stats[:3]:  # Show first 3
                    print(f"   {stat}")
            
            # Save annotated image if available
            if result.get('annotated_image'):
                annotated_data = result['annotated_image']
                if annotated_data.startswith('data:image'):
                    # Extract base64 data
                    base64_data = annotated_data.split(',')[1]
                    image_data = base64.b64decode(base64_data)
                    output_path = f"test_output_{Path(image_path).stem}_annotated.jpg"
                    with open(output_path, 'wb') as out:
                        out.write(image_data)
                    print(f"\n💾 Annotated image saved to: {output_path}")
            
            return True
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (image processing took too long)")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_prediction(video_path, confidence_threshold=0.5, label_mode="item_name", max_frames=50):
    """Test the video prediction endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing Video Prediction Endpoint")
    print(f"Video: {video_path}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Label Mode: {label_mode}")
    print(f"Max Frames: {max_frames}")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        # Prepare the file and form data
        with open(video_path, 'rb') as f:
            files = {
                'file': (Path(video_path).name, f, 'video/mp4')
            }
            data = {
                'confidence_threshold': confidence_threshold,
                'label_mode': label_mode,
                'max_frames': max_frames
            }
            
            print(f"📤 Sending request to {API_BASE_URL}/predict/video...")
            print("⏳ This may take a while for video processing...")
            response = requests.post(
                f"{API_BASE_URL}/predict/video",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout for video processing
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"✅ Filename: {result.get('filename', 'N/A')}")
            print(f"✅ Total Frames: {result.get('total_frames', 0)}")
            print(f"✅ Processed Frames: {result.get('processed_frames', 0)}")
            print(f"✅ Detections: {len(result.get('detections', {}))} item types found")
            
            # Print detections
            if result.get('detections'):
                print("\n📦 Detected Items:")
                for item_name, count in result.get('detections', {}).items():
                    print(f"   - {item_name}: {count} unique items")
            
            # Print statistics summary
            stats = result.get('statistics', [])
            if stats:
                print(f"\n📊 Statistics: {len(stats)} entries")
                for stat in stats[:3]:  # Show first 3
                    print(f"   {stat}")
            
            return True
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (video processing took too long)")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detect_and_identify(image_path, confidence_threshold=0.5, enable_ocr=True, enable_sku_matching=True):
    """Test the new detect-and-identify endpoint with OCR"""
    print("\n" + "=" * 60)
    print(f"Testing Detect-and-Identify Endpoint (with OCR)")
    print(f"Image: {image_path}")
    print("=" * 60)
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {
                'file': (Path(image_path).name, f, 'image/jpeg')
            }
            data = {
                'confidence_threshold': confidence_threshold,
                'enable_ocr': str(enable_ocr).lower(),
                'enable_sku_matching': str(enable_sku_matching).lower()
            }
            
            print(f"📤 Sending request to {API_BASE_URL}/detect-and-identify...")
            response = requests.post(
                f"{API_BASE_URL}/detect-and-identify",
                files=files,
                data=data,
                timeout=180
            )
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Success: {result.get('success', False)}")
            print(f"✅ Total Detections: {result.get('total_detections', 0)}")
            print(f"✅ Processing Time: {result.get('processing_time_ms', 0):.1f} ms")
            
            products = result.get('products', [])
            if products:
                print(f"\n📦 Detected Products ({len(products)}):")
                for i, product in enumerate(products[:3], 1):  # Show first 3
                    print(f"   {i}. {product.get('yolo_class', 'N/A')}")
                    print(f"      SKU: {product.get('sku', 'N/A')}")
                    print(f"      Confidence: {product.get('confidence', 0):.2%}")
                    ocr_text = product.get('ocr_text', '')
                    if ocr_text:
                        print(f"      OCR: '{ocr_text[:50]}...' if len(ocr_text) > 50 else f\"      OCR: '{ocr_text}'\"")
                    sku_match = product.get('sku_match_confidence')
                    if sku_match:
                        print(f"      SKU Match: {sku_match:.2%}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Inventory Tracking API Test Suite")
    print(f"Testing API at: {API_BASE_URL}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health checks
    results.append(("Health Check", test_health_check()))
    results.append(("Detailed Health", test_detailed_health()))
    results.append(("Model Info", test_model_info()))
    
    # Test 2: Image prediction (if test images exist)
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        if test_images:
            # Test with first available image
            test_image = test_images[0]
            results.append(("Image Prediction (Legacy)", test_image_prediction(str(test_image))))
            results.append(("Detect-and-Identify (New)", test_detect_and_identify(str(test_image))))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    main()

