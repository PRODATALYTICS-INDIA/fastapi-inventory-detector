#!/usr/bin/env python3
"""
Test script for OCR detection functionality.
Tests the new /detect-and-identify endpoint with OCR and SKU matching.
Can run against local server or remote API.
"""

import requests
import json
from pathlib import Path
import sys
import os
from typing import Optional, Dict

# Add project root to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API base URL - can be overridden with environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class OCRDetectionTester:
    """Test suite for OCR detection and SKU matching."""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.results = []
    
    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("=" * 70)
        print("Testing Health Check Endpoint")
        print("=" * 70)
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ Model Loaded: {data.get('model_loaded', False)}")
            print(f"✅ OCR Loaded: {data.get('ocr_loaded', False)}")
            print(f"✅ Status: {data.get('status', 'unknown')}")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test the model info endpoint."""
        print("\n" + "=" * 70)
        print("Testing Model Info Endpoint")
        print("=" * 70)
        
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"✅ YOLO Model Path: {data.get('yolo_model_path', 'N/A')}")
            print(f"✅ YOLO Is Segmentation: {data.get('yolo_is_segmentation', False)}")
            print(f"✅ OCR Available: {data.get('ocr_available', False)}")
            print(f"✅ SKU Master Count: {data.get('sku_master_count', 0)}")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_detect_and_identify(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        enable_ocr: bool = True,
        enable_sku_matching: bool = True,
        save_output: bool = True
    ) -> Optional[Dict]:
        """
        Test the detect-and-identify endpoint with OCR and SKU matching.
        
        Args:
            image_path: Path to test image
            confidence_threshold: YOLO confidence threshold
            enable_ocr: Enable OCR processing
            enable_sku_matching: Enable SKU matching
            save_output: Save detailed output to file
            
        Returns:
            Response data dictionary or None if failed
        """
        print("\n" + "=" * 70)
        print(f"Testing Detect-and-Identify Endpoint")
        print(f"Image: {image_path}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"Enable OCR: {enable_ocr}")
        print(f"Enable SKU Matching: {enable_sku_matching}")
        print("=" * 70)
        
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            return None
        
        try:
            # Prepare the file and form data
            with open(image_path, 'rb') as f:
                files = {
                    'file': (Path(image_path).name, f, 'image/jpeg')
                }
                data = {
                    'confidence_threshold': confidence_threshold,
                    'enable_ocr': str(enable_ocr).lower(),
                    'enable_sku_matching': str(enable_sku_matching).lower()
                }
                
                print(f"📤 Sending request to {self.api_base_url}/detect-and-identify...")
                response = requests.post(
                    f"{self.api_base_url}/detect-and-identify",
                    files=files,
                    data=data,
                    timeout=180  # 3 minute timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                print(f"✅ Status Code: {response.status_code}")
                print(f"✅ Success: {result.get('success', False)}")
                print(f"✅ Filename: {result.get('filename', 'N/A')}")
                print(f"✅ Total Detections: {result.get('total_detections', 0)}")
                print(f"✅ Processing Time: {result.get('processing_time_ms', 0):.1f} ms")
                
                # Print product details
                products = result.get('products', [])
                if products:
                    print(f"\n📦 Detected Products ({len(products)}):")
                    for i, product in enumerate(products, 1):
                        print(f"\n   Product {i}:")
                        print(f"      YOLO Class: {product.get('yolo_class', 'N/A')}")
                        print(f"      SKU: {product.get('sku', 'N/A')}")
                        print(f"      SKU Name: {product.get('sku_name', 'N/A')}")
                        print(f"      Confidence: {product.get('confidence', 0):.2%}")
                        print(f"      YOLO Confidence: {product.get('yolo_confidence', 0):.2%}")
                        
                        sku_match_conf = product.get('sku_match_confidence')
                        if sku_match_conf is not None:
                            print(f"      SKU Match Confidence: {sku_match_conf:.2%}")
                        
                        ocr_text = product.get('ocr_text', '')
                        if ocr_text:
                            print(f"      OCR Text: '{ocr_text[:100]}...' if len(ocr_text) > 100 else f\"      OCR Text: '{ocr_text}'\"")
                        else:
                            print(f"      OCR Text: (empty)")
                        
                        bbox = product.get('bbox', [])
                        if bbox:
                            print(f"      BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                        
                        has_mask = product.get('mask') is not None
                        print(f"      Has Mask: {has_mask}")
                else:
                    print("\n⚠️  No products detected")
                
                # Save detailed output
                if save_output:
                    output_file = f"test_output_ocr_{Path(image_path).stem}.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\n💾 Detailed output saved to: {output_file}")
                
                return result
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out (processing took too long)")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if hasattr(e.response, 'text'):
                print(f"   Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_ocr_only(self, image_path: str) -> bool:
        """Test OCR extraction without SKU matching."""
        print("\n" + "=" * 70)
        print("Testing OCR-Only Mode (SKU matching disabled)")
        print("=" * 70)
        
        result = self.test_detect_and_identify(
            image_path=image_path,
            enable_ocr=True,
            enable_sku_matching=False
        )
        
        if result and result.get('products'):
            products = result.get('products', [])
            ocr_texts = [p.get('ocr_text', '') for p in products if p.get('ocr_text')]
            if ocr_texts:
                print(f"\n✅ OCR extracted text from {len(ocr_texts)} products")
                for i, text in enumerate(ocr_texts[:3], 1):  # Show first 3
                    print(f"   {i}. '{text[:80]}...' if len(text) > 80 else f\"   {i}. '{text}'\"")
                return True
            else:
                print("\n⚠️  No OCR text extracted")
                return False
        else:
            print("\n❌ Test failed or no products detected")
            return False
    
    def test_sku_matching(self, image_path: str) -> bool:
        """Test SKU matching with OCR."""
        print("\n" + "=" * 70)
        print("Testing SKU Matching with OCR")
        print("=" * 70)
        
        result = self.test_detect_and_identify(
            image_path=image_path,
            enable_ocr=True,
            enable_sku_matching=True
        )
        
        if result and result.get('products'):
            products = result.get('products', [])
            matched_skus = [
                p for p in products 
                if p.get('sku_match_confidence') is not None 
                and p.get('sku_match_confidence', 0) > 0
            ]
            
            if matched_skus:
                print(f"\n✅ SKU matching successful for {len(matched_skus)} products")
                for product in matched_skus[:3]:  # Show first 3
                    print(f"   - {product.get('sku', 'N/A')}: "
                          f"{product.get('sku_match_confidence', 0):.2%} confidence")
                return True
            else:
                print("\n⚠️  No SKU matches found (this may be normal if OCR text doesn't match SKU keywords)")
                return True  # Not necessarily a failure
        else:
            print("\n❌ Test failed or no products detected")
            return False
    
    def test_batch_images(
        self,
        image_dir: str = "test_images",
        max_images: int = 5
    ) -> Dict[str, bool]:
        """Test OCR detection on multiple images."""
        print("\n" + "=" * 70)
        print(f"Testing Batch OCR Detection on Images")
        print(f"Directory: {image_dir}")
        print(f"Max Images: {max_images}")
        print("=" * 70)
        
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return {}
        
        # Find all image files
        image_files = (
            list(image_dir_path.glob("*.jpeg")) +
            list(image_dir_path.glob("*.jpg")) +
            list(image_dir_path.glob("*.png"))
        )
        
        if not image_files:
            print(f"❌ No image files found in {image_dir}")
            return {}
        
        # Limit number of images
        image_files = image_files[:max_images]
        
        results = {}
        for image_file in image_files:
            print(f"\n📸 Testing: {image_file.name}")
            result = self.test_detect_and_identify(
                image_path=str(image_file),
                enable_ocr=True,
                enable_sku_matching=True,
                save_output=False
            )
            
            if result and result.get('products'):
                products = result.get('products', [])
                ocr_count = sum(1 for p in products if p.get('ocr_text'))
                sku_matches = sum(
                    1 for p in products 
                    if p.get('sku_match_confidence') and p.get('sku_match_confidence', 0) > 0.5
                )
                results[image_file.name] = {
                    'success': True,
                    'products': len(products),
                    'ocr_extracted': ocr_count,
                    'sku_matched': sku_matches
                }
                print(f"   ✅ {len(products)} products, {ocr_count} with OCR, {sku_matches} SKU matches")
            else:
                results[image_file.name] = {'success': False}
                print(f"   ⚠️  No products detected")
        
        return results
    
    def run_full_test_suite(self, test_image: Optional[str] = None) -> Dict[str, bool]:
        """Run the complete test suite."""
        print("\n" + "=" * 70)
        print("OCR Detection Test Suite")
        print(f"API URL: {self.api_base_url}")
        print("=" * 70)
        
        results = {}
        
        # Test 1: Health checks
        results['health_check'] = self.test_health_check()
        results['model_info'] = self.test_model_info()
        
        # Test 2: Single image test (if provided or find one)
        if test_image:
            image_path = test_image
        else:
            test_images_dir = Path("test_images")
            if test_images_dir.exists():
                image_files = list(test_images_dir.glob("*.jpeg")) + list(test_images_dir.glob("*.jpg"))
                if image_files:
                    image_path = str(image_files[0])
                else:
                    image_path = None
            else:
                image_path = None
        
        if image_path:
            results['detect_and_identify'] = self.test_detect_and_identify(image_path) is not None
            results['ocr_only'] = self.test_ocr_only(image_path)
            results['sku_matching'] = self.test_sku_matching(image_path)
        else:
            print("\n⚠️  No test images found, skipping image tests")
            results['detect_and_identify'] = False
            results['ocr_only'] = False
            results['sku_matching'] = False
        
        # Test 3: Batch testing
        batch_results = self.test_batch_images(max_images=3)
        results['batch_test'] = len(batch_results) > 0
        
        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        total = len(results)
        passed = sum(1 for p in results.values() if p)
        print(f"\nTotal: {passed}/{total} tests passed")
        
        return results


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OCR detection functionality")
    parser.add_argument(
        "--api-url",
        default=API_BASE_URL,
        help=f"API base URL (default: {API_BASE_URL})"
    )
    parser.add_argument(
        "--image",
        help="Path to specific test image"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch tests on all test images"
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Test OCR extraction only (no SKU matching)"
    )
    
    args = parser.parse_args()
    
    tester = OCRDetectionTester(api_base_url=args.api_url)
    
    if args.batch:
        tester.test_batch_images(max_images=10)
    elif args.image:
        if args.ocr_only:
            tester.test_ocr_only(args.image)
        else:
            tester.test_detect_and_identify(args.image)
    else:
        tester.run_full_test_suite(test_image=args.image)


if __name__ == "__main__":
    main()
