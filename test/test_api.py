#!/usr/bin/env python3
"""
API Test Suite - Tests detection via FastAPI endpoint.

This test script:
1. Ensures FastAPI service is running (starts it if needed)
2. Processes each image from test_images/ directory via API
3. Saves only api_response.json to test_output/api_call/<image_name>/

Usage:
    python test/test_api.py [--api-url http://localhost:8000] [--image path/to/image.jpg]
"""

import sys
import time
import subprocess
import requests
from pathlib import Path
from typing import Optional
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class APITestRunner:
    """Runs complete detection pipeline via FastAPI endpoint."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        auto_start_server: bool = True
    ):
        """
        Initialize the API test runner.
        
        Args:
            api_url: Base URL of the FastAPI service
            auto_start_server: Automatically start server if not running
        """
        self.api_url = api_url.rstrip('/')
        self.output_base = PROJECT_ROOT / "test" / "test_output" / "api_call"
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.server_process = None
        
        # Ensure API is running
        if auto_start_server:
            self._ensure_server_running()
    
    def _ensure_server_running(self) -> bool:
        """
        Check if API server is running, start it if not.
        
        Returns:
            True if server is running, False otherwise
        """
        print("=" * 80)
        print("Checking API Server Status")
        print("=" * 80)
        
        # Check if server is already running
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API server is running at {self.api_url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Server not running, try to start it
        print(f"⚠️  API server not running at {self.api_url}")
        print("Attempting to start server...")
        
        try:
            # Start server in background (using v2.0 service from app/main.py)
            app_main_py = PROJECT_ROOT / "app" / "main.py"
            if not app_main_py.exists():
                print("❌ Cannot find app/main.py to start server")
                print("   Please start the server manually: uvicorn app.main:app --reload")
                return False
            
            # Start server process (using v2.0 service from app/main.py)
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start (max 30 seconds)
            print("⏳ Waiting for server to start...")
            for _ in range(30):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Server started successfully at {self.api_url}")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            print("❌ Server failed to start within 30 seconds")
            if self.server_process:
                self.server_process.terminate()
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            print("   Please start manually: uvicorn app.main:app --reload")
            return False
    
    def _stop_server(self):
        """Stop the server if we started it."""
        if self.server_process:
            print("\n🛑 Stopping server...")
            self.server_process.terminate()
            self.server_process.wait()
            print("✅ Server stopped")
    
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image through the API and save only api_response.json.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print(f"Processing: {image_path.name}")
        print("=" * 80)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return False
        
        print(f"✅ Image found: {image_path}")
        
        # Create output directory (same structure as local_call)
        output_dir = self.output_base / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call API endpoint /detect-and-count (new unified endpoint)
        print("\n📤 Sending request to API...")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                
                # Use new /detect-and-count endpoint (no parameters needed)
                endpoint = f"{self.api_url}/detect-and-count"
                print(f"   Endpoint: {endpoint}")
                
                response = requests.post(
                    endpoint,
                    files=files,
                    timeout=300  # 5 minute timeout (OCR can be slow)
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Save JSON response to file (only output needed)
                json_output_path = output_dir / "api_response.json"
                try:
                    # Use orjson for exact FastAPI serialization if available
                    try:
                        import orjson
                        json_bytes = orjson.dumps(result, option=orjson.OPT_INDENT_2)
                        with open(json_output_path, 'wb') as f:
                            f.write(json_bytes)
                    except ImportError:
                        # Fallback to standard json
                        with open(json_output_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ API JSON response saved: {json_output_path}")
                except Exception as e:
                    print(f"❌ Failed to save API JSON response: {e}")
                    return False
                
                # Print summary
                if 'summary' in result:
                    summary = result.get('summary', {})
                    total_detections = summary.get('total_detections', 0)
                    validated_detections = summary.get('validated_detections', 0)
                    processing_time = result.get('processing_time_ms', 0)
                    
                    print(f"   Status: {result.get('status', 'success')}")
                    print(f"   Total detections: {total_detections}")
                    print(f"   Validated detections: {validated_detections}")
                    print(f"   Processing time: {processing_time:.1f} ms")
                    
                    # Print SKU summary
                    sku_counts = summary.get('sku_counts', {})
                    if sku_counts:
                        print("   SKU Summary:")
                        for sku_slug, sku_info in sku_counts.items():
                            count = sku_info.get('count', 0)
                            avg_conf = sku_info.get('confidence', 0.0)
                            sku_id = sku_info.get('sku_id', sku_slug)
                            print(f"     - {sku_id}: {count} items (avg confidence: {avg_conf:.2%})")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error detail: {error_detail}")
                except Exception:
                    print(f"   Response: {e.response.text[:200]}")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    
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
        try:
            for image_path in image_files:
                try:
                    if self.process_image(image_path):
                        success_count += 1
                except Exception as e:
                    print(f"❌ Error processing {image_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            # Stop server if we started it
            self._stop_server()
        
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
        description="API test suite - saves api_response.json for each image"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Process a single image (default: process all images in test_images/)"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't automatically start server if not running"
    )
    
    args = parser.parse_args()
    
    try:
        runner = APITestRunner(
            api_url=args.api_url,
            auto_start_server=not args.no_auto_start
        )
        
        if args.image:
            # Process single image - resolve path to test/test_images
            image_path = Path(args.image)
            if not image_path.is_absolute():
                # Extract just the filename (handle cases like "test_images/file.jpg" or just "file.jpg")
                image_name = image_path.name
                image_path = PROJECT_ROOT / "test" / "test_images" / image_name
            runner.process_image(image_path)
            runner._stop_server()
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
