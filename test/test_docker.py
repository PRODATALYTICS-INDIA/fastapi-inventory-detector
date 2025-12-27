#!/usr/bin/env python3
"""
Docker API Test Suite - Tests detection via FastAPI endpoint running in Docker.

This test script:
1. Assumes Docker container is already running (managed by run_test_docker.sh)
2. Processes each image from test_images/ directory via API
3. Saves only api_response.json to test_output/docker_call/<image_name>/

Usage:
    python test/test_docker.py [--api-url http://localhost:8000] [--image path/to/image.jpg]
"""

import sys
import os
import requests
from pathlib import Path
from typing import Optional
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class DockerAPITestRunner:
    """Runs complete detection pipeline via FastAPI endpoint in Docker container."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000"
    ):
        """
        Initialize the Docker API test runner.
        
        Args:
            api_url: Base URL of the FastAPI service (Docker container)
        """
        self.api_url = api_url.rstrip('/')
        self.output_base = PROJECT_ROOT / "test" / "test_output" / "docker_call"
        self.output_base.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_base}")
    
    def check_docker_health(self) -> bool:
        """
        Check if Docker container is running and healthy.
        
        Returns:
            True if container is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Docker container is running at {self.api_url}")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Model loaded: {health_data.get('model_loaded', False)}")
                print(f"   OCR loaded: {health_data.get('ocr_loaded', False)}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Docker container not accessible at {self.api_url}")
            print(f"   Error: {e}")
            return False
        return False
    
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image through the Docker API and save only api_response.json.
        
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
        
        # Create output directory (same structure as other test outputs)
        # Structure: test/test_output/docker_call/<image_name>/api_response.json
        output_dir = self.output_base / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory for this image: {output_dir}")
        print(f"   Absolute path: {output_dir.absolute()}")
        print(f"   Directory exists: {output_dir.exists()}")
        
        # Call API endpoint /detect-and-count
        print("\n📤 Sending request to Docker API...")
        try:
            # Read file into memory first to avoid connection issues
            with open(image_path, 'rb') as f:
                file_data = f.read()
            
            files = {'file': (image_path.name, file_data, 'image/jpeg')}
            
            # Use /detect-and-count endpoint
            endpoint = f"{self.api_url}/detect-and-count"
            print(f"   Endpoint: {endpoint}")
            
            # Add connection retry logic for Docker container stability
            max_retries = 3
            retry_delay = 2
            response = None
            
            for attempt in range(max_retries):
                try:
                    # Re-read file for each retry attempt
                    if attempt > 0:
                        with open(image_path, 'rb') as f:
                            file_data = f.read()
                        files = {'file': (image_path.name, file_data, 'image/jpeg')}
                    
                    response = requests.post(
                        endpoint,
                        files=files,
                        timeout=300,  # 5 minute timeout (OCR can be slow)
                        stream=False  # Ensure full response is received
                    )
                    # If we get here, the request completed
                    break
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.Timeout) as e:
                    # Check if container is still running
                    try:
                        health_check = requests.get(f"{self.api_url}/health", timeout=2)
                        container_running = True
                    except:
                        container_running = False
                    
                    if not container_running:
                        print(f"   ❌ Container appears to have stopped/crashed!")
                        print(f"   Please check container status:")
                        print(f"   docker ps -a | grep inventory-tracker-api")
                        print(f"   docker logs inventory-tracker-api --tail 50")
                        raise requests.exceptions.ConnectionError(
                            f"Container stopped during processing. Original error: {e}"
                        )
                    
                    if attempt < max_retries - 1:
                        print(f"   ⚠️  Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"   Container is still running, retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Last attempt failed, re-raise
                        raise
            
            if response is None:
                raise requests.exceptions.RequestException("Failed to get response after retries")
            
            print(f"   Response status: {response.status_code}")
            
            # Parse response (save even if error for debugging)
            try:
                result = response.json()
                print(f"   Response received: {len(str(result))} characters")
            except Exception as e:
                print(f"❌ Failed to parse JSON response: {e}")
                print(f"   Response text (first 500 chars): {response.text[:500]}")
                # Create error response to save
                result = {
                    "success": False,
                    "status": "error",
                    "error": f"Failed to parse JSON response: {str(e)}",
                    "response_status": response.status_code,
                    "response_text": response.text[:1000] if len(response.text) <= 1000 else response.text[:1000] + "... (truncated)"
                }
            
            # Check if response indicates success (but still save for debugging)
            if response.status_code != 200:
                print(f"⚠️  API returned error status: {response.status_code}")
                if isinstance(result, dict):
                    print(f"   Error detail: {result.get('error', result.get('detail', 'Unknown error'))}")
                # Continue to save the error response
            
            if not isinstance(result, dict):
                print(f"⚠️  Response is not a dictionary: {type(result)}")
                # Convert to dict for saving
                result = {
                    "success": False,
                    "status": "error",
                    "error": f"Response is not a dictionary: {type(result)}",
                    "response_data": str(result)[:1000]
                }
            
            if result.get('success') is False:
                print(f"⚠️  API response indicates failure: {result.get('status', 'unknown')}")
                print(f"   Error: {result.get('error', 'No error message')}")
                # Still save the error response for debugging
            
            # Save JSON response to file
            json_output_path = output_dir / "api_response.json"
            print(f"   Saving to: {json_output_path}")
            print(f"   Absolute path: {json_output_path.absolute()}")
            
            # Ensure output directory exists (double-check)
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.exists():
                print(f"❌ Failed to create output directory: {output_dir}")
                return False
            
            try:
                # Use orjson for exact FastAPI serialization if available
                try:
                    import orjson
                    json_bytes = orjson.dumps(result, option=orjson.OPT_INDENT_2)
                    with open(json_output_path, 'wb') as f:
                        f.write(json_bytes)
                    print(f"✅ API JSON response saved (orjson): {json_output_path}")
                except ImportError:
                    # Fallback to standard json
                    with open(json_output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"✅ API JSON response saved (json): {json_output_path}")
                
                # File should be written and closed by now
                
                # Verify file was created and is readable
                import time
                time.sleep(0.1)  # Brief pause to ensure file system sync
                
                if json_output_path.exists():
                    file_size = json_output_path.stat().st_size
                    if file_size > 0:
                        print(f"   ✓ File verified: {file_size} bytes at {json_output_path.absolute()}")
                        
                        # Try to read it back to verify it's valid JSON
                        try:
                            with open(json_output_path, 'r', encoding='utf-8') as f:
                                test_read = json.load(f)
                            print(f"   ✓ File is valid JSON with {len(str(test_read))} characters")
                        except Exception as e:
                            print(f"   ⚠️  Warning: File exists but is not valid JSON: {e}")
                    else:
                        print(f"   ⚠️  Warning: File exists but is empty (0 bytes)")
                        return False
                else:
                    print(f"   ❌ ERROR: File was not created")
                    print(f"   Expected path: {json_output_path.absolute()}")
                    print(f"   Parent directory exists: {json_output_path.parent.exists()}")
                    print(f"   Parent directory: {json_output_path.parent.absolute()}")
                    print(f"   Parent directory is writable: {os.access(json_output_path.parent, os.W_OK)}")
                    return False
            except Exception as e:
                print(f"❌ Failed to save API JSON response: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Output directory: {output_dir.absolute()}")
                print(f"   Output directory exists: {output_dir.exists()}")
                print(f"   JSON path: {json_output_path.absolute()}")
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
            error_response = {
                "success": False,
                "status": "error",
                "error": f"API request failed: {str(e)}",
                "input": {
                    "type": "image",
                    "filename": image_path.name,
                    "frame_count": 1
                }
            }
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_response["error_detail"] = error_detail
                    print(f"   Error detail: {error_detail}")
                except Exception:
                    error_response["response_text"] = e.response.text[:1000]
                    print(f"   Response: {e.response.text[:200]}")
            
            # Save error response
            json_output_path = output_dir / "api_response.json"
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_response, f, indent=2, ensure_ascii=False)
                print(f"⚠️  Error response saved: {json_output_path}")
            except Exception as save_error:
                print(f"❌ Failed to save error response: {save_error}")
            
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error response
            error_response = {
                "success": False,
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "input": {
                    "type": "image",
                    "filename": image_path.name,
                    "frame_count": 1
                }
            }
            
            json_output_path = output_dir / "api_response.json"
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(error_response, f, indent=2, ensure_ascii=False)
                print(f"⚠️  Error response saved: {json_output_path}")
            except Exception as save_error:
                print(f"❌ Failed to save error response: {save_error}")
            
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
        print(f"📁 Output base directory: {self.output_base}")
        print(f"📁 Each image will have its own subdirectory with api_response.json\n")
        
        # Process each image
        success_count = 0
        import time
        for idx, image_path in enumerate(image_files):
            try:
                # Add small delay between requests to avoid overwhelming the container
                if idx > 0:
                    time.sleep(1)
                
                # Check if API is still accessible before each request
                try:
                    health_check = requests.get(f"{self.api_url}/health", timeout=5)
                    if health_check.status_code != 200:
                        print(f"⚠️  Warning: API health check failed before processing {image_path.name}")
                        print(f"   Status: {health_check.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"❌ ERROR: API is not accessible before processing {image_path.name}")
                    print(f"   Error: {e}")
                    print(f"   The Docker container may have crashed. Please check:")
                    print(f"   docker ps -a | grep inventory-tracker-api")
                    print(f"   docker logs inventory-tracker-api")
                    print(f"   Then restart the container and try again.")
                    return
                
                if self.process_image(image_path):
                    success_count += 1
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Check if container is still running after error
                try:
                    requests.get(f"{self.api_url}/health", timeout=2)
                except requests.exceptions.RequestException:
                    print(f"❌ Container appears to have crashed after processing {image_path.name}")
                    print(f"   Please check container status and logs:")
                    print(f"   docker ps -a | grep inventory-tracker-api")
                    print(f"   docker logs inventory-tracker-api --tail 50")
                    break
        
        # Final summary
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total images: {len(image_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(image_files) - success_count}")
        print(f"Results saved to: {self.output_base}")
        print(f"  Each image has its own subdirectory with api_response.json")
        print(f"  Example: {self.output_base}/20251010-1/api_response.json")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Docker API test suite - saves api_response.json for each image"
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
    
    args = parser.parse_args()
    
    try:
        runner = DockerAPITestRunner(api_url=args.api_url)
        
        # Check if Docker container is running
        if not runner.check_docker_health():
            print("\n❌ Docker container is not running or not accessible")
            print("   Please ensure the Docker container is running before running tests")
            print("   Use run_test_docker.sh to automatically manage the container")
            sys.exit(1)
        
        if args.image:
            # Process single image - resolve path to test/test_images
            image_path = Path(args.image)
            if not image_path.is_absolute():
                # Extract just the filename
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

