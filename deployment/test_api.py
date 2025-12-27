#!/usr/bin/env python3
"""
Simple script to send image requests to the FastAPI service and save JSON responses.

Usage:
    python3 test_api.py <image_path> [--api-url API_URL]
    
Example:
    python3 test_api.py test_images/20251010-1.jpeg
    python3 test_api.py test_images/20251010-1.jpeg --api-url http://13.203.145.155:8000
"""

import json
import os
import sys

try:
    import requests
except ImportError:
    print("❌ Error: 'requests' library not found!")
    print("   Install it with: pip install requests")
    sys.exit(1)


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 test_api.py <image_path> [--api-url API_URL]")
        print("Example: python3 test_api.py test_images/20251010-1.jpeg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    # Default to localhost when running on server, external IP when running from local machine
    # Check if we're likely on the server by checking hostname, username, or trying localhost connection
    import socket
    
    hostname = socket.gethostname()
    username = os.getenv('USER', os.getenv('USERNAME', ''))
    
    # Check multiple indicators that we're on the server
    is_on_server = False
    
    # Check hostname patterns (AWS EC2 instances typically have 'ip-' prefix)
    if 'ip-' in hostname.lower() or hostname.startswith('ip-'):
        is_on_server = True
    
    # Check username
    if username.lower() == 'ubuntu':
        is_on_server = True
    
    # Try to detect by checking if localhost:8000 is accessible (most reliable)
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(0.5)
        result = test_socket.connect_ex(('localhost', 8000))
        test_socket.close()
        if result == 0:
            is_on_server = True
    except Exception:
        pass
    
    # Set default API URL
    if is_on_server:
        api_url = "http://localhost:8000"
    else:
        api_url = "http://13.203.145.155:8000"
    
    # Check for custom API URL (overrides default)
    if len(sys.argv) > 2 and sys.argv[2] == "--api-url" and len(sys.argv) > 3:
        api_url = sys.argv[3]
    
    # Debug: Show detection result (can be removed later)
    # Uncomment to debug:
    # print(f"DEBUG: hostname={hostname}, username={username}, is_on_server={is_on_server}, api_url={api_url}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Get image directory and name (handle relative paths)
    if os.path.isabs(image_path):
        image_abs_path = image_path
    else:
        image_abs_path = os.path.abspath(image_path)
    
    image_dir = os.path.dirname(image_abs_path) or os.getcwd()
    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]
    
    # API endpoint
    endpoint = "/detect-and-count"
    url = f"{api_url.rstrip('/')}{endpoint}"
    
    print(f"📤 Sending request to: {url}")
    print(f"   Image: {image_path}")
    
    # Send request
    try:
        with open(image_path, 'rb') as image_file:
            # Determine content type from file extension
            content_type = 'image/jpeg'
            if image_path.lower().endswith('.png'):
                content_type = 'image/png'
            elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            
            files = {'file': (image_name, image_file, content_type)}
            response = requests.post(url, files=files, timeout=120)
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out (API may be processing)")
        print(f"   URL: {url}")
        print("   💡 If running from inside the server, use: --api-url http://localhost:8000")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to API at {url}")
        print("   Check if the service is running and accessible")
        print("")
        print("   🔍 Quick troubleshooting:")
        print("     1. Check if container is running:")
        print("        docker ps | grep inventory-tracker-api")
        print("     2. Check container logs:")
        print("        docker logs inventory-tracker-api --tail 20")
        print("     3. Check if port is listening:")
        print("        netstat -tuln | grep 8000")
        print("     4. Test health endpoint:")
        print("        curl http://localhost:8000/health")
        print("")
        print("   💡 If container is not running, deploy it:")
        print("      From local machine: ./deploy.sh run")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: Request failed - {e}")
        sys.exit(1)
    
    print(f"📥 Response: HTTP {response.status_code}")
    
    # Save JSON response
    response_dir = os.path.join(image_dir, f"{image_base}_response")
    os.makedirs(response_dir, exist_ok=True)
    
    json_path = os.path.join(response_dir, "api_response.json")
    try:
        response_json = response.json()
        with open(json_path, 'w') as f:
            json.dump(response_json, f, indent=2)
        print(f"✅ Response saved: {json_path}")
    except json.JSONDecodeError:
        print("⚠️  Response is not JSON, saving as text")
        with open(json_path, 'w') as f:
            f.write(response.text)
        print(f"✅ Response saved: {json_path}")
    
    if response.status_code == 200:
        print("✅ Success!")
    else:
        print(f"⚠️  Status: {response.status_code}")
        if response.text:
            print(f"   Error message: {response.text[:200]}")


if __name__ == "__main__":
    main()
