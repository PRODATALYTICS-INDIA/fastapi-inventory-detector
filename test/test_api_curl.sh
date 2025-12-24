#!/bin/bash
# Test script for Inventory Tracking API using curl
# API URL: https://inventory-tracker-api-latest.onrender.com/

API_BASE_URL="https://inventory-tracker-api-latest.onrender.com"

echo "=========================================="
echo "Testing Inventory Tracking API with curl"
echo "API URL: $API_BASE_URL"
echo "=========================================="

# Test 1: Health Check
echo ""
echo "1. Testing Health Check (GET /)"
echo "--------------------------------"
curl -X GET "$API_BASE_URL/" \
  -H "Accept: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  | jq '.' 2>/dev/null || cat

# Test 2: Detailed Health Check
echo ""
echo "2. Testing Detailed Health Check (GET /health)"
echo "--------------------------------"
curl -X GET "$API_BASE_URL/health" \
  -H "Accept: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  | jq '.' 2>/dev/null || cat

# Test 3: Model Info
echo ""
echo "3. Testing Model Info (GET /model/info)"
echo "--------------------------------"
curl -X GET "$API_BASE_URL/model/info" \
  -H "Accept: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  | jq '.' 2>/dev/null || cat

# Test 4: Image Prediction (Legacy endpoint)
if [ -f "test_images/20251010-1.jpeg" ]; then
  echo ""
  echo "4. Testing Image Prediction (POST /predict/image) - Legacy"
  echo "--------------------------------"
  echo "Uploading: test_images/20251010-1.jpeg"
  curl -X POST "$API_BASE_URL/predict/image" \
    -F "file=@test_images/20251010-1.jpeg" \
    -F "confidence_threshold=0.5" \
    -F "label_mode=item_name" \
    -H "Accept: application/json" \
    -w "\nHTTP Status: %{http_code}\n" \
    --max-time 120 \
    | jq '.' 2>/dev/null || cat
else
  echo ""
  echo "4. Skipping Image Prediction (no test image found)"
  echo "   To test, place an image in test_images/ directory"
fi

# Test 5: Detect-and-Identify (New endpoint with OCR)
if [ -f "test_images/20251010-1.jpeg" ]; then
  echo ""
  echo "5. Testing Detect-and-Identify (POST /detect-and-identify) - New with OCR"
  echo "--------------------------------"
  echo "Uploading: test_images/20251010-1.jpeg"
  curl -X POST "$API_BASE_URL/detect-and-identify" \
    -F "file=@test_images/20251010-1.jpeg" \
    -F "confidence_threshold=0.5" \
    -F "enable_ocr=true" \
    -F "enable_sku_matching=true" \
    -H "Accept: application/json" \
    -w "\nHTTP Status: %{http_code}\n" \
    --max-time 180 \
    | jq '.' 2>/dev/null || cat
else
  echo ""
  echo "5. Skipping Detect-and-Identify (no test image found)"
fi

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="

