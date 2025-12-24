#!/bin/bash
# Batch test script for OCR detection on all test images
# Tests the new /detect-and-identify endpoint with OCR and SKU matching

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
TEST_IMAGE_DIR="test_images"

echo "=========================================="
echo "Batch OCR Detection Test"
echo "API URL: $API_BASE_URL"
echo "Image Directory: $TEST_IMAGE_DIR"
echo "=========================================="

if [ ! -d "$TEST_IMAGE_DIR" ]; then
  echo "❌ Test images directory not found: $TEST_IMAGE_DIR"
  exit 1
fi

# Find all image files
IMAGE_FILES=$(find "$TEST_IMAGE_DIR" -type f \( -name "*.jpeg" -o -name "*.jpg" -o -name "*.png" \) | head -10)

if [ -z "$IMAGE_FILES" ]; then
  echo "❌ No image files found in $TEST_IMAGE_DIR"
  exit 1
fi

echo ""
echo "Found $(echo "$IMAGE_FILES" | wc -l) images to test"
echo ""

# Test each image
SUCCESS_COUNT=0
TOTAL_COUNT=0

for IMAGE_FILE in $IMAGE_FILES; do
  TOTAL_COUNT=$((TOTAL_COUNT + 1))
  IMAGE_NAME=$(basename "$IMAGE_FILE")
  
  echo "----------------------------------------"
  echo "Testing: $IMAGE_NAME"
  echo "----------------------------------------"
  
  RESPONSE=$(curl -s -X POST "$API_BASE_URL/detect-and-identify" \
    -F "file=@$IMAGE_FILE" \
    -F "confidence_threshold=0.5" \
    -F "enable_ocr=true" \
    -F "enable_sku_matching=true" \
    -H "Accept: application/json" \
    --max-time 180 \
    -w "\nHTTP_STATUS:%{http_code}")
  
  HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
  BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS/d')
  
  if [ "$HTTP_STATUS" = "200" ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    
    # Extract key information using jq if available, otherwise use grep
    if command -v jq &> /dev/null; then
      TOTAL_DETECTIONS=$(echo "$BODY" | jq -r '.total_detections // 0')
      PROCESSING_TIME=$(echo "$BODY" | jq -r '.processing_time_ms // 0')
      
      echo "✅ Success"
      echo "   Detections: $TOTAL_DETECTIONS"
      echo "   Processing Time: ${PROCESSING_TIME}ms"
      
      # Count products with OCR text
      OCR_COUNT=$(echo "$BODY" | jq '[.products[] | select(.ocr_text != "")] | length')
      SKU_MATCH_COUNT=$(echo "$BODY" | jq '[.products[] | select(.sku_match_confidence != null and .sku_match_confidence > 0.5)] | length')
      
      echo "   Products with OCR: $OCR_COUNT"
      echo "   Products with SKU match: $SKU_MATCH_COUNT"
      
      # Show sample OCR texts
      echo "$BODY" | jq -r '.products[0:3][] | "   OCR: \(.ocr_text // "none")"' 2>/dev/null || true
    else
      echo "✅ Success (HTTP $HTTP_STATUS)"
      echo "$BODY" | head -20
    fi
  else
    echo "❌ Failed (HTTP $HTTP_STATUS)"
    echo "$BODY" | head -10
  fi
  
  echo ""
done

echo "=========================================="
echo "Batch Test Summary"
echo "=========================================="
echo "Total Images: $TOTAL_COUNT"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $((TOTAL_COUNT - SUCCESS_COUNT))"
echo "Success Rate: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%"
echo "=========================================="
