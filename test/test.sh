#!/bin/bash

# =============================================================================
# FastAPI Inventory Detector - Test Script
# =============================================================================
# This script processes all images in the test_images folder and saves results
# Run this script from the test/ directory
# =============================================================================

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
TEST_FOLDER="test_images"
CONFIDENCE_THRESHOLD="0.5"
LABEL_MODE="item_name"

# Get current date in yyyymmdd format
DATE_STR=$(date +%Y%m%d)
OUTPUT_FILE="test_output_${DATE_STR}.md"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}FastAPI Inventory Detector - Test Runner${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Check if test_images folder exists
if [ ! -d "$TEST_FOLDER" ]; then
    echo -e "${RED}❌ Error: Test images folder '$TEST_FOLDER' not found!${NC}"
    exit 1
fi

# Check if API is running
echo -e "${GREEN}✓ Checking if API is running...${NC}"
if ! curl -s -f "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: API is not running at ${API_URL}${NC}"
    echo -e "${YELLOW}   Please start the API first using: ../run.sh (from project root)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API is running${NC}"
echo ""

# Find all image files in test_images folder
IMAGE_FILES=($(find "$TEST_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \)))

if [ ${#IMAGE_FILES[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No image files found in '$TEST_FOLDER' folder${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${#IMAGE_FILES[@]} image file(s) to process${NC}"
echo ""

# Create output file with header
cat > "$OUTPUT_FILE" << EOF
# Test Results - $(date +"%Y-%m-%d %H:%M:%S")

## Test Configuration
- **API URL**: ${API_URL}
- **Confidence Threshold**: ${CONFIDENCE_THRESHOLD}
- **Label Mode**: ${LABEL_MODE}
- **Total Images**: ${#IMAGE_FILES[@]}

---

EOF

# Process each image
SUCCESS_COUNT=0
FAIL_COUNT=0

for image_file in "${IMAGE_FILES[@]}"; do
    filename=$(basename "$image_file")
    echo -e "${BLUE}Processing: ${filename}...${NC}"
    
    # Call API and capture response
    response=$(curl -s -X POST "${API_URL}/predict/image" \
        -F "file=@${image_file}" \
        -F "confidence_threshold=${CONFIDENCE_THRESHOLD}" \
        -F "label_mode=${LABEL_MODE}" 2>&1)
    
    # Check if request was successful
    if echo "$response" | grep -q '"success":true'; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Extract relevant information from JSON response
        detections=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    detections = data.get('detections', {})
    stats = data.get('statistics', [])
    
    # Format detections
    print('## Detections:')
    if detections:
        for item, info in detections.items():
            if isinstance(info, dict):
                count = info.get('count', 0)
                conf_avg = info.get('confidence_avg', 0)
                print(f'- **{item}**: {count} items (avg confidence: {conf_avg:.1f}%)')
            else:
                print(f'- **{item}**: {info}')
    else:
        print('No detections found')
    
    # Format statistics
    if stats:
        print('')
        print('### Statistics:')
        print('| Item | Count | Confidence (%) | Frame Presence (%) |')
        print('|------|-------|----------------|-------------------|')
        for stat in stats:
            item = stat.get('item_name', 'N/A')
            count = stat.get('count', 0)
            conf = stat.get('confidence(%)', 'N/A')
            presence = stat.get('frame_presence(%)', 'N/A')
            print(f'| {item} | {count} | {conf} | {presence} |')
except Exception as e:
    print(f'Error parsing response: {e}')
" 2>/dev/null || echo "Error parsing detections")
        
        # Append to output file
        cat >> "$OUTPUT_FILE" << EOF
## Image: ${filename}

**File**: \`${filename}\`  
**Timestamp**: $(date +"%Y-%m-%d %H:%M:%S")

${detections}

---

EOF
        echo -e "${GREEN}  ✓ Success${NC}"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        error_msg=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('detail', 'Unknown error'))
except:
    print('Failed to parse error response')
" 2>/dev/null || echo "Unknown error")
        
        # Append error to output file
        cat >> "$OUTPUT_FILE" << EOF
## Image: ${filename}

**File**: \`${filename}\`  
**Status**: ❌ **FAILED**  
**Error**: ${error_msg}

---

EOF
        echo -e "${RED}  ✗ Failed: ${error_msg}${NC}"
    fi
done

# Add summary to output file
cat >> "$OUTPUT_FILE" << EOF
## Test Summary

- **Total Images**: ${#IMAGE_FILES[@]}
- **Successful**: ${SUCCESS_COUNT}
- **Failed**: ${FAIL_COUNT}
- **Success Rate**: $(awk "BEGIN {printf \"%.1f\", (${SUCCESS_COUNT}/${#IMAGE_FILES[@]})*100}")%

---
*Generated on $(date +"%Y-%m-%d %H:%M:%S")*
EOF

echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}✓ Test completed!${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo -e "Results saved to: ${GREEN}${OUTPUT_FILE}${NC}"
echo -e "Summary: ${GREEN}${SUCCESS_COUNT}${NC} successful, ${RED}${FAIL_COUNT}${NC} failed out of ${#IMAGE_FILES[@]} total"
echo ""

