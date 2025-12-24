#!/usr/bin/env python3
"""End-to-end local test: detection → OCR → product prediction.

This script:
- Loads an image from `test/test_images` (or a specific `--image` path)
- Runs the detection model to get object segments / crops
- Runs OCR on each detected object area
- Uses SKU matching to predict the product for each detection

It uses the same internal components as the FastAPI app, but runs everything
locally in a single Python process (no HTTP involved).
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Path setup so we can import from `app.*`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_model_path  # noqa: E402
from app.models.detection_model import DetectionModel  # noqa: E402
from app.models.ocr_model import OCRModel  # noqa: E402
from app.services.detection_service import DetectionService  # noqa: E402
from app.services.ocr_service import OCRService  # noqa: E402
from app.services.sku_matching_service import SKUMatchingService  # noqa: E402


def find_default_image(image_dir: Path) -> Optional[Path]:
    """Return the first image file in `image_dir`, if any."""
    if not image_dir.exists():
        return None

    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        files = sorted(image_dir.glob(pattern))
        if files:
            return files[0]
    return None


def run_pipeline(image_path: Path, confidence_threshold: float = 0.5) -> None:
    """Run detection → OCR → SKU prediction on a single image.

    Saves each detected crop under:
        test/test_output/<image_stem>/crop_<idx>.jpg
    so you can visually inspect what OCR sees.
    """
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 80)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    # -----------------------------------------------------------------------
    # Load image
    # -----------------------------------------------------------------------
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    print(f"✅ Image loaded: shape={image.shape}")

    # -----------------------------------------------------------------------
    # Initialize detection, OCR, and SKU matching components
    # -----------------------------------------------------------------------
    try:
        model_path = get_model_path()
        print(f"📦 Loading detection model from: {model_path}")
        detection_model = DetectionModel(model_path)
        detection_service = DetectionService(detection_model)
        print("✅ Detection model initialized")
    except Exception as e:  # pragma: no cover - diagnostic
        print(f"❌ Failed to initialize detection model: {e}")
        return

    try:
        print("📦 Initializing OCR model (PaddleOCR)...")
        ocr_model = OCRModel(use_gpu=False)
        ocr_service = OCRService(ocr_model, use_multiprocessing=False)
        print("✅ OCR model initialized")
    except Exception as e:  # pragma: no cover - diagnostic
        print(f"❌ Failed to initialize OCR model: {e}")
        print("   Make sure PaddleOCR is installed: pip install paddlepaddle paddleocr")
        return

    # SKU master is resolved internally; falls back to app/data/sku_master.json
    try:
        print("📦 Initializing SKU matching service...")
        sku_service = SKUMatchingService(
            catalog_path=None,  # use default path or provided CatalogService
        )
        print("✅ SKU matching service initialized")
    except Exception as e:  # pragma: no cover - diagnostic
        print(f"❌ Failed to initialize SKU matching service: {e}")
        return

    # -----------------------------------------------------------------------
    # Step 1: Detection → get crops
    # -----------------------------------------------------------------------
    print("\n🔍 Running detection...")
    detections, crops = detection_service.detect(
        image,
        confidence_threshold=confidence_threshold,
    )
    print(f"✅ Detected {len(detections)} objects (with {sum(c is not None for c in crops)} valid crops)")

    if not detections:
        print("⚠️ No detections found; nothing to OCR or match.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Save crops for visual inspection
    # -----------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "test" / "test_output" / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving crops to: {output_dir}")
    for idx, crop in enumerate(crops, start=1):
        if crop is None:
            continue
        crop_path = output_dir / f"crop_{idx}.jpg"
        cv2.imwrite(str(crop_path), crop)

    # -----------------------------------------------------------------------
    # Step 3: OCR on each crop
    # -----------------------------------------------------------------------
    print("\n🔍 Running OCR on detected object crops...")
    ocr_results = ocr_service.extract_text_batch(crops)

    # Save OCR text per crop for inspection
    ocr_summary_path = output_dir / "ocr_results.txt"
    with ocr_summary_path.open("w", encoding="utf-8") as f:
        for idx, (ocr_text, ocr_details) in enumerate(ocr_results, start=1):
            f.write(f"Crop {idx}:\n")
            if ocr_text:
                f.write(f"  text: {ocr_text}\n")
                f.write(f"  lines: {len(ocr_details)}\n")
                for j, d in enumerate(ocr_details, start=1):
                    f.write(
                        f"    {j}. '{d.get('text','')}' "
                        f"(conf={d.get('confidence',0.0):.3f})\n"
                    )
            else:
                f.write("  text: (none)\n")
            f.write("\n")

    # -----------------------------------------------------------------------
    # Step 4: SKU matching / product prediction
    # -----------------------------------------------------------------------
    print("\n📦 Per-object results (detection + OCR + SKU match):")
    print(f"(Detailed OCR output saved to: {ocr_summary_path})")
    print("-" * 80)

    for idx, (det, (ocr_text, ocr_details)) in enumerate(zip(detections, ocr_results), start=1):
        detection_sku = det.get("sku", "N/A")
        detection_conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", [])

        print(f"Object {idx}:")
        print(f"  Detection SKU: {detection_sku}")
        print(f"  Detection confidence: {detection_conf:.2%}")
        if bbox:
            print(f"  BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

        if ocr_text:
            short_text = (ocr_text[:80] + "...") if len(ocr_text) > 80 else ocr_text
            print(f"  OCR text: '{short_text}'")
        else:
            print("  OCR text: (none)")

        # SKU matching
        if ocr_text:
            matched_sku, match_conf, match_info = sku_service.find_best_sku_match(ocr_text)
            if matched_sku:
                sku_info = sku_service.get_sku_info(matched_sku) or {}
                sku_name = sku_info.get("item_name", matched_sku)
                print(f"  Predicted SKU: {matched_sku} ({sku_name})")
                print(f"  SKU match confidence: {match_conf:.2%}")
            else:
                print("  Predicted SKU: (no confident match)")
                print(f"  Best match confidence: {match_conf:.2%}")
        else:
            print("  Predicted SKU: (skipped – no OCR text)")

        print("-" * 80)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run local detection→OCR→SKU prediction pipeline on an image from "
            "test/test_images or a custom --image path."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a specific test image (overrides default discovery).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )

    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
    else:
        default_dir = PROJECT_ROOT / "test" / "test_images"
        image_path = find_default_image(default_dir)
        if image_path is None:
            print(f"❌ No images found in {default_dir}")
            return

    run_pipeline(image_path, confidence_threshold=args.confidence)


if __name__ == "__main__":
    main()
