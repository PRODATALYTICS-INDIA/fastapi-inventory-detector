"""
Decision Engine Service
Implements deterministic, explainable final SKU decision logic that fuses
detection and OCR signals into a single final SKU prediction.

Core Principle: Detection provides a visual prior. OCR provides confirmation.
This is evidence fusion, not voting.
"""
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
import logging

from app.schemas.response import ValidationStatus
from app.config import (
    DETECTION_MIN,
    OCR_MATCH_MIN,
    OCR_TOKEN_MIN,
    CONFLICT_DELTA,
    W_DET_DEFAULT,
    W_OCR_DEFAULT,
    W_DET_OCR_WEAK,
    W_OCR_OCR_WEAK,
    W_DET_OCR_STRONG,
    W_OCR_OCR_STRONG
)

logger = logging.getLogger(__name__)


class DecisionResult(NamedTuple):
    """Result from decision engine (without bbox, which is added later)."""
    sku_id: Optional[str]
    sku_name: Optional[str]
    confidence: float
    validation_status: ValidationStatus
    decision_reason: List[str]


class DecisionEngine:
    """
    Decision engine for fusing detection and OCR evidence into final SKU predictions.
    
    Implements deterministic, explainable decision logic with:
    - Weighted evidence fusion
    - Dynamic weight adjustment based on signal quality
    - Conflict resolution
    - Validation status determination
    """
    
    def __init__(
        self,
        catalog_service=None
    ):
        """
        Initialize decision engine.
        
        Args:
            catalog_service: Optional CatalogService for SKU name lookup
        """
        self.catalog_service = catalog_service
    
    def make_final_decision(
        self,
        detection_class: str,
        detection_confidence: float,
        ocr_text_raw: str,
        ocr_token_confidence_avg: float,
        matched_sku_id: Optional[str],
        ocr_match_confidence: Optional[float]
    ) -> DecisionResult:
        """
        Make final SKU decision by fusing detection and OCR evidence.
        
        This is a pure function - no model calls, deterministic output.
        
        Args:
            detection_class: Detection model's class prediction
            detection_confidence: Detection confidence score [0,1]
            ocr_text_raw: Raw OCR text (empty string if no OCR)
            ocr_token_confidence_avg: Average OCR token confidence [0,1]
            matched_sku_id: Matched SKU ID from OCR matching (None if no match)
            ocr_match_confidence: OCR match confidence [0,1] (None if no match)
            
        Returns:
            DecisionResult with sku_id, sku_name, confidence, decision_reason, validation_status
            (bbox is not included - it's added in the calling code)
        """
        # Normalize inputs
        ocr_text_raw = ocr_text_raw or ""
        ocr_token_confidence_avg = ocr_token_avg if (ocr_token_avg := ocr_token_confidence_avg) is not None else 0.0
        ocr_match_confidence = ocr_match_conf if (ocr_match_conf := ocr_match_confidence) is not None else 0.0
        
        # Track decision reasons for explainability
        decision_reasons = []
        
        # ====================================================================
        # DYNAMIC WEIGHT ADJUSTMENT
        # ====================================================================
        w_det, w_ocr = self._calculate_weights(
            ocr_text_raw,
            ocr_token_confidence_avg,
            ocr_match_confidence
        )
        
        # ====================================================================
        # FINAL CONFIDENCE CALCULATION (Weighted Evidence Fusion)
        # ====================================================================
        final_confidence = (
            w_det * detection_confidence +
            w_ocr * ocr_match_confidence
        )
        
        # Helper: Check if detection_class is a valid SKU (not unknown/-9999)
        detection_is_valid_sku = detection_class not in ("unknown", "-9999", None)
        
        # ====================================================================
        # DECISION LOGIC (Updated with new validation statuses)
        # ====================================================================
        
        # Case 1: Both weak → NOT_VALIDATED
        if detection_confidence < DETECTION_MIN and ocr_match_confidence < OCR_MATCH_MIN:
            final_sku_id = None
            validation_status = ValidationStatus.NOT_VALIDATED
            decision_reasons.append("both_signals_weak")
            decision_reasons.append(f"detection_confidence_{detection_confidence:.2f}_below_{DETECTION_MIN}")
            decision_reasons.append(f"ocr_match_confidence_{ocr_match_confidence:.2f}_below_{OCR_MATCH_MIN}")
            
            # Get SKU name (will be None)
            final_sku_name = None
        
        # Case 2: Detection & OCR agree → FULLY_VALIDATED
        elif matched_sku_id is not None and matched_sku_id == detection_class:
            final_sku_id = detection_class
            validation_status = ValidationStatus.FULLY_VALIDATED
            decision_reasons.append("detection_ocr_agreement")
            decision_reasons.append(f"both_predicted_{detection_class}")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id)
        
        # Case 3: Detection strong, OCR missing or weak
        elif detection_confidence >= DETECTION_MIN and (not ocr_text_raw or ocr_match_confidence < OCR_MATCH_MIN):
            final_sku_id = detection_class
            if detection_is_valid_sku:
                validation_status = ValidationStatus.ONLY_DETECTION_VALIDATED
            else:
                validation_status = ValidationStatus.DETECTION_NOT_VALIDATED
            
            decision_reasons.append("detection_strong_ocr_weak")
            decision_reasons.append(f"detection_confidence_{detection_confidence:.2f}_above_{DETECTION_MIN}")
            
            if not ocr_text_raw:
                decision_reasons.append("ocr_text_empty")
            else:
                decision_reasons.append(f"ocr_match_confidence_{ocr_match_confidence:.2f}_below_{OCR_MATCH_MIN}")
            
            # Apply detection-only penalty
            final_confidence = min(1.0, final_confidence * DETECTION_MIN)
            decision_reasons.append(f"confidence_penalty_applied_{DETECTION_MIN}")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id)
        
        # Case 4: OCR strong, detection weak
        elif ocr_match_confidence >= OCR_MATCH_MIN and detection_confidence < DETECTION_MIN:
            final_sku_id = matched_sku_id
            if detection_is_valid_sku:
                validation_status = ValidationStatus.OCR_VALIDATED
            else:
                validation_status = ValidationStatus.ONLY_OCR_VALIDATED
            
            decision_reasons.append("ocr_strong_detection_weak")
            decision_reasons.append(f"ocr_match_confidence_{ocr_match_confidence:.2f}_above_{OCR_MATCH_MIN}")
            decision_reasons.append(f"detection_confidence_{detection_confidence:.2f}_below_{DETECTION_MIN}")
            
            # Apply OCR-only penalty
            final_confidence = min(1.0, final_confidence * OCR_MATCH_MIN)
            decision_reasons.append(f"confidence_penalty_applied_{OCR_MATCH_MIN}")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id) if final_sku_id else None
        
        # Case 5: Detection & OCR conflict
        elif matched_sku_id is not None and matched_sku_id != detection_class:
            conflict_delta = ocr_match_confidence - detection_confidence
            
            if conflict_delta > CONFLICT_DELTA:
                # OCR significantly stronger
                final_sku_id = matched_sku_id
                if detection_is_valid_sku:
                    validation_status = ValidationStatus.OCR_VALIDATED
                else:
                    validation_status = ValidationStatus.ONLY_OCR_VALIDATED
                decision_reasons.append("ocr_stronger_in_conflict")
                decision_reasons.append(f"ocr_delta_{conflict_delta:.2f}_above_{CONFLICT_DELTA}")
            else:
                # Detection wins or tie
                final_sku_id = detection_class
                if detection_is_valid_sku:
                    validation_status = ValidationStatus.DETECTION_VALIDATED
                else:
                    validation_status = ValidationStatus.DETECTION_NOT_VALIDATED
                decision_reasons.append("detection_wins_conflict")
                decision_reasons.append(f"ocr_delta_{conflict_delta:.2f}_below_{CONFLICT_DELTA}")
            
            decision_reasons.append("detection_ocr_conflict")
            decision_reasons.append(f"detection_class_{detection_class}_vs_ocr_sku_{matched_sku_id}")
            
            # Apply conflict penalty
            final_confidence = max(0.0, final_confidence - 0.15)
            decision_reasons.append("conflict_penalty_applied_0.15")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id)
        
        # Case 6: Detection matched SKU, but OCR found text that didn't match any SKU
        elif detection_is_valid_sku and ocr_text_raw and matched_sku_id is None:
            final_sku_id = detection_class
            validation_status = ValidationStatus.OCR_NOT_VALIDATED
            decision_reasons.append("detection_valid_ocr_text_no_match")
            decision_reasons.append("ocr_found_text_but_no_sku_match")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id)
        
        # Fallback: Detection only (no OCR match)
        else:
            final_sku_id = detection_class
            if detection_is_valid_sku:
                validation_status = ValidationStatus.ONLY_DETECTION_VALIDATED
            else:
                validation_status = ValidationStatus.DETECTION_NOT_VALIDATED
            decision_reasons.append("detection_fallback")
            decision_reasons.append("no_ocr_match_available")
            
            # Get SKU name from catalog
            final_sku_name = self._get_sku_name(final_sku_id)
        
        # Ensure confidence is in valid range
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Log decision for auditability
        logger.debug(
            f"Decision: sku_id={final_sku_id}, confidence={final_confidence:.3f}, "
            f"status={validation_status}, reasons={decision_reasons}"
        )
        
        return DecisionResult(
            sku_id=final_sku_id,
            sku_name=final_sku_name,
            confidence=final_confidence,
            validation_status=validation_status,
            decision_reason=decision_reasons
        )
    
    def _calculate_weights(
        self,
        ocr_text_raw: str,
        ocr_token_confidence_avg: float,
        ocr_match_confidence: float
    ) -> Tuple[float, float]:
        """
        Calculate dynamic weights for detection and OCR based on signal quality.
        
        Args:
            ocr_text_raw: Raw OCR text
            ocr_token_confidence_avg: Average OCR token confidence
            ocr_match_confidence: OCR match confidence
            
        Returns:
            Tuple of (w_det, w_ocr) weights
        """
        # Case A: OCR weak or missing
        if ocr_token_confidence_avg < OCR_TOKEN_MIN or not ocr_text_raw:
            return W_DET_OCR_WEAK, W_OCR_OCR_WEAK
        
        # Case B: OCR strong (brand keyword hit)
        if ocr_match_confidence >= 0.8:
            return W_DET_OCR_STRONG, W_OCR_OCR_STRONG
        
        # Default weights
        return W_DET_DEFAULT, W_OCR_DEFAULT
    
    def _get_sku_name(self, sku_id: Optional[str]) -> Optional[str]:
        """
        Get SKU display name from catalog service.
        
        Args:
            sku_id: SKU ID
            
        Returns:
            SKU display name or None
        """
        if not sku_id:
            return None
        
        if self.catalog_service:
            try:
                sku_info = self.catalog_service.get_sku_info(sku_id)
                if sku_info:
                    return sku_info.get('item_name', sku_id)
            except Exception as e:
                logger.debug(f"Failed to get SKU name for {sku_id}: {str(e)}")
        
        # Fallback to SKU ID itself
        return sku_id

