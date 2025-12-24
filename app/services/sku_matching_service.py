"""
SKU Matching Service
Handles fuzzy matching of OCR text to SKU master list.
Uses CatalogService for efficient catalog data access.
"""
from typing import List, Dict, Tuple, Optional
import os
import re
import logging
from collections import defaultdict

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logging.warning("RapidFuzz not available. Install with: pip install rapidfuzz")

from app.services.catalog_service import CatalogService

logger = logging.getLogger(__name__)


class SKUMatchingService:
    """
    Service for matching OCR text to SKU master list using fuzzy matching.
    Uses CatalogService for efficient catalog data access.
    """
    
    def __init__(
        self,
        catalog_path: Optional[str] = None,
        catalog_service: Optional[CatalogService] = None,
        min_confidence: float = 0.65,
        use_tfidf: bool = False
    ):
        """
        Initialize SKU matching service.
        
        Args:
            catalog_path: Path to catalog JSON file (if catalog_service not provided)
            catalog_service: Pre-initialized CatalogService instance (optional)
            min_confidence: Minimum fuzzy match confidence (0.0-1.0)
            use_tfidf: Whether to use TF-IDF for text similarity (optional enhancement)
        """
        self.min_confidence = min_confidence
        self.use_tfidf = use_tfidf
        
        # Initialize catalog service
        if catalog_service:
            self.catalog_service = catalog_service
        elif catalog_path:
            self.catalog_service = CatalogService(
                catalog_path=catalog_path,
                use_database=True  # Use DuckDB for large datasets
            )
        else:
            # Default to JSON file in app/data folder
            default_path = "app/data/sku_master.json"
            if os.path.exists(default_path):
                self.catalog_service = CatalogService(
                    catalog_path=default_path,
                    use_database=True
                )
            else:
                raise FileNotFoundError(f"Catalog file not found: {default_path}")
        
        self.sku_keywords: Dict[str, List[str]] = {}
        
        # Load SKU data from catalog service
        self._load_sku_data()
        self._build_keyword_index()
    
    def _load_sku_data(self):
        """Load SKU data from catalog service."""
        try:
            # Get all SKUs from catalog service
            self.sku_master = self.catalog_service.get_all_skus()
            
            # Build lookup dictionary by code (or sku_code for backward compatibility)
            self.sku_lookup = {}
            for sku in self.sku_master:
                # Support both new 'code' and old 'sku_code' field names
                sku_code = sku.get('code') or sku.get('sku_code', '')
                if sku_code:
                    self.sku_lookup[sku_code] = sku
            
            logger.info(f"Loaded {len(self.sku_master)} SKUs from catalog service")
            
        except Exception as e:
            logger.error(f"Failed to load SKU data: {str(e)}")
            raise
    
    def _build_keyword_index(self):
        """
        Build keyword index for each SKU.
        Extracts keywords from item_name, brand, category, etc.
        """
        for sku in self.sku_master:
            # Support both new 'code' and old 'sku_code' field names
            sku_code = sku.get('code') or sku.get('sku_code', '')
            keywords = []
            
            # Extract keywords from various fields
            # Support both new 'code' and old 'sku_code' field names
            fields_to_extract = [
                'item_name', 'brand', 'category', 'sub_category',
                'code', 'sku_code', 'additional_info'
            ]
            
            for field in fields_to_extract:
                value = sku.get(field, '')
                if value and isinstance(value, str):
                    # Split by common delimiters and add to keywords
                    words = re.split(r'[-\s_]+', value.lower())
                    keywords.extend([w for w in words if len(w) > 2])
            
            # Extract numbers (sizes, weights)
            numbers = self._extract_numbers(sku)
            keywords.extend(numbers)
            
            self.sku_keywords[sku_code] = list(set(keywords))  # Remove duplicates
        
        logger.info(f"Built keyword index for {len(self.sku_keywords)} SKUs")
    
    def _extract_numbers(self, sku: Dict) -> List[str]:
        """
        Extract numbers (sizes, weights) from SKU data.
        
        Args:
            sku: SKU dictionary
            
        Returns:
            List of number strings (e.g., ['100', 'g', '500', 'ml'])
        """
        numbers = []
        
        # Check quantity_per_unit field
        qty = sku.get('quantity_per_unit', '')
        if qty and isinstance(qty, str):
            # Extract numbers and units
            matches = re.findall(r'(\d+)\s*([a-zA-Z]+)?', qty.lower())
            for num, unit in matches:
                numbers.append(num)
                if unit:
                    numbers.append(unit)
        
        # Extract from item_name
        item_name = sku.get('item_name', '')
        if item_name:
            matches = re.findall(r'(\d+)\s*([a-zA-Z]+)?', item_name.lower())
            for num, unit in matches:
                numbers.append(num)
                if unit:
                    numbers.append(unit)
        
        return numbers
    
    def find_best_sku_match(
        self,
        ocr_text: str
    ) -> Tuple[Optional[str], float, Dict]:
        """
        Find the best matching SKU for given OCR text.
        
        Args:
            ocr_text: OCR-extracted text (lowercase, concatenated)
            
        Returns:
            Tuple of (sku_code, confidence, match_info)
            - sku_code: Best matching SKU code or None
            - confidence: Match confidence (0.0-1.0)
            - match_info: Dictionary with match details
        """
        if not ocr_text or len(ocr_text.strip()) < 3:
            return None, 0.0, {}
        
        if not RAPIDFUZZ_AVAILABLE:
            logger.warning("RapidFuzz not available, using basic matching")
            return self._basic_match(ocr_text)
        
        best_match = None
        best_score = 0.0
        best_sku_code = None
        
        # Try matching against each SKU's keywords
        for sku_code, keywords in self.sku_keywords.items():
            # Create searchable text from keywords
            searchable_text = ' '.join(keywords)
            
            # Calculate fuzzy match scores
            # Use partial ratio for substring matching (handles partial text)
            partial_score = fuzz.partial_ratio(ocr_text, searchable_text) / 100.0
            
            # Use token sort ratio for word order independence
            token_score = fuzz.token_sort_ratio(ocr_text, searchable_text) / 100.0
            
            # Use token set ratio for better handling of partial matches
            token_set_score = fuzz.token_set_ratio(ocr_text, searchable_text) / 100.0
            
            # Weighted combination
            combined_score = (
                0.3 * partial_score +
                0.3 * token_score +
                0.4 * token_set_score
            )
            
            # Also try direct matching against SKU fields
            sku_data = self.sku_lookup.get(sku_code, {})
            # Support both new 'code' and old 'sku_code' field names
            code_field = sku_data.get('code') or sku_data.get('sku_code', '')
            for field in ['item_name', 'brand']:
                field_value = str(sku_data.get(field, '')).lower()
                if field_value:
                    field_score = fuzz.partial_ratio(ocr_text, field_value) / 100.0
                    combined_score = max(combined_score, field_score * 0.9)
            # Also match against code field
            if code_field:
                code_value = str(code_field).lower()
                field_score = fuzz.partial_ratio(ocr_text, code_value) / 100.0
                combined_score = max(combined_score, field_score * 0.9)
            
            # Check for number matches (sizes, weights)
            number_match_score = self._match_numbers(ocr_text, sku_data)
            if number_match_score > 0:
                combined_score = min(1.0, combined_score + number_match_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_sku_code = sku_code
        
        # Check if score meets minimum threshold
        if best_score >= self.min_confidence:
            match_info = {
                'match_method': 'fuzzy',
                'score': best_score,
                'sku_data': self.sku_lookup.get(best_sku_code, {})
            }
            return best_sku_code, best_score, match_info
        else:
            return None, best_score, {}
    
    def _match_numbers(self, ocr_text: str, sku_data: Dict) -> float:
        """
        Match numbers (sizes, weights) between OCR text and SKU data.
        
        Args:
            ocr_text: OCR text
            sku_data: SKU data dictionary
            
        Returns:
            Match score (0.0-1.0)
        """
        # Extract numbers from OCR text
        ocr_numbers = re.findall(r'\d+', ocr_text)
        ocr_units = re.findall(r'\d+\s*([a-zA-Z]+)', ocr_text.lower())
        
        if not ocr_numbers:
            return 0.0
        
        # Extract numbers from SKU
        sku_numbers = []
        qty = str(sku_data.get('quantity_per_unit', '')).lower()
        if qty:
            sku_numbers.extend(re.findall(r'\d+', qty))
        
        # Check for matches
        matches = 0
        for ocr_num in ocr_numbers:
            if ocr_num in sku_numbers:
                matches += 1
        
        if not sku_numbers:
            return 0.0
        
        return matches / max(len(ocr_numbers), len(sku_numbers))
    
    def _basic_match(self, ocr_text: str) -> Tuple[Optional[str], float, Dict]:
        """
        Basic matching without RapidFuzz (fallback).
        
        Args:
            ocr_text: OCR text
            
        Returns:
            Tuple of (sku_code, confidence, match_info)
        """
        best_match = None
        best_score = 0.0
        
        for sku_code, keywords in self.sku_keywords.items():
            searchable_text = ' '.join(keywords)
            
            # Simple substring matching
            if ocr_text in searchable_text or searchable_text in ocr_text:
                score = min(len(ocr_text) / len(searchable_text), 1.0)
            else:
                # Count common words
                ocr_words = set(ocr_text.split())
                search_words = set(searchable_text.split())
                common = len(ocr_words & search_words)
                total = len(ocr_words | search_words)
                score = common / total if total > 0 else 0.0
            
            if score > best_score:
                best_score = score
                best_match = sku_code
        
        if best_score >= self.min_confidence:
            return best_match, best_score, {'match_method': 'basic'}
        else:
            return None, best_score, {}
    
    def get_sku_info(self, sku_code: str) -> Optional[Dict]:
        """
        Get SKU information by code.
        Uses catalog service for efficient lookup.
        
        Args:
            sku_code: SKU code
            
        Returns:
            SKU data dictionary or None
        """
        # Try catalog service first (supports database queries)
        sku_info = self.catalog_service.get_sku_info(sku_code)
        if sku_info:
            return sku_info
        
        # Fallback to in-memory lookup
        return self.sku_lookup.get(sku_code)
