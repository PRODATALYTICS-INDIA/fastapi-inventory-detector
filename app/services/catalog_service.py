"""
Catalog Service
Handles SKU catalog data loading and querying.
Designed to scale from small JSON files to large databases (200,000+ SKUs).

Supports:
- JSON file loading (initial/small datasets)
- DuckDB for efficient querying of large datasets
- Future: Remote server fetching
"""
from typing import List, Dict, Optional, Any
import os
import json
import logging
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.warning("DuckDB not available. Install with: pip install duckdb")

logger = logging.getLogger(__name__)


class CatalogService:
    """
    Service for managing SKU catalog data.
    Handles loading from JSON, converting to DuckDB for performance,
    and provides efficient querying interface.
    """
    
    def __init__(
        self,
        catalog_path: str,
        use_database: bool = True,
        db_path: Optional[str] = None
    ):
        """
        Initialize catalog service.
        
        Args:
            catalog_path: Path to catalog JSON file
            use_database: Whether to use DuckDB for large datasets (recommended)
            db_path: Path to DuckDB database file (auto-generated if None)
        """
        self.catalog_path = catalog_path
        self.use_database = use_database and DUCKDB_AVAILABLE
        # Ensure app/data directory exists
        if db_path is None:
            os.makedirs("app/data", exist_ok=True)
            self.db_path = "app/data/catalog.duckdb"
        else:
            self.db_path = db_path
        self.conn: Optional[Any] = None
        self.catalog_df: Optional[pd.DataFrame] = None
        self.sku_lookup: Dict[str, Dict] = {}
        
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from JSON and optionally convert to DuckDB."""
        try:
            if not os.path.exists(self.catalog_path):
                raise FileNotFoundError(f"Catalog file not found: {self.catalog_path}")
            
            logger.info(f"Loading catalog from: {self.catalog_path}")
            
            # Load from JSON
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                catalog_data = json.load(f)
            
            # Convert JSON list to DataFrame
            self.catalog_df = pd.DataFrame(catalog_data)
            logger.info(f"Loaded {len(self.catalog_df)} SKUs from JSON")
            
            # Build in-memory lookup for small datasets
            # Support both 'code' and 'sku_code' column names for backward compatibility
            code_column = "code" if "code" in self.catalog_df.columns else "sku_code"
            if len(self.catalog_df) < 10000:
                # For small datasets, keep in-memory dict for fast access
                self.sku_lookup = self.catalog_df.set_index(code_column).to_dict(orient="index")
                logger.info("Using in-memory lookup for small dataset")
            else:
                # For large datasets, use database
                logger.info("Large dataset detected, using database for queries")
            
            # Convert to DuckDB if enabled
            # For small datasets (< 10k), in-memory dict is faster
            # For larger datasets, DuckDB provides better performance
            if self.use_database:
                if len(self.catalog_df) >= 1000:
                    # Auto-create database for medium/large datasets
                    self._create_database()
                else:
                    logger.info("Small dataset - using in-memory lookup (faster than database)")
            
        except Exception as e:
            logger.error(f"Failed to load catalog: {str(e)}")
            raise
    
    def _create_database(self):
        """Create DuckDB database from catalog DataFrame."""
        try:
            if not DUCKDB_AVAILABLE:
                logger.warning("DuckDB not available, using in-memory lookup only")
                return
            
            if self.catalog_df is None or len(self.catalog_df) == 0:
                logger.warning("No catalog data to create database")
                return
            
            # Create connection
            self.conn = duckdb.connect(self.db_path)
            
            # Create table from DataFrame using register method
            # DuckDB can directly work with pandas DataFrames
            self.conn.register('catalog_df', self.catalog_df)
            self.conn.execute("DROP TABLE IF EXISTS sku_catalog")
            self.conn.execute("CREATE TABLE sku_catalog AS SELECT * FROM catalog_df")
            self.conn.unregister('catalog_df')
            
            # Verify table was created
            try:
                count = self.conn.execute("SELECT COUNT(*) FROM sku_catalog").fetchone()[0]
                if count != len(self.catalog_df):
                    logger.warning(f"Database row count mismatch: {count} vs {len(self.catalog_df)}")
            except Exception as e:
                logger.warning(f"Could not verify database row count: {str(e)}")
            
            # DuckDB automatically creates indexes, but we can create a primary key for faster lookups
            try:
                # Support both 'code' and 'sku_code' column names
                code_column = "code" if "code" in self.catalog_df.columns else "sku_code"
                # Create index on code column for fast lookups (DuckDB supports this)
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_code ON sku_catalog({code_column})")
            except Exception as e:
                # Some DuckDB versions may not support CREATE INDEX
                logger.debug(f"Index creation skipped: {str(e)}")
            
            logger.info(f"Created DuckDB database: {self.db_path}")
            logger.info(f"Database contains {len(self.catalog_df)} SKUs")
            
        except Exception as e:
            logger.warning(f"Failed to create database: {str(e)}")
            logger.warning("Falling back to in-memory lookup")
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
            self.conn = None
    
    def get_sku_info(self, sku_code: str) -> Optional[Dict]:
        """
        Get SKU information by SKU code.
        
        Args:
            sku_code: SKU code to lookup
            
        Returns:
            SKU information dictionary or None if not found
        """
        # Support both 'code' and 'sku_code' column names
        code_column = "code" if self.catalog_df is not None and "code" in self.catalog_df.columns else "sku_code"
        
        # Try in-memory lookup first (fastest for small datasets)
        if self.sku_lookup:
            return self.sku_lookup.get(sku_code)
        
        # Use database for large datasets
        if self.conn:
            try:
                # DuckDB uses $1, $2 for parameters, not ?
                result = self.conn.execute(
                    f"SELECT * FROM sku_catalog WHERE {code_column} = $1",
                    [sku_code]
                ).fetchone()
                
                if result:
                    # Get column names from DataFrame (DuckDB doesn't use PRAGMA)
                    columns = list(self.catalog_df.columns) if self.catalog_df is not None else []
                    if not columns:
                        # Fallback: try to get from query description
                        try:
                            columns = [desc[0] for desc in self.conn.description]
                        except:
                            columns = []
                    return dict(zip(columns, result))
            except Exception as e:
                logger.error(f"Database query failed: {str(e)}")
        
        # Fallback to DataFrame lookup
        if self.catalog_df is not None:
            matches = self.catalog_df[self.catalog_df[code_column] == sku_code]
            if not matches.empty:
                return matches.iloc[0].to_dict()
        
        return None
    
    def search_skus(
        self,
        search_term: str,
        fields: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search SKUs by text in specified fields.
        
        Args:
            search_term: Text to search for
            fields: Fields to search in (default: item_name, brand, sku_code)
            limit: Maximum number of results
            
        Returns:
            List of matching SKU dictionaries
        """
        if fields is None:
            # Support both 'code' and 'sku_code' column names
            code_column = "code" if self.catalog_df is not None and "code" in self.catalog_df.columns else "sku_code"
            fields = ['item_name', 'brand', code_column]
        
        search_term_lower = search_term.lower()
        
        # Use database for large datasets
        if self.conn:
            try:
                # Build SQL query with LIKE conditions
                # DuckDB uses $1, $2 for parameters
                conditions = " OR ".join([f"LOWER({field}) LIKE ${i+1}" for i, field in enumerate(fields)])
                params = [f"%{search_term_lower}%"] * len(fields)
                params.append(limit)
                
                query = f"""
                    SELECT * FROM sku_catalog 
                    WHERE {conditions}
                    LIMIT ${len(fields) + 1}
                """
                
                results = self.conn.execute(query, params).fetchall()
                
                if results:
                    # Get column names from DataFrame
                    columns = list(self.catalog_df.columns) if self.catalog_df is not None else []
                    if not columns:
                        # Fallback: try to get from query result description
                        try:
                            columns = [desc[0] for desc in self.conn.description]
                        except:
                            columns = []
                    return [dict(zip(columns, row)) for row in results]
            except Exception as e:
                logger.error(f"Database search failed: {str(e)}")
        
        # Fallback to DataFrame search
        if self.catalog_df is not None:
            mask = pd.Series([False] * len(self.catalog_df))
            for field in fields:
                if field in self.catalog_df.columns:
                    mask |= self.catalog_df[field].astype(str).str.lower().str.contains(
                        search_term_lower, na=False
                    )
            
            matches = self.catalog_df[mask].head(limit)
            return matches.to_dict('records')
        
        return []
    
    def get_all_skus(self) -> List[Dict]:
        """
        Get all SKUs (use with caution for large datasets).
        
        Returns:
            List of all SKU dictionaries
        """
        if self.catalog_df is not None:
            return self.catalog_df.to_dict('records')
        return []
    
    def get_catalog_stats(self) -> Dict[str, Any]:
        """
        Get catalog statistics.
        
        Returns:
            Dictionary with catalog statistics
        """
        stats = {
            "total_skus": len(self.catalog_df) if self.catalog_df is not None else 0,
            "using_database": self.conn is not None,
            "database_path": self.db_path if self.conn else None,
            "has_in_memory_lookup": len(self.sku_lookup) > 0
        }
        
        if self.catalog_df is not None:
            stats["columns"] = list(self.catalog_df.columns)
            # Support both 'code' and 'sku_code' column names
            code_column = "code" if "code" in self.catalog_df.columns else "sku_code"
            stats["sample_sku_codes"] = self.catalog_df[code_column].head(5).tolist() if code_column in self.catalog_df.columns else []
        
        return stats
    
    def refresh_from_json(self):
        """Reload catalog from JSON file (useful for updates)."""
        logger.info("Refreshing catalog from JSON file...")
        self._load_catalog()
    
    def refresh_from_server(self, server_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Refresh catalog from remote server (future implementation).
        
        This method is designed to fetch catalog data from a remote API server
        when the dataset becomes too large for local storage or needs real-time updates.
        
        Args:
            server_url: URL to fetch catalog from (e.g., "https://api.example.com/catalog")
            api_key: Optional API key for authentication
            
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement server fetching
        # This would:
        # 1. Fetch catalog data from remote API (JSON/CSV)
        # 2. Update local DuckDB database
        # 3. Maintain cache for offline access
        # 4. Support incremental updates
        
        logger.warning("Server fetching not yet implemented")
        logger.info("Future implementation will support:")
        logger.info("  - Remote API fetching")
        logger.info("  - Incremental updates")
        logger.info("  - Caching for offline access")
        raise NotImplementedError("Server fetching will be implemented in future version")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
