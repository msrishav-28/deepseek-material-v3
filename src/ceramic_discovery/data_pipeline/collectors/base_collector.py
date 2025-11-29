"""
BaseCollector Abstract Interface

Defines the contract for all Phase 2 data collectors.

Design Pattern: Template Method
- Abstract methods: collect(), get_source_name(), validate_config()
- Concrete methods: save_to_csv(), log_stats()

All collectors must inherit from this base class.

CRITICAL: This is Phase 2 wrapper around Phase 1 clients.
- Do NOT modify existing client classes
- Use composition (HAS-A) not inheritance (IS-A)
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    This is a Phase 2 wrapper around Phase 1 clients. When implementing
    collectors that use existing Phase 1 code:
    - Import the existing client class
    - Create an instance via composition (self.client = ExistingClient(...))
    - Do NOT inherit from the existing client
    - Do NOT modify the existing client code
    
    Attributes:
        config: Configuration dictionary for this collector
        source_name: Identifier for this data source
        
    Methods to Implement:
        collect(): Main collection logic, returns DataFrame
        get_source_name(): Return source identifier string
        validate_config(): Validate configuration has required keys
        
    Provided Methods:
        save_to_csv(): Save collected data to processed/ directory
        log_stats(): Log collection statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize collector with configuration.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration validation fails
        """
        self.config = config
        
        if not self.validate_config():
            raise ValueError(
                f"{self.__class__.__name__} configuration invalid. "
                f"Check required keys."
            )
        
        self.source_name = self.get_source_name()
        logger.info(f"Initialized {self.source_name} collector")
    
    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """
        Collect data from source.
        
        Must be implemented by subclasses.
        
        Returns:
            DataFrame with collected materials. Must include:
                - 'source' column with source identifier
                - 'formula' column with chemical formula
                - Source-specific property columns
        
        Raises:
            Exception: If collection fails (subclass-specific)
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """
        Return identifier for this data source.
        
        Returns:
            Source name (e.g., 'MaterialsProject', 'JARVIS', 'AFLOW')
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate configuration has required keys.
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def save_to_csv(self, df: pd.DataFrame, output_dir: Path = Path('data/processed')) -> Path:
        """
        Save collected data to CSV file.
        
        Args:
            df: DataFrame to save
            output_dir: Output directory (default: data/processed/)
            
        Returns:
            Path to saved CSV file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standardized filename: {source}_data.csv
        filename = f"{self.source_name.lower()}_data.csv"
        output_path = output_dir / filename
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} materials to {output_path}")
        
        return output_path
    
    def log_stats(self, df: pd.DataFrame):
        """
        Log collection statistics.
        
        Args:
            df: Collected DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"{self.source_name} Collection Statistics")
        logger.info(f"{'='*60}")
        logger.info(f"Total materials: {len(df)}")
        
        if 'formula' in df.columns:
            logger.info(f"Unique formulas: {df['formula'].nunique()}")
        
        # Log property coverage
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            logger.info(f"\nProperty coverage:")
            for col in numeric_cols:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100 if len(df) > 0 else 0
                logger.info(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")
        
        logger.info(f"{'='*60}")
