"""
Phase 2 NIST Baseline Collector

NEW FUNCTIONALITY: Reference baseline data for validation.
Provides well-established reference values for common ceramic materials.

IMPORTANT DISTINCTION:
- Phase 1 NISTClient: Temperature-dependent thermochemical properties from NIST database
- Phase 2 NISTBaselineCollector: Reference baseline values for validation (this file)

These are separate and serve different purposes:
- Phase 1: Dynamic data retrieval from NIST API
- Phase 2: Static reference values embedded in configuration

This collector loads baseline reference data from configuration rather than
querying an external API. The baseline data serves as validation anchors
for other collected data sources.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class NISTBaselineCollector(BaseCollector):
    """
    NIST baseline data collector for reference values.
    
    This is Phase 2 baseline data (separate from Phase 1 NISTClient).
    Loads reference baseline values from configuration for validation purposes.
    
    Phase 1 NISTClient provides:
    - Temperature-dependent thermochemical properties
    - Dynamic API queries to NIST database
    - Enthalpy, entropy, heat capacity vs temperature
    
    Phase 2 NISTBaselineCollector provides:
    - Static reference values for common ceramics
    - Baseline data for validation
    - Well-established property values from literature
    
    Configuration keys:
        baseline_materials: Dict mapping material names to property dictionaries
            Each material must have: formula, ceramic_type, density, hardness, K_IC, source
    
    Returns DataFrame with columns:
        - source: 'NIST'
        - formula: Chemical formula
        - ceramic_type: Material type (e.g., 'SiC', 'B4C')
        - density: Density (g/cm³)
        - hardness: Hardness (GPa)
        - K_IC: Fracture toughness (MPa·m^0.5)
        - data_source: Original source of baseline data (provenance)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NIST baseline collector.
        
        Args:
            config: Configuration dictionary with keys:
                - baseline_materials: Dict[str, Dict] of material properties
        """
        super().__init__(config)
        
        self.baseline_materials = config.get('baseline_materials', {})
    
    def collect(self) -> pd.DataFrame:
        """
        Collect baseline reference data from configuration.
        
        Returns:
            DataFrame with baseline material properties
        """
        materials_list = []
        
        for material_name, properties in self.baseline_materials.items():
            logger.info(f"Loading baseline data for: {material_name}")
            
            # Create material record
            material = {
                'source': 'NIST',
                'formula': properties.get('formula'),
                'ceramic_type': properties.get('ceramic_type'),
                'density': properties.get('density'),
                'hardness': properties.get('hardness'),
                'K_IC': properties.get('K_IC'),
                'data_source': properties.get('source', 'NIST Reference'),
            }
            
            materials_list.append(material)
        
        df = pd.DataFrame(materials_list)
        logger.info(f"Loaded {len(df)} baseline materials from NIST references")
        
        return df
    
    def get_source_name(self) -> str:
        """Return source identifier."""
        return "NIST"
    
    def validate_config(self) -> bool:
        """
        Validate configuration has required keys and complete data.
        
        Returns:
            True if valid, False otherwise
        """
        # baseline_materials is required
        if 'baseline_materials' not in self.config:
            logger.error("Configuration missing 'baseline_materials' key")
            return False
        
        # Validate baseline_materials is a dict
        if not isinstance(self.config['baseline_materials'], dict):
            logger.error("'baseline_materials' must be a dictionary")
            return False
        
        # Validate each material has required properties
        required_properties = ['formula', 'ceramic_type', 'density', 'hardness', 'K_IC', 'source']
        
        for material_name, properties in self.config['baseline_materials'].items():
            if not isinstance(properties, dict):
                logger.error(f"Properties for {material_name} must be a dictionary")
                return False
            
            for prop in required_properties:
                if prop not in properties:
                    logger.error(
                        f"Material {material_name} missing required property: {prop}"
                    )
                    return False
        
        return True
