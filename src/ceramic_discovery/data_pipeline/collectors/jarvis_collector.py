"""
Phase 2 JARVIS Collector

CRITICAL: This wraps the existing JarvisClient from Phase 1 SiC Integration (Task 2).
Do NOT modify src/ceramic_discovery/dft/jarvis_client.py
"""

import pandas as pd
from typing import Dict, Any, Set
import logging

# Import existing Phase 1 components (READ ONLY)
try:
    from ceramic_discovery.dft.jarvis_client import JarvisClient
except ImportError as e:
    raise ImportError(
        "Cannot import JarvisClient from Phase 1. "
        "Ensure Phase 1 SiC Integration code at src/ceramic_discovery/dft/jarvis_client.py exists. "
        f"Original error: {e}"
    )

try:
    from ceramic_discovery.utils.data_conversion import safe_float
except ImportError as e:
    raise ImportError(
        "Cannot import safe_float from Phase 1. "
        "Ensure Phase 1 utilities at src/ceramic_discovery/utils/data_conversion.py exist. "
        f"Original error: {e}"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class JarvisCollector(BaseCollector):
    """
    Wrapper around existing JarvisClient for carbide collection.
    
    Uses composition to leverage existing Phase 1 JARVIS integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get JARVIS file path from config
        jarvis_file_path = config.get('jarvis_file_path')
        if not jarvis_file_path:
            raise ValueError("JARVIS file path required in config['jarvis_file_path']")
        
        # Create instance of existing client (composition)
        self.client = JarvisClient(jarvis_file_path=jarvis_file_path)
        
        self.carbide_metals = set(config.get('carbide_metals', []))
        self.data_source = config.get('data_source', 'dft_3d')
    
    def collect(self) -> pd.DataFrame:
        """
        Collect carbide materials using existing JarvisClient.
        
        Returns:
            DataFrame with carbide materials
        """
        logger.info("Collecting JARVIS carbide data")
        
        try:
            # Check if existing client has carbide filtering (from Phase 1 Task 2)
            if hasattr(self.client, 'load_carbides'):
                # Use existing method if available
                logger.info("Using existing JarvisClient.load_carbides method")
                carbides = self.client.load_carbides(metal_elements=self.carbide_metals)
            else:
                # Add filtering logic here (without modifying client)
                # This is new Phase 2 functionality
                logger.info("Using fallback carbide filtering logic")
                # Get all data from client
                all_materials = self.client.data
                carbides = self._filter_carbides(all_materials)
            
            # Convert to DataFrame
            materials_list = []
            for carbide in carbides:
                # Handle both MaterialData objects and dict entries
                if hasattr(carbide, 'material_id'):
                    # MaterialData object from load_carbides
                    material_dict = {
                        'source': 'JARVIS',
                        'jid': carbide.material_id,
                        'formula': carbide.formula,
                        'formation_energy_jarvis': safe_float(carbide.formation_energy),
                        'energy_above_hull_jarvis': safe_float(carbide.energy_above_hull),
                        'band_gap_jarvis': safe_float(carbide.band_gap),
                        'density_jarvis': safe_float(carbide.density),
                        'bulk_modulus_jarvis': safe_float(carbide.properties.get('bulk_modulus')),
                        'shear_modulus_jarvis': safe_float(carbide.properties.get('shear_modulus')),
                    }
                else:
                    # Dict entry from fallback filtering
                    material_dict = {
                        'source': 'JARVIS',
                        'jid': carbide.get('jid'),
                        'formula': carbide.get('formula'),
                        'formation_energy_jarvis': safe_float(carbide.get('formation_energy_peratom')),
                        'energy_above_hull_jarvis': safe_float(carbide.get('ehull')),
                        'band_gap_jarvis': safe_float(carbide.get('optb88vdw_bandgap') or carbide.get('mbj_bandgap') or carbide.get('bandgap')),
                        'density_jarvis': safe_float(carbide.get('density')),
                        'bulk_modulus_jarvis': safe_float(carbide.get('bulk_modulus_kv')),
                        'shear_modulus_jarvis': safe_float(carbide.get('shear_modulus_gv')),
                    }
                materials_list.append(material_dict)
            
            df = pd.DataFrame(materials_list)
            logger.info(f"Collected {len(df)} carbides from JARVIS")
            
            return df
            
        except Exception as e:
            logger.error(f"JARVIS collection failed: {e}")
            return pd.DataFrame()  # Return empty on failure
    
    def _filter_carbides(self, materials: list) -> list:
        """
        Filter carbides from materials list.
        
        This is NEW Phase 2 logic added in collector, NOT in existing client.
        """
        carbides = []
        for mat in materials:
            # Get formula
            formula = mat.get('formula', '')
            if not formula:
                # Try to parse from jid
                jid = mat.get('jid', '')
                if jid:
                    formula = self.client.parse_formula_from_jid(jid)
            
            if not formula:
                continue
            
            # Check if it's a carbide
            if self._is_carbide(formula):
                carbides.append(mat)
        
        return carbides
    
    def _is_carbide(self, formula: str) -> bool:
        """
        Check if formula represents a carbide material.
        
        Args:
            formula: Chemical formula
            
        Returns:
            True if formula is a carbide with required metals
        """
        # Parse formula to extract elements
        import re
        elements = set(re.findall(r'[A-Z][a-z]?', formula))
        
        # Must contain carbon
        if 'C' not in elements:
            return False
        
        # If no metal filter specified, any carbide is valid
        if not self.carbide_metals:
            return True
        
        # Check if any specified metal is present
        for metal in self.carbide_metals:
            if metal in elements:
                return True
        
        return False
    
    def get_source_name(self) -> str:
        return "JARVIS"
    
    def validate_config(self) -> bool:
        required = ['jarvis_file_path', 'carbide_metals']
        return all(k in self.config for k in required)
