"""
Phase 2 Materials Project Collector

CRITICAL: This wraps the existing MaterialsProjectClient from Phase 1.
Do NOT modify src/ceramic_discovery/dft/materials_project_client.py
"""

import pandas as pd
from typing import Dict, Any, List
import logging

# Import existing Phase 1 client (DO NOT MODIFY THE IMPORTED CLASS)
try:
    from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient
except ImportError as e:
    raise ImportError(
        "Cannot import MaterialsProjectClient from Phase 1. "
        "Ensure Phase 1 code at src/ceramic_discovery/dft/materials_project_client.py exists. "
        f"Original error: {e}"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MaterialsProjectCollector(BaseCollector):
    """
    Wrapper around existing MaterialsProjectClient for batch data collection.
    
    Uses composition (HAS-A) not inheritance (IS-A) to avoid modifying Phase 1 code.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Create instance of existing client (composition pattern)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Materials Project API key required in config['api_key']")
        
        # Instantiate existing Phase 1 client
        self.client = MaterialsProjectClient(api_key=api_key)
        
        self.target_systems = config.get('target_systems', [])
        self.batch_size = config.get('batch_size', 1000)
        self.max_materials = config.get('max_materials', 5000)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect materials from Materials Project using existing client.
        
        Returns:
            DataFrame with columns matching Phase 1 MaterialData format
        """
        all_materials = []
        
        for system in self.target_systems:
            logger.info(f"Collecting Materials Project data for {system}")
            
            try:
                # Parse system (e.g., "Si-C" -> elements ["Si", "C"])
                elements = system.split('-') if '-' in system else None
                
                # Call existing client method (DO NOT MODIFY)
                if elements:
                    materials = self.client.search_materials(
                        elements=elements,
                        limit=min(self.batch_size, self.max_materials - len(all_materials))
                    )
                else:
                    materials = self.client.search_materials(
                        formula=system,
                        limit=min(self.batch_size, self.max_materials - len(all_materials))
                    )
                
                # Process results into DataFrame
                for mat in materials:
                    material_dict = {
                        'source': 'MaterialsProject',
                        'material_id': mat.material_id,
                        'formula': mat.formula,
                        'formation_energy_mp': mat.formation_energy,
                        'energy_above_hull_mp': mat.energy_above_hull,
                        'band_gap_mp': mat.band_gap,
                        'density_mp': mat.density,
                        # Extract additional properties from properties dict
                        'bulk_modulus_mp': mat.properties.get('bulk_modulus'),
                        'shear_modulus_mp': mat.properties.get('shear_modulus'),
                        'is_stable_mp': mat.properties.get('is_stable'),
                    }
                    all_materials.append(material_dict)
                    
                    if len(all_materials) >= self.max_materials:
                        logger.info(f"Reached max_materials limit of {self.max_materials}")
                        break
                
                if len(all_materials) >= self.max_materials:
                    break
                
            except Exception as e:
                logger.error(f"Failed to collect {system}: {e}")
                continue
        
        df = pd.DataFrame(all_materials)
        logger.info(f"Collected {len(df)} materials from Materials Project")
        
        return df
    
    def get_source_name(self) -> str:
        return "MaterialsProject"
    
    def validate_config(self) -> bool:
        required = ['api_key', 'target_systems']
        return all(k in self.config for k in required)
