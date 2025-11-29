"""
Phase 2 AFLOW Collector

NEW FUNCTIONALITY: No Phase 1 dependencies.
This collector adds thermal property data from AFLOW.

AFLOW (Automatic FLOW for Materials Discovery) provides:
- Thermal conductivity at 300K
- Thermal expansion at 300K
- Debye temperature
- Heat capacity at 300K

These properties are critical for armor applications where thermal
management affects performance.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

try:
    import aflow
except ImportError:
    raise ImportError(
        "aflow library not installed. Install with: pip install aflow>=0.0.10"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class AFLOWCollector(BaseCollector):
    """
    AFLOW data collector for thermal properties.
    
    This is completely new Phase 2 functionality with no Phase 1 dependencies.
    Collects thermal property data for ceramic materials from AFLOW database.
    
    Configuration keys:
        ceramics: Dict mapping ceramic names to element lists
        max_per_system: Maximum materials to collect per ceramic system
        exclude_ldau: Whether to exclude LDAU calculations (default: True)
    
    Returns DataFrame with columns:
        - source: 'AFLOW'
        - formula: Chemical formula
        - ceramic_type: Ceramic system name (e.g., 'SiC', 'B4C')
        - density_aflow: Density (g/cm³)
        - formation_energy_aflow: Formation energy per atom (eV/atom)
        - band_gap_aflow: Band gap (eV)
        - thermal_conductivity_300K: Thermal conductivity at 300K (W/m·K)
        - thermal_expansion_300K: Thermal expansion at 300K (1/K)
        - debye_temperature: Debye temperature (K)
        - heat_capacity_Cp_300K: Heat capacity at 300K (J/mol·K)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AFLOW collector.
        
        Args:
            config: Configuration dictionary with keys:
                - ceramics: Dict[str, List[str]] mapping names to elements
                - max_per_system: int (optional, default 500)
                - exclude_ldau: bool (optional, default True)
        """
        super().__init__(config)
        
        self.ceramics = config.get('ceramics', {
            'SiC': ['Si', 'C'],
            'B4C': ['B', 'C'],
            'WC': ['W', 'C'],
            'TiC': ['Ti', 'C'],
            'Al2O3': ['Al', 'O']
        })
        self.max_per_system = config.get('max_per_system', 500)
        self.exclude_ldau = config.get('exclude_ldau', True)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect materials with thermal properties from AFLOW.
        
        Returns:
            DataFrame with thermal property data
        """
        all_materials = []
        
        for name, elements in self.ceramics.items():
            logger.info(f"Querying AFLOW for {name} ({','.join(elements)})")
            
            try:
                # Build AFLOW query
                query = aflow.search(
                    species=','.join(elements)
                )
                
                # Exclude LDAU calculations if configured
                if self.exclude_ldau:
                    query = query.exclude(['*:LDAU*'])
                
                # Select properties
                query = query.select([
                    aflow.K.compound,
                    aflow.K.density,
                    aflow.K.enthalpy_formation_atom,
                    aflow.K.Egap,
                    # Thermal properties (AFLOW specialty)
                    aflow.K.agl_thermal_conductivity_300K,
                    aflow.K.agl_thermal_expansion_300K,
                    aflow.K.agl_debye,
                    aflow.K.agl_heat_capacity_Cp_300K,
                ])
                
                # Limit results
                query = query.limit(self.max_per_system)
                
                # Execute query
                entries = list(query)
                logger.info(f"  Found {len(entries)} materials for {name}")
                
                # Extract properties from each entry
                for entry in entries:
                    material = {
                        'source': 'AFLOW',
                        'formula': getattr(entry, 'compound', None),
                        'ceramic_type': name,
                        'density_aflow': getattr(entry, 'density', None),
                        'formation_energy_aflow': getattr(entry, 'enthalpy_formation_atom', None),
                        'band_gap_aflow': getattr(entry, 'Egap', None),
                        # Thermal properties (KEY DATA)
                        'thermal_conductivity_300K': getattr(entry, 'agl_thermal_conductivity_300K', None),
                        'thermal_expansion_300K': getattr(entry, 'agl_thermal_expansion_300K', None),
                        'debye_temperature': getattr(entry, 'agl_debye', None),
                        'heat_capacity_Cp_300K': getattr(entry, 'agl_heat_capacity_Cp_300K', None),
                    }
                    all_materials.append(material)
                
            except aflow.AFLOWAPIError as e:
                logger.error(f"AFLOW API error for {name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error collecting {name}: {e}")
                continue
        
        df = pd.DataFrame(all_materials)
        logger.info(f"Collected {len(df)} total materials from AFLOW")
        
        return df
    
    def get_source_name(self) -> str:
        """Return source identifier."""
        return "AFLOW"
    
    def validate_config(self) -> bool:
        """
        Validate configuration has required keys.
        
        Returns:
            True if valid, False otherwise
        """
        # ceramics is required
        if 'ceramics' not in self.config:
            logger.error("Configuration missing 'ceramics' key")
            return False
        
        # Validate ceramics is a dict
        if not isinstance(self.config['ceramics'], dict):
            logger.error("'ceramics' must be a dictionary")
            return False
        
        # Validate each ceramic has a list of elements
        for name, elements in self.config['ceramics'].items():
            if not isinstance(elements, list):
                logger.error(f"Elements for {name} must be a list")
                return False
            if len(elements) == 0:
                logger.error(f"Elements list for {name} is empty")
                return False
        
        return True
