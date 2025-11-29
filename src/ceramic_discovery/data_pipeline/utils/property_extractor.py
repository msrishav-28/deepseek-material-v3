"""
PropertyExtractor Utility

Extracts material properties from text using regex patterns.
Designed for mining literature and web sources for experimental data.

Extracts:
- V50 (ballistic limit velocity) in m/s
- Hardness in GPa (with HV to GPa conversion)
- Fracture toughness (K_IC) in MPa·m^0.5
- Material identification (SiC, B4C, WC, etc.)

Example:
    text = "Silicon carbide (SiC) has V50 = 850 m/s and hardness of 26.5 GPa"
    props = PropertyExtractor.extract_all(text)
    # {'v50': 850.0, 'hardness': 26.5, 'material': 'SiC'}
"""

import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PropertyExtractor:
    """
    Extract material properties from text using regex patterns.
    
    All methods are class methods for stateless operation.
    """
    
    # Regex patterns for V50 (ballistic limit velocity)
    V50_PATTERNS = [
        r'V[_\s]?50[:\s=]+(?:is\s+)?(\d+\.?\d*)\s*(m/s|ms)',  # Handles "V50 is 850 m/s"
        r'ballistic\s+limit[:\s]+(?:is\s+)?(\d+\.?\d*)\s*(m/s|ms)',
        r'penetration\s+velocity[:\s]+(\d+\.?\d*)\s*(m/s|ms)',
    ]
    
    # Regex patterns for hardness
    HARDNESS_PATTERNS = [
        r'hardness[:\s]+(?:of\s+)?(\d+\.?\d*)\s*GPa',  # Handles "hardness of 28 GPa"
        r'Vickers\s+hardness[:\s]+(\d+\.?\d*)\s*(HV|GPa)',
        r'HV[:\s]+(\d+\.?\d*)(?:\s|$)',  # HV followed by space or end
    ]
    
    # Regex patterns for fracture toughness
    KIC_PATTERNS = [
        r'fracture\s+toughness[:\s]+K[_\s]?IC[:\s]*=?\s*(\d+\.?\d*)\s*MPa',
        r'K[_\s]?IC[:\s]*=?\s*(\d+\.?\d*)\s*MPa',
        r'toughness[:\s]+(\d+\.?\d*)\s*MPa',
    ]
    
    # Material keywords for identification
    MATERIAL_KEYWORDS = {
        'SiC': ['sic', 'silicon carbide', 'silicon-carbide'],
        'B4C': ['b4c', 'b₄c', 'boron carbide', 'boron-carbide'],
        'WC': ['wc', 'tungsten carbide', 'tungsten-carbide'],
        'TiC': ['tic', 'titanium carbide', 'titanium-carbide'],
        'Al2O3': ['al2o3', 'al₂o₃', 'alumina', 'aluminum oxide', 'aluminium oxide'],
        'ZrO2': ['zro2', 'zirconia', 'zirconium oxide'],
        'TiB2': ['tib2', 'tib₂', 'titanium diboride'],
    }
    
    @classmethod
    def extract_v50(cls, text: str) -> Optional[float]:
        """
        Extract V50 (ballistic limit velocity) from text.
        
        Args:
            text: Text to search
            
        Returns:
            V50 value in m/s, or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for pattern in cls.V50_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    logger.debug(f"Extracted V50: {value} m/s")
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @classmethod
    def extract_hardness(cls, text: str) -> Optional[float]:
        """
        Extract hardness from text.
        
        Converts Vickers hardness (HV) to GPa using: GPa = HV / 100
        
        Args:
            text: Text to search
            
        Returns:
            Hardness in GPa, or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for pattern in cls.HARDNESS_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    
                    # Check if unit is HV (Vickers hardness) or if pattern is HV pattern
                    if 'HV' in pattern.upper() or (len(match.groups()) > 1 and match.group(2) and match.group(2).upper() == 'HV'):
                        # Convert HV to GPa
                        value = value / 100.0
                        logger.debug(f"Converted HV to GPa: {value} GPa")
                    
                    logger.debug(f"Extracted hardness: {value} GPa")
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @classmethod
    def extract_fracture_toughness(cls, text: str) -> Optional[float]:
        """
        Extract fracture toughness (K_IC) from text.
        
        Args:
            text: Text to search
            
        Returns:
            K_IC in MPa·m^0.5, or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for pattern in cls.KIC_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    logger.debug(f"Extracted K_IC: {value} MPa·m^0.5")
                    return value
                except (ValueError, IndexError):
                    continue
        
        return None
    
    @classmethod
    def identify_material(cls, text: str) -> Optional[str]:
        """
        Identify ceramic material from text.
        
        Args:
            text: Text to search
            
        Returns:
            Material formula (e.g., 'SiC', 'B4C'), or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check each material's keywords
        for material, keywords in cls.MATERIAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.debug(f"Identified material: {material}")
                    return material
        
        return None
    
    @classmethod
    def extract_all(cls, text: str) -> Dict[str, Any]:
        """
        Extract all properties from text at once.
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary with extracted properties:
            {
                'v50': float or None,
                'hardness': float or None,
                'K_IC': float or None,
                'material': str or None
            }
        """
        if not text:
            return {
                'v50': None,
                'hardness': None,
                'K_IC': None,
                'material': None
            }
        
        result = {
            'v50': cls.extract_v50(text),
            'hardness': cls.extract_hardness(text),
            'K_IC': cls.extract_fracture_toughness(text),
            'material': cls.identify_material(text)
        }
        
        # Log summary
        extracted_props = [k for k, v in result.items() if v is not None]
        if extracted_props:
            logger.debug(f"Extracted properties: {extracted_props}")
        
        return result
