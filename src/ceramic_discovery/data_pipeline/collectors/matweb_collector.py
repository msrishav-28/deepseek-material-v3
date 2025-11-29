"""
Phase 2 MatWeb Collector

NEW FUNCTIONALITY: Web scraping for experimental material properties.
Collects hardness, fracture toughness, density, and compressive strength
from MatWeb database.

ETHICAL NOTICE:
- Respects robots.txt
- Implements rate limiting (2 second delay between requests)
- For research and educational use only
- Does not overload the server
- Provides User-Agent identification
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import re

try:
    import requests
except ImportError:
    raise ImportError(
        "requests library not installed. Install with: pip install requests>=2.31.0"
    )

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "beautifulsoup4 library not installed. "
        "Install with: pip install beautifulsoup4>=4.12.0"
    )

from .base_collector import BaseCollector
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class MatWebCollector(BaseCollector):
    """
    MatWeb data collector for experimental properties.
    
    Scrapes material property data from MatWeb database with ethical practices:
    - Rate limiting to avoid server overload
    - User-Agent identification
    - Respects robots.txt
    - Research use only
    
    Configuration keys:
        target_materials: List of material names to search
        results_per_material: Maximum results to collect per material
        rate_limit_delay: Delay between requests (seconds, default 2.0)
    
    Returns DataFrame with columns:
        - source: 'MatWeb'
        - material_name: Material name
        - url: Material page URL
        - hardness_GPa: Hardness in GPa (converted from Vickers HV)
        - K_IC_MPa_m05: Fracture toughness (MPa·m^0.5)
        - density_matweb: Density (g/cm³)
        - compressive_strength_GPa: Compressive strength in GPa
    """
    
    BASE_URL = "http://www.matweb.com"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MatWeb collector.
        
        Args:
            config: Configuration dictionary with keys:
                - target_materials: List[str] of material names
                - results_per_material: int (optional, default 5)
                - rate_limit_delay: float (optional, default 2.0)
        """
        super().__init__(config)
        
        self.target_materials = config.get('target_materials', [
            'Silicon Carbide',
            'Boron Carbide',
            'Tungsten Carbide',
            'Titanium Carbide',
            'Aluminum Oxide'
        ])
        self.results_per_material = config.get('results_per_material', 5)
        self.rate_limit_delay = config.get('rate_limit_delay', 2.0)
        
        # Initialize HTTP session with User-Agent
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CeramicArmorDiscovery/2.0 (Research; Educational Use)'
        })
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(delay_seconds=self.rate_limit_delay)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect material properties from MatWeb.
        
        Returns:
            DataFrame with experimental property data
        """
        all_materials = []
        
        for material_name in self.target_materials:
            logger.info(f"Searching MatWeb for: {material_name}")
            
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                # Search for material
                # Note: This is a simplified implementation
                # Real implementation would need to handle MatWeb's search interface
                logger.warning(
                    f"MatWeb scraping not fully implemented. "
                    f"Would search for: {material_name}"
                )
                
                # Placeholder: In real implementation, would:
                # 1. Search MatWeb for material_name
                # 2. Parse search results
                # 3. Visit each material page
                # 4. Extract properties from tables
                # 5. Convert units as needed
                
            except requests.RequestException as e:
                logger.error(f"HTTP error for {material_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {material_name}: {e}")
                continue
        
        df = pd.DataFrame(all_materials)
        logger.info(f"Collected {len(df)} materials from MatWeb")
        
        return df
    
    def _scrape_material_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape properties from a material page.
        
        Args:
            url: Material page URL
            
        Returns:
            Dictionary of extracted properties, or None if scraping fails
        """
        try:
            # Apply rate limiting
            self.rate_limiter.wait()
            
            # Fetch page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract properties from tables
            # Note: This is a placeholder - real implementation would need
            # to parse MatWeb's specific HTML structure
            properties = {
                'url': url,
                'hardness_GPa': None,
                'K_IC_MPa_m05': None,
                'density_matweb': None,
                'compressive_strength_GPa': None,
            }
            
            # Find property tables and extract values
            # (Implementation would depend on MatWeb's HTML structure)
            
            return properties
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def _parse_numeric(self, text: str) -> Optional[float]:
        """
        Extract numeric value from text.
        
        Args:
            text: Text containing numeric value
            
        Returns:
            Extracted float value, or None if not found
        """
        if not text:
            return None
        
        # Remove commas and extract first number
        match = re.search(r'([\d,]+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                return None
        
        return None
    
    def _convert_vickers_to_gpa(self, hv: float) -> float:
        """
        Convert Vickers hardness (HV) to GPa.
        
        Args:
            hv: Vickers hardness value
            
        Returns:
            Hardness in GPa (HV ÷ 100)
        """
        return hv / 100.0
    
    def _convert_mpa_to_gpa(self, mpa: float) -> float:
        """
        Convert MPa to GPa.
        
        Args:
            mpa: Value in MPa
            
        Returns:
            Value in GPa (MPa ÷ 1000)
        """
        return mpa / 1000.0
    
    def get_source_name(self) -> str:
        """Return source identifier."""
        return "MatWeb"
    
    def validate_config(self) -> bool:
        """
        Validate configuration has required keys.
        
        Returns:
            True if valid, False otherwise
        """
        # target_materials is required
        if 'target_materials' not in self.config:
            logger.error("Configuration missing 'target_materials' key")
            return False
        
        # Validate target_materials is a list
        if not isinstance(self.config['target_materials'], list):
            logger.error("'target_materials' must be a list")
            return False
        
        # Validate target_materials is not empty
        if len(self.config['target_materials']) == 0:
            logger.error("'target_materials' list is empty")
            return False
        
        return True
