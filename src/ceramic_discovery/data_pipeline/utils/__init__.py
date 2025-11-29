"""
Phase 2 Utility Functions

Helper utilities for configuration, rate limiting, and property extraction.

This module provides supporting utilities for Phase 2 data collection:

Components:
    ConfigLoader: YAML configuration loading with environment variable substitution
        - Loads data_pipeline_config.yaml
        - Validates required sections
        - Substitutes ${ENV_VAR} placeholders
    
    RateLimiter: API rate limiting decorator
        - Enforces delays between API calls
        - Prevents API throttling/blocking
        - Used by MatWeb and Semantic Scholar collectors
    
    PropertyExtractor: Text mining for experimental properties
        - Extracts V50 (ballistic velocity) from literature
        - Extracts hardness (GPa) with HV conversion
        - Extracts fracture toughness K_IC
        - Identifies ceramic material types
        - Used by Semantic Scholar and MatWeb collectors

Example:
    from ceramic_discovery.data_pipeline.utils import ConfigLoader
    
    config = ConfigLoader.load('config/data_pipeline_config.yaml')
    mp_config = config['materials_project']
    
    from ceramic_discovery.data_pipeline.utils import RateLimiter
    
    @RateLimiter(delay_seconds=2.0)
    def fetch_data():
        # API call here
        pass
    
    from ceramic_discovery.data_pipeline.utils import PropertyExtractor
    
    text = "Silicon carbide with V50 = 850 m/s and hardness 26.5 GPa"
    props = PropertyExtractor.extract_all(text)
    # {'v50': 850.0, 'hardness': 26.5, 'material': 'SiC'}
"""

from .config_loader import ConfigLoader
from .rate_limiter import RateLimiter
from .property_extractor import PropertyExtractor

__all__ = [
    'ConfigLoader',
    'RateLimiter',
    'PropertyExtractor',
]
