"""
Phase 2 Data Collectors

Each collector wraps a data source and implements BaseCollector interface.

Collector Types:
1. Wrapper Collectors: Wrap existing Phase 1 clients
   - MaterialsProjectCollector (wraps Phase 1 MaterialsProjectClient)
   - JarvisCollector (wraps Phase 1 JarvisClient)

2. New Collectors: Direct API/scraping integration
   - AFLOWCollector (new AFLOW integration)
   - SemanticScholarCollector (literature mining)
   - MatWebCollector (web scraping)
   - NISTBaselineCollector (embedded reference data)

Design Pattern: Composition
    Phase 2 Collector HAS-A Phase 1 Client
    (not IS-A, which would require inheritance)

Example:
    # Wrapper pattern (uses existing Phase 1 code)
    mp_collector = MaterialsProjectCollector(config)
    df = mp_collector.collect()  # Internally uses MaterialsProjectClient
    
    # New functionality pattern
    aflow_collector = AFLOWCollector(config)
    df = aflow_collector.collect()  # Direct AFLOW API integration
"""

from .base_collector import BaseCollector
from .materials_project_collector import MaterialsProjectCollector
from .jarvis_collector import JarvisCollector
from .aflow_collector import AFLOWCollector
from .semantic_scholar_collector import SemanticScholarCollector
from .matweb_collector import MatWebCollector
from .nist_baseline_collector import NISTBaselineCollector

__all__ = [
    'BaseCollector',
    'MaterialsProjectCollector',
    'JarvisCollector',
    'AFLOWCollector',
    'SemanticScholarCollector',
    'MatWebCollector',
    'NISTBaselineCollector',
]
