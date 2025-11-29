"""
Phase 2: Isolated Multi-Source Data Collection Pipeline

This module provides automated data collection from 6 sources:
- Materials Project (DFT properties)
- JARVIS-DFT (carbide materials)
- AFLOW (thermal properties)
- Semantic Scholar (literature mining)
- MatWeb (experimental data)
- NIST (baseline references)

IMPORTANT: This module is ISOLATED from Phase 1.
- Imports Phase 1 modules (read-only)
- Phase 1 NEVER imports this module
- Creates master_dataset.csv for Phase 1 to consume

Architecture:
    data_pipeline/
    ├── collectors/      # 6 source collectors + base interface
    ├── integration/     # Multi-source merging
    ├── pipelines/       # Orchestration
    └── utils/           # Helper utilities

Usage:
    from ceramic_discovery.data_pipeline import CollectionOrchestrator
    from ceramic_discovery.data_pipeline.utils import ConfigLoader
    
    config = ConfigLoader.load('config/data_pipeline_config.yaml')
    orchestrator = CollectionOrchestrator(config)
    master_path = orchestrator.run_all()
"""

__version__ = "2.0.0"  # Phase 2 version

# Export main classes for convenience
from .pipelines.collection_orchestrator import CollectionOrchestrator
from .pipelines.integration_pipeline import IntegrationPipeline
from .utils.config_loader import ConfigLoader

# Export collectors
from .collectors.base_collector import BaseCollector
from .collectors.materials_project_collector import MaterialsProjectCollector
from .collectors.jarvis_collector import JarvisCollector
from .collectors.aflow_collector import AFLOWCollector
from .collectors.semantic_scholar_collector import SemanticScholarCollector
from .collectors.matweb_collector import MatWebCollector
from .collectors.nist_baseline_collector import NISTBaselineCollector

__all__ = [
    # Main orchestration
    'CollectionOrchestrator',
    'IntegrationPipeline',
    'ConfigLoader',
    
    # Collectors
    'BaseCollector',
    'MaterialsProjectCollector',
    'JarvisCollector',
    'AFLOWCollector',
    'SemanticScholarCollector',
    'MatWebCollector',
    'NISTBaselineCollector',
]
