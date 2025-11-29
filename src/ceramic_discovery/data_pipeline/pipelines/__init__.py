"""
Phase 2 Pipeline Orchestration

Collection orchestrator and integration pipeline.

This module provides the main orchestration components for Phase 2:

Components:
    CollectionOrchestrator: Coordinates all 6 data collectors
        - Runs collectors in sequence or parallel
        - Handles errors and retries
        - Saves individual source CSVs to data/processed/
    
    IntegrationPipeline: Merges all sources into master dataset
        - Loads all source CSVs
        - Performs fuzzy formula matching
        - Deduplicates across sources
        - Resolves conflicts using source priority
        - Creates combined property columns
        - Saves master_dataset.csv to data/final/

Example:
    from ceramic_discovery.data_pipeline import CollectionOrchestrator
    from ceramic_discovery.data_pipeline.utils import ConfigLoader
    
    # Load configuration
    config = ConfigLoader.load('config/data_pipeline_config.yaml')
    
    # Run collection
    orchestrator = CollectionOrchestrator(config)
    orchestrator.run_all()
    
    # Run integration
    from ceramic_discovery.data_pipeline import IntegrationPipeline
    pipeline = IntegrationPipeline(config)
    master_path = pipeline.integrate()
"""

from .collection_orchestrator import CollectionOrchestrator
from .integration_pipeline import IntegrationPipeline

__all__ = [
    'CollectionOrchestrator',
    'IntegrationPipeline',
]
