"""
Phase 2 Integration Module

Multi-source data integration utilities.

This module provides tools for:
- Formula matching and fuzzy comparison
- Multi-source data integration and deduplication
- Conflict resolution across data sources
- Quality validation

Components:
    FormulaMatcher: Fuzzy formula matching for deduplication
    IntegrationPipeline: Main integration orchestrator (imported from pipelines)

Example:
    from ceramic_discovery.data_pipeline.integration import FormulaMatcher
    
    matcher = FormulaMatcher()
    similarity = matcher.calculate_similarity("SiC", "Si1C1")
    best_match, score = matcher.find_best_match("SiC", ["Si1C1", "B4C", "WC"])
"""

from .formula_matcher import FormulaMatcher

__all__ = [
    'FormulaMatcher',
]
