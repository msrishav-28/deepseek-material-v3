"""
Phase 2 Utility: Formula Fuzzy Matching

NEW FUNCTIONALITY: No Phase 1 dependencies.
Uses SequenceMatcher for formula similarity comparison.
"""

from difflib import SequenceMatcher
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FormulaMatcher:
    """
    Fuzzy matching for chemical formulas.
    
    This is new Phase 2 utility for multi-source deduplication.
    Uses SequenceMatcher to calculate similarity between formulas.
    """
    
    @staticmethod
    def similarity(formula1: str, formula2: str) -> float:
        """
        Calculate similarity between two formulas (0-1).
        
        Args:
            formula1: First chemical formula
            formula2: Second chemical formula
            
        Returns:
            Similarity score from 0.0 (completely different) to 1.0 (identical)
            
        Examples:
            >>> FormulaMatcher.similarity("SiC", "SiC")
            1.0
            >>> FormulaMatcher.similarity("SiC", "B4C")
            0.333...
            >>> FormulaMatcher.similarity("SiC", "Si1C1")
            0.666...
        """
        return SequenceMatcher(None, formula1, formula2).ratio()
    
    @classmethod
    def find_best_match(
        cls, 
        formula: str, 
        candidates: List[str], 
        threshold: float = 0.95
    ) -> Tuple[Optional[str], float]:
        """
        Find best matching formula from candidates.
        
        Args:
            formula: Formula to match
            candidates: List of candidate formulas
            threshold: Minimum similarity score (default: 0.95)
        
        Returns:
            Tuple of (best_match, score) or (None, 0.0) if no match above threshold
            
        Examples:
            >>> FormulaMatcher.find_best_match("SiC", ["SiC", "B4C", "WC"])
            ("SiC", 1.0)
            >>> FormulaMatcher.find_best_match("Si1C1", ["SiC", "B4C"], threshold=0.6)
            ("SiC", 0.666...)
            >>> FormulaMatcher.find_best_match("SiC", ["B4C", "WC"], threshold=0.95)
            (None, 0.0)
        """
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = cls.similarity(formula, candidate)
            if score > best_score and score >= threshold:
                best_match = candidate
                best_score = score
        
        return best_match, best_score
