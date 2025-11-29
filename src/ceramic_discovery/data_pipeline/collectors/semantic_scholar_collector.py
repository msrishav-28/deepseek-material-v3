"""
Phase 2 Semantic Scholar Collector

NEW FUNCTIONALITY: Literature mining for experimental properties.
Extracts ballistic and mechanical properties from research papers.

This collector searches academic literature for experimental data on
ceramic armor materials, extracting properties like V50, hardness, and
fracture toughness from paper abstracts and titles.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import time

try:
    from semanticscholar import SemanticScholar
except ImportError:
    raise ImportError(
        "semanticscholar library not installed. "
        "Install with: pip install semanticscholar>=0.4.0"
    )

from .base_collector import BaseCollector
from ..utils.property_extractor import PropertyExtractor
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class SemanticScholarCollector(BaseCollector):
    """
    Semantic Scholar literature collector for experimental properties.
    
    Searches academic papers for experimental data on ceramic armor materials.
    Extracts properties from paper titles and abstracts using text mining.
    
    Configuration keys:
        queries: List of search queries for ceramic armor papers
        papers_per_query: Maximum papers to retrieve per query
        min_citations: Minimum citation count filter
        rate_limit_delay: Delay between API calls (seconds)
    
    Returns DataFrame with columns:
        - source: 'Literature'
        - paper_id: Semantic Scholar paper ID
        - title: Paper title
        - year: Publication year
        - citations: Citation count
        - url: Paper URL
        - ceramic_type: Identified ceramic material (e.g., 'SiC', 'B4C')
        - v50_literature: Ballistic limit velocity (m/s)
        - hardness_literature: Hardness (GPa)
        - K_IC_literature: Fracture toughness (MPa·m^0.5)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Semantic Scholar collector.
        
        Args:
            config: Configuration dictionary with keys:
                - queries: List[str] of search queries
                - papers_per_query: int (optional, default 50)
                - min_citations: int (optional, default 5)
                - rate_limit_delay: float (optional, default 1.0)
        """
        super().__init__(config)
        
        self.queries = config.get('queries', [
            'ballistic limit velocity ceramic armor',
            'silicon carbide armor penetration',
            'boron carbide ballistic performance',
            'ceramic armor hardness fracture toughness'
        ])
        self.papers_per_query = config.get('papers_per_query', 50)
        self.min_citations = config.get('min_citations', 5)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
        
        # Initialize Semantic Scholar client
        self.client = SemanticScholar()
        
        # Initialize property extractor
        self.extractor = PropertyExtractor()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(delay_seconds=self.rate_limit_delay)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect experimental properties from literature.
        
        Returns:
            DataFrame with extracted properties from papers
        """
        all_papers = []
        
        for query in self.queries:
            logger.info(f"Searching papers for: '{query}'")
            
            try:
                # Apply rate limiting
                self.rate_limiter.wait()
                
                # Search papers
                results = self.client.search_paper(
                    query,
                    limit=self.papers_per_query,
                    fields=['paperId', 'title', 'abstract', 'year', 'citationCount', 'url']
                )
                
                papers_found = 0
                papers_with_data = 0
                
                for paper in results:
                    papers_found += 1
                    
                    # Filter by citation count
                    citation_count = paper.get('citationCount', 0) or 0
                    if citation_count < self.min_citations:
                        continue
                    
                    # Extract text for property mining
                    title = paper.get('title', '')
                    abstract = paper.get('abstract', '')
                    text = f"{title} {abstract}"
                    
                    # Extract properties
                    properties = self.extractor.extract_all(text)
                    
                    # Only save if we extracted useful data AND identified material
                    has_properties = (
                        properties.get('v50') is not None or
                        properties.get('hardness') is not None or
                        properties.get('K_IC') is not None
                    )
                    has_material = properties.get('material') is not None
                    
                    if has_properties and has_material:
                        paper_data = {
                            'source': 'Literature',
                            'paper_id': paper.get('paperId'),
                            'title': title,
                            'year': paper.get('year'),
                            'citations': citation_count,
                            'url': paper.get('url'),
                            'ceramic_type': properties.get('material'),
                            'v50_literature': properties.get('v50'),
                            'hardness_literature': properties.get('hardness'),
                            'K_IC_literature': properties.get('K_IC'),
                        }
                        all_papers.append(paper_data)
                        papers_with_data += 1
                
                logger.info(
                    f"  Found {papers_found} papers, "
                    f"{papers_with_data} with extractable properties"
                )
                
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
                continue
        
        df = pd.DataFrame(all_papers)
        logger.info(f"Collected {len(df)} papers with experimental data")
        
        return df
    
    def get_source_name(self) -> str:
        """Return source identifier."""
        return "Literature"
    
    def validate_config(self) -> bool:
        """
        Validate configuration has required keys.
        
        Returns:
            True if valid, False otherwise
        """
        # queries is required
        if 'queries' not in self.config:
            logger.error("Configuration missing 'queries' key")
            return False
        
        # Validate queries is a list
        if not isinstance(self.config['queries'], list):
            logger.error("'queries' must be a list")
            return False
        
        # Validate queries is not empty
        if len(self.config['queries']) == 0:
            logger.error("'queries' list is empty")
            return False
        
        return True
