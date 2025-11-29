"""
Unit tests for SemanticScholarCollector

Tests the literature mining collector for experimental properties.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector import SemanticScholarCollector


class TestSemanticScholarCollector:
    """Test suite for SemanticScholarCollector"""
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration for Semantic Scholar collector"""
        return {
            'queries': [
                'ballistic limit velocity ceramic armor',
                'silicon carbide armor'
            ],
            'papers_per_query': 10,
            'min_citations': 5,
            'rate_limit_delay': 0.1
        }
    
    @pytest.fixture
    def mock_paper_with_properties(self):
        """Create a mock paper with extractable properties"""
        return {
            'paperId': 'abc123',
            'title': 'Silicon Carbide Armor Performance',
            'abstract': 'We tested SiC armor with V50 = 850 m/s, hardness of 26.5 GPa, and K_IC = 4.0 MPa·m^0.5',
            'year': 2020,
            'citationCount': 10,
            'url': 'https://example.com/paper'
        }
    
    @pytest.fixture
    def mock_paper_without_properties(self):
        """Create a mock paper without extractable properties"""
        return {
            'paperId': 'xyz789',
            'title': 'General Materials Science',
            'abstract': 'This paper discusses various materials.',
            'year': 2019,
            'citationCount': 15,
            'url': 'https://example.com/paper2'
        }
    
    def test_paper_search(self, valid_config, mock_paper_with_properties):
        """Test 1: Verify paper search is executed"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [mock_paper_with_properties]
                    mock_ss.return_value = mock_client
                    
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': 850.0,
                        'hardness': 26.5,
                        'K_IC': 4.0,
                        'material': 'SiC'
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Verify search_paper was called for each query
                    assert mock_client.search_paper.call_count == 2
                    
                    # Verify queries were executed
                    calls = mock_client.search_paper.call_args_list
                    assert calls[0][0][0] == 'ballistic limit velocity ceramic armor'
                    assert calls[1][0][0] == 'silicon carbide armor'
    
    def test_property_extraction_integration(self, valid_config, mock_paper_with_properties):
        """Test 2: Verify PropertyExtractor is called and properties extracted"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [mock_paper_with_properties]
                    mock_ss.return_value = mock_client
                    
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': 850.0,
                        'hardness': 26.5,
                        'K_IC': 4.0,
                        'material': 'SiC'
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Verify PropertyExtractor.extract_all was called
                    assert mock_extractor.extract_all.called
                    
                    # Verify extracted properties in DataFrame
                    assert len(df) > 0
                    assert 'v50_literature' in df.columns
                    assert 'hardness_literature' in df.columns
                    assert 'K_IC_literature' in df.columns
                    assert df.iloc[0]['v50_literature'] == 850.0
                    assert df.iloc[0]['hardness_literature'] == 26.5
                    assert df.iloc[0]['K_IC_literature'] == 4.0
    
    def test_citation_filtering(self, valid_config):
        """Test 3: Verify only papers >= min_citations are included"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Create papers with different citation counts
                    paper_low_citations = {
                        'paperId': 'low',
                        'title': 'SiC Test',
                        'abstract': 'V50 = 850 m/s',
                        'year': 2020,
                        'citationCount': 2,  # Below threshold
                        'url': 'https://example.com/low'
                    }
                    
                    paper_high_citations = {
                        'paperId': 'high',
                        'title': 'SiC Test',
                        'abstract': 'V50 = 900 m/s',
                        'year': 2020,
                        'citationCount': 10,  # Above threshold
                        'url': 'https://example.com/high'
                    }
                    
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [paper_low_citations, paper_high_citations]
                    mock_ss.return_value = mock_client
                    
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': 850.0,
                        'material': 'SiC'
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Only high citation paper should be included
                    # (2 queries * 1 paper each = 2 total)
                    assert len(df) == 2
                    assert all(df['citations'] >= 5)
    
    def test_provenance_tracking(self, valid_config, mock_paper_with_properties):
        """Test 4: Verify paper_id, title, year, citations, URL are tracked"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [mock_paper_with_properties]
                    mock_ss.return_value = mock_client
                    
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': 850.0,
                        'material': 'SiC'
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Verify provenance columns exist
                    required_columns = ['paper_id', 'title', 'year', 'citations', 'url']
                    for col in required_columns:
                        assert col in df.columns, f"Missing column: {col}"
                    
                    # Verify values
                    assert df.iloc[0]['paper_id'] == 'abc123'
                    assert df.iloc[0]['title'] == 'Silicon Carbide Armor Performance'
                    assert df.iloc[0]['year'] == 2020
                    assert df.iloc[0]['citations'] == 10
                    assert df.iloc[0]['url'] == 'https://example.com/paper'
    
    def test_rate_limiting(self, valid_config, mock_paper_with_properties):
        """Test 5: Verify rate limiting is applied between queries"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [mock_paper_with_properties]
                    mock_ss.return_value = mock_client
                    
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': 850.0,
                        'material': 'SiC'
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Verify rate limiter wait was called (once per query)
                    assert mock_limiter.wait.call_count == 2
    
    def test_skip_papers_without_properties(self, valid_config, mock_paper_without_properties):
        """Test 6: Verify papers without extractable properties are skipped"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar') as mock_ss:
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor') as mock_pe:
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter') as mock_rl:
                    # Setup mocks
                    mock_client = Mock()
                    mock_client.search_paper.return_value = [mock_paper_without_properties]
                    mock_ss.return_value = mock_client
                    
                    # Extractor returns no properties
                    mock_extractor = Mock()
                    mock_extractor.extract_all.return_value = {
                        'v50': None,
                        'hardness': None,
                        'K_IC': None,
                        'material': None
                    }
                    mock_pe.return_value = mock_extractor
                    
                    mock_limiter = Mock()
                    mock_limiter.wait.return_value = None
                    mock_rl.return_value = mock_limiter
                    
                    collector = SemanticScholarCollector(valid_config)
                    df = collector.collect()
                    
                    # Should return empty DataFrame (no papers with properties)
                    assert len(df) == 0
    
    def test_validate_config_valid(self, valid_config):
        """Test that valid configuration passes validation"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar'):
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor'):
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter'):
                    collector = SemanticScholarCollector(valid_config)
                    assert collector.validate_config() is True
    
    def test_validate_config_missing_queries(self):
        """Test that missing queries key fails validation"""
        config = {'papers_per_query': 10}
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar'):
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor'):
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter'):
                    with pytest.raises(ValueError):
                        SemanticScholarCollector(config)
    
    def test_validate_config_invalid_queries_type(self):
        """Test that invalid queries type fails validation"""
        config = {'queries': 'not a list'}
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar'):
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor'):
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter'):
                    with pytest.raises(ValueError):
                        SemanticScholarCollector(config)
    
    def test_validate_config_empty_queries(self):
        """Test that empty queries list fails validation"""
        config = {'queries': []}
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar'):
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor'):
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter'):
                    with pytest.raises(ValueError):
                        SemanticScholarCollector(config)
    
    def test_get_source_name(self, valid_config):
        """Test that source name is correct"""
        with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.SemanticScholar'):
            with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.PropertyExtractor'):
                with patch('ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector.RateLimiter'):
                    collector = SemanticScholarCollector(valid_config)
                    assert collector.get_source_name() == "Literature"
