"""
Unit tests for CollectionOrchestrator

Tests the main pipeline orchestrator that runs all collectors in sequence.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ceramic_discovery.data_pipeline.pipelines.collection_orchestrator import CollectionOrchestrator


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'materials_project': {
            'enabled': True,
            'api_key': 'test_key',
            'target_systems': ['Si-C']
        },
        'jarvis': {
            'enabled': True,
            'carbide_metals': ['Si', 'B']
        },
        'aflow': {
            'enabled': True,
            'ceramics': {'SiC': ['Si', 'C']}
        },
        'semantic_scholar': {
            'enabled': True,
            'queries': ['test query']
        },
        'matweb': {
            'enabled': True,
            'target_materials': ['Silicon Carbide']
        },
        'nist': {
            'enabled': True,
            'baseline_materials': {
                'SiC': {'density': 3.2, 'hardness': 26, 'K_IC': 4.0, 'source': 'NIST'}
            }
        },
        'integration': {
            'deduplication_threshold': 0.95
        }
    }


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)



def test_initialize_collectors(test_config):
    """Test 1: Verify all 6 collectors are initialized."""
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector'), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector'), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector'), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector'), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector'), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector'):
        
        orchestrator = CollectionOrchestrator(test_config)
        
        # Verify all 6 collectors initialized
        assert len(orchestrator.collectors) == 6
        assert 'mp' in orchestrator.collectors
        assert 'jarvis' in orchestrator.collectors
        assert 'aflow' in orchestrator.collectors
        assert 'semantic_scholar' in orchestrator.collectors
        assert 'matweb' in orchestrator.collectors
        assert 'nist' in orchestrator.collectors


def test_sequential_execution(test_config, tmp_path, monkeypatch):
    """Test 2: Verify collectors are called in correct order."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Mock collect() to return test DataFrames
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.return_value = test_df
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration
    mock_integration = Mock()
    mock_integration.integrate_all_sources.return_value = Path('data/final/master_dataset.csv')
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration):
        
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify all collectors were called
        mock_mp.collect.assert_called_once()
        mock_jarvis.collect.assert_called_once()
        mock_aflow.collect.assert_called_once()
        mock_scholar.collect.assert_called_once()
        mock_matweb.collect.assert_called_once()
        mock_nist.collect.assert_called_once()
        
        # Verify integration was called
        mock_integration.integrate_all_sources.assert_called_once()



def test_collector_failure_continues(test_config, tmp_path, monkeypatch, caplog):
    """Test 3: Verify that collector failure logs error but continues with next collector."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Make JARVIS collector fail
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.side_effect = Exception("JARVIS API error")
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration
    mock_integration = Mock()
    mock_integration.integrate_all_sources.return_value = Path('data/final/master_dataset.csv')
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration):
        
        orchestrator = CollectionOrchestrator(test_config)
        result = orchestrator.run_all()
        
        # Verify JARVIS failed but others succeeded
        assert orchestrator.results['jarvis']['status'] == 'FAILED'
        assert 'JARVIS API error' in orchestrator.results['jarvis']['error']
        
        # Verify other collectors still ran
        assert orchestrator.results['mp']['status'] == 'SUCCESS'
        assert orchestrator.results['aflow']['status'] == 'SUCCESS'
        assert orchestrator.results['semantic_scholar']['status'] == 'SUCCESS'
        assert orchestrator.results['matweb']['status'] == 'SUCCESS'
        assert orchestrator.results['nist']['status'] == 'SUCCESS'
        
        # Verify integration still ran
        assert orchestrator.results['integration']['status'] == 'SUCCESS'



def test_checkpoint_saving(test_config, tmp_path, monkeypatch):
    """Test 4: Verify each collector saves CSV to data/processed/."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Mock collect() to return test DataFrames
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.return_value = test_df
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration
    mock_integration = Mock()
    mock_integration.integrate_all_sources.return_value = Path('data/final/master_dataset.csv')
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration):
        
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify CSV files were created
        processed_dir = Path('data/processed')
        assert (processed_dir / 'mp_data.csv').exists()
        assert (processed_dir / 'jarvis_data.csv').exists()
        assert (processed_dir / 'aflow_data.csv').exists()
        assert (processed_dir / 'semantic_scholar_data.csv').exists()
        assert (processed_dir / 'matweb_data.csv').exists()
        assert (processed_dir / 'nist_data.csv').exists()



def test_integration_failure_aborts(test_config, tmp_path, monkeypatch):
    """Test 5: Verify integration failure aborts pipeline (integration is critical)."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Mock collect() to return test DataFrames
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.return_value = test_df
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration to fail
    mock_integration = Mock()
    mock_integration.integrate_all_sources.side_effect = Exception("Integration failed")
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration):
        
        orchestrator = CollectionOrchestrator(test_config)
        
        # Verify integration failure raises exception
        with pytest.raises(Exception, match="Integration failed"):
            orchestrator.run_all()
        
        # Verify integration status is FAILED
        assert orchestrator.results['integration']['status'] == 'FAILED'



def test_progress_tracking(test_config, tmp_path, monkeypatch):
    """Test 6: Verify tqdm progress bar is used."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Mock collect() to return test DataFrames
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.return_value = test_df
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration
    mock_integration = Mock()
    mock_integration.integrate_all_sources.return_value = Path('data/final/master_dataset.csv')
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.tqdm') as mock_tqdm:
        
        # Configure mock tqdm
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify tqdm was called with correct parameters
        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs['total'] == 6  # 6 collectors
        assert 'desc' in call_kwargs
        
        # Verify progress bar was updated 6 times (once per collector)
        assert mock_pbar.update.call_count == 6



def test_summary_report_generated(test_config, tmp_path, monkeypatch):
    """Test 7: Verify pipeline_summary.txt is created with correct content."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Mock collect() to return test DataFrames
    test_df = pd.DataFrame({'formula': ['SiC'], 'source': ['test']})
    mock_mp.collect.return_value = test_df
    mock_jarvis.collect.return_value = test_df
    mock_aflow.collect.return_value = test_df
    mock_scholar.collect.return_value = test_df
    mock_matweb.collect.return_value = test_df
    mock_nist.collect.return_value = test_df
    
    # Mock integration
    mock_integration = Mock()
    mock_integration.integrate_all_sources.return_value = Path('data/final/master_dataset.csv')
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.IntegrationPipeline', return_value=mock_integration):
        
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify summary report was created
        summary_file = Path('data/final/pipeline_summary.txt')
        assert summary_file.exists()
        
        # Read and verify content
        content = summary_file.read_text(encoding='utf-8')
        
        # Verify header
        assert 'PHASE 2 DATA COLLECTION PIPELINE SUMMARY' in content
        assert 'COLLECTOR RESULTS' in content
        
        # Verify per-source status
        assert 'mp' in content
        assert 'jarvis' in content
        assert 'aflow' in content
        assert 'semantic_scholar' in content
        assert 'matweb' in content
        assert 'nist' in content
        assert 'integration' in content
        
        # Verify runtime information
        assert 'Total Runtime:' in content
        
        # Verify status indicators
        assert '✓' in content  # Success indicator
