"""
End-to-end integration tests for Phase 2 data collection pipeline.

Tests the complete pipeline from collection through integration to master dataset.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ceramic_discovery.data_pipeline.pipelines.collection_orchestrator import CollectionOrchestrator
from ceramic_discovery.data_pipeline.pipelines.integration_pipeline import IntegrationPipeline


@pytest.fixture
def test_config():
    """Create comprehensive test configuration."""
    return {
        'materials_project': {
            'enabled': True,
            'api_key': 'test_mp_key',
            'target_systems': ['Si-C', 'B-C'],
            'batch_size': 100,
            'max_materials': 500
        },
        'jarvis': {
            'enabled': True,
            'carbide_metals': ['Si', 'B', 'W', 'Ti'],
            'data_source': 'dft_3d'
        },
        'aflow': {
            'enabled': True,
            'ceramics': {
                'SiC': ['Si', 'C'],
                'B4C': ['B', 'C'],
                'WC': ['W', 'C']
            },
            'max_per_system': 200
        },
        'semantic_scholar': {
            'enabled': True,
            'queries': ['ballistic limit velocity ceramic armor', 'silicon carbide armor'],
            'papers_per_query': 20,
            'min_citations': 5
        },
        'matweb': {
            'enabled': True,
            'target_materials': ['Silicon Carbide', 'Boron Carbide'],
            'rate_limit_delay': 2,
            'results_per_material': 5
        },
        'nist': {
            'enabled': True,
            'baseline_materials': {
                'SiC': {
                    'formula': 'SiC',
                    'ceramic_type': 'SiC',
                    'density': 3.21,
                    'hardness': 26.0,
                    'K_IC': 4.0,
                    'source': 'NIST SRM'
                },
                'B4C': {
                    'formula': 'B4C',
                    'ceramic_type': 'B4C',
                    'density': 2.52,
                    'hardness': 30.0,
                    'K_IC': 3.5,
                    'source': 'NIST SRM'
                }
            }
        },
        'integration': {
            'deduplication_threshold': 0.95,
            'source_priority': ['Literature', 'MatWeb', 'NIST', 'MaterialsProject', 'JARVIS', 'AFLOW']
        }
    }


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Create temporary workspace with data directories."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    # Create directory structure
    (tmp_path / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'data' / 'final').mkdir(parents=True, exist_ok=True)
    
    yield tmp_path


@pytest.fixture
def mock_collector_data():
    """Create mock data for each collector."""
    return {
        'mp': pd.DataFrame({
            'source': ['MaterialsProject'] * 3,
            'material_id': ['mp-1', 'mp-2', 'mp-3'],
            'formula': ['SiC', 'B4C', 'WC'],
            'formation_energy_mp': [-0.5, -0.6, -0.7],
            'band_gap_mp': [2.3, 2.1, 0.0],
            'density_mp': [3.21, 2.52, 15.6]
        }),
        'jarvis': pd.DataFrame({
            'source': ['JARVIS'] * 3,
            'jid': ['jid-1', 'jid-2', 'jid-3'],
            'formula': ['SiC', 'B4C', 'TiC'],
            'formation_energy_jarvis': [-0.48, -0.58, -0.65],
            'bulk_modulus_jarvis': [220, 240, 260],
            'density_jarvis': [3.20, 2.51, 4.93]
        }),
        'aflow': pd.DataFrame({
            'source': ['AFLOW'] * 3,
            'formula': ['SiC', 'WC', 'TiC'],
            'ceramic_type': ['SiC', 'WC', 'TiC'],
            'density_aflow': [3.22, 15.63, 4.91],
            'thermal_conductivity_300K': [120, 110, 25],
            'thermal_expansion_300K': [4.5e-6, 5.2e-6, 7.4e-6],
            'debye_temperature': [1200, 400, 600]
        }),
        'semantic_scholar': pd.DataFrame({
            'source': ['Literature'] * 2,
            'paper_id': ['paper-1', 'paper-2'],
            'title': ['SiC Armor Study', 'B4C Ballistic Performance'],
            'ceramic_type': ['SiC', 'B4C'],
            'v50_literature': [850, 920],
            'hardness_literature': [26.5, 30.2],
            'K_IC_literature': [4.0, 3.5]
        }),
        'matweb': pd.DataFrame({
            'source': ['MatWeb'] * 2,
            'material_name': ['Silicon Carbide', 'Boron Carbide'],
            'formula': ['SiC', 'B4C'],
            'hardness_GPa': [26.0, 30.0],
            'K_IC_MPa_m05': [4.0, 3.5],
            'density_matweb': [3.21, 2.52],
            'compressive_strength_GPa': [3.9, 2.8]
        }),
        'nist': pd.DataFrame({
            'source': ['NIST'] * 2,
            'formula': ['SiC', 'B4C'],
            'ceramic_type': ['SiC', 'B4C'],
            'density': [3.21, 2.52],
            'hardness': [26.0, 30.0],
            'K_IC': [4.0, 3.5]
        })
    }


def test_complete_pipeline_with_mocked_apis(test_config, temp_workspace, mock_collector_data):
    """
    Test 1: test_complete_pipeline_with_mocked_apis()
    
    Mock all external APIs and run complete pipeline.
    Verify all collectors execute and master_dataset.csv is created.
    """
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Configure mocks to return test data
    mock_mp.collect.return_value = mock_collector_data['mp']
    mock_jarvis.collect.return_value = mock_collector_data['jarvis']
    mock_aflow.collect.return_value = mock_collector_data['aflow']
    mock_scholar.collect.return_value = mock_collector_data['semantic_scholar']
    mock_matweb.collect.return_value = mock_collector_data['matweb']
    mock_nist.collect.return_value = mock_collector_data['nist']
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist):
        
        # Run complete pipeline
        orchestrator = CollectionOrchestrator(test_config)
        master_path = orchestrator.run_all()
        
        # Verify all collectors were called
        mock_mp.collect.assert_called_once()
        mock_jarvis.collect.assert_called_once()
        mock_aflow.collect.assert_called_once()
        mock_scholar.collect.assert_called_once()
        mock_matweb.collect.assert_called_once()
        mock_nist.collect.assert_called_once()
        
        # Verify integration ran
        assert 'integration' in orchestrator.results
        assert orchestrator.results['integration']['status'] == 'SUCCESS'
        
        # Verify master_dataset.csv was created
        assert master_path.exists()
        assert master_path.name == 'master_dataset.csv'
        
        # Verify all collectors succeeded
        for collector_name in ['mp', 'jarvis', 'aflow', 'semantic_scholar', 'matweb', 'nist']:
            assert orchestrator.results[collector_name]['status'] == 'SUCCESS'
            assert orchestrator.results[collector_name]['count'] > 0


def test_master_dataset_structure(test_config, temp_workspace, mock_collector_data):
    """
    Test 2: test_master_dataset_structure()
    
    Load master_dataset.csv and verify required columns and derived properties.
    """
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Configure mocks to return test data
    mock_mp.collect.return_value = mock_collector_data['mp']
    mock_jarvis.collect.return_value = mock_collector_data['jarvis']
    mock_aflow.collect.return_value = mock_collector_data['aflow']
    mock_scholar.collect.return_value = mock_collector_data['semantic_scholar']
    mock_matweb.collect.return_value = mock_collector_data['matweb']
    mock_nist.collect.return_value = mock_collector_data['nist']
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist):
        
        # Run pipeline
        orchestrator = CollectionOrchestrator(test_config)
        master_path = orchestrator.run_all()
        
        # Load master dataset
        df = pd.read_csv(master_path)
        
        # Verify required columns present
        assert 'formula' in df.columns, "Missing 'formula' column"
        
        # Verify combined property columns
        assert 'hardness_combined' in df.columns, "Missing 'hardness_combined' column"
        assert 'K_IC_combined' in df.columns, "Missing 'K_IC_combined' column"
        assert 'density_combined' in df.columns, "Missing 'density_combined' column"
        
        # Verify derived properties
        assert 'specific_hardness' in df.columns, "Missing 'specific_hardness' column"
        assert 'ballistic_efficacy' in df.columns, "Missing 'ballistic_efficacy' column"
        assert 'dop_resistance' in df.columns, "Missing 'dop_resistance' column"
        
        # Verify data is present (not all NaN)
        assert df['formula'].notna().sum() > 0, "No formulas in dataset"
        assert df['hardness_combined'].notna().sum() > 0, "No hardness values"
        assert df['density_combined'].notna().sum() > 0, "No density values"
        
        # Verify derived properties are calculated correctly
        # specific_hardness = hardness / density
        for idx, row in df.iterrows():
            if pd.notna(row['hardness_combined']) and pd.notna(row['density_combined']):
                expected_specific = row['hardness_combined'] / row['density_combined']
                if pd.notna(row['specific_hardness']):
                    assert abs(row['specific_hardness'] - expected_specific) < 0.01, \
                        f"Incorrect specific_hardness calculation at row {idx}"


def test_phase1_can_consume_output(test_config, temp_workspace, mock_collector_data):
    """
    Test 3: test_phase1_can_consume_output()
    
    Verify master_dataset.csv can be loaded and used by Phase 1 ML pipeline.
    """
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Configure mocks to return test data
    mock_mp.collect.return_value = mock_collector_data['mp']
    mock_jarvis.collect.return_value = mock_collector_data['jarvis']
    mock_aflow.collect.return_value = mock_collector_data['aflow']
    mock_scholar.collect.return_value = mock_collector_data['semantic_scholar']
    mock_matweb.collect.return_value = mock_collector_data['matweb']
    mock_nist.collect.return_value = mock_collector_data['nist']
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist):
        
        # Run pipeline
        orchestrator = CollectionOrchestrator(test_config)
        master_path = orchestrator.run_all()
        
        # Load with pandas (as Phase 1 would)
        df = pd.read_csv(master_path)
        
        # Verify DataFrame is valid
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Verify data types are compatible with Phase 1
        # Numeric columns should be numeric
        numeric_cols = ['hardness_combined', 'K_IC_combined', 'density_combined',
                       'specific_hardness', 'ballistic_efficacy', 'dop_resistance']
        
        for col in numeric_cols:
            if col in df.columns:
                # Check that non-null values are numeric
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    assert pd.api.types.is_numeric_dtype(df[col]), \
                        f"Column {col} should be numeric for Phase 1 compatibility"
        
        # Verify formula column is string type
        assert 'formula' in df.columns
        assert df['formula'].dtype == object or pd.api.types.is_string_dtype(df['formula'])
        
        # Verify no completely empty rows
        assert not df.isna().all(axis=1).any(), "Found completely empty rows"


def test_integration_report_completeness(test_config, temp_workspace, mock_collector_data):
    """
    Test 4: test_integration_report_completeness()
    
    Verify integration_report.txt contains all required information.
    """
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Configure mocks to return test data
    mock_mp.collect.return_value = mock_collector_data['mp']
    mock_jarvis.collect.return_value = mock_collector_data['jarvis']
    mock_aflow.collect.return_value = mock_collector_data['aflow']
    mock_scholar.collect.return_value = mock_collector_data['semantic_scholar']
    mock_matweb.collect.return_value = mock_collector_data['matweb']
    mock_nist.collect.return_value = mock_collector_data['nist']
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist):
        
        # Run pipeline
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify integration report exists
        report_path = Path('data/final/integration_report.txt')
        assert report_path.exists(), "integration_report.txt not created"
        
        # Read report content
        content = report_path.read_text(encoding='utf-8')
        
        # Verify header
        assert 'PHASE 2 DATA INTEGRATION REPORT' in content
        
        # Verify dataset summary section
        assert 'DATASET SUMMARY' in content
        assert 'Total materials:' in content
        assert 'Unique formulas:' in content
        assert 'Total columns:' in content
        
        # Verify data sources section
        assert 'DATA SOURCES' in content
        
        # Verify property coverage section
        assert 'PROPERTY COVERAGE' in content
        assert 'hardness_combined' in content or 'With hardness_combined' in content
        assert 'K_IC_combined' in content or 'With K_IC_combined' in content
        assert 'density_combined' in content or 'With density_combined' in content
        
        # Verify derived properties section
        assert 'DERIVED PROPERTIES' in content
        assert 'Specific hardness' in content or 'specific_hardness' in content
        assert 'Ballistic efficacy' in content or 'ballistic_efficacy' in content
        assert 'DOP resistance' in content or 'dop_resistance' in content


def test_pipeline_summary_completeness(test_config, temp_workspace, mock_collector_data):
    """
    Test 5: test_pipeline_summary_completeness()
    
    Verify pipeline_summary.txt contains execution details for all collectors.
    """
    # Create mock collectors
    mock_mp = Mock()
    mock_jarvis = Mock()
    mock_aflow = Mock()
    mock_scholar = Mock()
    mock_matweb = Mock()
    mock_nist = Mock()
    
    # Configure mocks to return test data
    mock_mp.collect.return_value = mock_collector_data['mp']
    mock_jarvis.collect.return_value = mock_collector_data['jarvis']
    mock_aflow.collect.return_value = mock_collector_data['aflow']
    mock_scholar.collect.return_value = mock_collector_data['semantic_scholar']
    mock_matweb.collect.return_value = mock_collector_data['matweb']
    mock_nist.collect.return_value = mock_collector_data['nist']
    
    with patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MaterialsProjectCollector', return_value=mock_mp), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.JarvisCollector', return_value=mock_jarvis), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.AFLOWCollector', return_value=mock_aflow), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.SemanticScholarCollector', return_value=mock_scholar), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.MatWebCollector', return_value=mock_matweb), \
         patch('ceramic_discovery.data_pipeline.pipelines.collection_orchestrator.NISTBaselineCollector', return_value=mock_nist):
        
        # Run pipeline
        orchestrator = CollectionOrchestrator(test_config)
        orchestrator.run_all()
        
        # Verify summary report exists
        summary_path = Path('data/final/pipeline_summary.txt')
        assert summary_path.exists(), "pipeline_summary.txt not created"
        
        # Read summary content
        content = summary_path.read_text(encoding='utf-8')
        
        # Verify header
        assert 'PHASE 2 DATA COLLECTION PIPELINE SUMMARY' in content
        
        # Verify execution timestamp
        assert 'Execution Time:' in content
        
        # Verify collector results section
        assert 'COLLECTOR RESULTS' in content
        
        # Verify per-collector status (SUCCESS/FAILED)
        collector_names = ['mp', 'jarvis', 'aflow', 'semantic_scholar', 'matweb', 'nist', 'integration']
        for name in collector_names:
            assert name in content, f"Collector '{name}' not in summary"
        
        # Verify status indicators present
        assert '✓' in content or 'SUCCESS' in content, "No success indicators in summary"
        
        # Verify runtime information
        assert 'Total Runtime:' in content
        
        # Verify runtime for each collector is present (format: X.Xs)
        # Should have runtime values like "0.1s" or "1.5s"
        import re
        runtime_pattern = r'\d+\.\d+s'
        runtimes = re.findall(runtime_pattern, content)
        # Should have at least 7 runtimes (6 collectors + 1 integration)
        assert len(runtimes) >= 7, f"Expected at least 7 runtime values, found {len(runtimes)}"
