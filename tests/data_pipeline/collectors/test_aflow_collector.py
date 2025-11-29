"""
Unit tests for AFLOWCollector

Tests the AFLOW data collector for thermal properties.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from ceramic_discovery.data_pipeline.collectors.aflow_collector import AFLOWCollector


class TestAFLOWCollector:
    """Test suite for AFLOWCollector"""
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration for AFLOW collector"""
        return {
            'ceramics': {
                'SiC': ['Si', 'C'],
                'B4C': ['B', 'C']
            },
            'max_per_system': 10,
            'exclude_ldau': True
        }
    
    @pytest.fixture
    def mock_aflow_entry(self):
        """Create a mock AFLOW entry with thermal properties"""
        entry = Mock()
        entry.compound = 'SiC'
        entry.density = 3.21
        entry.enthalpy_formation_atom = -0.5
        entry.Egap = 2.3
        entry.agl_thermal_conductivity_300K = 120.0
        entry.agl_thermal_expansion_300K = 4.5e-6
        entry.agl_debye = 1200.0
        entry.agl_heat_capacity_Cp_300K = 25.0
        return entry
    
    def test_aflow_query_construction(self, valid_config, mock_aflow_entry):
        """Test 1: Verify AFLOW query is built with correct species"""
        with patch('ceramic_discovery.data_pipeline.collectors.aflow_collector.aflow') as mock_aflow:
            # Setup mock query chain
            mock_query = MagicMock()
            mock_query.exclude.return_value = mock_query
            mock_query.select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.__iter__.return_value = iter([mock_aflow_entry])
            
            mock_aflow.search.return_value = mock_query
            mock_aflow.K = Mock()
            mock_aflow.K.compound = 'compound'
            mock_aflow.K.density = 'density'
            mock_aflow.K.enthalpy_formation_atom = 'enthalpy_formation_atom'
            mock_aflow.K.Egap = 'Egap'
            mock_aflow.K.agl_thermal_conductivity_300K = 'agl_thermal_conductivity_300K'
            mock_aflow.K.agl_thermal_expansion_300K = 'agl_thermal_expansion_300K'
            mock_aflow.K.agl_debye = 'agl_debye'
            mock_aflow.K.agl_heat_capacity_Cp_300K = 'agl_heat_capacity_Cp_300K'
            
            collector = AFLOWCollector(valid_config)
            df = collector.collect()
            
            # Verify search was called with correct species
            assert mock_aflow.search.call_count == 2  # SiC and B4C
            calls = mock_aflow.search.call_args_list
            assert calls[0][1]['species'] == 'Si,C'
            assert calls[1][1]['species'] == 'B,C'
            
            # Verify LDAU exclusion
            assert mock_query.exclude.called
            assert mock_query.exclude.call_args[0][0] == ['*:LDAU*']
    
    def test_thermal_property_extraction(self, valid_config, mock_aflow_entry):
        """Test 2: Verify thermal properties are extracted correctly"""
        with patch('ceramic_discovery.data_pipeline.collectors.aflow_collector.aflow') as mock_aflow:
            # Setup mock
            mock_query = MagicMock()
            mock_query.exclude.return_value = mock_query
            mock_query.select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.__iter__.return_value = iter([mock_aflow_entry])
            
            mock_aflow.search.return_value = mock_query
            mock_aflow.K = Mock()
            mock_aflow.K.compound = 'compound'
            mock_aflow.K.density = 'density'
            mock_aflow.K.enthalpy_formation_atom = 'enthalpy_formation_atom'
            mock_aflow.K.Egap = 'Egap'
            mock_aflow.K.agl_thermal_conductivity_300K = 'agl_thermal_conductivity_300K'
            mock_aflow.K.agl_thermal_expansion_300K = 'agl_thermal_expansion_300K'
            mock_aflow.K.agl_debye = 'agl_debye'
            mock_aflow.K.agl_heat_capacity_Cp_300K = 'agl_heat_capacity_Cp_300K'
            
            collector = AFLOWCollector(valid_config)
            df = collector.collect()
            
            # Verify DataFrame has expected thermal properties
            assert 'thermal_conductivity_300K' in df.columns
            assert 'thermal_expansion_300K' in df.columns
            assert 'debye_temperature' in df.columns
            assert 'heat_capacity_Cp_300K' in df.columns
            
            # Verify values extracted correctly
            assert df.iloc[0]['thermal_conductivity_300K'] == 120.0
            assert df.iloc[0]['thermal_expansion_300K'] == 4.5e-6
            assert df.iloc[0]['debye_temperature'] == 1200.0
            assert df.iloc[0]['heat_capacity_Cp_300K'] == 25.0
    
    def test_error_handling_api_error(self, valid_config):
        """Test 3: Verify collector handles AFLOW API errors gracefully"""
        with patch('ceramic_discovery.data_pipeline.collectors.aflow_collector.aflow') as mock_aflow:
            # Setup mock to raise AFLOWAPIError
            mock_aflow.AFLOWAPIError = Exception  # Mock the exception class
            mock_aflow.search.side_effect = mock_aflow.AFLOWAPIError("API Error")
            
            collector = AFLOWCollector(valid_config)
            
            # Should not raise exception, should log error and continue
            df = collector.collect()
            
            # Should return empty DataFrame since all queries failed
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
    
    def test_dataframe_structure(self, valid_config, mock_aflow_entry):
        """Test 4: Verify DataFrame has correct structure and columns"""
        with patch('ceramic_discovery.data_pipeline.collectors.aflow_collector.aflow') as mock_aflow:
            # Setup mock
            mock_query = MagicMock()
            mock_query.exclude.return_value = mock_query
            mock_query.select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.__iter__.return_value = iter([mock_aflow_entry])
            
            mock_aflow.search.return_value = mock_query
            mock_aflow.K = Mock()
            mock_aflow.K.compound = 'compound'
            mock_aflow.K.density = 'density'
            mock_aflow.K.enthalpy_formation_atom = 'enthalpy_formation_atom'
            mock_aflow.K.Egap = 'Egap'
            mock_aflow.K.agl_thermal_conductivity_300K = 'agl_thermal_conductivity_300K'
            mock_aflow.K.agl_thermal_expansion_300K = 'agl_thermal_expansion_300K'
            mock_aflow.K.agl_debye = 'agl_debye'
            mock_aflow.K.agl_heat_capacity_Cp_300K = 'agl_heat_capacity_Cp_300K'
            
            collector = AFLOWCollector(valid_config)
            df = collector.collect()
            
            # Verify required columns
            required_columns = [
                'source', 'formula', 'ceramic_type', 'density_aflow',
                'thermal_conductivity_300K', 'thermal_expansion_300K',
                'debye_temperature', 'heat_capacity_Cp_300K'
            ]
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            # Verify source is 'AFLOW'
            assert all(df['source'] == 'AFLOW')
            
            # Verify ceramic_type is set
            assert df.iloc[0]['ceramic_type'] in ['SiC', 'B4C']
    
    def test_max_per_system_limit(self, valid_config):
        """Test 5: Verify max_per_system limit is respected"""
        with patch('ceramic_discovery.data_pipeline.collectors.aflow_collector.aflow') as mock_aflow:
            # Create multiple mock entries
            mock_entries = []
            for i in range(20):  # More than max_per_system
                entry = Mock()
                entry.compound = f'SiC_{i}'
                entry.density = 3.21
                entry.enthalpy_formation_atom = -0.5
                entry.Egap = 2.3
                entry.agl_thermal_conductivity_300K = 120.0
                entry.agl_thermal_expansion_300K = 4.5e-6
                entry.agl_debye = 1200.0
                entry.agl_heat_capacity_Cp_300K = 25.0
                mock_entries.append(entry)
            
            # Setup mock query
            mock_query = MagicMock()
            mock_query.exclude.return_value = mock_query
            mock_query.select.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.__iter__.return_value = iter(mock_entries[:10])  # Limit returns only 10
            
            mock_aflow.search.return_value = mock_query
            mock_aflow.K = Mock()
            mock_aflow.K.compound = 'compound'
            mock_aflow.K.density = 'density'
            mock_aflow.K.enthalpy_formation_atom = 'enthalpy_formation_atom'
            mock_aflow.K.Egap = 'Egap'
            mock_aflow.K.agl_thermal_conductivity_300K = 'agl_thermal_conductivity_300K'
            mock_aflow.K.agl_thermal_expansion_300K = 'agl_thermal_expansion_300K'
            mock_aflow.K.agl_debye = 'agl_debye'
            mock_aflow.K.agl_heat_capacity_Cp_300K = 'agl_heat_capacity_Cp_300K'
            
            collector = AFLOWCollector(valid_config)
            df = collector.collect()
            
            # Verify limit was called with correct value
            assert mock_query.limit.called
            assert mock_query.limit.call_args[0][0] == 10
            
            # Verify we got at most max_per_system * num_ceramics results
            # (10 per system * 2 systems = 20 max)
            assert len(df) <= 20
    
    def test_validate_config_valid(self, valid_config):
        """Test that valid configuration passes validation"""
        collector = AFLOWCollector(valid_config)
        assert collector.validate_config() is True
    
    def test_validate_config_missing_ceramics(self):
        """Test that missing ceramics key fails validation"""
        config = {'max_per_system': 10}
        with pytest.raises(ValueError):
            AFLOWCollector(config)
    
    def test_validate_config_invalid_ceramics_type(self):
        """Test that invalid ceramics type fails validation"""
        config = {'ceramics': 'not a dict'}
        with pytest.raises(ValueError):
            AFLOWCollector(config)
    
    def test_validate_config_empty_elements(self):
        """Test that empty elements list fails validation"""
        config = {'ceramics': {'SiC': []}}
        with pytest.raises(ValueError):
            AFLOWCollector(config)
    
    def test_get_source_name(self, valid_config):
        """Test that source name is correct"""
        collector = AFLOWCollector(valid_config)
        assert collector.get_source_name() == "AFLOW"
