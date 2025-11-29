"""
Unit tests for MaterialsProjectCollector.

Tests verify that the collector properly wraps Phase 1 MaterialsProjectClient
using composition pattern without modifying existing code.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from ceramic_discovery.data_pipeline.collectors.materials_project_collector import MaterialsProjectCollector
from ceramic_discovery.dft.materials_project_client import MaterialData


class TestMaterialsProjectCollector:
    """Test suite for MaterialsProjectCollector."""
    
    def test_collector_uses_composition_not_inheritance(self):
        """Verify collector uses composition (HAS-A) not inheritance (IS-A)."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
            # Create mock instance
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance
            
            # Create collector with test config
            config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C'],
                'batch_size': 100,
                'max_materials': 1000
            }
            collector = MaterialsProjectCollector(config)
            
            # Verify MaterialsProjectClient.__init__ was called (composition)
            MockClient.assert_called_once_with(api_key='test_key')
            
            # Verify collector has client as attribute (composition)
            assert hasattr(collector, 'client')
            assert collector.client == mock_client_instance
            
            # Verify collector is NOT instance of MaterialsProjectClient (not inheritance)
            # We can't directly test this with mocks, but we verify it doesn't inherit
            assert not hasattr(MaterialsProjectCollector, 'search_materials')
            assert not hasattr(MaterialsProjectCollector, 'get_material_by_id')
    
    def test_collect_returns_dataframe(self):
        """Verify collect() returns DataFrame with required columns."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
            # Create mock client
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Create mock MaterialData objects
            mock_material1 = MaterialData(
                material_id='mp-149',
                formula='SiC',
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={'bulk_modulus': 220, 'shear_modulus': 190, 'is_stable': True},
                metadata={}
            )
            mock_material2 = MaterialData(
                material_id='mp-150',
                formula='Si2C',
                crystal_structure={},
                formation_energy=-0.4,
                energy_above_hull=0.1,
                band_gap=1.5,
                density=3.0,
                properties={'bulk_modulus': 200, 'is_stable': False},
                metadata={}
            )
            
            # Mock search_materials to return test data
            mock_client.search_materials.return_value = [mock_material1, mock_material2]
            
            # Create collector and collect
            config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C'],
                'batch_size': 100,
                'max_materials': 1000
            }
            collector = MaterialsProjectCollector(config)
            df = collector.collect()
            
            # Verify returns pandas DataFrame
            assert isinstance(df, pd.DataFrame)
            
            # Verify DataFrame has required columns
            assert 'source' in df.columns
            assert 'material_id' in df.columns
            assert 'formula' in df.columns
            
            # Verify data content
            assert len(df) == 2
            assert df['source'].iloc[0] == 'MaterialsProject'
            assert df['material_id'].iloc[0] == 'mp-149'
            assert df['formula'].iloc[0] == 'SiC'
            assert df['formation_energy_mp'].iloc[0] == -0.5
            assert df['band_gap_mp'].iloc[0] == 2.3
    
    def test_batch_processing(self):
        """Verify batch_size and max_materials limits are respected."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Create many mock materials
            mock_materials = []
            for i in range(150):
                mock_materials.append(MaterialData(
                    material_id=f'mp-{i}',
                    formula='SiC',
                    crystal_structure={},
                    formation_energy=-0.5,
                    energy_above_hull=0.0,
                    band_gap=2.3,
                    density=3.2,
                    properties={},
                    metadata={}
                ))
            
            mock_client.search_materials.return_value = mock_materials
            
            # Create collector with max_materials limit
            config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C'],
                'batch_size': 50,
                'max_materials': 100
            }
            collector = MaterialsProjectCollector(config)
            df = collector.collect()
            
            # Verify max_materials limit respected
            assert len(df) <= 100
            
            # Verify batch_size was passed to search_materials
            call_args = mock_client.search_materials.call_args
            assert call_args[1]['limit'] <= 50
    
    def test_error_handling_api_failure(self):
        """Verify collector logs error and continues on API failure."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Mock search_materials to raise exception
            mock_client.search_materials.side_effect = Exception("API Error")
            
            # Create collector
            config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C', 'B-C'],
                'batch_size': 100,
                'max_materials': 1000
            }
            collector = MaterialsProjectCollector(config)
            
            # Should not raise exception, should return empty DataFrame
            df = collector.collect()
            
            # Verify returns DataFrame (empty due to errors)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
    
    def test_missing_api_key(self):
        """Verify ValueError raised when api_key missing."""
        config = {
            'target_systems': ['Si-C'],
            'batch_size': 100,
            'max_materials': 1000
        }
        
        # Should raise ValueError (from base class validation or api_key check)
        with pytest.raises(ValueError):
            MaterialsProjectCollector(config)
    
    def test_import_error_handling(self):
        """Verify ImportError has helpful message if Phase 1 client missing."""
        # This test verifies the import error message is helpful
        # We can't actually test the import failure without breaking the environment
        # But we can verify the error message format in the code
        
        # Read the source file to verify error message
        import inspect
        from ceramic_discovery.data_pipeline.collectors import materials_project_collector
        
        source = inspect.getsource(materials_project_collector)
        
        # Verify helpful error message exists
        assert "Cannot import MaterialsProjectClient from Phase 1" in source
        assert "Ensure Phase 1 code at src/ceramic_discovery/dft/materials_project_client.py exists" in source
    
    def test_validate_config(self):
        """Verify config validation works correctly."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient'):
            # Valid config
            valid_config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C'],
                'batch_size': 100,
                'max_materials': 1000
            }
            collector = MaterialsProjectCollector(valid_config)
            assert collector.validate_config() is True
            
            # Invalid config - missing target_systems
            invalid_config = {
                'api_key': 'test_key',
                'batch_size': 100
            }
            # Should raise ValueError during __init__ due to validation
            with pytest.raises(ValueError):
                MaterialsProjectCollector(invalid_config)
    
    def test_get_source_name(self):
        """Verify get_source_name returns correct identifier."""
        with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient'):
            config = {
                'api_key': 'test_key',
                'target_systems': ['Si-C'],
                'batch_size': 100,
                'max_materials': 1000
            }
            collector = MaterialsProjectCollector(config)
            assert collector.get_source_name() == "MaterialsProject"
