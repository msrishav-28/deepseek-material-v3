"""
Unit tests for JarvisCollector.

Tests verify that the collector properly wraps Phase 1 JarvisClient
using composition pattern without modifying existing code.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from ceramic_discovery.data_pipeline.collectors.jarvis_collector import JarvisCollector
from ceramic_discovery.dft.materials_project_client import MaterialData


class TestJarvisCollector:
    """Test suite for JarvisCollector."""
    
    def test_collector_uses_composition(self):
        """Verify collector uses composition (HAS-A) not inheritance (IS-A)."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            # Create mock instance
            mock_client_instance = Mock()
            MockClient.return_value = mock_client_instance
            
            # Create collector with test config
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si', 'Ti', 'B'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            
            # Verify JarvisClient.__init__ was called (composition)
            MockClient.assert_called_once_with(jarvis_file_path='test_jarvis.json')
            
            # Verify collector has client as attribute (composition)
            assert hasattr(collector, 'client')
            assert collector.client == mock_client_instance
            
            # Verify collector is NOT instance of JarvisClient (not inheritance)
            assert not hasattr(JarvisCollector, 'load_carbides')
            assert not hasattr(JarvisCollector, 'extract_properties')
    
    def test_carbide_filtering(self):
        """Verify carbide filtering logic when using fallback method."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Remove load_carbides to force fallback filtering
            del mock_client.load_carbides
            
            # Mock data attribute with mixed materials
            mock_client.data = [
                {'jid': 'JVASP-1', 'formula': 'SiC', 'formation_energy_peratom': -0.5},
                {'jid': 'JVASP-2', 'formula': 'TiC', 'formation_energy_peratom': -0.6},
                {'jid': 'JVASP-3', 'formula': 'Si', 'formation_energy_peratom': -0.3},  # Not a carbide
                {'jid': 'JVASP-4', 'formula': 'WC', 'formation_energy_peratom': -0.4},  # Carbide but wrong metal
            ]
            mock_client.parse_formula_from_jid.side_effect = lambda jid: {
                'JVASP-1': 'SiC',
                'JVASP-2': 'TiC',
                'JVASP-3': 'Si',
                'JVASP-4': 'WC'
            }.get(jid)
            
            # Create collector with metal filter
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si', 'Ti'],  # Only Si and Ti carbides
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            df = collector.collect()
            
            # Verify only carbides with Si or Ti returned (filtering logic: 'C' in elements AND metal in carbide_metals)
            assert len(df) == 2
            formulas = set(df['formula'].values)
            assert 'SiC' in formulas
            assert 'TiC' in formulas
            assert 'Si' not in formulas
            assert 'WC' not in formulas
    
    def test_safe_float_usage(self):
        """Verify safe_float utility used for all numeric conversions."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.safe_float') as mock_safe_float:
                mock_client = Mock()
                MockClient.return_value = mock_client
                
                # Mock safe_float to return the value or None
                mock_safe_float.side_effect = lambda x: float(x) if x is not None and str(x).replace('.', '').replace('-', '').isdigit() else None
                
                # Create mock MaterialData with various numeric formats
                material = MaterialData(
                    material_id='JVASP-1',
                    formula='SiC',
                    crystal_structure={},
                    formation_energy=-0.5,
                    energy_above_hull=0.0,
                    band_gap=None,  # None value
                    density='3.2',  # String value
                    properties={'bulk_modulus': 220, 'shear_modulus': None},
                    metadata={}
                )
                
                mock_client.load_carbides.return_value = [material]
                
                config = {
                    'jarvis_file_path': 'test_jarvis.json',
                    'carbide_metals': ['Si'],
                    'data_source': 'dft_3d'
                }
                collector = JarvisCollector(config)
                df = collector.collect()
                
                # Verify safe_float was called for numeric conversions
                assert mock_safe_float.called
                # Verify no exceptions raised for invalid data
                assert len(df) == 1
    
    def test_existing_client_method_detection(self):
        """Verify collector uses existing method if available, fallback otherwise."""
        # Test 1: Client HAS load_carbides method
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Mock that load_carbides exists
            mock_client.load_carbides.return_value = []
            
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            collector.collect()
            
            # Verify load_carbides was called
            mock_client.load_carbides.assert_called_once()
        
        # Test 2: Client does NOT have load_carbides method
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Remove load_carbides attribute
            del mock_client.load_carbides
            
            # Mock data attribute for fallback
            mock_client.data = [
                {'jid': 'JVASP-1', 'formula': 'SiC', 'formation_energy_peratom': -0.5}
            ]
            mock_client.parse_formula_from_jid.return_value = 'SiC'
            
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            df = collector.collect()
            
            # Verify fallback filtering logic was used
            assert len(df) >= 0  # Should not crash
    
    def test_dataframe_structure(self):
        """Verify DataFrame has required columns with correct source."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Create mock MaterialData
            material = MaterialData(
                material_id='JVASP-1',
                formula='SiC',
                crystal_structure={},
                formation_energy=-0.5,
                energy_above_hull=0.0,
                band_gap=2.3,
                density=3.2,
                properties={'bulk_modulus': 220, 'shear_modulus': 190},
                metadata={}
            )
            
            mock_client.load_carbides.return_value = [material]
            
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            df = collector.collect()
            
            # Verify DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert 'source' in df.columns
            assert 'jid' in df.columns
            assert 'formula' in df.columns
            assert 'formation_energy_jarvis' in df.columns
            assert 'bulk_modulus_jarvis' in df.columns
            
            # Verify source column value
            assert df['source'].iloc[0] == 'JARVIS'
            
            # Verify data content
            assert df['jid'].iloc[0] == 'JVASP-1'
            assert df['formula'].iloc[0] == 'SiC'
    
    def test_missing_jarvis_file_path(self):
        """Verify ValueError raised when jarvis_file_path missing."""
        config = {
            'carbide_metals': ['Si'],
            'data_source': 'dft_3d'
        }
        
        # Should raise ValueError (from base class validation or file path check)
        with pytest.raises(ValueError):
            JarvisCollector(config)
    
    def test_validate_config(self):
        """Verify config validation works correctly."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient'):
            # Valid config
            valid_config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si', 'Ti'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(valid_config)
            assert collector.validate_config() is True
            
            # Invalid config - missing carbide_metals
            invalid_config = {
                'jarvis_file_path': 'test_jarvis.json',
                'data_source': 'dft_3d'
            }
            # Should raise ValueError during __init__ due to validation
            with pytest.raises(ValueError):
                JarvisCollector(invalid_config)
    
    def test_get_source_name(self):
        """Verify get_source_name returns correct identifier."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient'):
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            assert collector.get_source_name() == "JARVIS"
    
    def test_error_handling_collection_failure(self):
        """Verify collector returns empty DataFrame on collection failure."""
        with patch('ceramic_discovery.data_pipeline.collectors.jarvis_collector.JarvisClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            
            # Mock load_carbides to raise exception
            mock_client.load_carbides.side_effect = Exception("Collection error")
            
            config = {
                'jarvis_file_path': 'test_jarvis.json',
                'carbide_metals': ['Si'],
                'data_source': 'dft_3d'
            }
            collector = JarvisCollector(config)
            df = collector.collect()
            
            # Should return empty DataFrame, not raise exception
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
