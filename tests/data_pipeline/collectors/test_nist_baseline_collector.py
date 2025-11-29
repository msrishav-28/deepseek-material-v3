"""
Unit tests for NISTBaselineCollector

Tests the NIST baseline reference data collector.
"""

import pytest
import pandas as pd
from ceramic_discovery.data_pipeline.collectors.nist_baseline_collector import NISTBaselineCollector


class TestNISTBaselineCollector:
    """Test suite for NISTBaselineCollector"""
    
    @pytest.fixture
    def valid_config(self):
        """Valid configuration with 5 baseline materials"""
        return {
            'baseline_materials': {
                'SiC': {
                    'formula': 'SiC',
                    'ceramic_type': 'SiC',
                    'density': 3.21,
                    'hardness': 26.5,
                    'K_IC': 4.0,
                    'source': 'NIST SRM 2100'
                },
                'B4C': {
                    'formula': 'B4C',
                    'ceramic_type': 'B4C',
                    'density': 2.52,
                    'hardness': 30.0,
                    'K_IC': 3.5,
                    'source': 'NIST Reference'
                },
                'WC': {
                    'formula': 'WC',
                    'ceramic_type': 'WC',
                    'density': 15.6,
                    'hardness': 22.0,
                    'K_IC': 5.0,
                    'source': 'NIST Database'
                },
                'TiC': {
                    'formula': 'TiC',
                    'ceramic_type': 'TiC',
                    'density': 4.93,
                    'hardness': 28.0,
                    'K_IC': 4.5,
                    'source': 'NIST Reference'
                },
                'Al2O3': {
                    'formula': 'Al2O3',
                    'ceramic_type': 'Al2O3',
                    'density': 3.98,
                    'hardness': 20.0,
                    'K_IC': 4.0,
                    'source': 'NIST SRM 676a'
                }
            }
        }
    
    def test_baseline_data_loading(self, valid_config):
        """Test 1: Verify baseline data loads correctly"""
        collector = NISTBaselineCollector(valid_config)
        df = collector.collect()
        
        # Verify DataFrame has 5 rows (one per material)
        assert len(df) == 5
        
        # Verify all materials present
        formulas = set(df['formula'])
        assert formulas == {'SiC', 'B4C', 'WC', 'TiC', 'Al2O3'}
    
    def test_validation_complete_data(self, valid_config):
        """Test 2: Verify validation passes with complete data"""
        collector = NISTBaselineCollector(valid_config)
        assert collector.validate_config() is True
    
    def test_validation_missing_property(self):
        """Test 3: Verify validation fails when property is missing"""
        config = {
            'baseline_materials': {
                'SiC': {
                    'formula': 'SiC',
                    'ceramic_type': 'SiC',
                    'density': 3.21,
                    # Missing 'hardness'
                    'K_IC': 4.0,
                    'source': 'NIST'
                }
            }
        }
        
        with pytest.raises(ValueError):
            NISTBaselineCollector(config)
    
    def test_no_conflict_with_phase1_nist(self):
        """Test 4: Verify no conflict with Phase 1 NISTClient"""
        # Verify we can import both classes
        try:
            from ceramic_discovery.dft.nist_client import NISTClient
            from ceramic_discovery.data_pipeline.collectors.nist_baseline_collector import NISTBaselineCollector
            
            # Verify they are different classes
            assert NISTClient != NISTBaselineCollector
            assert NISTClient.__name__ == 'NISTClient'
            assert NISTBaselineCollector.__name__ == 'NISTBaselineCollector'
            
        except ImportError as e:
            # Phase 1 NISTClient might not exist yet, which is okay
            pytest.skip(f"Phase 1 NISTClient not available: {e}")
    
    def test_dataframe_structure(self, valid_config):
        """Test 5: Verify DataFrame has correct structure"""
        collector = NISTBaselineCollector(valid_config)
        df = collector.collect()
        
        # Verify required columns
        required_columns = [
            'source', 'formula', 'ceramic_type', 
            'density', 'hardness', 'K_IC', 'data_source'
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Verify source is 'NIST'
        assert all(df['source'] == 'NIST')
        
        # Verify data_source (provenance) is present
        assert all(df['data_source'].notna())
    
    def test_validate_config_missing_baseline_materials(self):
        """Test that missing baseline_materials key fails validation"""
        config = {}
        with pytest.raises(ValueError):
            NISTBaselineCollector(config)
    
    def test_validate_config_invalid_baseline_materials_type(self):
        """Test that invalid baseline_materials type fails validation"""
        config = {'baseline_materials': 'not a dict'}
        with pytest.raises(ValueError):
            NISTBaselineCollector(config)
    
    def test_validate_config_invalid_properties_type(self):
        """Test that invalid properties type fails validation"""
        config = {
            'baseline_materials': {
                'SiC': 'not a dict'
            }
        }
        with pytest.raises(ValueError):
            NISTBaselineCollector(config)
    
    def test_get_source_name(self, valid_config):
        """Test that source name is correct"""
        collector = NISTBaselineCollector(valid_config)
        assert collector.get_source_name() == "NIST"
    
    def test_property_values(self, valid_config):
        """Test that property values are correctly loaded"""
        collector = NISTBaselineCollector(valid_config)
        df = collector.collect()
        
        # Find SiC row
        sic_row = df[df['formula'] == 'SiC'].iloc[0]
        
        # Verify property values
        assert sic_row['density'] == 3.21
        assert sic_row['hardness'] == 26.5
        assert sic_row['K_IC'] == 4.0
        assert sic_row['ceramic_type'] == 'SiC'
        assert 'NIST' in sic_row['data_source']
