"""
Unit tests for BaseCollector abstract class.

Tests verify:
1. BaseCollector cannot be instantiated directly (abstract)
2. Abstract methods must be implemented by subclasses
3. Concrete subclass with all methods works correctly
4. save_to_csv() saves DataFrame correctly
5. log_stats() logs statistics correctly
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import logging
from ceramic_discovery.data_pipeline.collectors.base_collector import BaseCollector


class TestBaseCollectorAbstract:
    """Test that BaseCollector is properly abstract."""
    
    def test_base_collector_cannot_instantiate(self):
        """Verify TypeError raised when trying to instantiate BaseCollector directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseCollector({})
        
        # Verify error message mentions abstract methods
        error_msg = str(exc_info.value)
        assert "abstract" in error_msg.lower() or "Can't instantiate" in error_msg
    
    def test_abstract_methods_required(self):
        """Create minimal concrete subclass missing one abstract method."""
        
        # Missing validate_config method
        class IncompleteCollector(BaseCollector):
            def collect(self) -> pd.DataFrame:
                return pd.DataFrame()
            
            def get_source_name(self) -> str:
                return "Test"
        
        # Should raise TypeError because validate_config is not implemented
        with pytest.raises(TypeError):
            IncompleteCollector({})


class TestConcreteCollector:
    """Test a complete concrete implementation of BaseCollector."""
    
    @pytest.fixture
    def concrete_collector_class(self):
        """Create a complete concrete subclass for testing."""
        
        class TestCollector(BaseCollector):
            def collect(self) -> pd.DataFrame:
                return pd.DataFrame({
                    'source': ['Test', 'Test'],
                    'formula': ['SiC', 'B4C'],
                    'property1': [1.0, 2.0],
                    'property2': [3.0, None]
                })
            
            def get_source_name(self) -> str:
                return "TestSource"
            
            def validate_config(self) -> bool:
                return 'required_key' in self.config
        
        return TestCollector
    
    def test_concrete_subclass_works(self, concrete_collector_class):
        """Verify instantiation succeeds with all abstract methods implemented."""
        config = {'required_key': 'value'}
        collector = concrete_collector_class(config)
        
        # Verify instantiation succeeded
        assert collector is not None
        assert collector.source_name == "TestSource"
        
        # Verify collect() returns DataFrame
        df = collector.collect()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'source' in df.columns
        assert 'formula' in df.columns
    
    def test_validation_failure_raises_error(self, concrete_collector_class):
        """Verify ValueError raised when validate_config returns False."""
        config = {}  # Missing required_key
        
        with pytest.raises(ValueError) as exc_info:
            concrete_collector_class(config)
        
        error_msg = str(exc_info.value)
        assert "configuration invalid" in error_msg.lower()


class TestSaveToCSV:
    """Test save_to_csv() method."""
    
    @pytest.fixture
    def test_collector(self):
        """Create a test collector instance."""
        
        class TestCollector(BaseCollector):
            def collect(self) -> pd.DataFrame:
                return pd.DataFrame()
            
            def get_source_name(self) -> str:
                return "TestSource"
            
            def validate_config(self) -> bool:
                return True
        
        return TestCollector({})
    
    def test_save_to_csv(self, test_collector):
        """Create test DataFrame and verify CSV saved correctly."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'source': ['TestSource', 'TestSource', 'TestSource'],
            'formula': ['SiC', 'B4C', 'WC'],
            'property1': [1.0, 2.0, 3.0],
            'property2': [4.0, 5.0, 6.0]
        })
        
        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save to CSV
            output_path = test_collector.save_to_csv(test_df, output_dir)
            
            # Verify file created with correct name
            assert output_path.exists()
            assert output_path.name == "testsource_data.csv"
            
            # Verify CSV content matches DataFrame
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == len(test_df)
            assert list(loaded_df.columns) == list(test_df.columns)
            assert loaded_df['formula'].tolist() == test_df['formula'].tolist()
    
    def test_save_to_csv_creates_directory(self, test_collector):
        """Verify save_to_csv creates output directory if it doesn't exist."""
        test_df = pd.DataFrame({
            'source': ['TestSource'],
            'formula': ['SiC']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use non-existent subdirectory
            output_dir = Path(tmpdir) / "subdir" / "nested"
            
            # Should create directory
            output_path = test_collector.save_to_csv(test_df, output_dir)
            
            assert output_dir.exists()
            assert output_path.exists()


class TestLogStats:
    """Test log_stats() method."""
    
    @pytest.fixture
    def test_collector(self):
        """Create a test collector instance."""
        
        class TestCollector(BaseCollector):
            def collect(self) -> pd.DataFrame:
                return pd.DataFrame()
            
            def get_source_name(self) -> str:
                return "TestSource"
            
            def validate_config(self) -> bool:
                return True
        
        return TestCollector({})
    
    def test_log_stats(self, test_collector, caplog):
        """Create test DataFrame and verify statistics logged."""
        # Create test DataFrame with various properties
        test_df = pd.DataFrame({
            'source': ['TestSource'] * 5,
            'formula': ['SiC', 'B4C', 'WC', 'TiC', 'SiC'],
            'property1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'property2': [10.0, None, 30.0, None, 50.0],
            'property3': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        # Capture log output
        with caplog.at_level(logging.INFO):
            test_collector.log_stats(test_df)
        
        # Verify statistics logged
        log_text = caplog.text
        
        # Check total materials count
        assert "Total materials: 5" in log_text
        
        # Check unique formulas count
        assert "Unique formulas: 4" in log_text
        
        # Check property coverage
        assert "Property coverage:" in log_text
        assert "property1:" in log_text
        assert "5/5 (100.0%)" in log_text  # All values present
        assert "property2:" in log_text
        assert "3/5 (60.0%)" in log_text  # 3 out of 5 values present
        assert "property3:" in log_text
        assert "5/5 (100.0%)" in log_text  # All values present
    
    def test_log_stats_empty_dataframe(self, test_collector, caplog):
        """Verify log_stats handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame()
        
        with caplog.at_level(logging.INFO):
            test_collector.log_stats(empty_df)
        
        log_text = caplog.text
        assert "Total materials: 0" in log_text
    
    def test_log_stats_no_numeric_columns(self, test_collector, caplog):
        """Verify log_stats handles DataFrame with no numeric columns."""
        test_df = pd.DataFrame({
            'source': ['TestSource', 'TestSource'],
            'formula': ['SiC', 'B4C']
        })
        
        with caplog.at_level(logging.INFO):
            test_collector.log_stats(test_df)
        
        log_text = caplog.text
        assert "Total materials: 2" in log_text
        assert "Unique formulas: 2" in log_text
