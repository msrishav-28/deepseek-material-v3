"""
Unit tests for ConfigLoader utility.

Tests verify:
1. Loading valid configuration files
2. Environment variable substitution
3. Missing config file error handling
4. Validation of missing sections
5. Validation of missing API keys
"""

import pytest
import os
import tempfile
from pathlib import Path
from ceramic_discovery.data_pipeline.utils.config_loader import ConfigLoader


class TestConfigLoaderBasic:
    """Test basic configuration loading."""
    
    def test_load_valid_config(self):
        """Create temp YAML file with valid Phase 2 config and verify loading."""
        valid_config = """
materials_project:
  api_key: test_mp_key
  target_systems:
    - Si-C
    - B-C
  batch_size: 1000
  max_materials: 5000

jarvis:
  carbide_metals:
    - Si
    - B
    - W
  data_source: dft_3d

aflow:
  ceramics:
    SiC:
      - Si
      - C
  max_per_system: 500

semantic_scholar:
  queries:
    - ballistic limit velocity ceramic armor
  papers_per_query: 50
  min_citations: 5

matweb:
  target_materials:
    - Silicon Carbide
    - Boron Carbide
  rate_limit_delay: 2

nist_baseline:
  baseline_materials:
    SiC:
      density: 3.21
      hardness: 26.5
      K_IC: 4.0

integration:
  deduplication_threshold: 0.95
  source_priority:
    - Literature
    - MatWeb
    - NIST
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(valid_config)
            config_path = f.name
        
        try:
            # Load config
            config = ConfigLoader.load(config_path)
            
            # Verify all sections present
            assert 'materials_project' in config
            assert 'jarvis' in config
            assert 'aflow' in config
            assert 'semantic_scholar' in config
            assert 'matweb' in config
            assert 'nist_baseline' in config
            assert 'integration' in config
            
            # Verify specific values
            assert config['materials_project']['api_key'] == 'test_mp_key'
            assert config['materials_project']['batch_size'] == 1000
            assert 'Si-C' in config['materials_project']['target_systems']
            assert config['jarvis']['data_source'] == 'dft_3d'
            
        finally:
            # Cleanup
            os.unlink(config_path)


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution."""
    
    def test_environment_variable_substitution(self):
        """Create config with ${TEST_VAR} and verify substitution works."""
        config_with_env_var = """
materials_project:
  api_key: ${TEST_MP_API_KEY}
  target_systems:
    - Si-C

jarvis:
  carbide_metals:
    - Si
  data_source: dft_3d

aflow:
  ceramics:
    SiC:
      - Si
      - C

semantic_scholar:
  queries:
    - test query

matweb:
  target_materials:
    - Silicon Carbide

nist_baseline:
  baseline_materials:
    SiC:
      density: 3.21

integration:
  deduplication_threshold: 0.95
  source_priority:
    - Literature
"""
        
        # Set environment variable
        os.environ['TEST_MP_API_KEY'] = 'substituted_api_key_value'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_with_env_var)
            config_path = f.name
        
        try:
            # Load config
            config = ConfigLoader.load(config_path)
            
            # Verify substitution worked
            assert config['materials_project']['api_key'] == 'substituted_api_key_value'
            
        finally:
            # Cleanup
            os.unlink(config_path)
            del os.environ['TEST_MP_API_KEY']
    
    def test_multiple_env_var_substitution(self):
        """Test multiple environment variables in same config."""
        config_with_multiple_vars = """
materials_project:
  api_key: ${TEST_VAR_1}
  target_systems:
    - Si-C

jarvis:
  carbide_metals:
    - Si
  data_source: ${TEST_VAR_2}

aflow:
  ceramics:
    SiC:
      - Si
      - C

semantic_scholar:
  queries:
    - test query

matweb:
  target_materials:
    - Silicon Carbide

nist_baseline:
  baseline_materials:
    SiC:
      density: 3.21

integration:
  deduplication_threshold: 0.95
  source_priority:
    - Literature
"""
        
        # Set environment variables
        os.environ['TEST_VAR_1'] = 'value_1'
        os.environ['TEST_VAR_2'] = 'value_2'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_with_multiple_vars)
            config_path = f.name
        
        try:
            config = ConfigLoader.load(config_path)
            
            assert config['materials_project']['api_key'] == 'value_1'
            assert config['jarvis']['data_source'] == 'value_2'
            
        finally:
            os.unlink(config_path)
            del os.environ['TEST_VAR_1']
            del os.environ['TEST_VAR_2']


class TestErrorHandling:
    """Test error handling for invalid configurations."""
    
    def test_missing_config_file(self):
        """Call load() with non-existent path and verify FileNotFoundError."""
        non_existent_path = '/path/that/does/not/exist/config.yaml'
        
        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader.load(non_existent_path)
        
        error_msg = str(exc_info.value)
        assert "Configuration file not found" in error_msg
        assert "config.yaml" in error_msg
    
    def test_validation_missing_section(self):
        """Create config missing required section and verify ValueError."""
        incomplete_config = """
materials_project:
  api_key: test_key
  target_systems:
    - Si-C

# Missing jarvis, aflow, semantic_scholar, matweb, nist_baseline, integration sections
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(incomplete_config)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(config_path)
            
            error_msg = str(exc_info.value)
            assert "missing required sections" in error_msg.lower()
            
        finally:
            os.unlink(config_path)
    
    def test_validation_missing_api_key(self):
        """Create config with empty MP api_key and verify ValueError."""
        config_without_api_key = """
materials_project:
  target_systems:
    - Si-C
  # api_key is missing

jarvis:
  carbide_metals:
    - Si

aflow:
  ceramics:
    SiC:
      - Si
      - C

semantic_scholar:
  queries:
    - test

matweb:
  target_materials:
    - Silicon Carbide

nist_baseline:
  baseline_materials:
    SiC:
      density: 3.21

integration:
  deduplication_threshold: 0.95
  source_priority:
    - Literature
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_without_api_key)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(config_path)
            
            error_msg = str(exc_info.value)
            assert "api key" in error_msg.lower()
            
        finally:
            os.unlink(config_path)
    
    def test_validation_unresolved_env_var(self):
        """Create config with unresolved ${VAR} and verify ValueError."""
        config_with_unresolved_var = """
materials_project:
  api_key: ${UNSET_ENV_VAR_THAT_DOES_NOT_EXIST}
  target_systems:
    - Si-C

jarvis:
  carbide_metals:
    - Si

aflow:
  ceramics:
    SiC:
      - Si
      - C

semantic_scholar:
  queries:
    - test

matweb:
  target_materials:
    - Silicon Carbide

nist_baseline:
  baseline_materials:
    SiC:
      density: 3.21

integration:
  deduplication_threshold: 0.95
  source_priority:
    - Literature
"""
        
        # Make sure the env var is NOT set
        if 'UNSET_ENV_VAR_THAT_DOES_NOT_EXIST' in os.environ:
            del os.environ['UNSET_ENV_VAR_THAT_DOES_NOT_EXIST']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_with_unresolved_var)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(config_path)
            
            error_msg = str(exc_info.value)
            assert "not resolved" in error_msg.lower()
            
        finally:
            os.unlink(config_path)
    
    def test_empty_config_file(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            config_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                ConfigLoader.load(config_path)
            
            error_msg = str(exc_info.value)
            assert "empty" in error_msg.lower()
            
        finally:
            os.unlink(config_path)
