"""
ConfigLoader Utility

Loads and validates Phase 2 data pipeline configuration from YAML files.
Supports environment variable substitution for sensitive values like API keys.

Example:
    config = ConfigLoader.load('config/data_pipeline_config.yaml')
    mp_api_key = config['materials_project']['api_key']
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader with environment variable substitution.
    
    Features:
    - Loads YAML configuration files
    - Substitutes environment variables (${VAR_NAME} syntax)
    - Validates required configuration sections
    - Provides helpful error messages
    """
    
    @staticmethod
    def load(config_path: str | Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary with environment variables substituted
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected location: {config_path.absolute()}\n"
                f"Please create the configuration file or check the path."
            )
        
        logger.info(f"Loading configuration from {config_path}")
        
        # Load YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
        
        # Substitute environment variables
        config = ConfigLoader._substitute_env_vars(config)
        
        # Validate configuration
        ConfigLoader._validate_config(config)
        
        logger.info("Configuration loaded and validated successfully")
        return config
    
    @staticmethod
    def _substitute_env_vars(config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports ${VAR_NAME} syntax. If environment variable is not set,
        keeps the original ${VAR_NAME} string.
        
        Args:
            config: Configuration value (dict, list, str, or other)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {k: ConfigLoader._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Pattern: ${VAR_NAME}
            pattern = r'\$\{([^}]+)\}'
            
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is not None:
                    return env_value
                else:
                    # Keep original if not found
                    logger.warning(
                        f"Environment variable ${{{var_name}}} not set, "
                        f"keeping original value"
                    )
                    return match.group(0)
            
            return re.sub(pattern, replace_env_var, config)
        else:
            return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration has required sections and keys.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required top-level sections
        required_sections = [
            'materials_project',
            'jarvis',
            'aflow',
            'semantic_scholar',
            'matweb',
            'nist_baseline',
            'integration'
        ]
        
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            raise ValueError(
                f"Configuration missing required sections: {missing_sections}\n"
                f"Required sections: {required_sections}"
            )
        
        # Validate Materials Project section
        mp_config = config.get('materials_project', {})
        if not mp_config.get('api_key'):
            raise ValueError(
                "Materials Project API key is required.\n"
                "Set 'api_key' in materials_project section or "
                "set MP_API_KEY environment variable."
            )
        
        # Check if API key is still a placeholder
        api_key = mp_config.get('api_key', '')
        if api_key.startswith('${'):
            raise ValueError(
                f"Materials Project API key not resolved: {api_key}\n"
                "Please set the corresponding environment variable."
            )
        
        logger.debug("Configuration validation passed")
