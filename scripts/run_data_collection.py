#!/usr/bin/env python3
"""
Phase 2 Data Collection Pipeline - Main Execution Script

This script runs the complete multi-source data collection pipeline with
comprehensive safety checks to ensure Phase 1 functionality remains intact.

CRITICAL SAFETY FEATURES:
- Verifies all Phase 1 tests pass BEFORE starting collection
- Verifies all Phase 1 tests pass AFTER collection completes
- Checks all required dependencies are installed
- Provides detailed logging and error reporting

Usage:
    # Run complete pipeline with all sources
    python scripts/run_data_collection.py
    
    # Run specific sources only
    python scripts/run_data_collection.py --sources mp,jarvis,aflow
    
    # Skip Phase 1 verification (NOT RECOMMENDED)
    python scripts/run_data_collection.py --skip-verification
    
    # Use custom config file
    python scripts/run_data_collection.py --config path/to/config.yaml

Requirements:
    - All Phase 1 tests must pass (157+ tests)
    - All Phase 2 dependencies must be installed
    - Configuration file must be valid

Output:
    - data/processed/{source}_data.csv for each source
    - data/final/master_dataset.csv (main output)
    - data/final/integration_report.txt
    - data/final/pipeline_summary.txt
    - logs/collection_{timestamp}.log
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import importlib.util

# Setup logging
def setup_logging(log_dir: Path = Path('logs')) -> logging.Logger:
    """
    Configure logging to both file and console.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'collection_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('data_collection')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


def verify_dependencies() -> List[str]:
    """
    Verify all required Phase 2 dependencies are installed.
    
    Returns:
        List of missing package names (empty if all present)
    """
    required_packages = [
        'pandas',
        'numpy',
        'pyyaml',
        'tqdm',
        'requests',
        'beautifulsoup4',
        'lxml',
        # Phase 2 specific
        'aflow',
        'semanticscholar',
    ]
    
    missing = []
    
    for package in required_packages:
        # Handle package name variations
        import_name = package
        if package == 'beautifulsoup4':
            import_name = 'bs4'
        elif package == 'pyyaml':
            import_name = 'yaml'
        
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(package)
    
    return missing


def verify_phase1_tests(logger: logging.Logger) -> bool:
    """
    Run Phase 1 test suite to verify existing functionality is intact.
    
    Args:
        logger: Logger instance
        
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("=" * 70)
    logger.info("PHASE 1 TEST VERIFICATION")
    logger.info("=" * 70)
    logger.info("Running Phase 1 test suite to verify existing functionality...")
    logger.info("This may take a few minutes...")
    
    try:
        # Run pytest with verbose output and stop on first failure
        result = subprocess.run(
            ['pytest', 'tests/', '-v', '--tb=short', '-x'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Log output
        logger.debug("STDOUT:")
        logger.debug(result.stdout)
        
        if result.returncode == 0:
            logger.info("✓ All Phase 1 tests PASSED")
            logger.info("=" * 70)
            return True
        else:
            logger.error("✗ Phase 1 tests FAILED")
            logger.error("=" * 70)
            logger.error("STDERR:")
            logger.error(result.stderr)
            logger.error("\nFailed test output:")
            logger.error(result.stdout)
            logger.error("=" * 70)
            logger.error("\nPHASE 1 INTEGRITY COMPROMISED!")
            logger.error("Do NOT proceed with Phase 2 implementation.")
            logger.error("Fix Phase 1 tests before continuing.")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Phase 1 tests TIMED OUT (>10 minutes)")
        logger.error("This may indicate a hanging test or system issue.")
        return False
    except FileNotFoundError:
        logger.error("✗ pytest not found. Install with: pip install pytest")
        return False
    except Exception as e:
        logger.error(f"✗ Error running Phase 1 tests: {e}")
        return False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run Phase 2 Multi-Source Data Collection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with all sources
  python scripts/run_data_collection.py
  
  # Run specific sources only
  python scripts/run_data_collection.py --sources mp,jarvis,aflow
  
  # Skip Phase 1 verification (NOT RECOMMENDED)
  python scripts/run_data_collection.py --skip-verification
  
  # Use custom config file
  python scripts/run_data_collection.py --config path/to/config.yaml

Output Files:
  - data/processed/{source}_data.csv for each source
  - data/final/master_dataset.csv (main output)
  - data/final/integration_report.txt
  - data/final/pipeline_summary.txt
  - logs/collection_{timestamp}.log
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_pipeline_config.yaml',
        help='Path to Phase 2 configuration file (default: config/data_pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip Phase 1 test verification (NOT RECOMMENDED - use only for debugging)'
    )
    
    parser.add_argument(
        '--sources',
        type=str,
        default=None,
        help='Comma-separated list of sources to run (e.g., mp,jarvis,aflow). '
             'Available: mp, jarvis, aflow, scholar, matweb, nist. '
             'Default: all sources'
    )
    
    return parser.parse_args()


def filter_sources(config: Dict[str, Any], sources_str: Optional[str], logger: logging.Logger) -> Dict[str, Any]:
    """
    Filter configuration to only include specified sources.
    
    Args:
        config: Full configuration dictionary
        sources_str: Comma-separated source names (e.g., "mp,jarvis,aflow")
        logger: Logger instance
        
    Returns:
        Modified configuration with only specified sources enabled
    """
    if sources_str is None:
        logger.info("Running all sources")
        return config
    
    # Parse source list
    requested_sources = [s.strip().lower() for s in sources_str.split(',')]
    
    # Map short names to config keys
    source_map = {
        'mp': 'materials_project',
        'jarvis': 'jarvis',
        'aflow': 'aflow',
        'scholar': 'semantic_scholar',
        'matweb': 'matweb',
        'nist': 'nist_baseline'
    }
    
    # Validate sources
    valid_sources = set(source_map.keys())
    invalid = set(requested_sources) - valid_sources
    if invalid:
        logger.warning(f"Invalid sources specified: {invalid}")
        logger.warning(f"Valid sources: {', '.join(valid_sources)}")
    
    # Filter config
    enabled_sources = []
    for short_name, config_key in source_map.items():
        if short_name in requested_sources:
            enabled_sources.append(config_key)
            logger.info(f"  ✓ {config_key} enabled")
        else:
            # Disable in config by setting enabled=False
            if config_key in config:
                config[config_key]['enabled'] = False
                logger.info(f"  ✗ {config_key} disabled")
    
    if not enabled_sources:
        logger.error("No valid sources specified!")
        sys.exit(1)
    
    return config


def main():
    """
    Main execution function.
    
    Workflow:
        1. Setup logging
        2. Parse arguments
        3. Verify dependencies
        4. Verify Phase 1 tests BEFORE (unless skipped)
        5. Load Phase 2 configuration
        6. Filter sources if specified
        7. Run CollectionOrchestrator
        8. Verify Phase 1 tests AFTER (unless skipped)
        9. Report success or failure
    """
    # Step 1: Setup logging
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("PHASE 2 DATA COLLECTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Step 2: Parse arguments
    args = parse_arguments()
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Skip verification: {args.skip_verification}")
    logger.info(f"Sources filter: {args.sources if args.sources else 'all'}")
    logger.info("")
    
    # Step 3: Verify dependencies
    logger.info("Step 1/7: Verifying dependencies...")
    missing = verify_dependencies()
    if missing:
        logger.error(f"✗ Missing required packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        sys.exit(1)
    logger.info("✓ All dependencies installed")
    logger.info("")
    
    # Step 4: Verify Phase 1 tests BEFORE
    if not args.skip_verification:
        logger.info("Step 2/7: Verifying Phase 1 tests BEFORE collection...")
        if not verify_phase1_tests(logger):
            logger.error("\n" + "!" * 70)
            logger.error("CRITICAL: Phase 1 tests are failing!")
            logger.error("Fix Phase 1 before running Phase 2 collection.")
            logger.error("!" * 70)
            sys.exit(1)
        logger.info("")
    else:
        logger.warning("Step 2/7: SKIPPED Phase 1 verification (--skip-verification)")
        logger.warning("This is NOT RECOMMENDED. Proceed at your own risk.")
        logger.info("")
    
    # Step 5: Load Phase 2 configuration
    logger.info("Step 3/7: Loading Phase 2 configuration...")
    try:
        from ceramic_discovery.data_pipeline.utils import ConfigLoader
        
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"✗ Configuration file not found: {config_path}")
            sys.exit(1)
        
        config = ConfigLoader.load(str(config_path))
        logger.info(f"✓ Configuration loaded from {config_path}")
        logger.info("")
    except ImportError as e:
        logger.error(f"✗ Cannot import ConfigLoader: {e}")
        logger.error("Ensure Phase 2 data_pipeline module is installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error loading configuration: {e}")
        sys.exit(1)
    
    # Step 6: Filter sources if specified
    logger.info("Step 4/7: Configuring sources...")
    config = filter_sources(config, args.sources, logger)
    logger.info("")
    
    # Step 7: Run CollectionOrchestrator
    logger.info("Step 5/7: Running data collection pipeline...")
    logger.info("=" * 70)
    try:
        from ceramic_discovery.data_pipeline.pipelines import CollectionOrchestrator
        
        orchestrator = CollectionOrchestrator(config)
        master_path = orchestrator.run_all()
        
        logger.info("=" * 70)
        logger.info(f"✓ Pipeline completed successfully!")
        logger.info(f"✓ Master dataset saved to: {master_path}")
        logger.info("")
    except ImportError as e:
        logger.error(f"✗ Cannot import CollectionOrchestrator: {e}")
        logger.error("Ensure Phase 2 data_pipeline module is installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    
    # Step 8: Verify Phase 1 tests AFTER
    if not args.skip_verification:
        logger.info("Step 6/7: Verifying Phase 1 tests AFTER collection...")
        if not verify_phase1_tests(logger):
            logger.error("\n" + "!" * 70)
            logger.error("CRITICAL: Phase 1 tests are NOW failing!")
            logger.error("Phase 2 collection may have broken existing functionality.")
            logger.error("Investigate and fix immediately.")
            logger.error("!" * 70)
            sys.exit(1)
        logger.info("")
    else:
        logger.warning("Step 6/7: SKIPPED Phase 1 verification (--skip-verification)")
        logger.info("")
    
    # Step 9: Report success
    logger.info("Step 7/7: Final report")
    logger.info("=" * 70)
    logger.info("✓ PHASE 2 DATA COLLECTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - Master dataset: data/final/master_dataset.csv")
    logger.info(f"  - Integration report: data/final/integration_report.txt")
    logger.info(f"  - Pipeline summary: data/final/pipeline_summary.txt")
    logger.info(f"  - Individual source CSVs: data/processed/")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review integration report for data quality")
    logger.info("  2. Validate output with: python scripts/validate_collection_output.py")
    logger.info("  3. Use master_dataset.csv with Phase 1 ML pipeline")
    logger.info("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
