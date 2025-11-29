"""
Phase 2 Collection Orchestrator

Runs all Phase 2 data collectors in sequence.

IMPORTANT: This does NOT modify existing Phase 1 code.
It creates a new master_dataset.csv that Phase 1 ML pipeline can consume.
"""

import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

from ..collectors.materials_project_collector import MaterialsProjectCollector
from ..collectors.jarvis_collector import JarvisCollector
from ..collectors.aflow_collector import AFLOWCollector
from ..collectors.semantic_scholar_collector import SemanticScholarCollector
from ..collectors.matweb_collector import MatWebCollector
from ..collectors.nist_baseline_collector import NISTBaselineCollector
from .integration_pipeline import IntegrationPipeline

logger = logging.getLogger(__name__)


class CollectionOrchestrator:
    """
    Orchestrate all Phase 2 data collection.
    
    This is new Phase 2 functionality that does NOT modify existing Phase 1 code.
    
    Attributes:
        config: Configuration dictionary
        processed_dir: Directory for intermediate CSV files
        collectors: Dictionary of initialized collectors
        results: Dictionary tracking execution results
    """
    
    def __init__(self, config: Dict):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary with sections for each collector
        """
        self.config = config
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.collectors = self._initialize_collectors()
        self.results = {}
    
    def _initialize_collectors(self) -> Dict:
        """
        Initialize all collectors from config.
        
        Returns:
            Dictionary mapping collector names to collector instances
        """
        collectors = {}
        
        # Materials Project
        if self.config.get('materials_project', {}).get('enabled', True):
            try:
                collectors['mp'] = MaterialsProjectCollector(
                    self.config['materials_project']
                )
                logger.info("✓ Initialized Materials Project collector")
            except Exception as e:
                logger.warning(f"Could not initialize Materials Project collector: {e}")
        
        # JARVIS
        if self.config.get('jarvis', {}).get('enabled', True):
            try:
                collectors['jarvis'] = JarvisCollector(
                    self.config['jarvis']
                )
                logger.info("✓ Initialized JARVIS collector")
            except Exception as e:
                logger.warning(f"Could not initialize JARVIS collector: {e}")
        
        # AFLOW
        if self.config.get('aflow', {}).get('enabled', True):
            try:
                collectors['aflow'] = AFLOWCollector(
                    self.config['aflow']
                )
                logger.info("✓ Initialized AFLOW collector")
            except Exception as e:
                logger.warning(f"Could not initialize AFLOW collector: {e}")
        
        # Semantic Scholar
        if self.config.get('semantic_scholar', {}).get('enabled', True):
            try:
                collectors['semantic_scholar'] = SemanticScholarCollector(
                    self.config['semantic_scholar']
                )
                logger.info("✓ Initialized Semantic Scholar collector")
            except Exception as e:
                logger.warning(f"Could not initialize Semantic Scholar collector: {e}")
        
        # MatWeb
        if self.config.get('matweb', {}).get('enabled', True):
            try:
                collectors['matweb'] = MatWebCollector(
                    self.config['matweb']
                )
                logger.info("✓ Initialized MatWeb collector")
            except Exception as e:
                logger.warning(f"Could not initialize MatWeb collector: {e}")
        
        # NIST Baseline
        if self.config.get('nist', {}).get('enabled', True):
            try:
                collectors['nist'] = NISTBaselineCollector(
                    self.config['nist']
                )
                logger.info("✓ Initialized NIST Baseline collector")
            except Exception as e:
                logger.warning(f"Could not initialize NIST Baseline collector: {e}")
        
        logger.info(f"Initialized {len(collectors)} collectors")
        return collectors
    
    def run_all(self) -> Path:
        """
        Run all collectors and integration.
        
        Executes collectors in order: MP → JARVIS → AFLOW → Scholar → MatWeb → NIST → Integration
        
        Returns:
            Path to master_dataset.csv
            
        Raises:
            Exception: If integration fails (critical step)
        """
        logger.info("=" * 80)
        logger.info("Starting Phase 2 data collection pipeline")
        logger.info("=" * 80)
        start_time = time.time()
        
        # Run collectors with progress bar
        collector_names = list(self.collectors.keys())
        with tqdm(total=len(collector_names), desc="Collection Progress", unit="source") as pbar:
            for name in collector_names:
                collector = self.collectors[name]
                logger.info(f"\n{'='*60}")
                logger.info(f"Running {name} collector...")
                logger.info(f"{'='*60}")
                
                try:
                    collector_start = time.time()
                    
                    # Collect data
                    df = collector.collect()
                    
                    # Save checkpoint to processed/
                    output_file = self.processed_dir / f"{name}_data.csv"
                    df.to_csv(output_file, index=False)
                    
                    collector_time = time.time() - collector_start
                    
                    self.results[name] = {
                        'status': 'SUCCESS',
                        'count': len(df),
                        'runtime': collector_time,
                        'output': str(output_file)
                    }
                    
                    logger.info(f"✓ {name}: {len(df)} materials collected in {collector_time:.1f}s")
                    logger.info(f"  Saved to: {output_file}")
                    
                except Exception as e:
                    logger.error(f"✗ {name} collector failed: {e}", exc_info=True)
                    self.results[name] = {
                        'status': 'FAILED',
                        'error': str(e),
                        'runtime': 0
                    }
                
                pbar.update(1)
        
        # Integration (critical step - must succeed)
        logger.info(f"\n{'='*60}")
        logger.info("Running integration pipeline...")
        logger.info(f"{'='*60}")
        
        try:
            integration_start = time.time()
            
            integration = IntegrationPipeline(self.config.get('integration', {}))
            master_path = integration.integrate_all_sources()
            
            integration_time = time.time() - integration_start
            
            self.results['integration'] = {
                'status': 'SUCCESS',
                'output': str(master_path),
                'runtime': integration_time
            }
            
            logger.info(f"✓ Integration completed in {integration_time:.1f}s")
            logger.info(f"  Master dataset: {master_path}")
            
        except Exception as e:
            logger.error(f"✗ Integration failed: {e}", exc_info=True)
            self.results['integration'] = {
                'status': 'FAILED',
                'error': str(e),
                'runtime': 0
            }
            raise  # Integration failure is critical - abort pipeline
        
        # Generate final summary report
        total_time = time.time() - start_time
        self._generate_summary_report(total_time)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE COMPLETE in {total_time/60:.1f} minutes")
        logger.info(f"{'='*80}")
        
        return Path('data/final/master_dataset.csv')
    
    def _generate_summary_report(self, total_time: float):
        """
        Generate pipeline execution summary report.
        
        Args:
            total_time: Total pipeline execution time in seconds
        """
        report_lines = [
            "=" * 80,
            "PHASE 2 DATA COLLECTION PIPELINE SUMMARY",
            f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "COLLECTOR RESULTS",
            "-" * 80,
        ]
        
        # Add results for each collector
        for name, result in self.results.items():
            status = result['status']
            if status == 'SUCCESS':
                if name == 'integration':
                    runtime = result.get('runtime', 0)
                    output = result.get('output', 'N/A')
                    report_lines.append(f"  ✓ {name:20s} SUCCESS  {runtime:6.1f}s  {output}")
                else:
                    count = result.get('count', 0)
                    runtime = result.get('runtime', 0)
                    output = result.get('output', 'N/A')
                    report_lines.append(f"  ✓ {name:20s} {count:5d} materials  {runtime:6.1f}s  {output}")
            else:
                error = result.get('error', 'Unknown error')
                report_lines.append(f"  ✗ {name:20s} FAILED: {error}")
        
        report_lines.extend([
            "",
            "-" * 80,
            f"Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)",
            "=" * 80
        ])
        
        # Write report to file
        report_dir = Path('data/final')
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / 'pipeline_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
