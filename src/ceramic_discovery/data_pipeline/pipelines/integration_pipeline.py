"""
Phase 2 Integration Pipeline

Merges all Phase 2 collected data sources into master dataset.

IMPORTANT: This is separate from existing DataCombiner (Phase 1 Task 5).
- Existing DataCombiner: Combines MP + JARVIS + NIST + Literature (Phase 1)
- This IntegrationPipeline: Combines Phase 2 collected data (MP + JARVIS + AFLOW + Semantic Scholar + MatWeb + NIST)

Both can coexist - this does NOT modify or replace existing DataCombiner.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..integration.formula_matcher import FormulaMatcher

logger = logging.getLogger(__name__)


class IntegrationPipeline:
    """
    Integrate all Phase 2 collected data sources.
    
    This is new Phase 2 functionality separate from existing Phase 1 DataCombiner.
    Does NOT modify existing data_combiner.py
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.matcher = FormulaMatcher()
        
        self.processed_dir = Path('data/processed')
        self.output_dir = Path('data/final')
        
        self.dedup_threshold = config.get('deduplication_threshold', 0.95)
        self.source_priority = config.get('source_priority', [
            'Literature', 'MatWeb', 'NIST', 'MaterialsProject', 'JARVIS', 'AFLOW'
        ])
    
    def integrate_all_sources(self) -> pd.DataFrame:
        """
        Integrate all Phase 2 data sources into master dataset.
        
        Returns:
            Master DataFrame with all sources merged
        """
        logger.info("Starting multi-source integration")
        
        # Load all source CSVs
        sources = self._load_all_sources()
        
        if not sources:
            raise ValueError("No data sources found in data/processed/")
        
        # Start with highest priority source or concatenate all
        df_master = self._initialize_master(sources)
        
        # Merge other sources with deduplication
        df_master = self._merge_sources(df_master, sources)
        
        # Create combined property columns
        df_master = self._create_combined_columns(df_master)
        
        # Calculate derived properties
        df_master = self._calculate_derived_properties(df_master)
        
        # Save master dataset
        self._save_master_dataset(df_master)
        
        # Generate integration report
        self._generate_report(df_master, sources)
        
        logger.info(f"Integration complete: {len(df_master)} materials in master dataset")
        
        return df_master
    
    def _load_all_sources(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from processed directory."""
        sources = {}
        
        if not self.processed_dir.exists():
            logger.warning(f"Processed directory does not exist: {self.processed_dir}")
            return sources
        
        for csv_file in self.processed_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                source_name = csv_file.stem
                sources[source_name] = df
                logger.info(f"Loaded {source_name}: {len(df)} entries")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
        
        return sources
    
    def _initialize_master(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Initialize master dataset with first source or concatenation."""
        # Try to start with Materials Project if available
        if 'mp_data' in sources:
            logger.info("Initializing with Materials Project data")
            return sources['mp_data'].copy()
        elif 'materials_project_data' in sources:
            logger.info("Initializing with Materials Project data")
            return sources['materials_project_data'].copy()
        else:
            # Concatenate all sources
            logger.info("Concatenating all sources")
            return pd.concat(sources.values(), ignore_index=True)
    
    def _merge_sources(self, df_master: pd.DataFrame, 
                      sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge additional sources with fuzzy formula matching."""
        
        # Skip sources already in master
        skip_sources = {'mp_data', 'materials_project_data'}
        
        for source_name, df_source in sources.items():
            if source_name in skip_sources:
                continue  # Already in master
            
            logger.info(f"Merging {source_name}...")
            
            # Ensure formula column exists
            if 'formula' not in df_source.columns:
                logger.warning(f"  {source_name} has no 'formula' column, skipping")
                continue
            
            new_materials = []
            merged_count = 0
            
            for _, row in df_source.iterrows():
                formula = row.get('formula', '')
                if not formula or pd.isna(formula):
                    continue
                
                # Find matching material in master
                if 'formula' in df_master.columns:
                    master_formulas = df_master['formula'].dropna().unique().tolist()
                    best_match, score = self.matcher.find_best_match(
                        str(formula), master_formulas, self.dedup_threshold
                    )
                    
                    if best_match:
                        # Merge properties into existing material
                        idx = df_master[df_master['formula'] == best_match].index[0]
                        for col in row.index:
                            # Merge source-specific columns
                            if pd.notna(row[col]) and col not in ['formula', 'source']:
                                # If column doesn't exist in master, add it
                                if col not in df_master.columns:
                                    df_master[col] = np.nan
                                # Only update if master value is NaN
                                if pd.isna(df_master.at[idx, col]):
                                    df_master.at[idx, col] = row[col]
                        merged_count += 1
                    else:
                        # New material, add to list
                        new_materials.append(row)
                else:
                    # No formula column in master yet, just add all
                    new_materials.append(row)
            
            # Append new materials
            if new_materials:
                df_new = pd.DataFrame(new_materials)
                df_master = pd.concat([df_master, df_new], ignore_index=True)
            
            logger.info(f"  Merged {merged_count} existing, added {len(new_materials)} new")
            logger.info(f"  After {source_name}: {len(df_master)} materials")
        
        return df_master
    
    def _create_combined_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined property columns averaging across sources."""
        
        # Hardness combined
        hardness_cols = [c for c in df.columns if 'hardness' in c.lower() and 
                        c not in ['hardness_combined', 'specific_hardness']]
        if hardness_cols:
            df['hardness_combined'] = df[hardness_cols].mean(axis=1, skipna=True)
            logger.info(f"Created hardness_combined from {len(hardness_cols)} sources: {hardness_cols}")
        
        # K_IC combined
        kic_cols = [c for c in df.columns if ('k_ic' in c.lower() or 
                   'fracture_toughness' in c.lower() or 'kic' in c.lower()) and 
                   c != 'K_IC_combined']
        if kic_cols:
            df['K_IC_combined'] = df[kic_cols].mean(axis=1, skipna=True)
            logger.info(f"Created K_IC_combined from {len(kic_cols)} sources: {kic_cols}")
        
        # Density combined (prefer MP, then others)
        density_cols = [c for c in df.columns if 'density' in c.lower() and c != 'density_combined']
        if density_cols:
            # Try priority order
            density_priority = ['density_mp', 'density_jarvis', 'density_aflow', 
                              'density_matweb', 'density']
            available_priority = [c for c in density_priority if c in df.columns]
            
            if available_priority:
                df['density_combined'] = df[available_priority].bfill(axis=1).iloc[:, 0]
            else:
                # Just use mean of all density columns
                df['density_combined'] = df[density_cols].mean(axis=1, skipna=True)
            
            logger.info(f"Created density_combined from {len(density_cols)} sources")
        
        return df
    
    def _calculate_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived ballistic-relevant properties."""
        
        # Specific hardness (hardness / density)
        if 'hardness_combined' in df.columns and 'density_combined' in df.columns:
            df['specific_hardness'] = df['hardness_combined'] / df['density_combined']
            valid_count = df['specific_hardness'].notna().sum()
            logger.info(f"Calculated specific_hardness ({valid_count} values)")
        
        # Ballistic efficacy (simplified: hardness / density)
        if 'hardness_combined' in df.columns and 'density_combined' in df.columns:
            df['ballistic_efficacy'] = df['hardness_combined'] / df['density_combined']
            valid_count = df['ballistic_efficacy'].notna().sum()
            logger.info(f"Calculated ballistic_efficacy ({valid_count} values)")
        
        # DOP resistance (depth of penetration resistance)
        if all(c in df.columns for c in ['hardness_combined', 'K_IC_combined', 'density_combined']):
            df['dop_resistance'] = (
                np.sqrt(df['hardness_combined']) * 
                df['K_IC_combined'] / 
                df['density_combined']
            )
            valid_count = df['dop_resistance'].notna().sum()
            logger.info(f"Calculated dop_resistance ({valid_count} values)")
        
        return df
    
    def _save_master_dataset(self, df: pd.DataFrame):
        """Save master dataset to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.output_dir / 'master_dataset.csv'
        df.to_csv(output_file, index=False)
        
        logger.info(f"Master dataset saved: {output_file}")
        logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    def _generate_report(self, df: pd.DataFrame, sources: Dict[str, pd.DataFrame]):
        """Generate comprehensive integration report."""
        
        report_lines = [
            "=" * 80,
            "PHASE 2 DATA INTEGRATION REPORT",
            "=" * 80,
            "",
            "DATASET SUMMARY",
            "-" * 80,
            f"Total materials: {len(df)}",
            f"Unique formulas: {df['formula'].nunique() if 'formula' in df.columns else 'N/A'}",
            f"Total columns: {len(df.columns)}",
            "",
            "DATA SOURCES",
            "-" * 80,
        ]
        
        for source_name, source_df in sources.items():
            report_lines.append(f"  {source_name}: {len(source_df)} entries")
        
        # Property coverage
        report_lines.extend([
            "",
            "PROPERTY COVERAGE",
            "-" * 80,
        ])
        
        # Check for combined properties
        if 'hardness_combined' in df.columns:
            count = df['hardness_combined'].notna().sum()
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"  With hardness_combined: {count} ({pct:.1f}%)")
        
        if 'K_IC_combined' in df.columns:
            count = df['K_IC_combined'].notna().sum()
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"  With K_IC_combined: {count} ({pct:.1f}%)")
        
        if 'density_combined' in df.columns:
            count = df['density_combined'].notna().sum()
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"  With density_combined: {count} ({pct:.1f}%)")
        
        # Check for source-specific properties
        if 'v50_literature' in df.columns:
            count = df['v50_literature'].notna().sum()
            report_lines.append(f"  With V50 (literature): {count}")
        
        if 'thermal_conductivity_300K' in df.columns:
            count = df['thermal_conductivity_300K'].notna().sum()
            report_lines.append(f"  With thermal_conductivity: {count}")
        
        # Derived properties
        report_lines.extend([
            "",
            "DERIVED PROPERTIES",
            "-" * 80,
        ])
        
        if 'specific_hardness' in df.columns:
            count = df['specific_hardness'].notna().sum()
            report_lines.append(f"  Specific hardness: {count}")
        
        if 'ballistic_efficacy' in df.columns:
            count = df['ballistic_efficacy'].notna().sum()
            report_lines.append(f"  Ballistic efficacy: {count}")
        
        if 'dop_resistance' in df.columns:
            count = df['dop_resistance'].notna().sum()
            report_lines.append(f"  DOP resistance: {count}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "Integration complete!",
            "=" * 80
        ])
        
        # Write report
        report_file = self.output_dir / 'integration_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Report saved: {report_file}")
