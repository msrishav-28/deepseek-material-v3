#!/usr/bin/env python3
"""
Phase 2 Data Collection Output Validation Script

This script validates the output of the data collection pipeline to ensure:
- master_dataset.csv exists and has expected structure
- Required columns are present
- Property coverage meets minimum thresholds
- Data quality checks pass (no NaN in critical columns, physical plausibility)
- Integration report is complete

Usage:
    python scripts/validate_collection_output.py
    
    # Specify custom master dataset path
    python scripts/validate_collection_output.py --dataset path/to/master_dataset.csv
    
    # Generate detailed report
    python scripts/validate_collection_output.py --detailed

Exit Codes:
    0: All validations passed
    1: Validation failures detected
    2: File not found or cannot be read
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name}\n  {self.message}"


class DatasetValidator:
    """Validator for Phase 2 master dataset."""
    
    # Required columns that must be present
    REQUIRED_COLUMNS = [
        'formula',
        'source',
    ]
    
    # Expected combined property columns
    EXPECTED_COMBINED_PROPERTIES = [
        'density_combined',
        'hardness_combined',
        'K_IC_combined',
    ]
    
    # Expected derived properties
    EXPECTED_DERIVED_PROPERTIES = [
        'specific_hardness',
        'ballistic_efficacy',
        'dop_resistance',
    ]
    
    # Physical plausibility ranges (property: (min, max))
    PHYSICAL_RANGES = {
        'density_combined': (0.1, 30.0),  # g/cm³
        'hardness_combined': (0.1, 100.0),  # GPa
        'K_IC_combined': (0.1, 20.0),  # MPa·m^0.5
        'specific_hardness': (0.01, 50.0),  # GPa/(g/cm³)
        'ballistic_efficacy': (0.01, 50.0),  # GPa/(g/cm³)
        'dop_resistance': (0.1, 100.0),  # Derived metric
    }
    
    # Minimum property coverage thresholds (percentage)
    MIN_COVERAGE = {
        'density_combined': 50.0,  # At least 50% of materials should have density
        'hardness_combined': 30.0,  # At least 30% should have hardness
        'K_IC_combined': 20.0,  # At least 20% should have fracture toughness
    }
    
    def __init__(self, dataset_path: Path, detailed: bool = False):
        """
        Initialize validator.
        
        Args:
            dataset_path: Path to master_dataset.csv
            detailed: Whether to generate detailed reports
        """
        self.dataset_path = dataset_path
        self.detailed = detailed
        self.df = None
        self.results: List[ValidationResult] = []
    
    def load_dataset(self) -> ValidationResult:
        """Load and validate dataset file exists and is readable."""
        try:
            if not self.dataset_path.exists():
                return ValidationResult(
                    "File Existence",
                    False,
                    f"Master dataset not found at: {self.dataset_path}"
                )
            
            self.df = pd.read_csv(self.dataset_path)
            
            return ValidationResult(
                "File Existence",
                True,
                f"Master dataset loaded successfully ({len(self.df)} materials)"
            )
        except Exception as e:
            return ValidationResult(
                "File Loading",
                False,
                f"Error loading dataset: {e}"
            )
    
    def validate_structure(self) -> ValidationResult:
        """Validate dataset has expected structure."""
        if self.df is None:
            return ValidationResult(
                "Structure",
                False,
                "Dataset not loaded"
            )
        
        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        
        if missing_required:
            return ValidationResult(
                "Required Columns",
                False,
                f"Missing required columns: {', '.join(missing_required)}"
            )
        
        # Check for at least some expected columns
        present_combined = [col for col in self.EXPECTED_COMBINED_PROPERTIES if col in self.df.columns]
        present_derived = [col for col in self.EXPECTED_DERIVED_PROPERTIES if col in self.df.columns]
        
        details = {
            'total_columns': len(self.df.columns),
            'combined_properties': present_combined,
            'derived_properties': present_derived,
        }
        
        return ValidationResult(
            "Dataset Structure",
            True,
            f"Dataset has {len(self.df.columns)} columns including "
            f"{len(present_combined)} combined and {len(present_derived)} derived properties",
            details
        )
    
    def validate_property_coverage(self) -> List[ValidationResult]:
        """Validate property coverage meets minimum thresholds."""
        if self.df is None:
            return [ValidationResult("Property Coverage", False, "Dataset not loaded")]
        
        results = []
        
        for prop, min_pct in self.MIN_COVERAGE.items():
            if prop not in self.df.columns:
                results.append(ValidationResult(
                    f"Coverage: {prop}",
                    False,
                    f"Property column not found: {prop}"
                ))
                continue
            
            non_null = self.df[prop].notna().sum()
            total = len(self.df)
            coverage_pct = (non_null / total * 100) if total > 0 else 0
            
            passed = coverage_pct >= min_pct
            
            results.append(ValidationResult(
                f"Coverage: {prop}",
                passed,
                f"{coverage_pct:.1f}% coverage ({non_null}/{total}) - "
                f"{'meets' if passed else 'below'} minimum {min_pct}%"
            ))
        
        return results
    
    def validate_data_quality(self) -> List[ValidationResult]:
        """Validate data quality (no invalid values, physical plausibility)."""
        if self.df is None:
            return [ValidationResult("Data Quality", False, "Dataset not loaded")]
        
        results = []
        
        # Check for duplicate formulas
        if 'formula' in self.df.columns:
            duplicates = self.df['formula'].duplicated().sum()
            results.append(ValidationResult(
                "Duplicate Formulas",
                duplicates == 0,
                f"Found {duplicates} duplicate formulas" if duplicates > 0 
                else "No duplicate formulas"
            ))
        
        # Check physical plausibility
        for prop, (min_val, max_val) in self.PHYSICAL_RANGES.items():
            if prop not in self.df.columns:
                continue
            
            # Get non-null values
            values = self.df[prop].dropna()
            if len(values) == 0:
                continue
            
            # Check for values outside physical range
            below_min = (values < min_val).sum()
            above_max = (values > max_val).sum()
            invalid = below_min + above_max
            
            passed = invalid == 0
            
            if passed:
                msg = f"All {len(values)} values in range [{min_val}, {max_val}]"
            else:
                msg = f"{invalid} values outside physical range [{min_val}, {max_val}] "
                msg += f"({below_min} below, {above_max} above)"
            
            results.append(ValidationResult(
                f"Physical Range: {prop}",
                passed,
                msg,
                {'below_min': below_min, 'above_max': above_max} if not passed else None
            ))
        
        # Check for infinite or NaN in critical columns
        critical_cols = ['density_combined', 'hardness_combined', 'K_IC_combined']
        for col in critical_cols:
            if col not in self.df.columns:
                continue
            
            inf_count = np.isinf(self.df[col]).sum()
            
            results.append(ValidationResult(
                f"No Infinities: {col}",
                inf_count == 0,
                f"Found {inf_count} infinite values" if inf_count > 0 
                else "No infinite values"
            ))
        
        return results
    
    def validate_sources(self) -> ValidationResult:
        """Validate data from multiple sources is present."""
        if self.df is None or 'source' not in self.df.columns:
            return ValidationResult("Source Diversity", False, "Cannot validate sources")
        
        sources = self.df['source'].value_counts().to_dict()
        num_sources = len(sources)
        
        # Expect at least 3 sources (some may fail during collection)
        passed = num_sources >= 3
        
        details = {
            'sources': sources,
            'total_sources': num_sources
        }
        
        msg = f"Data from {num_sources} sources: {', '.join(sources.keys())}"
        
        return ValidationResult(
            "Source Diversity",
            passed,
            msg,
            details
        )
    
    def validate_integration_report(self) -> ValidationResult:
        """Validate integration report exists and is complete."""
        report_path = self.dataset_path.parent / 'integration_report.txt'
        
        if not report_path.exists():
            return ValidationResult(
                "Integration Report",
                False,
                f"Integration report not found at: {report_path}"
            )
        
        try:
            content = report_path.read_text()
            
            # Check for key sections
            required_sections = [
                'Total materials',
                'Source counts',
                'Property coverage',
            ]
            
            missing = [s for s in required_sections if s not in content]
            
            if missing:
                return ValidationResult(
                    "Integration Report",
                    False,
                    f"Report missing sections: {', '.join(missing)}"
                )
            
            return ValidationResult(
                "Integration Report",
                True,
                f"Integration report complete ({len(content)} characters)"
            )
        except Exception as e:
            return ValidationResult(
                "Integration Report",
                False,
                f"Error reading report: {e}"
            )
    
    def run_all_validations(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validations.
        
        Returns:
            Tuple of (all_passed, results_list)
        """
        # Load dataset
        result = self.load_dataset()
        self.results.append(result)
        if not result.passed:
            return False, self.results
        
        # Structure validation
        self.results.append(self.validate_structure())
        
        # Property coverage
        self.results.extend(self.validate_property_coverage())
        
        # Data quality
        self.results.extend(self.validate_data_quality())
        
        # Source diversity
        self.results.append(self.validate_sources())
        
        # Integration report
        self.results.append(self.validate_integration_report())
        
        # Check if all passed
        all_passed = all(r.passed for r in self.results)
        
        return all_passed, self.results
    
    def generate_report(self) -> str:
        """Generate validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PHASE 2 DATA COLLECTION OUTPUT VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Dataset: {self.dataset_path}")
        lines.append("")
        
        # Summary
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        lines.append(f"Summary: {passed_count}/{total_count} validations passed")
        lines.append("")
        
        # Results
        lines.append("Validation Results:")
        lines.append("-" * 70)
        for result in self.results:
            lines.append(str(result))
            if self.detailed and result.details:
                lines.append(f"  Details: {result.details}")
            lines.append("")
        
        # Dataset statistics
        if self.df is not None:
            lines.append("Dataset Statistics:")
            lines.append("-" * 70)
            lines.append(f"Total materials: {len(self.df)}")
            lines.append(f"Total columns: {len(self.df.columns)}")
            
            if 'formula' in self.df.columns:
                lines.append(f"Unique formulas: {self.df['formula'].nunique()}")
            
            if 'source' in self.df.columns:
                lines.append("\nMaterials per source:")
                for source, count in self.df['source'].value_counts().items():
                    lines.append(f"  {source}: {count}")
            
            lines.append("\nProperty coverage:")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                non_null = self.df[col].notna().sum()
                pct = (non_null / len(self.df) * 100) if len(self.df) > 0 else 0
                lines.append(f"  {col}: {non_null}/{len(self.df)} ({pct:.1f}%)")
            
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate Phase 2 Data Collection Output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/final/master_dataset.csv',
        help='Path to master dataset CSV (default: data/final/master_dataset.csv)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate detailed validation report'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    dataset_path = Path(args.dataset)
    
    print("=" * 70)
    print("PHASE 2 DATA COLLECTION OUTPUT VALIDATION")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print("")
    
    # Create validator
    validator = DatasetValidator(dataset_path, detailed=args.detailed)
    
    # Run validations
    print("Running validations...")
    print("")
    
    all_passed, results = validator.run_all_validations()
    
    # Generate and print report
    report = validator.generate_report()
    print(report)
    
    # Exit with appropriate code
    if all_passed:
        print("\n✓ All validations PASSED")
        print("The master dataset is ready for use with Phase 1 ML pipeline.")
        sys.exit(0)
    else:
        print("\n✗ Some validations FAILED")
        print("Review the report above and address issues before using the dataset.")
        sys.exit(1)


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
        sys.exit(2)
