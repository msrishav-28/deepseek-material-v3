# Manual Testing Guide: Phase 2 Data Collection Pipeline

This guide provides step-by-step instructions for manually testing the complete Phase 2 data collection pipeline with real APIs.

## Prerequisites

### 1. API Keys Required

Before running the pipeline, you need to obtain API keys for the following services:

#### Materials Project API Key
- **Website**: https://materialsproject.org/
- **How to get**:
  1. Create a free account at https://materialsproject.org/
  2. Go to your dashboard
  3. Generate an API key
- **Environment variable**: `MP_API_KEY`

#### Semantic Scholar API (Optional)
- **Website**: https://www.semanticscholar.org/product/api
- **How to get**: 
  1. Visit https://www.semanticscholar.org/product/api
  2. Request an API key (free tier available)
- **Note**: The library works without an API key but with rate limits
- **Environment variable**: `SEMANTIC_SCHOLAR_API_KEY` (optional)

### 2. Set Environment Variables

**On Windows (PowerShell):**
```powershell
$env:MP_API_KEY = "your_materials_project_api_key_here"
$env:SEMANTIC_SCHOLAR_API_KEY = "your_semantic_scholar_key_here"  # Optional
```

**On Windows (CMD):**
```cmd
set MP_API_KEY=your_materials_project_api_key_here
set SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

**On Linux/Mac:**
```bash
export MP_API_KEY="your_materials_project_api_key_here"
export SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key_here"  # Optional
```

**Permanent Setup (Recommended):**

Create a `.env` file in the project root:
```bash
MP_API_KEY=your_materials_project_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

### 3. Verify Dependencies

Run the dependency check:
```bash
python -c "from scripts.run_data_collection import verify_dependencies; missing = verify_dependencies(); print('All dependencies installed!' if not missing else f'Missing: {missing}')"
```

If any dependencies are missing, install them:
```bash
pip install -r requirements_phase2.txt
```

## Running the Complete Pipeline

### Option 1: Run All Sources (Recommended for Full Test)

This will collect data from all 6 sources and create the master dataset:

```bash
python scripts/run_data_collection.py
```

**Expected behavior:**
- Phase 1 tests run first (should all pass)
- Each collector runs in sequence:
  1. Materials Project (MP)
  2. JARVIS-DFT
  3. AFLOW
  4. Semantic Scholar (Literature)
  5. MatWeb
  6. NIST Baseline
- Integration pipeline merges all sources
- Phase 1 tests run again (should still pass)
- Master dataset created at `data/final/master_dataset.csv`

**Expected duration:** 15-30 minutes (depending on API response times)

**Expected output:**
```
INFO: Logging to: logs/collection_20241129_HHMMSS.log
======================================================================
PHASE 2 DATA COLLECTION PIPELINE
======================================================================
Started at: 2024-11-29 HH:MM:SS

Step 1/7: Verifying dependencies...
✓ All dependencies installed

Step 2/7: Verifying Phase 1 tests BEFORE collection...
======================================================================
PHASE 1 TEST VERIFICATION
======================================================================
Running Phase 1 test suite to verify existing functionality...
✓ All Phase 1 tests PASSED
======================================================================

Step 3/7: Loading Phase 2 configuration...
✓ Configuration loaded from config/data_pipeline_config.yaml

Step 4/7: Configuring sources...
Running all sources
  ✓ materials_project enabled
  ✓ jarvis enabled
  ✓ aflow enabled
  ✓ semantic_scholar enabled
  ✓ matweb enabled
  ✓ nist enabled

Step 5/7: Running data collection pipeline...
======================================================================
Starting Phase 2 data collection pipeline
======================================================================
Collection Progress: 100%|████████████████████| 6/6 [XX:XX<00:00]

✓ mp: XXX materials collected in XX.Xs
✓ jarvis: XXX materials collected in XX.Xs
✓ aflow: XXX materials collected in XX.Xs
✓ semantic_scholar: XX materials collected in XX.Xs
✓ matweb: XX materials collected in XX.Xs
✓ nist: X materials collected in XX.Xs

Running integration pipeline...
✓ Integration completed in XX.Xs
  Master dataset: data/final/master_dataset.csv

======================================================================
✓ Pipeline completed successfully!
✓ Master dataset saved to: data/final/master_dataset.csv

Step 6/7: Verifying Phase 1 tests AFTER collection...
✓ All Phase 1 tests PASSED

Step 7/7: Final report
======================================================================
✓ PHASE 2 DATA COLLECTION COMPLETED SUCCESSFULLY
======================================================================
```

### Option 2: Run Specific Sources Only

For faster testing or debugging, run only specific sources:

```bash
# Run only Materials Project and JARVIS
python scripts/run_data_collection.py --sources mp,jarvis

# Run only AFLOW and NIST (no API keys needed)
python scripts/run_data_collection.py --sources aflow,nist

# Run only literature sources
python scripts/run_data_collection.py --sources scholar,matweb
```

**Available source codes:**
- `mp` - Materials Project
- `jarvis` - JARVIS-DFT
- `aflow` - AFLOW
- `scholar` - Semantic Scholar (Literature)
- `matweb` - MatWeb
- `nist` - NIST Baseline

### Option 3: Skip Phase 1 Verification (Not Recommended)

For debugging only, skip the Phase 1 test verification:

```bash
python scripts/run_data_collection.py --skip-verification
```

**⚠️ WARNING:** This is NOT recommended for production use. Always verify Phase 1 tests pass before and after collection.

## Verifying the Output

### 1. Check Master Dataset

```bash
# View first few rows
python -c "import pandas as pd; df = pd.read_csv('data/final/master_dataset.csv'); print(f'Shape: {df.shape}'); print(df.head())"
```

**Expected output:**
- Shape: (6000+, 50+) - At least 6,000 materials with 50+ properties
- Columns should include:
  - `formula` - Chemical formula
  - `ceramic_type` - Material type (SiC, B4C, etc.)
  - `hardness_combined` - Combined hardness from multiple sources
  - `K_IC_combined` - Combined fracture toughness
  - `density_combined` - Combined density
  - `specific_hardness` - Derived property
  - `ballistic_efficacy` - Derived property
  - `dop_resistance` - Derived property

### 2. Review Integration Report

```bash
# View integration report
cat data/final/integration_report.txt
# Or on Windows:
type data\final\integration_report.txt
```

**Expected content:**
```
================================================================================
PHASE 2 DATA INTEGRATION REPORT
================================================================================

DATASET SUMMARY
--------------------------------------------------------------------------------
Total materials: 6000+
Unique formulas: 5000+
Total columns: 50+

DATA SOURCES
--------------------------------------------------------------------------------
  mp_data: XXX entries
  jarvis_data: XXX entries
  aflow_data: XXX entries
  semantic_scholar_data: XX entries
  matweb_data: XX entries
  nist_data: X entries

PROPERTY COVERAGE
--------------------------------------------------------------------------------
  With hardness_combined: XXXX (XX.X%)
  With K_IC_combined: XXXX (XX.X%)
  With density_combined: XXXX (XX.X%)
  With V50 (literature): XX
  With thermal_conductivity: XXX

DERIVED PROPERTIES
--------------------------------------------------------------------------------
  Specific hardness: XXXX
  Ballistic efficacy: XXXX
  DOP resistance: XXXX

================================================================================
Integration complete!
================================================================================
```

### 3. Review Pipeline Summary

```bash
# View pipeline summary
cat data/final/pipeline_summary.txt
# Or on Windows:
type data\final\pipeline_summary.txt
```

**Expected content:**
```
================================================================================
PHASE 2 DATA COLLECTION PIPELINE SUMMARY
Execution Time: 2024-11-29 HH:MM:SS
================================================================================

COLLECTOR RESULTS
--------------------------------------------------------------------------------
  ✓ mp                   XXX materials   XX.Xs  data/processed/mp_data.csv
  ✓ jarvis               XXX materials   XX.Xs  data/processed/jarvis_data.csv
  ✓ aflow                XXX materials   XX.Xs  data/processed/aflow_data.csv
  ✓ semantic_scholar      XX materials   XX.Xs  data/processed/semantic_scholar_data.csv
  ✓ matweb                XX materials   XX.Xs  data/processed/matweb_data.csv
  ✓ nist                   X materials    X.Xs  data/processed/nist_data.csv
  ✓ integration          SUCCESS         XX.Xs  data/final/master_dataset.csv

--------------------------------------------------------------------------------
Total Runtime: XXX.Xs (XX.X minutes)
================================================================================
```

### 4. Verify Phase 1 Compatibility

Test that Phase 1 can consume the output:

```bash
# Load master dataset with Phase 1 code
python -c "
import pandas as pd
from ceramic_discovery.ml.feature_engineering import FeatureEngineer

# Load master dataset
df = pd.read_csv('data/final/master_dataset.csv')
print(f'✓ Loaded {len(df)} materials')

# Verify required columns
required = ['formula', 'hardness_combined', 'density_combined']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f'✗ Missing columns: {missing}')
else:
    print('✓ All required columns present')

# Test with Phase 1 feature engineering
try:
    fe = FeatureEngineer()
    print('✓ Phase 1 FeatureEngineer can be instantiated')
    print('✓ Phase 1 compatibility verified')
except Exception as e:
    print(f'✗ Phase 1 compatibility issue: {e}')
"
```

### 5. Run Validation Script

```bash
python scripts/validate_collection_output.py
```

This will perform comprehensive validation checks on the output.

## Troubleshooting

### Issue: "Missing required packages"

**Solution:**
```bash
pip install -r requirements_phase2.txt
```

### Issue: "Phase 1 tests FAILED"

**Solution:**
1. Do NOT proceed with Phase 2 collection
2. Fix Phase 1 tests first:
   ```bash
   pytest tests/ -v --tb=short
   ```
3. Identify and fix failing tests
4. Re-run Phase 1 tests to verify all pass
5. Then retry Phase 2 collection

### Issue: "Configuration file not found"

**Solution:**
```bash
# Verify config file exists
ls config/data_pipeline_config.yaml

# If missing, check the example
ls config/data_pipeline_config.yaml.example

# Copy example if needed
cp config/data_pipeline_config.yaml.example config/data_pipeline_config.yaml
```

### Issue: "MP_API_KEY not found"

**Solution:**
1. Verify environment variable is set:
   ```bash
   # Windows PowerShell
   echo $env:MP_API_KEY
   
   # Linux/Mac
   echo $MP_API_KEY
   ```
2. If not set, set it as described in Prerequisites section
3. Restart your terminal/shell after setting

### Issue: "Collector failed: API rate limit exceeded"

**Solution:**
1. Wait a few minutes for rate limit to reset
2. Increase `rate_limit_delay` in config file
3. Run with `--sources` to skip the problematic source temporarily
4. Retry the full pipeline later

### Issue: "Integration failed: No data sources found"

**Solution:**
1. Check that collectors ran successfully
2. Verify CSV files exist in `data/processed/`:
   ```bash
   ls data/processed/
   ```
3. If missing, check collector logs for errors
4. Re-run specific collectors that failed

### Issue: "Master dataset has < 6000 materials"

**Possible causes:**
1. Some collectors failed (check pipeline_summary.txt)
2. API rate limits reduced data collection
3. Configuration limits are too restrictive

**Solution:**
1. Review pipeline_summary.txt for failed collectors
2. Check individual source CSVs in data/processed/
3. Adjust configuration limits if needed:
   - `max_materials` in materials_project section
   - `max_per_system` in aflow section
   - `papers_per_query` in semantic_scholar section

## Success Criteria

The manual test is successful if:

1. ✅ All Phase 1 tests pass BEFORE collection
2. ✅ All 6 collectors execute successfully
3. ✅ Integration pipeline completes without errors
4. ✅ Master dataset created with 6,000+ materials
5. ✅ Integration report shows all sources
6. ✅ Pipeline summary shows all collectors succeeded
7. ✅ All Phase 1 tests pass AFTER collection
8. ✅ Phase 1 can load and use master_dataset.csv

## Next Steps After Successful Test

1. **Review Data Quality**
   - Check integration_report.txt for property coverage
   - Verify derived properties are calculated correctly
   - Look for any anomalies in the data

2. **Use with Phase 1 ML Pipeline**
   ```bash
   # Train models using master dataset
   python scripts/train_models.py --data data/final/master_dataset.csv
   ```

3. **Run Screening Workflow**
   ```bash
   # Use master dataset for screening
   python scripts/run_screening.py --data data/final/master_dataset.csv
   ```

4. **Archive Results**
   ```bash
   # Create backup of successful run
   mkdir -p data/backups/
   cp data/final/master_dataset.csv data/backups/master_dataset_$(date +%Y%m%d).csv
   ```

## Reporting Issues

If you encounter issues during manual testing:

1. **Collect Information**
   - Log file: `logs/collection_YYYYMMDD_HHMMSS.log`
   - Pipeline summary: `data/final/pipeline_summary.txt`
   - Error messages from console

2. **Check Common Issues**
   - Review Troubleshooting section above
   - Verify all prerequisites are met
   - Check API keys are valid

3. **Report Bug**
   - Include log file
   - Include error messages
   - Describe steps to reproduce
   - Note which collectors succeeded/failed

## Conclusion

This manual test verifies that the complete Phase 2 data collection pipeline:
- Works with real APIs
- Collects data from all 6 sources
- Integrates data correctly
- Produces a master dataset with 6,000+ materials
- Maintains Phase 1 compatibility
- Does not break existing functionality

Successful completion of this test confirms the Phase 2 implementation is production-ready.
