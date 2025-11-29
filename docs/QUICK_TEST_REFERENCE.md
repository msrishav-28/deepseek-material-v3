# Quick Test Reference: Phase 2 Manual Testing

## Quick Start (5 minutes)

### 1. Set API Key
```bash
# Windows PowerShell
$env:MP_API_KEY = "your_key_here"

# Linux/Mac
export MP_API_KEY="your_key_here"
```

### 2. Run Pipeline
```bash
python scripts/run_data_collection.py
```

### 3. Verify Output
```bash
# Check master dataset exists and has data
python -c "import pandas as pd; df = pd.read_csv('data/final/master_dataset.csv'); print(f'✓ {len(df)} materials collected')"
```

## Expected Results

| Metric | Expected Value |
|--------|---------------|
| Total materials | 6,000+ |
| Unique formulas | 5,000+ |
| Total columns | 50+ |
| Runtime | 15-30 minutes |
| Phase 1 tests | All pass (before & after) |

## Output Files

```
data/
├── processed/
│   ├── mp_data.csv              # Materials Project data
│   ├── jarvis_data.csv          # JARVIS-DFT data
│   ├── aflow_data.csv           # AFLOW data
│   ├── semantic_scholar_data.csv # Literature data
│   ├── matweb_data.csv          # MatWeb data
│   └── nist_data.csv            # NIST baseline data
│
└── final/
    ├── master_dataset.csv       # ★ MAIN OUTPUT
    ├── integration_report.txt   # Data quality report
    └── pipeline_summary.txt     # Execution summary
```

## Quick Validation

```bash
# 1. Check master dataset shape
python -c "import pandas as pd; print(pd.read_csv('data/final/master_dataset.csv').shape)"
# Expected: (6000+, 50+)

# 2. Check required columns
python -c "import pandas as pd; df = pd.read_csv('data/final/master_dataset.csv'); print([c for c in ['formula', 'hardness_combined', 'K_IC_combined', 'density_combined'] if c in df.columns])"
# Expected: ['formula', 'hardness_combined', 'K_IC_combined', 'density_combined']

# 3. View integration report
cat data/final/integration_report.txt  # Linux/Mac
type data\final\integration_report.txt  # Windows

# 4. View pipeline summary
cat data/final/pipeline_summary.txt  # Linux/Mac
type data\final\pipeline_summary.txt  # Windows
```

## Common Commands

### Run specific sources only
```bash
# Just MP and JARVIS (fastest)
python scripts/run_data_collection.py --sources mp,jarvis

# Just AFLOW and NIST (no API key needed)
python scripts/run_data_collection.py --sources aflow,nist
```

### Skip Phase 1 verification (debugging only)
```bash
python scripts/run_data_collection.py --skip-verification
```

### Use custom config
```bash
python scripts/run_data_collection.py --config path/to/config.yaml
```

## Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| Missing packages | `pip install -r requirements_phase2.txt` |
| Phase 1 tests fail | Fix Phase 1 first: `pytest tests/ -v` |
| API key not found | Set `MP_API_KEY` environment variable |
| Rate limit exceeded | Wait 5 minutes, then retry |
| No data collected | Check logs: `logs/collection_*.log` |

## Success Checklist

- [ ] Phase 1 tests pass BEFORE collection
- [ ] All 6 collectors execute successfully
- [ ] Integration completes without errors
- [ ] Master dataset has 6,000+ materials
- [ ] Integration report shows all sources
- [ ] Pipeline summary shows all SUCCESS
- [ ] Phase 1 tests pass AFTER collection
- [ ] Phase 1 can load master_dataset.csv

## Next Steps

After successful test:

1. **Review data quality**
   ```bash
   cat data/final/integration_report.txt
   ```

2. **Use with Phase 1 ML**
   ```bash
   python scripts/train_models.py --data data/final/master_dataset.csv
   ```

3. **Archive results**
   ```bash
   cp data/final/master_dataset.csv data/backups/master_dataset_$(date +%Y%m%d).csv
   ```

## Need Help?

See full manual testing guide: `docs/MANUAL_TESTING_GUIDE.md`
