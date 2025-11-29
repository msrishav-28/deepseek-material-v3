# Phase 2 Data Pipeline User Guide

## Overview

The Phase 2 Data Collection Pipeline is an isolated module that extends the Ceramic Armor Discovery Framework with automated multi-source data collection capabilities. It collects data from 6 authoritative sources and produces a master dataset for Phase 1 ML pipeline consumption.

**Key Features:**
- ✅ Completely isolated from Phase 1 code (no modifications to existing files)
- ✅ Automated collection from 6 data sources
- ✅ Intelligent deduplication and conflict resolution
- ✅ Comprehensive property coverage (50+ properties)
- ✅ Graceful error handling and resumability
- ✅ Expected output: 6,000-7,000 materials

## Data Sources

### 1. Materials Project (DFT Properties)
- **Source:** Materials Project API
- **Data Type:** DFT-calculated properties
- **Key Properties:** Formation energy, band gap, elastic moduli, density
- **Expected Materials:** ~2,000-3,000 carbides and oxides
- **Configuration:** Requires API key

### 2. JARVIS-DFT (Carbide Materials)
- **Source:** JARVIS database
- **Data Type:** DFT properties for carbides
- **Key Properties:** Formation energy, bulk modulus, shear modulus
- **Expected Materials:** ~1,000-1,500 carbides
- **Configuration:** Uses local JSON file

### 3. AFLOW (Thermal Properties)
- **Source:** AFLOW database
- **Data Type:** Thermal properties
- **Key Properties:** Thermal conductivity, thermal expansion, Debye temperature, heat capacity
- **Expected Materials:** ~500-1,000 per system
- **Configuration:** No API key required

### 4. Semantic Scholar (Literature Data)
- **Source:** Academic literature via Semantic Scholar API
- **Data Type:** Experimental properties from papers
- **Key Properties:** V₅₀ (ballistic limit), hardness, fracture toughness
- **Expected Materials:** ~50-100 with experimental data
- **Configuration:** No API key required (rate limited)

### 5. MatWeb (Experimental Data)
- **Source:** MatWeb material database
- **Data Type:** Manufacturer and experimental data
- **Key Properties:** Hardness, fracture toughness, density, compressive strength
- **Expected Materials:** ~50-100 commercial ceramics
- **Configuration:** Web scraping with rate limiting

### 6. NIST (Baseline References)
- **Source:** Embedded reference data
- **Data Type:** Validated baseline values
- **Key Properties:** Density, hardness, fracture toughness
- **Expected Materials:** 5 reference materials (SiC, B₄C, WC, TiC, Al₂O₃)
- **Configuration:** Embedded in config file

## Installation

### Prerequisites
```bash
# Ensure Phase 1 is working
pytest tests/ -v --ignore=tests/test_notebook_widgets.py --ignore=tests/test_performance_benchmarks.py

# Should see 716+ Phase 1 tests passing
```

### Install Phase 2 Dependencies
```bash
pip install -r requirements_phase2.txt
```

**New dependencies:**
- `aflow>=0.0.10` - AFLOW API client
- `semanticscholar>=0.4.0` - Literature search
- `requests>=2.31.0` - HTTP requests
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=4.9.0` - XML/HTML parser
- `tqdm>=4.65.0` - Progress bars

## Configuration

### Setup Configuration File

The pipeline uses `config/data_pipeline_config.yaml` for all settings.

**Required Environment Variables:**
```bash
# Windows (PowerShell)
$env:MP_API_KEY = "your-materials-project-api-key"

# Linux/Mac
export MP_API_KEY="your-materials-project-api-key"
```

**Get Materials Project API Key:**
1. Visit https://materialsproject.org/
2. Create free account
3. Go to Dashboard → API
4. Copy your API key

### Configuration Options

#### Materials Project Section
```yaml
materials_project:
  api_key: ${MP_API_KEY}  # Environment variable
  target_systems:
    - "Si-C"
    - "B-C"
    - "W-C"
    - "Ti-C"
    - "Al-O"
  batch_size: 1000
  max_materials: 5000
```

#### JARVIS Section
```yaml
jarvis:
  data_file: "data/jdft_3d-7-7-2018.json"
  carbide_metals:
    - "Si"
    - "B"
    - "W"
    - "Ti"
    - "Zr"
    - "Hf"
    - "V"
    - "Nb"
    - "Ta"
    - "Cr"
    - "Mo"
```

#### AFLOW Section
```yaml
aflow:
  ceramics:
    SiC: ["Si", "C"]
    B4C: ["B", "C"]
    WC: ["W", "C"]
    TiC: ["Ti", "C"]
    Al2O3: ["Al", "O"]
  max_per_system: 500
  exclude_ldau: true
```

#### Semantic Scholar Section
```yaml
semantic_scholar:
  queries:
    - "ballistic limit velocity ceramic armor"
    - "silicon carbide armor penetration"
    - "boron carbide ballistic performance"
  papers_per_query: 50
  min_citations: 5
```

#### MatWeb Section
```yaml
matweb:
  target_materials:
    - "Silicon Carbide"
    - "Boron Carbide"
    - "Tungsten Carbide"
    - "Titanium Carbide"
    - "Aluminum Oxide"
  results_per_material: 10
  rate_limit_delay: 2  # seconds between requests
```

#### NIST Baseline Section
```yaml
nist_baseline:
  baseline_materials:
    SiC:
      formula: "SiC"
      ceramic_type: "SiC"
      density: 3.21
      hardness: 26.0
      K_IC: 4.0
      source: "NIST SRM"
    # ... (other materials)
```

#### Integration Section
```yaml
integration:
  deduplication_threshold: 0.95  # Formula similarity threshold
  source_priority:  # Higher priority = preferred for conflicts
    - "Literature"
    - "MatWeb"
    - "NIST"
    - "MaterialsProject"
    - "JARVIS"
    - "AFLOW"
```

## Usage

### Quick Start

```bash
# Run complete pipeline
python scripts/run_data_collection.py

# Expected output:
# - data/processed/*.csv (6 source files)
# - data/final/master_dataset.csv (integrated dataset)
# - data/final/integration_report.txt
# - data/final/pipeline_summary.txt
```

### Step-by-Step Execution

```python
from ceramic_discovery.data_pipeline import CollectionOrchestrator, ConfigLoader

# Load configuration
config = ConfigLoader.load('config/data_pipeline_config.yaml')

# Initialize orchestrator
orchestrator = CollectionOrchestrator(config)

# Run all collectors
master_path = orchestrator.run_all()

print(f"Master dataset saved to: {master_path}")
```

### Run Individual Collectors

```python
from ceramic_discovery.data_pipeline.collectors import (
    MaterialsProjectCollector,
    JarvisCollector,
    AFLOWCollector
)

# Materials Project
mp_config = config['materials_project']
mp_collector = MaterialsProjectCollector(mp_config)
mp_df = mp_collector.collect()
mp_collector.save_to_csv(mp_df)

# JARVIS
jarvis_config = config['jarvis']
jarvis_collector = JarvisCollector(jarvis_config)
jarvis_df = jarvis_collector.collect()
jarvis_collector.save_to_csv(jarvis_df)

# AFLOW
aflow_config = config['aflow']
aflow_collector = AFLOWCollector(aflow_config)
aflow_df = aflow_collector.collect()
aflow_collector.save_to_csv(aflow_df)
```

### Run Integration Only

```python
from ceramic_discovery.data_pipeline.pipelines import IntegrationPipeline

# Assumes data/processed/*.csv files exist
integration_config = config['integration']
pipeline = IntegrationPipeline(integration_config)
master_df = pipeline.integrate()

print(f"Integrated {len(master_df)} materials")
```

## Output Files

### Processed Data (Intermediate)
```
data/processed/
├── materialsproject_data.csv  # Materials Project data
├── jarvis_data.csv            # JARVIS data
├── aflow_data.csv             # AFLOW data
├── semanticscholar_data.csv   # Literature data
├── matweb_data.csv            # MatWeb data
└── nist_data.csv              # NIST baseline
```

### Final Output
```
data/final/
├── master_dataset.csv         # ★ MAIN OUTPUT - Use this for Phase 1 ML
├── integration_report.txt     # Deduplication and merging statistics
└── pipeline_summary.txt       # Collection summary
```

### Master Dataset Structure

**Columns:**
- `formula` - Chemical formula (primary key)
- `ceramic_type` - Material classification (SiC, B4C, etc.)
- `source_count` - Number of sources with data for this material
- `sources` - Comma-separated list of sources

**DFT Properties:**
- `formation_energy_combined` - Formation energy (eV/atom)
- `band_gap_combined` - Band gap (eV)
- `bulk_modulus_combined` - Bulk modulus (GPa)
- `shear_modulus_combined` - Shear modulus (GPa)
- `density_combined` - Density (g/cm³)

**Mechanical Properties:**
- `hardness_combined` - Hardness (GPa)
- `K_IC_combined` - Fracture toughness (MPa·m^0.5)
- `compressive_strength_combined` - Compressive strength (GPa)

**Thermal Properties:**
- `thermal_conductivity_300K` - Thermal conductivity at 300K (W/m·K)
- `thermal_expansion_300K` - Thermal expansion at 300K (10^-6/K)
- `debye_temperature` - Debye temperature (K)
- `heat_capacity_Cp_300K` - Heat capacity at 300K (J/mol·K)

**Ballistic Properties:**
- `v50_literature` - Ballistic limit velocity (m/s) from literature
- `v50_predicted` - Predicted V₅₀ (if ML model available)

**Derived Properties:**
- `specific_hardness` - Hardness/density (GPa·cm³/g)
- `ballistic_efficacy` - Hardness/density (armor metric)
- `dop_resistance` - √(hardness) × K_IC / density

## Validation

### Verify Output

```bash
python scripts/validate_collection_output.py
```

**Checks:**
- ✅ Master dataset exists
- ✅ Minimum material count (>1000)
- ✅ Required columns present
- ✅ Property coverage thresholds met
- ✅ No duplicate formulas
- ✅ Valid numeric ranges

### Test Phase 1 Compatibility

```python
import pandas as pd

# Load master dataset
df = pd.read_csv('data/final/master_dataset.csv')

# Verify Phase 1 can consume it
from ceramic_discovery.ml.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_data(df)  # Should work without errors
```

## Troubleshooting

### Common Issues

#### 1. Materials Project API Key Error
```
ValueError: Materials Project API key required
```

**Solution:**
```bash
# Set environment variable
$env:MP_API_KEY = "your-api-key-here"

# Verify
python -c "import os; print(os.environ.get('MP_API_KEY'))"
```

#### 2. JARVIS File Not Found
```
FileNotFoundError: data/jdft_3d-7-7-2018.json not found
```

**Solution:**
```bash
# Download JARVIS data
# Visit: https://jarvis.nist.gov/
# Or use existing file from Phase 1
```

#### 3. Rate Limiting Errors
```
HTTPError: 429 Too Many Requests
```

**Solution:**
- Increase `rate_limit_delay` in config
- Wait and retry
- Use checkpointing to resume

#### 4. Import Errors
```
ImportError: No module named 'aflow'
```

**Solution:**
```bash
pip install -r requirements_phase2.txt
```

#### 5. Phase 1 Tests Failing
```
FAILED tests/test_*.py
```

**Solution:**
- Phase 2 should NOT break Phase 1
- Check if tests were already failing
- Compare with baseline: `logs/phase1_baseline_tests.log`
- Report issue if Phase 2 caused regression

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline
orchestrator.run_all()
```

### Checkpoint Recovery

If collection fails partway through:

```python
# Checkpoints saved to: data/processed/*.csv

# Resume from checkpoint
orchestrator = CollectionOrchestrator(config)
orchestrator.run_all()  # Skips completed collectors
```

## Performance

### Expected Runtime
- **Materials Project:** 5-10 minutes (API rate limits)
- **JARVIS:** 1-2 minutes (local file)
- **AFLOW:** 10-15 minutes (API queries)
- **Semantic Scholar:** 5-10 minutes (rate limited)
- **MatWeb:** 10-15 minutes (web scraping, rate limited)
- **NIST:** <1 minute (embedded data)
- **Integration:** 2-5 minutes (deduplication)

**Total:** ~45-60 minutes for complete pipeline

### Optimization Tips

1. **Reduce target systems** in config for faster testing
2. **Lower max_materials** limits
3. **Use cached data** from `data/processed/` if available
4. **Run collectors in parallel** (advanced)

## Integration with Phase 1

### ML Pipeline Integration

```python
# Phase 1 ML pipeline can consume master dataset directly
from ceramic_discovery.ml.model_trainer import ModelTrainer
import pandas as pd

# Load Phase 2 output
df = pd.read_csv('data/final/master_dataset.csv')

# Use in Phase 1 ML pipeline
trainer = ModelTrainer()
trainer.load_data(df)
trainer.train()
```

### Screening Integration

```python
from ceramic_discovery.screening.screening_engine import ScreeningEngine

# Load Phase 2 output
df = pd.read_csv('data/final/master_dataset.csv')

# Use in Phase 1 screening
engine = ScreeningEngine()
candidates = engine.screen(df, criteria={'hardness_combined': '>25'})
```

## API Reference

### CollectionOrchestrator

```python
class CollectionOrchestrator:
    """Orchestrates all data collectors and integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        
    def run_all(self) -> Path:
        """
        Run all collectors and integration.
        
        Returns:
            Path to master_dataset.csv
        """
        
    def run_collectors(self) -> List[Path]:
        """Run all collectors, return paths to CSV files."""
        
    def run_integration(self) -> Path:
        """Run integration pipeline, return path to master dataset."""
```

### BaseCollector

```python
class BaseCollector(ABC):
    """Abstract base class for all collectors."""
    
    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Collect data from source."""
        
    @abstractmethod
    def get_source_name(self) -> str:
        """Return source identifier."""
        
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration."""
        
    def save_to_csv(self, df: pd.DataFrame, output_dir: Path) -> Path:
        """Save collected data to CSV."""
```

### IntegrationPipeline

```python
class IntegrationPipeline:
    """Integrates data from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with integration config."""
        
    def integrate(self) -> pd.DataFrame:
        """
        Integrate all source data.
        
        Returns:
            Integrated DataFrame
        """
        
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate materials using fuzzy matching."""
        
    def resolve_conflicts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve property conflicts using source priority."""
```

## Best Practices

### 1. Always Verify Phase 1 First
```bash
# Before running Phase 2
pytest tests/ -v --ignore=tests/test_notebook_widgets.py --ignore=tests/test_performance_benchmarks.py
```

### 2. Use Environment Variables for Secrets
```yaml
# Good
api_key: ${MP_API_KEY}

# Bad (hardcoded)
api_key: "abc123..."
```

### 3. Start with Small Limits for Testing
```yaml
materials_project:
  max_materials: 100  # Test with small number first
```

### 4. Monitor Progress
```python
# Use tqdm progress bars (built-in)
orchestrator.run_all()  # Shows progress automatically
```

### 5. Validate Output
```bash
# Always validate after collection
python scripts/validate_collection_output.py
```

## Support

### Documentation
- **Phase 1 Guide:** `docs/USER_GUIDE.md`
- **Phase 2 Guide:** `docs/DATA_PIPELINE_GUIDE.md` (this file)
- **API Reference:** `docs/API_REFERENCE.md`
- **Testing Guide:** `docs/MANUAL_TESTING_GUIDE.md`

### Reporting Issues

If you encounter issues:

1. Check troubleshooting section above
2. Verify Phase 1 tests still pass
3. Check logs in `logs/collection_*.log`
4. Review integration report in `data/final/integration_report.txt`

### Contributing

Phase 2 follows strict isolation principles:
- ✅ Add new collectors in `data_pipeline/collectors/`
- ✅ Add new utilities in `data_pipeline/utils/`
- ❌ DO NOT modify Phase 1 files
- ❌ DO NOT import Phase 2 from Phase 1

## Appendix

### File Locations

**Phase 2 Code:**
```
src/ceramic_discovery/data_pipeline/
├── collectors/
├── integration/
├── pipelines/
└── utils/
```

**Phase 2 Tests:**
```
tests/data_pipeline/
├── collectors/
├── integration/
├── pipelines/
└── utils/
```

**Configuration:**
```
config/data_pipeline_config.yaml
```

**Scripts:**
```
scripts/
├── run_data_collection.py
└── validate_collection_output.py
```

### Version History

- **v2.0.0** - Initial Phase 2 release
  - 6 data source collectors
  - Intelligent integration pipeline
  - Complete isolation from Phase 1
  - 143 new tests
  - Expected output: 6,000-7,000 materials

---

**Last Updated:** 2024
**Phase:** 2 (Data Collection Pipeline)
**Status:** Production Ready
