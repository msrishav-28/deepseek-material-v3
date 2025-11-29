# Requirements Document: Phase 2 - Isolated Multi-Source Data Collection Pipeline

## ⚠️ CRITICAL: PRESERVATION OF EXISTING WORK

**THIS IS PHASE 2 - EXISTING FRAMEWORK MUST REMAIN UNTOUCHED**

The target system has **34 completed tasks** across two projects:
- **Base Framework**: 14 tasks (Project setup, DFT collection, ML pipeline, screening, validation, deployment)
- **SiC Integration**: 20 tasks (JARVIS client, NIST client, multi-source integration, application ranking)

**ALL 157+ existing tests must continue to pass after this implementation.**

***

## Introduction

This specification defines **Phase 2: Automated Multi-Source Data Collection Pipeline** as a **completely isolated module** that extends the Ceramic Armor Discovery Framework's capabilities without modifying any existing code from the 34 completed tasks.

### Design Philosophy: Strict Isolation

**Isolation Principle:** New code imports FROM existing modules but existing modules NEVER import new code.

```
Existing Framework (UNTOUCHABLE)     New Pipeline (ISOLATED)
================================     =======================
src/ceramic_discovery/               src/ceramic_discovery/
├── dft/              [FROZEN]  ←──imports── data_pipeline/
├── ml/               [FROZEN]  ←──imports── ├── collectors/
├── screening/        [FROZEN]  ←──imports── ├── integration/
├── validation/       [FROZEN]  ←──imports── └── pipelines/
└── utils/            [FROZEN]  ←──imports──
```

**Dependency Flow:** data_pipeline → existing modules (ONE WAY ONLY)

***

## Glossary

- **Phase 1 Code**: All existing framework code from 34 completed tasks (MUST NOT BE MODIFIED)
- **Phase 2 Code**: New data collection pipeline code (THIS SPECIFICATION)
- **Collector**: New class that wraps existing clients for batch data collection
- **Composition Pattern**: Using existing classes via import/instantiation, not inheritance/modification
- **Isolated Module**: Code that lives in separate namespace and doesn't alter existing files
- **Master Dataset**: Final integrated CSV file that Phase 1 ML pipeline can consume

***

## DO NOT MODIFY - PROTECTED FILES LIST

### 🔒 EXISTING FILES THAT ARE OFF LIMITS

**Before making ANY changes, verify these files exist and DO NOT modify them:**

```bash
# Phase 1 Base Framework (14 tasks) - PROTECTED
src/ceramic_discovery/dft/materials_project_client.py    [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/dft/stability_analyzer.py          [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/ml/feature_engineering.py          [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/ml/model_trainer.py                [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/screening/screening_engine.py      [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/validation/validator.py            [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/reporting/report_generator.py      [EXISTS - DO NOT MODIFY]

# Phase 1 SiC Integration (20 tasks) - PROTECTED  
src/ceramic_discovery/dft/jarvis_client.py               [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/dft/nist_client.py                 [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/dft/data_combiner.py               [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/ml/composition_descriptors.py      [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/ml/structure_descriptors.py        [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/screening/application_ranker.py    [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/validation/experimental_planner.py [EXISTS - DO NOT MODIFY]
src/ceramic_discovery/utils/data_conversion.py           [EXISTS - DO NOT MODIFY]

# ALL existing test files - PROTECTED
tests/**/*.py                                            [EXISTS - DO NOT MODIFY]

# Existing configuration - PROTECTED (may need to read, but don't overwrite)
config/*.yaml                                            [EXISTS - READ ONLY]
```

**Verification Command:**
```bash
# Run this BEFORE starting implementation to verify nothing breaks
pytest tests/ -v
# Expected: All 157+ tests pass
# If any fail, DO NOT PROCEED - report to user
```

***

## NEW FILES ONLY - What This Spec Creates

### ✅ NEW Files to Create (Phase 2)

```bash
# New isolated module namespace
src/ceramic_discovery/data_pipeline/              [NEW DIRECTORY]
├── __init__.py                                   [NEW]
│
├── collectors/                                   [NEW DIRECTORY]
│   ├── __init__.py                              [NEW]
│   ├── base_collector.py                        [NEW - abstract interface]
│   ├── materials_project_collector.py           [NEW - wraps existing client]
│   ├── jarvis_collector.py                      [NEW - wraps existing client]
│   ├── aflow_collector.py                       [NEW - new functionality]
│   ├── semantic_scholar_collector.py            [NEW - new functionality]
│   ├── matweb_collector.py                      [NEW - new functionality]
│   └── nist_baseline_collector.py               [NEW - new functionality]
│
├── integration/                                  [NEW DIRECTORY]
│   ├── __init__.py                              [NEW]
│   ├── formula_matcher.py                       [NEW - fuzzy matching utility]
│   ├── deduplicator.py                          [NEW - multi-source dedup]
│   ├── conflict_resolver.py                     [NEW - property merging]
│   └── quality_validator.py                     [NEW - data validation]
│
├── pipelines/                                    [NEW DIRECTORY]
│   ├── __init__.py                              [NEW]
│   ├── collection_orchestrator.py               [NEW - runs all collectors]
│   └── integration_pipeline.py                  [NEW - merges all sources]
│
└── utils/                                        [NEW DIRECTORY]
    ├── __init__.py                              [NEW]
    ├── property_extractor.py                    [NEW - text parsing for literature]
    ├── rate_limiter.py                          [NEW - API throttling]
    └── config_loader.py                         [NEW - Phase 2 config]

# New configuration for Phase 2 only
config/data_pipeline_config.yaml                  [NEW - separate from existing config]

# New standalone scripts
scripts/                                          [NEW DIRECTORY]
├── run_data_collection.py                       [NEW - main entry point]
└── validate_collection_output.py                [NEW - verify master dataset]

# New tests for Phase 2 only
tests/data_pipeline/                              [NEW DIRECTORY]
├── collectors/                                   [NEW]
├── integration/                                  [NEW]
└── pipelines/                                    [NEW]
```

***

## System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: EXISTING FRAMEWORK (34 tasks - FROZEN)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  dft/materials_project_client.py  ← imports (read only)   │
│  dft/jarvis_client.py             ← imports (read only)   │
│  dft/data_combiner.py             ← imports (read only)   │
│  ml/model_trainer.py              ← consumes master.csv   │
│  utils/data_conversion.py         ← imports (read only)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        ↑
                        │ imports only (one-way)
                        │
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: NEW DATA PIPELINE (THIS SPEC - ISOLATED)         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  data_pipeline/collectors/mp_collector.py                  │
│    ├─ imports MaterialsProjectClient (doesn't modify it)  │
│    └─ adds batch collection logic                          │
│                                                             │
│  data_pipeline/collectors/jarvis_collector.py              │
│    ├─ imports JarvisClient (doesn't modify it)            │
│    └─ adds carbide filtering at collection time            │
│                                                             │
│  data_pipeline/collectors/aflow_collector.py               │
│    └─ new AFLOW integration (no existing dependency)       │
│                                                             │
│  data_pipeline/integration/integration_pipeline.py         │
│    ├─ imports data_combiner.DataCombiner (read only)      │
│    └─ creates master_dataset.csv                           │
│                                                             │
│  Output: data/final/master_dataset.csv                     │
│    └─ consumed by existing ML pipeline (no changes to ML) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

***

## Requirements

### Requirement 1: Base Collector Interface

**User Story:** As a pipeline developer, I want an abstract base class for all collectors, so that all data sources follow a consistent interface without modifying existing clients.

#### Acceptance Criteria

1. WHEN creating base collector THEN the system SHALL define abstract class `BaseCollector` in NEW file `data_pipeline/collectors/base_collector.py`
2. WHEN defining interface THEN the class SHALL require abstract methods: `collect()` returning `pd.DataFrame`, `get_source_name()` returning `str`, `validate_config()` returning `bool`
3. WHEN implementing collectors THEN each collector SHALL inherit from `BaseCollector` and implement required methods
4. WHEN existing code is present THEN the system SHALL NOT modify any existing client classes (`MaterialsProjectClient`, `JarvisClient`)
5. WHEN using ABC THEN the system SHALL import from `abc` module and use `@abstractmethod` decorator

**New File to Create:**
```python
# src/ceramic_discovery/data_pipeline/collectors/base_collector.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseCollector(ABC):
    """
    Abstract base class for all Phase 2 data collectors.
    
    IMPORTANT: This is a wrapper around existing Phase 1 clients.
    Do NOT modify existing client classes (MaterialsProjectClient, JarvisClient, etc.)
    Import and use them via composition.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """
        Collect data from source and return as DataFrame.
        
        Returns:
            pd.DataFrame with columns: material_id, formula, source, properties...
        """
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return source identifier (e.g., 'MaterialsProject', 'JARVIS')"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration has required keys for this collector"""
        pass
```

**Integration Pattern:**
- ✅ New abstract base class (no existing code dependency)
- ✅ Other collectors inherit from this
- ❌ Does NOT modify any existing classes

**Testing:**
```python
# tests/data_pipeline/collectors/test_base_collector.py
def test_base_collector_cannot_instantiate():
    """Verify BaseCollector is abstract"""
    with pytest.raises(TypeError):
        BaseCollector({})
```

***

### Requirement 2: Materials Project Collector (Wrapper Pattern)

**User Story:** As a data collector, I want to use the existing MaterialsProjectClient without modifying it, so that Phase 1 functionality remains intact.

#### Acceptance Criteria

1. WHEN creating MP collector THEN the system SHALL create NEW file `data_pipeline/collectors/materials_project_collector.py`
2. WHEN using existing client THEN the collector SHALL import `MaterialsProjectClient` from `ceramic_discovery.dft.materials_project_client` (existing file)
3. WHEN instantiating THEN the collector SHALL create instance of `MaterialsProjectClient` internally using composition (NOT inheritance)
4. WHEN collecting data THEN the collector SHALL call existing client methods without modification and add batch processing logic
5. WHEN existing client is unavailable THEN the collector SHALL raise `ImportError` with clear message indicating Phase 1 dependency

**New File to Create:**
```python
# src/ceramic_discovery/data_pipeline/collectors/materials_project_collector.py
"""
Phase 2 Materials Project Collector

CRITICAL: This wraps the existing MaterialsProjectClient from Phase 1.
Do NOT modify src/ceramic_discovery/dft/materials_project_client.py
"""

import pandas as pd
from typing import Dict, Any, List
import logging

# Import existing Phase 1 client (DO NOT MODIFY THE IMPORTED CLASS)
try:
    from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient
except ImportError as e:
    raise ImportError(
        "Cannot import MaterialsProjectClient from Phase 1. "
        "Ensure Phase 1 code at src/ceramic_discovery/dft/materials_project_client.py exists. "
        f"Original error: {e}"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MaterialsProjectCollector(BaseCollector):
    """
    Wrapper around existing MaterialsProjectClient for batch data collection.
    
    Uses composition (HAS-A) not inheritance (IS-A) to avoid modifying Phase 1 code.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Create instance of existing client (composition pattern)
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Materials Project API key required in config['api_key']")
        
        # Instantiate existing Phase 1 client
        self.client = MaterialsProjectClient(api_key=api_key)
        
        self.target_systems = config.get('target_systems', [])
        self.batch_size = config.get('batch_size', 1000)
        self.max_materials = config.get('max_materials', 5000)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect materials from Materials Project using existing client.
        
        Returns:
            DataFrame with columns matching Phase 1 MaterialData format
        """
        all_materials = []
        
        for system in self.target_systems:
            logger.info(f"Collecting Materials Project data for {system}")
            
            try:
                # Call existing client method (DO NOT MODIFY)
                materials = self.client.get_materials(
                    chemsys=system,
                    # Add any other parameters the existing method supports
                )
                
                # Process results into DataFrame
                for mat in materials:
                    material_dict = {
                        'source': 'MaterialsProject',
                        'material_id': mat.material_id,
                        'formula': mat.formula,
                        # Extract other properties as needed
                        # (match existing MaterialData structure)
                    }
                    all_materials.append(material_dict)
                    
                    if len(all_materials) >= self.max_materials:
                        break
                
            except Exception as e:
                logger.error(f"Failed to collect {system}: {e}")
                continue
        
        df = pd.DataFrame(all_materials)
        logger.info(f"Collected {len(df)} materials from Materials Project")
        
        return df
    
    def get_source_name(self) -> str:
        return "MaterialsProject"
    
    def validate_config(self) -> bool:
        required = ['api_key', 'target_systems']
        return all(k in self.config for k in required)
```

**Integration Pattern:**
- ✅ Imports existing `MaterialsProjectClient` (read-only)
- ✅ Creates instance via composition (`self.client = MaterialsProjectClient(...)`)
- ✅ Calls existing methods without modification
- ✅ Adds batch processing logic in NEW file
- ❌ Does NOT inherit from `MaterialsProjectClient`
- ❌ Does NOT modify `materials_project_client.py`

**Verification:**
```bash
# After implementation, verify Phase 1 still works
python -c "from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient; print('✓ Phase 1 client intact')"

# Verify Phase 2 works
python -c "from ceramic_discovery.data_pipeline.collectors.materials_project_collector import MaterialsProjectCollector; print('✓ Phase 2 collector created')"
```

**Testing:**
```python
# tests/data_pipeline/collectors/test_materials_project_collector.py
def test_mp_collector_uses_existing_client(mocker):
    """Verify collector uses existing Phase 1 client"""
    # Mock the existing client
    mock_client = mocker.patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient')
    
    config = {'api_key': 'test', 'target_systems': ['Si-C']}
    collector = MaterialsProjectCollector(config)
    
    # Verify existing client was instantiated (not inherited)
    mock_client.assert_called_once_with(api_key='test')
```

***

### Requirement 3: JARVIS Collector (Wrapper Pattern)

**User Story:** As a data collector, I want to use the existing JarvisClient without modifying it, so that Phase 1 SiC integration remains functional.

#### Acceptance Criteria

1. WHEN creating JARVIS collector THEN the system SHALL create NEW file `data_pipeline/collectors/jarvis_collector.py`
2. WHEN using existing client THEN the collector SHALL import `JarvisClient` from `ceramic_discovery.dft.jarvis_client` (existing from Tasks 1-20)
3. WHEN filtering carbides THEN the collector SHALL call existing client methods and add filtering logic at collection time (NOT by modifying client)
4. WHEN using safe conversion THEN the collector SHALL import `safe_float` from `ceramic_discovery.utils.data_conversion` (existing utility)
5. WHEN existing client has carbide methods THEN the collector SHALL use them; otherwise add filtering in collector

**New File to Create:**
```python
# src/ceramic_discovery/data_pipeline/collectors/jarvis_collector.py
"""
Phase 2 JARVIS Collector

CRITICAL: This wraps the existing JarvisClient from Phase 1 SiC Integration (Task 2).
Do NOT modify src/ceramic_discovery/dft/jarvis_client.py
"""

import pandas as pd
from typing import Dict, Any, Set
import logging

# Import existing Phase 1 components (READ ONLY)
try:
    from ceramic_discovery.dft.jarvis_client import JarvisClient
except ImportError as e:
    raise ImportError(
        "Cannot import JarvisClient from Phase 1. "
        "Ensure Phase 1 SiC Integration code at src/ceramic_discovery/dft/jarvis_client.py exists. "
        f"Original error: {e}"
    )

try:
    from ceramic_discovery.utils.data_conversion import safe_float
except ImportError as e:
    raise ImportError(
        "Cannot import safe_float from Phase 1. "
        "Ensure Phase 1 utilities at src/ceramic_discovery/utils/data_conversion.py exist. "
        f"Original error: {e}"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class JarvisCollector(BaseCollector):
    """
    Wrapper around existing JarvisClient for carbide collection.
    
    Uses composition to leverage existing Phase 1 JARVIS integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Create instance of existing client (composition)
        self.client = JarvisClient()  # Existing client from Phase 1
        
        self.carbide_metals = set(config.get('carbide_metals', []))
        self.data_source = config.get('data_source', 'dft_3d')
    
    def collect(self) -> pd.DataFrame:
        """
        Collect carbide materials using existing JarvisClient.
        
        Returns:
            DataFrame with carbide materials
        """
        logger.info("Collecting JARVIS carbide data")
        
        try:
            # Check if existing client has carbide filtering (from Phase 1 Task 2)
            if hasattr(self.client, 'load_carbides'):
                # Use existing method if available
                carbides = self.client.load_carbides(metal_elements=self.carbide_metals)
            else:
                # Add filtering logic here (without modifying client)
                # This is new Phase 2 functionality
                all_materials = self.client.load_all_materials()  # Assuming this exists
                carbides = self._filter_carbides(all_materials)
            
            # Convert to DataFrame
            materials_list = []
            for carbide in carbides:
                material_dict = {
                    'source': 'JARVIS',
                    'jid': carbide.get('jid'),
                    'formula': carbide.get('formula'),
                    # Use existing safe_float utility (Phase 1)
                    'formation_energy_jarvis': safe_float(carbide.get('formation_energy')),
                    'bulk_modulus_jarvis': safe_float(carbide.get('bulk_modulus')),
                    # ... other properties
                }
                materials_list.append(material_dict)
            
            df = pd.DataFrame(materials_list)
            logger.info(f"Collected {len(df)} carbides from JARVIS")
            
            return df
            
        except Exception as e:
            logger.error(f"JARVIS collection failed: {e}")
            return pd.DataFrame()  # Return empty on failure
    
    def _filter_carbides(self, materials: list) -> list:
        """
        Filter carbides from materials list.
        
        This is NEW Phase 2 logic added in collector, NOT in existing client.
        """
        carbides = []
        for mat in materials:
            elements = set(mat.get('elements', []))
            if 'C' in elements and elements.intersection(self.carbide_metals):
                carbides.append(mat)
        return carbides
    
    def get_source_name(self) -> str:
        return "JARVIS"
    
    def validate_config(self) -> bool:
        return 'carbide_metals' in self.config
```

**Integration Pattern:**
- ✅ Imports existing `JarvisClient` (read-only)
- ✅ Imports existing `safe_float` utility (read-only)
- ✅ Creates client instance via composition
- ✅ Checks if existing client has methods (graceful detection)
- ✅ Adds new filtering logic in collector if needed
- ❌ Does NOT modify `jarvis_client.py`
- ❌ Does NOT modify `data_conversion.py`

**Verification:**
```bash
# Verify Phase 1 SiC integration still works
pytest tests/test_jarvis_client.py -v
# Expected: All existing JARVIS tests pass

# Verify Phase 2 collector works
python -c "from ceramic_discovery.data_pipeline.collectors.jarvis_collector import JarvisCollector; print('✓ Phase 2 JARVIS collector OK')"
```

***

### Requirement 4: AFLOW Collector (New Functionality)

**User Story:** As a researcher, I want to collect thermal property data from AFLOW without any existing code dependencies, so that I can add new capabilities safely.

#### Acceptance Criteria

1. WHEN creating AFLOW collector THEN the system SHALL create NEW file `data_pipeline/collectors/aflow_collector.py` with no dependencies on existing Phase 1 code
2. WHEN using AFLOW API THEN the collector SHALL use `aflow` Python library (new dependency) directly
3. WHEN collecting thermal properties THEN the collector SHALL request thermal_conductivity_300K, thermal_expansion_300K, debye_temperature, heat_capacity
4. WHEN handling errors THEN the collector SHALL catch `aflow.AFLOWAPIError` and continue with other systems
5. WHEN saving results THEN the collector SHALL return DataFrame with columns matching Phase 2 naming conventions (suffix `_aflow`)

**New File to Create:**
```python
# src/ceramic_discovery/data_pipeline/collectors/aflow_collector.py
"""
Phase 2 AFLOW Collector

NEW FUNCTIONALITY: No Phase 1 dependencies.
This collector adds thermal property data from AFLOW.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

try:
    import aflow
except ImportError:
    raise ImportError(
        "aflow library not installed. Install with: pip install aflow>=0.0.10"
    )

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class AFLOWCollector(BaseCollector):
    """
    AFLOW data collector for thermal properties.
    
    This is completely new Phase 2 functionality with no Phase 1 dependencies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.ceramics = config.get('ceramics', {
            'SiC': ['Si', 'C'],
            'B4C': ['B', 'C'],
            'WC': ['W', 'C'],
            'TiC': ['Ti', 'C'],
            'Al2O3': ['Al', 'O']
        })
        self.max_per_system = config.get('max_per_system', 500)
        self.exclude_ldau = config.get('exclude_ldau', True)
    
    def collect(self) -> pd.DataFrame:
        """
        Collect materials with thermal properties from AFLOW.
        
        Returns:
            DataFrame with thermal property data
        """
        all_materials = []
        
        for name, elements in self.ceramics.items():
            logger.info(f"Querying AFLOW for {name}")
            
            try:
                # Query AFLOW API
                query = aflow.search(
                    species=','.join(elements),
                    exclude=['*:LDAU*'] if self.exclude_ldau else []
                )
                
                # Select properties
                query = query.select([
                    aflow.K.compound,
                    aflow.K.density,
                    aflow.K.enthalpy_formation_atom,
                    aflow.K.Egap,
                    # Thermal properties (AFLOW specialty)
                    aflow.K.agl_thermal_conductivity_300K,
                    aflow.K.agl_thermal_expansion_300K,
                    aflow.K.agl_debye,
                    aflow.K.agl_heat_capacity_Cp_300K,
                ])
                
                # Limit results
                query = query.limit(self.max_per_system)
                
                # Execute
                entries = list(query)
                logger.info(f"  Found {len(entries)} materials for {name}")
                
                # Extract properties
                for entry in entries:
                    material = {
                        'source': 'AFLOW',
                        'formula': entry.compound,
                        'ceramic_type': name,
                        'density_aflow': entry.density,
                        'formation_energy_aflow': entry.enthalpy_formation_atom,
                        'band_gap_aflow': entry.Egap,
                        # Thermal properties (KEY DATA)
                        'thermal_conductivity_300K': entry.agl_thermal_conductivity_300K,
                        'thermal_expansion_300K': entry.agl_thermal_expansion_300K,
                        'debye_temperature': entry.agl_debye,
                        'heat_capacity_Cp_300K': entry.agl_heat_capacity_Cp_300K,
                    }
                    all_materials.append(material)
                
            except aflow.AFLOWAPIError as e:
                logger.error(f"AFLOW query failed for {name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for {name}: {e}")
                continue
        
        df = pd.DataFrame(all_materials)
        logger.info(f"Collected {len(df)} materials from AFLOW")
        
        return df
    
    def get_source_name(self) -> str:
        return "AFLOW"
    
    def validate_config(self) -> bool:
        return 'ceramics' in self.config
```

**Integration Pattern:**
- ✅ No Phase 1 dependencies (completely new)
- ✅ Uses external `aflow` library
- ✅ Returns DataFrame matching Phase 2 format
- ❌ Does NOT touch any existing code

**New Dependency:**
```txt
# Add to requirements.txt (new dependency)
aflow>=0.0.10
```

***

### Requirement 5: Semantic Scholar Collector (New Functionality)

**User Story:** As a researcher, I want automated extraction of experimental properties from literature without modifying existing code, so that I can add validation data safely.

#### Acceptance Criteria

1. WHEN creating literature collector THEN the system SHALL create NEW file `data_pipeline/collectors/semantic_scholar_collector.py`
2. WHEN extracting properties THEN the collector SHALL use NEW utility file `data_pipeline/utils/property_extractor.py` for text parsing (no existing dependencies)
3. WHEN searching papers THEN the collector SHALL use `semanticscholar` library with configurable queries
4. WHEN extracting V₅₀ THEN the system SHALL use regex: `V[_\s]?50[:\s=]+(\d+\.?\d*)\s*(m/s|ms)`
5. WHEN extracting hardness THEN the system SHALL use regex: `hardness[:\s]+(\d+\.?\d*)\s*(GPa)` with HV-to-GPa conversion (÷100)
6. WHEN extracting K_IC THEN the system SHALL use regex: `fracture\s+toughness[:\s]+(\d+\.?\d*)\s*(MPa[·\s]?m)`
7. WHEN determining material THEN the system SHALL match keywords: "sic", "silicon carbide", "b4c", "boron carbide", etc.
8. WHEN saving results THEN the collector SHALL include provenance: paper_id, title, year, citations, URL

**New Files to Create:**

```python
# src/ceramic_discovery/data_pipeline/utils/property_extractor.py
"""
Phase 2 Utility: Property Extraction from Text

NEW FUNCTIONALITY: No Phase 1 dependencies.
Extracts material properties from research paper text using regex.
"""

import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PropertyExtractor:
    """
    Extract material properties from unstructured text.
    
    This is new Phase 2 functionality for literature mining.
    """
    
    # V50 patterns
    V50_PATTERNS = [
        r'V[_\s]?50[:\s=]+(\d+\.?\d*)\s*(m/s|ms)',
        r'ballistic\s+limit[:\s]+(\d+\.?\d*)\s*(m/s|ms)',
        r'penetration\s+velocity[:\s]+(\d+\.?\d*)\s*(m/s|ms)'
    ]
    
    # Hardness patterns
    HARDNESS_PATTERNS = [
        r'hardness[:\s]+(\d+\.?\d*)\s*(GPa)',
        r'Vickers\s+hardness[:\s]+(\d+\.?\d*)\s*(HV|GPa)',
        r'HV[:\s]+(\d+\.?\d*)',
    ]
    
    # Fracture toughness patterns
    KIC_PATTERNS = [
        r'fracture\s+toughness[:\s]+(\d+\.?\d*)\s*(MPa[·\s]?m)',
        r'K[_\s]?IC[:\s]+(\d+\.?\d*)\s*(MPa)',
        r'toughness[:\s]+(\d+\.?\d*)\s*(MPa[·\s]?m)',
    ]
    
    # Material identification
    MATERIAL_KEYWORDS = {
        'SiC': ['sic', 'silicon carbide'],
        'B4C': ['b4c', 'b₄c', 'boron carbide'],
        'WC': ['wc', 'tungsten carbide'],
        'TiC': ['tic', 'titanium carbide'],
        'Al2O3': ['al2o3', 'al₂o₃', 'alumina', 'aluminum oxide']
    }
    
    @classmethod
    def extract_v50(cls, text: str) -> Optional[float]:
        """Extract V50 ballistic limit velocity from text."""
        for pattern in cls.V50_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0][0])
                except (ValueError, IndexError):
                    continue
        return None
    
    @classmethod
    def extract_hardness(cls, text: str) -> Optional[float]:
        """Extract hardness from text with unit conversion."""
        for pattern in cls.HARDNESS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0][0])
                    unit = matches[0][1] if len(matches[0]) > 1 else 'GPa'
                    
                    # Convert HV to GPa (approximate)
                    if 'HV' in unit.upper():
                        value = value / 100.0
                    
                    return value
                except (ValueError, IndexError):
                    continue
        return None
    
    @classmethod
    def extract_fracture_toughness(cls, text: str) -> Optional[float]:
        """Extract fracture toughness K_IC."""
        for pattern in cls.KIC_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0][0])
                except (ValueError, IndexError):
                    continue
        return None
    
    @classmethod
    def identify_material(cls, text: str) -> Optional[str]:
        """Identify which ceramic material is discussed."""
        text_lower = text.lower()
        
        for material, keywords in cls.MATERIAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return material
        
        return None
    
    @classmethod
    def extract_all(cls, text: str) -> Dict[str, Any]:
        """Extract all properties from text."""
        return {
            'v50': cls.extract_v50(text),
            'hardness': cls.extract_hardness(text),
            'K_IC': cls.extract_fracture_toughness(text),
            'ceramic_type': cls.identify_material(text)
        }
```

```python
# src/ceramic_discovery/data_pipeline/collectors/semantic_scholar_collector.py
"""
Phase 2 Semantic Scholar Collector

NEW FUNCTIONALITY: Literature mining for experimental data.
Uses PropertyExtractor utility (new Phase 2 code).
"""

import pandas as pd
from typing import Dict, Any, List
import logging
import time

try:
    from semanticscholar import SemanticScholar
except ImportError:
    raise ImportError(
        "semanticscholar library not installed. Install with: pip install semanticscholar>=0.4.0"
    )

from .base_collector import BaseCollector
from ..utils.property_extractor import PropertyExtractor  # New Phase 2 utility

logger = logging.getLogger(__name__)


class SemanticScholarCollector(BaseCollector):
    """
    Semantic Scholar literature mining for experimental properties.
    
    This is new Phase 2 functionality with no Phase 1 dependencies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.queries = config.get('queries', [])
        self.papers_per_query = config.get('papers_per_query', 50)
        self.min_citations = config.get('min_citations', 5)
        self.rate_limit_delay = config.get('rate_limit_delay', 1)  # seconds
        
        self.sch = SemanticScholar()
        self.extractor = PropertyExtractor()
    
    def collect(self) -> pd.DataFrame:
        """
        Mine literature for experimental ballistic and mechanical data.
        
        Returns:
            DataFrame with extracted properties from papers
        """
        all_data = []
        processed_papers = set()
        
        for query in self.queries:
            logger.info(f"Searching Semantic Scholar: '{query}'")
            
            try:
                results = self.sch.search_paper(
                    query,
                    limit=self.papers_per_query,
                    fields=['title', 'abstract', 'year', 'authors', 
                           'citationCount', 'paperId', 'url']
                )
                
                for paper in results:
                    # Skip duplicates
                    if paper.paperId in processed_papers:
                        continue
                    processed_papers.add(paper.paperId)
                    
                    # Skip low-citation papers
                    if paper.citationCount < self.min_citations:
                        continue
                    
                    # Extract properties from title + abstract
                    text = f"{paper.title} {paper.abstract if paper.abstract else ''}"
                    properties = self.extractor.extract_all(text)
                    
                    # Only save if we extracted something useful
                    if any([properties['v50'], properties['hardness'], 
                           properties['K_IC']]) and properties['ceramic_type']:
                        
                        entry = {
                            'source': 'Literature',
                            'paper_id': paper.paperId,
                            'title': paper.title,
                            'year': paper.year,
                            'citations': paper.citationCount,
                            'url': paper.url if hasattr(paper, 'url') else None,
                            'ceramic_type': properties['ceramic_type'],
                            'v50_literature': properties['v50'],
                            'hardness_literature': properties['hardness'],
                            'K_IC_literature': properties['K_IC'],
                            'extraction_confidence': 'medium'  # Could enhance with NLP
                        }
                        
                        all_data.append(entry)
                        logger.info(f"  ✓ Extracted from: {paper.title[:60]}...")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Query failed '{query}': {e}")
                continue
        
        df = pd.DataFrame(all_data)
        logger.info(f"Literature mining: {len(df)} data points extracted")
        
        return df
    
    def get_source_name(self) -> str:
        return "SemanticScholar"
    
    def validate_config(self) -> bool:
        return 'queries' in self.config
```

**Integration Pattern:**
- ✅ No Phase 1 dependencies
- ✅ Uses new utility class (PropertyExtractor)
- ✅ External library (`semanticscholar`)
- ❌ Does NOT touch existing code

**New Dependencies:**
```txt
# Add to requirements.txt
semanticscholar>=0.4.0
```

***

### Requirement 6: MatWeb Scraper (New Functionality)

**User Story:** As a data collector, I want to scrape MatWeb for experimental ceramic properties without modifying existing code, so that I can add commercial data safely.

#### Acceptance Criteria

1. WHEN creating scraper THEN the system SHALL create NEW file `data_pipeline/collectors/matweb_collector.py`
2. WHEN using rate limiting THEN the system SHALL use NEW utility `data_pipeline/utils/rate_limiter.py`
3. WHEN making HTTP requests THEN the system SHALL use `requests` + `BeautifulSoup` with User-Agent header
4. WHEN parsing tables THEN the system SHALL extract hardness (Vickers), K_IC, density, compressive strength
5. WHEN converting units THEN the system SHALL normalize: Vickers HV to GPa (÷100), MPa to GPa (÷1000)
6. WHEN respecting limits THEN the system SHALL implement configurable delay between requests (default: 2 seconds)
7. WHEN handling failures THEN the system SHALL log HTTP errors, skip entry, continue with next material

**New Files to Create:**

```python
# src/ceramic_discovery/data_pipeline/utils/rate_limiter.py
"""
Phase 2 Utility: Rate Limiting for API/Web Requests

NEW FUNCTIONALITY: No Phase 1 dependencies.
"""

import time
from typing import Callable, Any
import functools
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API and web scraping requests.
    
    This is new Phase 2 utility for respectful data collection.
    """
    
    def __init__(self, delay_seconds: float = 1.0):
        self.delay = delay_seconds
        self.last_request_time = 0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for rate-limited functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            self.wait()
            return func(*args, **kwargs)
        return wrapper
```

```python
# src/ceramic_discovery/data_pipeline/collectors/matweb_collector.py
"""
Phase 2 MatWeb Scraper

NEW FUNCTIONALITY: Web scraping for experimental ceramic properties.
No Phase 1 dependencies.

ETHICAL NOTICE: 
- Respects robots.txt
- Rate limiting to avoid server load
- For research/non-commercial use only
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging
import re

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Required libraries not installed. Install with: "
        "pip install requests beautifulsoup4 lxml"
    )

from .base_collector import BaseCollector
from ..utils.rate_limiter import RateLimiter  # New Phase 2 utility

logger = logging.getLogger(__name__)


class MatWebCollector(BaseCollector):
    """
    MatWeb web scraper for experimental ceramic properties.
    
    This is new Phase 2 functionality with no Phase 1 dependencies.
    """
    
    BASE_URL = "http://www.matweb.com"
    SEARCH_URL = f"{BASE_URL}/search/QuickText.aspx"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.target_materials = config.get('target_materials', [])
        self.results_per_material = config.get('results_per_material', 20)
        
        # Rate limiter (respect server)
        delay = config.get('rate_limit_delay', 2)
        self.rate_limiter = RateLimiter(delay_seconds=delay)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot - Ceramic Discovery Framework)'
        })
    
    def collect(self) -> pd.DataFrame:
        """
        Scrape experimental data from MatWeb.
        
        Returns:
            DataFrame with experimental ceramic properties
        """
        all_data = []
        
        for material_name in self.target_materials:
            logger.info(f"Scraping MatWeb for: {material_name}")
            
            try:
                # Search with rate limiting
                self.rate_limiter.wait()
                response = self.session.get(
                    self.SEARCH_URL,
                    params={'SearchText': material_name},
                    timeout=15
                )
                
                if response.status_code != 200:
                    logger.warning(f"  Search failed for {material_name}: HTTP {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find result links
                result_links = soup.find_all('a', href=re.compile(r'/search/DataSheet\.aspx\?MatGUID='))
                logger.info(f"  Found {len(result_links)} results")
                
                # Process results
                for i, link in enumerate(result_links[:self.results_per_material]):
                    material_data = self._scrape_material_page(link['href'])
                    if material_data:
                        material_data['ceramic_type'] = material_name.split()[0]
                        all_data.append(material_data)
                
            except Exception as e:
                logger.error(f"  Error with {material_name}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        logger.info(f"MatWeb scraping: {len(df)} materials collected")
        
        return df
    
    def _scrape_material_page(self, path: str) -> Optional[Dict[str, Any]]:
        """Scrape individual material page."""
        try:
            # Rate limit
            self.rate_limiter.wait()
            
            url = self.BASE_URL + path
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            material_name = title_elem.text.strip() if title_elem else "Unknown"
            
            properties = {
                'source': 'MatWeb',
                'material_name': material_name,
                'url': url
            }
            
            # Parse property tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        prop_name = cells[0].get_text().strip().lower()
                        prop_value_text = cells[1].get_text().strip()
                        
                        # Hardness
                        if 'hardness' in prop_name and 'vickers' in prop_name:
                            value = self._parse_numeric(prop_value_text)
                            if value:
                                # Convert Vickers to GPa if needed
                                if 'kg/mm' in prop_value_text.lower():
                                    value = value / 100.0
                                properties['hardness_GPa'] = value
                        
                        # Fracture toughness
                        elif 'fracture' in prop_name and 'toughness' in prop_name:
                            value = self._parse_numeric(prop_value_text)
                            if value:
                                properties['K_IC_MPa_m05'] = value
                        
                        # Density
                        elif 'density' in prop_name and 'g/cc' in prop_value_text.lower():
                            value = self._parse_numeric(prop_value_text)
                            if value:
                                properties['density_matweb'] = value
                        
                        # Compressive strength
                        elif 'compressive' in prop_name and 'strength' in prop_name:
                            value = self._parse_numeric(prop_value_text)
                            if value:
                                # Convert MPa to GPa
                                if 'mpa' in prop_value_text.lower():
                                    value = value / 1000.0
                                properties['compressive_strength_GPa'] = value
            
            # Only return if we got useful data
            if any(k in properties for k in ['hardness_GPa', 'K_IC_MPa_m05']):
                logger.info(f"    ✓ Scraped: {material_name[:50]}...")
                return properties
            
            return None
            
        except Exception as e:
            logger.warning(f"    Failed to scrape page: {e}")
            return None
    
    @staticmethod
    def _parse_numeric(text: str) -> Optional[float]:
        """Extract first numeric value from text."""
        text = text.replace(',', '').strip()
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def get_source_name(self) -> str:
        return "MatWeb"
    
    def validate_config(self) -> bool:
        return 'target_materials' in self.config
```

**Integration Pattern:**
- ✅ No Phase 1 dependencies
- ✅ Uses new RateLimiter utility
- ✅ External libraries (requests, beautifulsoup4)
- ✅ Respects web scraping ethics
- ❌ Does NOT touch existing code

**New Dependencies:**
```txt
# Add to requirements.txt
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

***

### Requirement 7: NIST Baseline Collector (New Functionality)

**User Story:** As a validation engineer, I want embedded NIST reference data without modifying existing NIST client, so that I have baseline values without breaking Phase 1.

#### Acceptance Criteria

1. WHEN creating baseline collector THEN the system SHALL create NEW file `data_pipeline/collectors/nist_baseline_collector.py`
2. WHEN loading baseline data THEN the collector SHALL read from Phase 2 configuration file `config/data_pipeline_config.yaml` (NOT existing Phase 1 config)
3. WHEN existing NIST client exists THEN the collector SHALL NOT import or use it (Phase 1 has different NIST integration from Tasks 1-20)
4. WHEN defining properties THEN the system SHALL include 5 reference materials with density, hardness, K_IC
5. WHEN validating THEN the collector SHALL ensure all baseline materials have complete data

**New File to Create:**

```python
# src/ceramic_discovery/data_pipeline/collectors/nist_baseline_collector.py
"""
Phase 2 NIST Baseline Collector

NEW FUNCTIONALITY: Embedded reference data.
Does NOT use existing NISTClient from Phase 1 (different purpose).

Phase 1 NISTClient (Task 3): Temperature-dependent thermochemical properties
Phase 2 NISTBaselineCollector: Reference baseline values for validation
"""

import pandas as pd
from typing import Dict, Any
import logging

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class NISTBaselineCollector(BaseCollector):
    """
    NIST baseline reference data collector.
    
    This is new Phase 2 functionality separate from existing Phase 1 NISTClient.
    Provides gold-standard reference values for validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Baseline data from config
        self.baseline_materials = config.get('baseline_materials', {})
    
    def collect(self) -> pd.DataFrame:
        """
        Create DataFrame from embedded NIST baseline data.
        
        Returns:
            DataFrame with reference baseline values
        """
        logger.info("Loading NIST baseline reference data")
        
        records = []
        for formula, props in self.baseline_materials.items():
            record = {
                'source': 'NIST',
                'formula': formula,
                'ceramic_type': formula,
                **props  # density, hardness, K_IC, source
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"NIST baseline: {len(df)} reference materials")
        
        return df
    
    def get_source_name(self) -> str:
        return "NIST_Baseline"
    
    def validate_config(self) -> bool:
        """Ensure all baseline materials have required properties."""
        required_props = ['density', 'hardness', 'K_IC', 'source']
        
        for formula, props in self.baseline_materials.items():
            if not all(prop in props for prop in required_props):
                logger.error(f"Baseline material {formula} missing required properties")
                return False
        
        return True
```

**Integration Pattern:**
- ✅ No Phase 1 dependencies
- ✅ Does NOT conflict with existing `NISTClient` (different purpose)
- ✅ Reads from Phase 2 config only
- ❌ Does NOT touch Phase 1 NIST integration

**Configuration (in Phase 2 config file):**
```yaml
# config/data_pipeline_config.yaml (NEW FILE)
nist:
  baseline_materials:
    SiC:
      density: 3.21
      hardness: 26.0
      K_IC: 4.0
      source: "NIST SRD 30"
    B4C:
      density: 2.52
      hardness: 35.0
      K_IC: 2.9
      source: "NIST SRD 30"
    WC:
      density: 15.6
      hardness: 22.0
      K_IC: 8.5
      source: "NIST SRD 30"
    TiC:
      density: 4.93
      hardness: 28.0
      K_IC: 5.0
      source: "NIST SRD 30"
    Al2O3:
      density: 3.96
      hardness: 18.5
      K_IC: 4.0
      source: "NIST SRD 30"
```

***

### Requirement 8: Multi-Source Integration Pipeline

**User Story:** As a data scientist, I want intelligent integration of all Phase 2 sources without modifying existing DataCombiner, so that I can create a master dataset safely.

#### Acceptance Criteria

1. WHEN creating integration pipeline THEN the system SHALL create NEW file `data_pipeline/pipelines/integration_pipeline.py`
2. WHEN using fuzzy matching THEN the system SHALL create NEW utility `data_pipeline/integration/formula_matcher.py` (NOT modify existing code)
3. WHEN existing DataCombiner exists THEN the pipeline MAY import it (read-only) for reference but SHALL NOT call or modify it
4. WHEN merging sources THEN the pipeline SHALL load ALL CSV files from `data/processed/` directory
5. WHEN deduplicating THEN the system SHALL use SequenceMatcher for formula similarity with threshold 0.95
6. WHEN resolving conflicts THEN the system SHALL apply source priority: Literature > MatWeb > NIST > MP > JARVIS > AFLOW
7. WHEN calculating derived properties THEN the system SHALL compute: specific_hardness, ballistic_efficacy, dop_resistance
8. WHEN saving output THEN the system SHALL create `data/final/master_dataset.csv` with integration report

**New Files to Create:**

```python
# src/ceramic_discovery/data_pipeline/integration/formula_matcher.py
"""
Phase 2 Utility: Formula Fuzzy Matching

NEW FUNCTIONALITY: No Phase 1 dependencies.
"""

from difflib import SequenceMatcher
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FormulaMatcher:
    """
    Fuzzy matching for chemical formulas.
    
    This is new Phase 2 utility for multi-source deduplication.
    """
    
    @staticmethod
    def similarity(formula1: str, formula2: str) -> float:
        """Calculate similarity between two formulas (0-1)."""
        return SequenceMatcher(None, formula1, formula2).ratio()
    
    @classmethod
    def find_best_match(cls, formula: str, candidates: List[str], 
                       threshold: float = 0.95) -> Tuple[str, float]:
        """
        Find best matching formula from candidates.
        
        Args:
            formula: Formula to match
            candidates: List of candidate formulas
            threshold: Minimum similarity score
        
        Returns:
            Tuple of (best_match, score) or (None, 0) if no match
        """
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            score = cls.similarity(formula, candidate)
            if score > best_score and score >= threshold:
                best_match = candidate
                best_score = score
        
        return best_match, best_score
```

```python
# src/ceramic_discovery/data_pipeline/pipelines/integration_pipeline.py
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
from typing import Dict, List
import logging

from ..integration.formula_matcher import FormulaMatcher  # New Phase 2 utility

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
        else:
            # Concatenate all sources
            logger.info("Concatenating all sources")
            return pd.concat(sources.values(), ignore_index=True)
    
    def _merge_sources(self, df_master: pd.DataFrame, 
                      sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge additional sources with fuzzy formula matching."""
        
        for source_name, df_source in sources.items():
            if source_name == 'mp_data':
                continue  # Already in master
            
            logger.info(f"Merging {source_name}...")
            
            new_materials = []
            
            for _, row in df_source.iterrows():
                formula = row.get('formula', '')
                if not formula:
                    continue
                
                # Find matching material in master
                master_formulas = df_master['formula'].dropna().unique()
                best_match, score = self.matcher.find_best_match(
                    formula, master_formulas, self.dedup_threshold
                )
                
                if best_match:
                    # Merge properties into existing material
                    idx = df_master[df_master['formula'] == best_match].index[0]
                    for col in row.index:
                        if col.endswith(('_jarvis', '_aflow', '_literature', '_matweb', '_nist')):
                            if pd.notna(row[col]):
                                df_master.at[idx, col] = row[col]
                else:
                    # New material, add to list
                    new_materials.append(row)
            
            # Append new materials
            if new_materials:
                df_new = pd.DataFrame(new_materials)
                df_master = pd.concat([df_master, df_new], ignore_index=True)
            
            logger.info(f"  After {source_name}: {len(df_master)} materials")
        
        return df_master
    
    def _create_combined_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combined property columns averaging across sources."""
        
        # Hardness combined
        hardness_cols = [c for c in df.columns if 'hardness' in c.lower() and c != 'hardness_combined']
        if hardness_cols:
            df['hardness_combined'] = df[hardness_cols].mean(axis=1, skipna=True)
            logger.info(f"Created hardness_combined from {len(hardness_cols)} sources")
        
        # K_IC combined
        kic_cols = [c for c in df.columns if 'k_ic' in c.lower() or 'fracture_toughness' in c.lower()]
        if kic_cols:
            df['K_IC_combined'] = df[kic_cols].mean(axis=1, skipna=True)
            logger.info(f"Created K_IC_combined from {len(kic_cols)} sources")
        
        # Density combined (prefer MP, then others)
        density_priority = ['density_mp', 'density_jarvis', 'density_aflow', 'density_matweb', 'density']
        df['density_combined'] = df[density_priority].bfill(axis=1).iloc[:, 0]
        
        return df
    
    def _calculate_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived ballistic-relevant properties."""
        
        # Specific hardness (hardness / density)
        if 'hardness_combined' in df.columns and 'density_combined' in df.columns:
            df['specific_hardness'] = df['hardness_combined'] / df['density_combined']
            logger.info("Calculated specific_hardness")
        
        # Ballistic efficacy (simplified: hardness / density)
        if 'hardness_combined' in df.columns and 'density_combined' in df.columns:
            df['ballistic_efficacy'] = df['hardness_combined'] / df['density_combined']
            logger.info("Calculated ballistic_efficacy")
        
        # DOP resistance (depth of penetration resistance)
        if all(c in df.columns for c in ['hardness_combined', 'K_IC_combined', 'density_combined']):
            df['dop_resistance'] = (
                np.sqrt(df['hardness_combined']) * 
                df['K_IC_combined'] / 
                df['density_combined']
            )
            logger.info("Calculated dop_resistance")
        
        return df
    
    def _save_master_dataset(self, df: pd.DataFrame):
        """Save master dataset to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = self.output_dir / 'master_dataset.csv'
        df.to_csv(output_file, index=False)
        
        logger.info(f"Master dataset saved: {output_file}")
    
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
            "",
            "DATA SOURCES",
            "-" * 80,
        ]
        
        for source_name, source_df in sources.items():
            report_lines.append(f"  {source_name}: {len(source_df)} entries")
        
        report_lines.extend([
            "",
            "PROPERTY COVERAGE",
            "-" * 80,
            f"  With hardness_combined: {df['hardness_combined'].notna().sum() if 'hardness_combined' in df.columns else 0}",
            f"  With K_IC_combined: {df['K_IC_combined'].notna().sum() if 'K_IC_combined' in df.columns else 0}",
            f"  With V50 (literature): {df['v50_literature'].notna().sum() if 'v50_literature' in df.columns else 0}",
            f"  With thermal_conductivity: {df['thermal_conductivity_300K'].notna().sum() if 'thermal_conductivity_300K' in df.columns else 0}",
            "",
            "DERIVED PROPERTIES",
            "-" * 80,
            f"  Specific hardness: {df['specific_hardness'].notna().sum() if 'specific_hardness' in df.columns else 0}",
            f"  Ballistic efficacy: {df['ballistic_efficacy'].notna().sum() if 'ballistic_efficacy' in df.columns else 0}",
            f"  DOP resistance: {df['dop_resistance'].notna().sum() if 'dop_resistance' in df.columns else 0}",
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
```

**Integration Pattern:**
- ✅ Completely new Phase 2 pipeline
- ✅ Uses new FormulaMatcher utility
- ✅ Does NOT import or use existing `DataCombiner` (different purpose)
- ✅ Creates `master_dataset.csv` for Phase 1 ML pipeline to consume
- ❌ Does NOT modify any Phase 1 code

***

### Requirement 9: Collection Orchestrator

**User Story:** As a pipeline operator, I want a single command to run all data collection without modifying existing code, so that I can reproduce results safely.

#### Acceptance Criteria

1. WHEN creating orchestrator THEN the system SHALL create NEW file `data_pipeline/pipelines/collection_orchestrator.py`
2. WHEN running pipeline THEN the system SHALL execute collectors in order: MP → JARVIS → AFLOW → Semantic Scholar → MatWeb → NIST → Integration
3. WHEN a collector fails THEN the system SHALL log error, mark failed, continue with remaining collectors
4. WHEN tracking progress THEN the system SHALL use `tqdm` progress bars for user feedback
5. WHEN saving checkpoints THEN the system SHALL save intermediate CSV files to allow resuming
6. WHEN generating summary THEN the system SHALL create execution report with per-source runtime and status

**New File to Create:**

```python
# src/ceramic_discovery/data_pipeline/pipelines/collection_orchestrator.py
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
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.collectors = self._initialize_collectors()
        self.results = {}
    
    def _initialize_collectors(self) -> Dict:
        """Initialize all collectors from config."""
        collectors = {}
        
        # Materials Project
        if self.config.get('materials_project', {}).get('enabled', True):
            collectors['mp'] = MaterialsProjectCollector(
                self.config['materials_project']
            )
        
        # JARVIS
        if self.config.get('jarvis', {}).get('enabled', True):
            collectors['jarvis'] = JarvisCollector(
                self.config['jarvis']
            )
        
        # AFLOW
        if self.config.get('aflow', {}).get('enabled', True):
            collectors['aflow'] = AFLOWCollector(
                self.config['aflow']
            )
        
        # Semantic Scholar
        if self.config.get('semantic_scholar', {}).get('enabled', True):
            collectors['semantic_scholar'] = SemanticScholarCollector(
                self.config['semantic_scholar']
            )
        
        # MatWeb
        if self.config.get('matweb', {}).get('enabled', True):
            collectors['matweb'] = MatWebCollector(
                self.config['matweb']
            )
        
        # NIST Baseline
        if self.config.get('nist', {}).get('enabled', True):
            collectors['nist'] = NISTBaselineCollector(
                self.config['nist']
            )
        
        return collectors
    
    def run_all(self) -> Path:
        """
        Run all collectors and integration.
        
        Returns:
            Path to master_dataset.csv
        """
        logger.info("Starting Phase 2 data collection pipeline")
        start_time = time.time()
        
        # Run collectors
        with tqdm(total=len(self.collectors), desc="Collection Progress") as pbar:
            for name, collector in self.collectors.items():
                logger.info(f"\nRunning {name} collector...")
                
                try:
                    collector_start = time.time()
                    
                    # Collect data
                    df = collector.collect()
                    
                    # Save to processed/
                    output_file = self.processed_dir / f"{name}_data.csv"
                    df.to_csv(output_file, index=False)
                    
                    collector_time = time.time() - collector_start
                    
                    self.results[name] = {
                        'status': 'SUCCESS',
                        'count': len(df),
                        'runtime': collector_time,
                        'output': str(output_file)
                    }
                    
                    logger.info(f"✓ {name}: {len(df)} materials in {collector_time:.1f}s")
                    
                except Exception as e:
                    logger.error(f"✗ {name} failed: {e}")
                    self.results[name] = {
                        'status': 'FAILED',
                        'error': str(e),
                        'runtime': 0
                    }
                
                pbar.update(1)
        
        # Integration
        logger.info("\nRunning integration pipeline...")
        try:
            integration_start = time.time()
            
            integration = IntegrationPipeline(self.config.get('integration', {}))
            df_master = integration.integrate_all_sources()
            
            integration_time = time.time() - integration_start
            
            self.results['integration'] = {
                'status': 'SUCCESS',
                'count': len(df_master),
                'runtime': integration_time
            }
            
            logger.info(f"✓ Integration: {len(df_master)} materials in {integration_time:.1f}s")
            
        except Exception as e:
            logger.error(f"✗ Integration failed: {e}")
            self.results['integration'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            raise
        
        # Generate final report
        total_time = time.time() - start_time
        self._generate_summary_report(total_time)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE in {total_time/60:.1f} minutes")
        logger.info(f"{'='*60}")
        
        return Path('data/final/master_dataset.csv')
    
    def _generate_summary_report(self, total_time: float):
        """Generate pipeline execution summary."""
        
        report_lines = [
            "=" * 80,
            "PHASE 2 DATA COLLECTION PIPELINE SUMMARY",
            f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "COLLECTOR RESULTS",
            "-" * 80,
        ]
        
        for name, result in self.results.items():
            status = result['status']
            if status == 'SUCCESS':
                count = result.get('count', 0)
                runtime = result.get('runtime', 0)
                report_lines.append(f"  ✓ {name:20s} {count:5d} materials  {runtime:6.1f}s")
            else:
                error = result.get('error', 'Unknown error')
                report_lines.append(f"  ✗ {name:20s} FAILED: {error}")
        
        report_lines.extend([
            "",
            f"Total Runtime: {total_time/60:.1f} minutes",
            "=" * 80
        ])
        
        # Write report
        report_file = Path('data/final/pipeline_summary.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report: {report_file}")
```

**Integration Pattern:**
- ✅ New Phase 2 orchestrator
- ✅ Runs all new collectors in sequence
- ✅ Creates master_dataset.csv for Phase 1 to consume
- ❌ Does NOT call or modify any Phase 1 code

***

### Requirement 10: Configuration Management (Isolated)

**User Story:** As a framework user, I want Phase 2 configuration completely separate from Phase 1 config, so that I don't accidentally break existing settings.

#### Acceptance Criteria

1. WHEN creating Phase 2 config THEN the system SHALL create NEW file `config/data_pipeline_config.yaml` (separate from existing Phase 1 configs)
2. WHEN loading config THEN the system SHALL create NEW utility `data_pipeline/utils/config_loader.py`
3. WHEN existing config files exist THEN the system SHALL NOT read from or modify them
4. WHEN supporting env vars THEN the system SHALL allow `${VAR_NAME}` substitution for API keys
5. WHEN validating config THEN the system SHALL use schema validation with clear error messages

**New Files to Create:**

```python
# src/ceramic_discovery/data_pipeline/utils/config_loader.py
"""
Phase 2 Configuration Loader

NEW FUNCTIONALITY: Loads Phase 2 configuration.
Does NOT read or modify existing Phase 1 configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and validate Phase 2 configuration.
    
    This is new Phase 2 utility that does NOT interact with existing Phase 1 config.
    """
    
    @staticmethod
    def load(config_path: str = 'config/data_pipeline_config.yaml') -> Dict[str, Any]:
        """
        Load Phase 2 configuration from YAML file.
        
        Args:
            config_path: Path to Phase 2 config file
        
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Phase 2 config not found: {config_path}\n"
                f"Create config/data_pipeline_config.yaml with Phase 2 settings."
            )
        
        logger.info(f"Loading Phase 2 config from {config_path}")
        
        with open(config_file, 'r') as f:
            config_text = f.read()
        
        # Substitute environment variables
        config_text = ConfigLoader._substitute_env_vars(config_text)
        
        # Parse YAML
        config = yaml.safe_load(config_text)
        
        # Validate
        ConfigLoader._validate(config)
        
        logger.info("Phase 2 configuration loaded successfully")
        
        return config
    
    @staticmethod
    def _substitute_env_vars(text: str) -> str:
        """Substitute ${VAR_NAME} with environment variables."""
        pattern = r'\$\{(\w+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning(f"Environment variable {var_name} not set")
                return match.group(0)  # Keep ${VAR_NAME} if not found
            return value
        
        return re.sub(pattern, replacer, text)
    
    @staticmethod
    def _validate(config: Dict[str, Any]):
        """Validate Phase 2 configuration structure."""
        required_sections = ['materials_project', 'jarvis', 'aflow', 
                            'semantic_scholar', 'matweb', 'nist', 
                            'integration', 'output']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate Materials Project
        mp_config = config['materials_project']
        if 'api_key' not in mp_config or not mp_config['api_key']:
            raise ValueError("Materials Project API key required in config")
        
        logger.info("Configuration validation passed")
```

```yaml
# config/data_pipeline_config.yaml (NEW FILE - separate from Phase 1 config)
"""
Phase 2 Data Pipeline Configuration

IMPORTANT: This is SEPARATE from existing Phase 1 configuration.
Do NOT modify existing config files in config/*.yaml
"""

# Materials Project (Phase 2 batch collection)
materials_project:
  enabled: true
  api_key: ${MP_API_KEY}  # Set environment variable: export MP_API_KEY="your_key"
  target_systems:
    - "Si-C"
    - "B-C"
    - "W-C"
    - "Ti-C"
    - "Al-O"
  batch_size: 1000
  max_materials: 5000
  retry_attempts: 3
  retry_delay: 5

# JARVIS-DFT (Phase 2 carbide collection)
jarvis:
  enabled: true
  data_source: "dft_3d"
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

# AFLOW (Phase 2 thermal properties)
aflow:
  enabled: true
  ceramics:
    SiC: ["Si", "C"]
    B4C: ["B", "C"]
    WC: ["W", "C"]
    TiC: ["Ti", "C"]
    Al2O3: ["Al", "O"]
  max_per_system: 500
  thermal_properties: true
  exclude_ldau: true

# Semantic Scholar (Phase 2 literature mining)
semantic_scholar:
  enabled: true
  queries:
    - "ballistic limit velocity ceramic armor"
    - "V50 silicon carbide penetration"
    - "hardness fracture toughness ceramic"
    - "boron carbide armor ballistic"
  papers_per_query: 50
  min_citations: 5
  rate_limit_delay: 1

# MatWeb (Phase 2 web scraping)
matweb:
  enabled: true
  target_materials:
    - "Silicon Carbide"
    - "Boron Carbide"
    - "Tungsten Carbide"
    - "Titanium Carbide"
    - "Aluminum Oxide"
    - "Alumina"
  results_per_material: 20
  rate_limit_delay: 2

# NIST Baseline (Phase 2 reference data)
nist:
  enabled: true
  baseline_materials:
    SiC:
      density: 3.21
      hardness: 26.0
      K_IC: 4.0
      source: "NIST SRD 30"
    B4C:
      density: 2.52
      hardness: 35.0
      K_IC: 2.9
      source: "NIST SRD 30"
    WC:
      density: 15.6
      hardness: 22.0
      K_IC: 8.5
      source: "NIST SRD 30"
    TiC:
      density: 4.93
      hardness: 28.0
      K_IC: 5.0
      source: "NIST SRD 30"
    Al2O3:
      density: 3.96
      hardness: 18.5
      K_IC: 4.0
      source: "NIST SRD 30"

# Integration (Phase 2 multi-source merging)
integration:
  deduplication_threshold: 0.95
  conflict_resolution: "weighted_average"
  min_properties: 5
  source_priority:
    - "Literature"
    - "MatWeb"
    - "NIST"
    - "MaterialsProject"
    - "JARVIS"
    - "AFLOW"

# Output (Phase 2 results)
output:
  master_file: "data/final/master_dataset.csv"
  report_file: "data/final/integration_report.txt"
  summary_file: "data/final/pipeline_summary.txt"
  backup_dir: "data/backups/"
```

**Integration Pattern:**
- ✅ Completely separate configuration file
- ✅ New config loader utility
- ❌ Does NOT read or modify existing Phase 1 config files
- ❌ Does NOT conflict with existing configuration

***

### Requirement 11: Main Execution Script

**User Story:** As an end user, I want a single command to run all Phase 2 data collection, so that I can reproduce results without touching Phase 1 code.

#### Acceptance Criteria

1. WHEN creating main script THEN the system SHALL create NEW file `scripts/run_data_collection.py`
2. WHEN running script THEN it SHALL verify Phase 1 tests pass BEFORE starting collection
3. WHEN script completes THEN it SHALL verify Phase 1 tests STILL pass AFTER collection
4. WHEN any tests fail THEN the system SHALL abort

### Requirement 11: Main Execution Script (Continued)

#### Acceptance Criteria (Continued)

1. WHEN creating main script THEN the system SHALL create NEW file `scripts/run_data_collection.py`
2. WHEN running script THEN it SHALL verify Phase 1 tests pass BEFORE starting collection
3. WHEN script completes THEN it SHALL verify Phase 1 tests STILL pass AFTER collection
4. WHEN any tests fail THEN the system SHALL abort and report which phase broke
5. WHEN executing THEN the system SHALL support command-line arguments: `--config`, `--skip-verification`, `--sources`

**New File to Create:**

```python
# scripts/run_data_collection.py
"""
Phase 2 Data Collection - Main Execution Script

CRITICAL: This script verifies Phase 1 code remains intact.
- Runs Phase 1 tests BEFORE collection
- Runs Phase 1 tests AFTER collection
- Aborts if any Phase 1 tests fail

Usage:
    python scripts/run_data_collection.py
    python scripts/run_data_collection.py --config config/data_pipeline_config.yaml
    python scripts/run_data_collection.py --skip-verification  # NOT RECOMMENDED
    python scripts/run_data_collection.py --sources mp,jarvis,aflow
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ceramic_discovery.data_pipeline.utils.config_loader import ConfigLoader
from ceramic_discovery.data_pipeline.pipelines.collection_orchestrator import CollectionOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def verify_phase1_tests() -> bool:
    """
    Verify all existing Phase 1 tests still pass.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFYING PHASE 1 CODE INTEGRITY")
    logger.info("="*60)
    
    try:
        # Run existing test suite
        result = subprocess.run(
            ['pytest', 'tests/', '-v', '--tb=short', '-x'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✓ All Phase 1 tests PASSED")
            logger.info(f"  Test output: {result.stdout.splitlines()[-5:]}")  # Last 5 lines
            return True
        else:
            logger.error("✗ Phase 1 tests FAILED")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Test verification TIMEOUT (exceeded 5 minutes)")
        return False
    except FileNotFoundError:
        logger.error("✗ pytest not found. Install with: pip install pytest")
        return False
    except Exception as e:
        logger.error(f"✗ Test verification failed with error: {e}")
        return False


def verify_dependencies() -> bool:
    """Verify all required dependencies are installed."""
    logger.info("\n" + "="*60)
    logger.info("VERIFYING DEPENDENCIES")
    logger.info("="*60)
    
    required_packages = {
        'mp_api': 'mp-api',
        'jarvis': 'jarvis-tools',
        'aflow': 'aflow',
        'semanticscholar': 'semanticscholar',
        'requests': 'requests',
        'bs4': 'beautifulsoup4',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.warning(f"  ✗ {package} not installed")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("✓ All dependencies installed")
    return True


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Phase 2 Data Collection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with verification
  python scripts/run_data_collection.py
  
  # Use custom config
  python scripts/run_data_collection.py --config my_config.yaml
  
  # Skip verification (NOT RECOMMENDED - may break Phase 1)
  python scripts/run_data_collection.py --skip-verification
  
  # Run only specific sources
  python scripts/run_data_collection.py --sources mp,jarvis,aflow
        """
    )
    
    parser.add_argument(
        '--config',
        default='config/data_pipeline_config.yaml',
        help='Path to Phase 2 configuration file (default: config/data_pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip Phase 1 test verification (NOT RECOMMENDED - may break existing code)'
    )
    
    parser.add_argument(
        '--sources',
        help='Comma-separated list of sources to collect (e.g., mp,jarvis,aflow)'
    )
    
    return parser.parse_args()


def filter_sources(config: dict, sources: str) -> dict:
    """Filter configuration to only include specified sources."""
    if not sources:
        return config
    
    source_list = [s.strip().lower() for s in sources.split(',')]
    
    # Disable sources not in list
    source_map = {
        'mp': 'materials_project',
        'jarvis': 'jarvis',
        'aflow': 'aflow',
        'scholar': 'semantic_scholar',
        'matweb': 'matweb',
        'nist': 'nist'
    }
    
    for short_name, config_name in source_map.items():
        if short_name not in source_list and config_name in config:
            config[config_name]['enabled'] = False
            logger.info(f"Disabled source: {config_name}")
    
    return config


def main():
    """Main execution function."""
    args = parse_arguments()
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 DATA COLLECTION PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Step 1: Verify dependencies
    if not verify_dependencies():
        logger.error("\n❌ ABORTED: Missing dependencies")
        sys.exit(1)
    
    # Step 2: Verify Phase 1 tests BEFORE collection (unless skipped)
    if not args.skip_verification:
        logger.info("\nStep 1/3: Verifying Phase 1 code integrity BEFORE collection...")
        
        if not verify_phase1_tests():
            logger.error("\n❌ ABORTED: Phase 1 tests failing BEFORE collection")
            logger.error("Your existing code has issues. Fix them before running Phase 2.")
            sys.exit(1)
    else:
        logger.warning("\n⚠️  SKIPPING Phase 1 verification (NOT RECOMMENDED)")
        logger.warning("Proceeding without verification may break existing code!")
    
    # Step 3: Load Phase 2 configuration
    logger.info("\nStep 2/3: Loading Phase 2 configuration...")
    
    try:
        config = ConfigLoader.load(args.config)
        logger.info(f"✓ Configuration loaded from {args.config}")
        
        # Filter sources if specified
        if args.sources:
            config = filter_sources(config, args.sources)
        
    except Exception as e:
        logger.error(f"\n❌ ABORTED: Configuration loading failed: {e}")
        sys.exit(1)
    
    # Step 4: Run data collection
    logger.info("\nStep 3/3: Running data collection pipeline...")
    
    try:
        orchestrator = CollectionOrchestrator(config)
        master_dataset_path = orchestrator.run_all()
        
        logger.info(f"\n✓ Data collection complete!")
        logger.info(f"  Master dataset: {master_dataset_path}")
        logger.info(f"  Total materials: {len(open(master_dataset_path).readlines()) - 1}")
        
    except Exception as e:
        logger.error(f"\n❌ Data collection failed: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)
    
    # Step 5: Verify Phase 1 tests AFTER collection (unless skipped)
    if not args.skip_verification:
        logger.info("\nStep 4/4: Verifying Phase 1 code integrity AFTER collection...")
        
        if not verify_phase1_tests():
            logger.error("\n❌ WARNING: Phase 1 tests failing AFTER collection")
            logger.error("Phase 2 may have broken existing code!")
            logger.error("Investigate changes and restore if needed.")
            sys.exit(1)
        
        logger.info("\n✓ Phase 1 code integrity verified - no existing code broken!")
    
    # Success summary
    logger.info("\n" + "="*60)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review: data/final/integration_report.txt")
    logger.info(f"  2. Use master dataset: data/final/master_dataset.csv")
    logger.info(f"  3. Train ML models with existing Phase 1 pipeline:")
    logger.info(f"     python -m ceramic_discovery.ml.train_models \\")
    logger.info(f"       --input data/final/master_dataset.csv")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
```

**Integration Pattern:**
- ✅ Standalone script (no modifications to existing code)
- ✅ Verifies Phase 1 tests before AND after
- ✅ Aborts if any existing tests fail
- ✅ Clear user guidance
- ❌ Does NOT modify any Phase 1 files

***

### Requirement 12: Testing Framework for Phase 2

**User Story:** As a quality engineer, I want comprehensive tests for Phase 2 code only, so that I can ensure reliability without touching Phase 1 tests.

#### Acceptance Criteria

1. WHEN creating tests THEN the system SHALL create NEW directory `tests/data_pipeline/` (separate from existing `tests/`)
2. WHEN testing collectors THEN each collector SHALL have unit tests with mocked API responses
3. WHEN testing integration THEN the system SHALL use small sample datasets (10-50 materials)
4. WHEN running Phase 2 tests THEN the system SHALL use command: `pytest tests/data_pipeline/ -v`
5. WHEN measuring coverage THEN Phase 2 code SHALL achieve ≥85% coverage without affecting Phase 1 coverage

**New Test Structure:**

```
tests/
├── [EXISTING PHASE 1 TESTS - DO NOT MODIFY]
│   ├── test_materials_project_client.py       [EXISTS - DO NOT TOUCH]
│   ├── test_jarvis_client.py                  [EXISTS - DO NOT TOUCH]
│   ├── test_data_combiner.py                  [EXISTS - DO NOT TOUCH]
│   ├── ... (157+ existing tests)
│
└── data_pipeline/                              [NEW DIRECTORY]
    ├── __init__.py                             [NEW]
    ├── conftest.py                             [NEW - fixtures]
    │
    ├── collectors/                             [NEW]
    │   ├── __init__.py
    │   ├── test_base_collector.py              [NEW]
    │   ├── test_materials_project_collector.py [NEW]
    │   ├── test_jarvis_collector.py            [NEW]
    │   ├── test_aflow_collector.py             [NEW]
    │   ├── test_semantic_scholar_collector.py  [NEW]
    │   ├── test_matweb_collector.py            [NEW]
    │   └── test_nist_baseline_collector.py     [NEW]
    │
    ├── integration/                            [NEW]
    │   ├── __init__.py
    │   ├── test_formula_matcher.py             [NEW]
    │   └── test_quality_validator.py           [NEW]
    │
    ├── pipelines/                              [NEW]
    │   ├── __init__.py
    │   ├── test_collection_orchestrator.py     [NEW]
    │   ├── test_integration_pipeline.py        [NEW]
    │   └── test_end_to_end.py                  [NEW]
    │
    └── utils/                                  [NEW]
        ├── __init__.py
        ├── test_property_extractor.py          [NEW]
        ├── test_rate_limiter.py                [NEW]
        └── test_config_loader.py               [NEW]
```

**Example Test Files:**

```python
# tests/data_pipeline/conftest.py
"""
Phase 2 Test Fixtures

NEW: Fixtures for Phase 2 tests only.
Does NOT modify existing Phase 1 fixtures.
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_mp_config():
    """Sample Materials Project configuration."""
    return {
        'api_key': 'test_key',
        'target_systems': ['Si-C'],
        'batch_size': 10,
        'max_materials': 20
    }


@pytest.fixture
def sample_materials_df():
    """Sample materials DataFrame for testing."""
    return pd.DataFrame([
        {
            'source': 'MaterialsProject',
            'material_id': 'mp-1',
            'formula': 'SiC',
            'density_mp': 3.21,
            'formation_energy_mp': -0.5
        },
        {
            'source': 'JARVIS',
            'jid': 'JVASP-1',
            'formula': 'SiC',
            'density_jarvis': 3.20,
            'formation_energy_jarvis': -0.52
        }
    ])


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directories."""
    processed = tmp_path / "data" / "processed"
    final = tmp_path / "data" / "final"
    processed.mkdir(parents=True)
    final.mkdir(parents=True)
    return tmp_path
```

```python
# tests/data_pipeline/collectors/test_base_collector.py
"""
Test BaseCollector abstract interface.

NEW: Phase 2 tests only.
"""

import pytest
from ceramic_discovery.data_pipeline.collectors.base_collector import BaseCollector


def test_base_collector_is_abstract():
    """Verify BaseCollector cannot be instantiated."""
    with pytest.raises(TypeError, match="abstract"):
        BaseCollector({'test': 'config'})


def test_base_collector_requires_methods():
    """Verify subclass must implement abstract methods."""
    
    class IncompleteCollector(BaseCollector):
        def get_source_name(self):
            return "Test"
    
    with pytest.raises(TypeError, match="abstract"):
        IncompleteCollector({'test': 'config'})
```

```python
# tests/data_pipeline/collectors/test_materials_project_collector.py
"""
Test Materials Project Collector (Phase 2).

IMPORTANT: This tests the NEW Phase 2 wrapper collector.
Does NOT test existing Phase 1 MaterialsProjectClient.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from ceramic_discovery.data_pipeline.collectors.materials_project_collector import MaterialsProjectCollector


@pytest.fixture
def mp_config():
    return {
        'api_key': 'test_key',
        'target_systems': ['Si-C'],
        'batch_size': 10,
        'max_materials': 20
    }


def test_mp_collector_initialization(mp_config):
    """Test collector initializes with existing Phase 1 client."""
    
    with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
        collector = MaterialsProjectCollector(mp_config)
        
        # Verify existing client was instantiated (composition pattern)
        MockClient.assert_called_once_with(api_key='test_key')
        assert collector.target_systems == ['Si-C']


def test_mp_collector_collect(mp_config):
    """Test data collection returns DataFrame."""
    
    # Mock existing Phase 1 client
    with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
        
        # Mock materials returned by existing client
        mock_material = Mock()
        mock_material.material_id = 'mp-1'
        mock_material.formula = 'SiC'
        
        mock_instance = MockClient.return_value
        mock_instance.get_materials.return_value = [mock_material]
        
        # Create collector and collect
        collector = MaterialsProjectCollector(mp_config)
        df = collector.collect()
        
        # Verify
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'material_id' in df.columns
        assert df.iloc[0]['formula'] == 'SiC'
        
        # Verify existing client method was called (not modified)
        mock_instance.get_materials.assert_called()


def test_mp_collector_handles_api_errors(mp_config):
    """Test collector handles API failures gracefully."""
    
    with patch('ceramic_discovery.data_pipeline.collectors.materials_project_collector.MaterialsProjectClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.get_materials.side_effect = Exception("API Error")
        
        collector = MaterialsProjectCollector(mp_config)
        df = collector.collect()
        
        # Should return empty DataFrame, not crash
        assert isinstance(df, pd.DataFrame)
```

```python
# tests/data_pipeline/utils/test_property_extractor.py
"""
Test Property Extractor utility.

NEW: Phase 2 functionality.
"""

import pytest
from ceramic_discovery.data_pipeline.utils.property_extractor import PropertyExtractor


def test_extract_v50_basic():
    """Test V50 extraction from text."""
    text = "The material showed V50 = 850 m/s in ballistic tests"
    
    v50 = PropertyExtractor.extract_v50(text)
    
    assert v50 == 850.0


def test_extract_v50_alternative_formats():
    """Test V50 extraction with different formats."""
    test_cases = [
        ("V_50: 920 m/s", 920.0),
        ("ballistic limit: 875 m/s", 875.0),
        ("V50 = 1050 ms", 1050.0),
    ]
    
    for text, expected in test_cases:
        assert PropertyExtractor.extract_v50(text) == expected


def test_extract_hardness_gpa():
    """Test hardness extraction in GPa."""
    text = "The hardness measured was 26.5 GPa"
    
    hardness = PropertyExtractor.extract_hardness(text)
    
    assert hardness == 26.5


def test_extract_hardness_hv_conversion():
    """Test hardness extraction with HV to GPa conversion."""
    text = "Vickers hardness: 2800 HV"
    
    hardness = PropertyExtractor.extract_hardness(text)
    
    assert hardness == 28.0  # 2800 / 100


def test_extract_kic():
    """Test fracture toughness extraction."""
    text = "fracture toughness: 4.5 MPa·m^0.5"
    
    kic = PropertyExtractor.extract_fracture_toughness(text)
    
    assert kic == 4.5


def test_identify_material():
    """Test material identification from text."""
    test_cases = [
        ("Silicon carbide showed excellent properties", "SiC"),
        ("The B4C ceramic exhibited high hardness", "B4C"),
        ("Tungsten carbide (WC) is widely used", "WC"),
        ("Alumina (Al2O3) samples were tested", "Al2O3"),
    ]
    
    for text, expected in test_cases:
        assert PropertyExtractor.identify_material(text) == expected


def test_extract_all():
    """Test extracting all properties at once."""
    text = """
    Silicon carbide (SiC) was tested for ballistic applications.
    Vickers hardness: 2600 HV
    Fracture toughness: 4.0 MPa·m
    V50 = 850 m/s
    """
    
    result = PropertyExtractor.extract_all(text)
    
    assert result['ceramic_type'] == 'SiC'
    assert result['hardness'] == 26.0  # 2600 HV / 100
    assert result['K_IC'] == 4.0
    assert result['v50'] == 850.0
```

```python
# tests/data_pipeline/pipelines/test_end_to_end.py
"""
End-to-end test for Phase 2 pipeline.

NEW: Integration test for complete Phase 2 workflow.
Does NOT test Phase 1 functionality.
"""

import pytest
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock

from ceramic_discovery.data_pipeline.pipelines.collection_orchestrator import CollectionOrchestrator
from ceramic_discovery.data_pipeline.utils.config_loader import ConfigLoader


@pytest.fixture
def minimal_config():
    """Minimal config for testing."""
    return {
        'materials_project': {
            'enabled': True,
            'api_key': 'test',
            'target_systems': ['Si-C'],
            'batch_size': 10,
            'max_materials': 20
        },
        'jarvis': {
            'enabled': True,
            'carbide_metals': ['Si']
        },
        'aflow': {'enabled': False},
        'semantic_scholar': {'enabled': False},
        'matweb': {'enabled': False},
        'nist': {
            'enabled': True,
            'baseline_materials': {
                'SiC': {'density': 3.21, 'hardness': 26.0, 'K_IC': 4.0, 'source': 'NIST'}
            }
        },
        'integration': {
            'deduplication_threshold': 0.95,
            'source_priority': ['MaterialsProject', 'JARVIS', 'NIST']
        },
        'output': {
            'master_file': 'data/final/master_dataset.csv'
        }
    }


def test_end_to_end_pipeline(minimal_config, tmp_path, monkeypatch):
    """Test complete Phase 2 pipeline end-to-end."""
    
    # Patch paths to use tmp_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "final").mkdir(parents=True)
    
    # Mock collectors to return sample data
    with patch.multiple(
        'ceramic_discovery.data_pipeline.pipelines.collection_orchestrator',
        MaterialsProjectCollector=MagicMock(),
        JarvisCollector=MagicMock(),
        NISTBaselineCollector=MagicMock()
    ):
        # Create orchestrator
        orchestrator = CollectionOrchestrator(minimal_config)
        
        # Mock collector.collect() to return sample DataFrames
        for collector in orchestrator.collectors.values():
            collector.collect.return_value = pd.DataFrame([
                {'formula': 'SiC', 'source': 'Test', 'density': 3.21}
            ])
        
        # Run pipeline
        master_path = orchestrator.run_all()
        
        # Verify outputs
        assert master_path.exists()
        assert (tmp_path / "data" / "final" / "integration_report.txt").exists()
        
        # Verify master dataset
        df_master = pd.read_csv(master_path)
        assert len(df_master) > 0
        assert 'formula' in df_master.columns
```

**Running Phase 2 Tests:**

```bash
# Run ONLY Phase 2 tests (does NOT run Phase 1 tests)
pytest tests/data_pipeline/ -v

# Run with coverage for Phase 2 code only
pytest tests/data_pipeline/ --cov=src/ceramic_discovery/data_pipeline --cov-report=html

# Run Phase 1 tests to verify nothing broken
pytest tests/ --ignore=tests/data_pipeline/ -v

# Run ALL tests (Phase 1 + Phase 2)
pytest tests/ -v
```

**Integration Pattern:**
- ✅ Completely separate test directory
- ✅ Tests Phase 2 code only
- ✅ Does NOT modify or run existing Phase 1 tests
- ✅ Can be run independently

***

### Requirement 13: Documentation

**User Story:** As a framework user, I want clear documentation for Phase 2 that explains integration with Phase 1, so that I understand how both systems work together.

#### Acceptance Criteria

1. WHEN creating documentation THEN the system SHALL create NEW file `docs/data_pipeline_guide.md`
2. WHEN explaining architecture THEN documentation SHALL clearly distinguish Phase 1 vs Phase 2 components
3. WHEN providing examples THEN documentation SHALL show how to use Phase 2 without modifying Phase 1
4. WHEN troubleshooting THEN documentation SHALL include common issues and "Phase 1 vs Phase 2" debugging guide

**New Documentation Files:**

```markdown
# docs/data_pipeline_guide.md

# Phase 2 Data Pipeline Guide

## Overview

Phase 2 adds automated multi-source data collection to the Ceramic Armor Discovery Framework **without modifying any existing Phase 1 code**.

### Architecture: Phase 1 + Phase 2

```
┌─────────────────────────────────────────┐
│ PHASE 1 (Existing - 34 tasks complete) │
├─────────────────────────────────────────┤
│ -  DFT data collection                   │
│ -  ML pipeline                           │
│ -  Screening engine                      │
│ -  Validation framework                  │
│ -  All 157+ tests                        │
└─────────────────────────────────────────┘
           ↑ imports (read-only)
           │
┌─────────────────────────────────────────┐
│ PHASE 2 (New - this guide)              │
├─────────────────────────────────────────┤
│ -  6-source data collection              │
│ -  Automated integration                 │
│ -  Master dataset creation               │
│ -  Phase 2 tests                         │
└─────────────────────────────────────────┘
```

**Key Principle:** Phase 2 imports FROM Phase 1, but Phase 1 never imports Phase 2.

## Quick Start

### 1. Installation

```
# Install Phase 2 dependencies (adds to existing)
pip install aflow>=0.0.10 semanticscholar>=0.4.0 beautifulsoup4>=4.12.0 lxml>=4.9.0
```

### 2. Configuration

```
# Create Phase 2 config (separate from Phase 1 config)
cp config/data_pipeline_config.yaml.example config/data_pipeline_config.yaml

# Set Materials Project API key
export MP_API_KEY="your_api_key_here"
```

### 3. Run Data Collection

```
# Complete pipeline with verification
python scripts/run_data_collection.py

# This will:
# 1. Verify Phase 1 tests pass (before)
# 2. Collect data from 6 sources
# 3. Integrate into master_dataset.csv
# 4. Verify Phase 1 tests still pass (after)
```

### 4. Use with Phase 1 ML Pipeline

```
# Train models using Phase 2 data (existing Phase 1 code)
python -m ceramic_discovery.ml.train_models \
  --input data/final/master_dataset.csv \
  --targets hardness_combined K_IC_combined

# Existing Phase 1 screening still works
python -m ceramic_discovery.screening.screen_materials \
  --data data/final/master_dataset.csv
```

## Phase 1 vs Phase 2: What's What?

### Phase 1 Components (DO NOT MODIFY)

| Component | File | Purpose |
|-----------|------|---------|
| MaterialsProjectClient | `dft/materials_project_client.py` | Direct MP API access |
| JarvisClient | `dft/jarvis_client.py` | JARVIS data loading |
| DataCombiner | `dft/data_combiner.py` | Multi-source merging |
| ModelTrainer | `ml/model_trainer.py` | ML model training |
| ScreeningEngine | `screening/screening_engine.py` | Material screening |

### Phase 2 Components (NEW)

| Component | File | Purpose |
|-----------|------|---------|
| MaterialsProjectCollector | `data_pipeline/collectors/materials_project_collector.py` | **Wraps** Phase 1 client for batch collection |
| JarvisCollector | `data_pipeline/collectors/jarvis_collector.py` | **Wraps** Phase 1 client for carbide filtering |
| AFLOWCollector | `data_pipeline/collectors/aflow_collector.py` | New thermal property collection |
| SemanticScholarCollector | `data_pipeline/collectors/semantic_scholar_collector.py` | Literature mining |
| MatWebCollector | `data_pipeline/collectors/matweb_collector.py` | Web scraping |
| IntegrationPipeline | `data_pipeline/pipelines/integration_pipeline.py` | Creates master_dataset.csv |

**Key Difference:** Phase 2 collectors **use** Phase 1 clients (composition), they don't **replace** them.

## Common Workflows

### Workflow 1: Complete Data Collection

```
# Run all 6 sources
python scripts/run_data_collection.py
```

**Output:**
- `data/processed/mp_data.csv`
- `data/processed/jarvis_data.csv`
- `data/processed/aflow_data.csv`
- `data/processed/semantic_scholar_data.csv`
- `data/processed/matweb_data.csv`
- `data/processed/nist_baseline.csv`
- `data/final/master_dataset.csv` ← Use this for ML

### Workflow 2: Selective Source Collection

```
# Only Materials Project + JARVIS
python scripts/run_data_collection.py --sources mp,jarvis

# Only AFLOW (thermal properties)
python scripts/run_data_collection.py --sources aflow
```

### Workflow 3: Update Existing Data

```
# Re-run collection (overwrites data/processed/)
python scripts/run_data_collection.py

# Integration automatically merges with existing
```

## Troubleshooting

### "Phase 1 tests failing BEFORE collection"

**Problem:** Existing code has issues unrelated to Phase 2.

**Solution:**
```
# Identify which test is failing
pytest tests/ --ignore=tests/data_pipeline/ -v -x

# Fix Phase 1 issue first
# Then re-run Phase 2 collection
```

### "Phase 1 tests failing AFTER collection"

**Problem:** Phase 2 accidentally modified Phase 1 code.

**Solution:**
```
# Check git diff to see what changed
git diff src/ceramic_discovery/

# If Phase 2 modified Phase 1 files, report as bug
# Restore Phase 1 files:
git checkout src/ceramic_discovery/dft/
git checkout src/ceramic_discovery/ml/
# etc.
```

### "Cannot import MaterialsProjectClient"

**Problem:** Phase 1 code not installed or path issue.

**Solution:**
```
# Verify Phase 1 installation
python -c "from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient; print('OK')"

# If fails, install framework
cd /path/to/ceramic-armor-discovery
pip install -e .
```

### "API key not found"

**Problem:** Materials Project API key not set.

**Solution:**
```
# Set environment variable
export MP_API_KEY="your_key"

# Or add to .env file (gitignored)
echo "export MP_API_KEY='your_key'" >> .env
source .env
```

## Advanced Usage

### Custom Configuration

```
# config/my_custom_config.yaml
materials_project:
  enabled: true
  api_key: ${MP_API_KEY}
  target_systems: ["Si-C"]  # Only SiC
  max_materials: 1000       # Limit to 1000

# Disable other sources
aflow:
  enabled: false
semantic_scholar:
  enabled: false
matweb:
  enabled: false
```

```
python scripts/run_data_collection.py --config config/my_custom_config.yaml
```

### Programmatic Usage

```
# Use Phase 2 from Python code
from ceramic_discovery.data_pipeline.utils.config_loader import ConfigLoader
from ceramic_discovery.data_pipeline.pipelines.collection_orchestrator import CollectionOrchestrator

# Load config
config = ConfigLoader.load('config/data_pipeline_config.yaml')

# Run collection
orchestrator = CollectionOrchestrator(config)
master_path = orchestrator.run_all()

# Use with Phase 1 ML
import pandas as pd
df = pd.read_csv(master_path)

from ceramic_discovery.ml.model_trainer import ModelTrainer
trainer = ModelTrainer()
model = trainer.train(df, target='hardness_combined')
```

## Data Schema

### Master Dataset Columns

| Column | Description | Sources |
|--------|-------------|---------|
| `material_id` | Unique identifier | MP |
| `formula` | Chemical formula | All |
| `ceramic_type` | Base ceramic (SiC, B4C, etc.) | All |
| `density_mp` | Density from MP (g/cm³) | MP |
| `density_jarvis` | Density from JARVIS | JARVIS |
| `density_combined` | Average density | Computed |
| `formation_energy_mp` | Formation energy (eV/atom) | MP |
| `hull_distance_mp` | Energy above hull | MP |
| `hardness_combined` | Average hardness (GPa) | Literature, MatWeb, NIST |
| `K_IC_combined` | Average K_IC (MPa·m^0.5) | Literature, MatWeb, NIST |
| `thermal_conductivity_300K` | Thermal conductivity (W/m·K) | AFLOW |
| `v50_literature` | Ballistic V50 (m/s) | Literature |
| `specific_hardness` | hardness / density | Computed |
| `ballistic_efficacy` | hardness / density | Computed |
| `dop_resistance` | √hardness × K_IC / density | Computed |

## Best Practices

### 1. Always Verify Phase 1 First

```
# Before running Phase 2
pytest tests/ --ignore=tests/data_pipeline/ -v
```

### 2. Use Version Control

```
# Before Phase 2 collection
git add -A
git commit -m "Before Phase 2 data collection"

# After Phase 2 collection
git diff  # Should show NO changes to Phase 1 files
```

### 3. Backup Existing Data

```
# Before re-running collection
cp -r data/final data/backups/final_$(date +%Y%m%d)
```

### 4. Incremental Testing

```
# Test one source at a time
python scripts/run_data_collection.py --sources mp
pytest tests/data_pipeline/collectors/test_materials_project_collector.py -v

python scripts/run_data_collection.py --sources jarvis
pytest tests/data_pipeline/collectors/test_jarvis_collector.py -v

# etc.
```

## Performance

### Expected Runtimes

| Source | Materials | Runtime |
|--------|-----------|---------|
| Materials Project | 3,000-5,000 | 10-15 min |
| JARVIS | 400-600 | 8-12 min |
| AFLOW | 1,500-2,000 | 15-20 min |
| Semantic Scholar | 50-100 | 25-35 min |
| MatWeb | 30-50 | 40-50 min |
| NIST Baseline | 5 | <1 min |
| Integration | 6,000-7,000 | 3-5 min |
| **Total** | **~6,500** | **~2 hours** |

### Optimization Tips

```
# In config file, reduce limits for faster testing
materials_project:
  max_materials: 100  # Instead of 5000

aflow:
  max_per_system: 50  # Instead of 500

semantic_scholar:
  papers_per_query: 10  # Instead of 50
```

## Support

### Phase 1 Issues
- File: Existing framework code (`dft/`, `ml/`, etc.)
- Tests: `tests/test_*.py` (not in `tests/data_pipeline/`)
- Contact: Original framework maintainers

### Phase 2 Issues
- File: New pipeline code (`data_pipeline/`)
- Tests: `tests/data_pipeline/`
- Documentation: This guide

### Reporting Bugs

If Phase 2 breaks Phase 1:
```
# Verify issue
pytest tests/ --ignore=tests/data_pipeline/ -v

# Report with:
# 1. Phase 1 test failure output
# 2. Steps to reproduce
# 3. Git diff showing changes (should be none)
```

---

## Next Steps

1. **Run Phase 2 collection:**
   ```
   python scripts/run_data_collection.py
   ```

2. **Train ML models (Phase 1 code):**
   ```
   python -m ceramic_discovery.ml.train_models \
     --input data/final/master_dataset.csv
   ```

3. **Generate predictions and rankings:**
   ```
   python -m ceramic_discovery.screening.screen_materials \
     --data data/final/master_dataset.csv
   ```

4. **Write your paper with comprehensive data!**
```

***

### Requirement 14: Migration Safety Checklist

**User Story:** As a developer integrating Phase 2, I want a pre-flight checklist, so that I can verify nothing will break before starting.

#### Acceptance Criteria

1. WHEN creating checklist THEN the system SHALL create NEW file `docs/PRE_FLIGHT_CHECKLIST.md`
2. WHEN following checklist THEN each item SHALL be verifiable with a command
3. WHEN any item fails THEN the checklist SHALL indicate DO NOT PROCEED

**New Documentation File:**

```markdown
# docs/PRE_FLIGHT_CHECKLIST.md

# Phase 2 Integration - Pre-Flight Checklist

## ⚠️ CRITICAL: Complete ALL items before running Phase 2

**Purpose:** Verify Phase 1 code is intact and Phase 2 won't break anything.

---

## Pre-Integration Checklist

### ☐ 1. Verify Phase 1 Code Exists

```
# Check Phase 1 files exist
ls src/ceramic_discovery/dft/materials_project_client.py
ls src/ceramic_discovery/dft/jarvis_client.py
ls src/ceramic_discovery/dft/data_combiner.py
ls src/ceramic_discovery/ml/model_trainer.py

# Expected: All files exist
# If any missing: Phase 1 not complete, DO NOT PROCEED
```

### ☐ 2. Run Phase 1 Tests

```
# Run existing test suite
pytest tests/ --ignore=tests/data_pipeline/ -v

# Expected: All tests pass (157+ tests)
# If any fail: Fix Phase 1 first, DO NOT PROCEED
```

### ☐ 3. Verify No Uncommitted Changes

```
# Check git status
git status

# Expected: "nothing to commit, working tree clean"
# If changes exist: Commit them first for safety
git add -A
git commit -m "Pre-Phase-2 checkpoint"
```

### ☐ 4. Create Backup

```
# Backup entire codebase
cd ..
cp -r ceramic-armor-discovery ceramic-armor-discovery-backup-$(date +%Y%m%d)

# Expected: Backup directory created
ls -d ceramic-armor-discovery-backup-*
```

### ☐ 5. Verify Python Environment

```
# Check Python version
python --version

# Expected: Python 3.11+ (Phase 1 requirement)
# If < 3.11: Upgrade Python
```

### ☐ 6. Check Existing Dependencies

```
# Verify Phase 1 dependencies installed
pip list | grep -E "(mp-api|jarvis-tools|scikit-learn|pandas)"

# Expected: All present
# If missing: pip install -r requirements.txt
```

### ☐ 7. Verify Disk Space

```
# Check available space
df -h .

# Expected: >10GB free (for data collection)
# If < 10GB: Free up space
```

### ☐ 8. Test Phase 1 Functionality

```
# Test Phase 1 imports
python -c "from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient; print('✓ Phase 1 imports OK')"
python -c "from ceramic_discovery.ml.model_trainer import ModelTrainer; print('✓ Phase 1 ML OK')"

# Expected: Both print "✓"
# If ImportError: Fix Phase 1 installation
```

---

## Installation Checklist

### ☐ 9. Install Phase 2 Dependencies

```
# Install new dependencies
pip install aflow>=0.0.10 semanticscholar>=0.4.0 beautifulsoup4>=4.12.0 lxml>=4.9.0

# Expected: All install successfully
```

### ☐ 10. Create Phase 2 Configuration

```
# Copy template config
cp config/data_pipeline_config.yaml.example config/data_pipeline_config.yaml

# Set API key
export MP_API_KEY="your_api_key"

# Verify
python -c "import os; print('✓ API key set' if os.getenv('MP_API_KEY') else '✗ API key missing')"

# Expected: "✓ API key set"
```

### ☐ 11. Verify Phase 2 Structure

```
# Check Phase 2 files exist
ls src/ceramic_discovery/data_pipeline/collectors/base_collector.py
ls src/ceramic_discovery/data_pipeline/pipelines/collection_orchestrator.py
ls scripts/run_data_collection.py

# Expected: All files exist
# If missing: Phase 2 code not added yet
```

### ☐ 12. Test Phase 2 Imports

```
# Test Phase 2 imports (should NOT affect Phase 1)
python -c "from ceramic_discovery.data_pipeline.collectors.base_collector import BaseCollector; print('✓ Phase 2 imports OK')"

# Expected: "✓ Phase 2 imports OK"
# If ImportError: Check Phase 2 installation
```

---

## Verification Checklist

### ☐ 13. Run Phase 1 Tests Again

```
# Verify Phase 1 still works after Phase 2 installation
pytest tests/ --ignore=tests/data_pipeline/ -v

# Expected: All tests still pass
# If any fail: Phase 2 broke something, ROLLBACK
```

### ☐ 14. Test Phase 2 in Isolation

```
# Run Phase 2 tests only
pytest tests/data_pipeline/ -v

# Expected: Phase 2 tests pass
# If any fail: Fix Phase 2 code
```

### ☐ 15. Dry Run Data Collection

```
# Test collection with minimal config
python scripts/run_data_collection.py --sources nist

# Expected: Completes successfully, creates data/processed/nist_baseline.csv
# If error: Check logs, fix config
```

---

## Final Safety Check

### ☐ 16. Verify No Phase 1 Modifications

```
# Check that Phase 2 didn't modify Phase 1 files
git diff src/ceramic_discovery/dft/
git diff src/ceramic_discovery/ml/
git diff src/ceramic_discovery/screening/
git diff tests/test_*.py

# Expected: No output (no changes)
# If changes: Phase 2 modified Phase 1, ABORT AND ROLLBACK
```

### ☐ 17. Final Phase 1 Test

```
# One last verification
pytest tests/ --ignore=tests/data_pipeline/ -v -x

# Expected: All pass
# If any fail: DO NOT PROCEED with full collection
```

---

## ✅ All Items Complete?

**If ALL checkboxes checked:**
- ✅ Proceed with Phase 2 data collection:
  ```
  python scripts/run_data_collection.py
  ```

**If ANY item failed:**
- ❌ DO NOT PROCEED
- Fix issues first
- Re-run checklist from start

---

## Emergency Rollback

**If Phase 2 breaks something:**

```
# 1. Stop immediately
Ctrl+C

# 2. Restore from backup
cd ..
rm -rf ceramic-armor-discovery
cp -r ceramic-armor-discovery-backup-YYYYMMDD ceramic-armor-discovery
cd ceramic-armor-discovery

# 3. Verify Phase 1 works
pytest tests/ --ignore=tests/data_pipeline/ -v

# 4. Report issue with logs
cat logs/collection_*.log
```

---

## Post-Collection Verification

**After Phase 2 completes, verify:**

### ☐ 18. Master Dataset Created

```
ls data/final/master_dataset.csv

# Expected: File exists
```

### ☐ 19. Phase 1 Still Works

```
pytest tests/ --ignore=tests/data_pipeline/ -v

# Expected: All Phase 1 tests still pass
```

### ☐ 20. Use Master Dataset with Phase 1

```
# Test that Phase 1 can use Phase 2 data
python -m ceramic_discovery.ml.train_models \
  --input data/final/master_dataset.csv \
  --targets hardness_combined

# Expected: Training completes successfully
```

---

## Success Criteria

✅ **Phase 2 integration successful if:**
1. All 20 checklist items completed
2. Phase 1 tests pass before AND after
3. Master dataset created
4. Phase 1 code can use Phase 2 data
5. No Phase 1 files modified

❌ **Phase 2 integration failed if:**
1. Any Phase 1 test fails after Phase 2
2. Phase 1 files modified
3. Master dataset not created
4. Phase 1 code cannot use Phase 2 data

In case of failure: **ROLLBACK** to backup and report issue.
```

***

## Technical Implementation Standards

### Code Quality Requirements

**All Phase 2 code must meet:**

1. **Type Hints:** All function signatures must include type hints
   ```python
   def collect(self) -> pd.DataFrame:  # ✓ Correct
   def collect(self):                  # ✗ Missing type hint
   ```

2. **Docstrings:** Google-style docstrings for all classes and public methods
   ```python
   def collect(self) -> pd.DataFrame:
       """
       Collect data from source.
       
       Returns:
           DataFrame with collected materials
       
       Raises:
           DataCollectionError: If collection fails
       """
   ```

3. **Logging:** Use Python logging module, not print statements
   ```python
   logger.info("Collecting data...")  # ✓ Correct
   print("Collecting data...")        # ✗ Use logging
   ```

4. **Error Handling:** Specific exceptions, not bare except
   ```python
   except aflow.AFLOWAPIError as e:  # ✓ Correct
   except:                            # ✗ Too broad
   ```

### File Organization Standards

```
Phase 2 Module Structure:

src/ceramic_discovery/data_pipeline/
├── __init__.py                    # Module exports
├── collectors/
│   ├── __init__.py               # Collector exports
│   ├── base_collector.py         # Abstract base (50 lines)
│   ├── [source]_collector.py    # Each collector (100-200 lines)
│   └── ...
├── integration/
│   ├── __init__.py
│   ├── formula_matcher.py        # Single responsibility (50-100 lines)
│   └── ...
├── pipelines/
│   ├── __init__.py
│   ├── collection_orchestrator.py  # Main pipeline (200-300 lines)
│   └── integration_pipeline.py     # Integration (200-300 lines)
└── utils/
    ├── __init__.py
    ├── property_extractor.py     # Text extraction (100-150 lines)
    ├── rate_limiter.py           # API throttling (50-100 lines)
    └── config_loader.py          # Config management (100-150 lines)
```

**File Size Limits:**
- Collector files: 150-250 lines each
- Utility files: 50-150 lines each
- Pipeline files: 200-400 lines each
- Test files: 100-300 lines each

**If file exceeds limit:** Split into multiple files with clear responsibilities.

### Dependency Management

**Phase 2 adds these dependencies:**

```txt
# requirements_phase2.txt
aflow>=0.0.10
semanticscholar>=0.4.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Existing dependencies (Phase 1 - do not modify versions)
mp-api>=0.37.0
jarvis-tools>=2023.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
pytest>=7.4.0
```

**Installation:**
```bash
# Install Phase 2 dependencies without affecting Phase 1
pip install -r requirements_phase2.txt
```

### Performance Requirements

**Phase 2 must meet these targets:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total Runtime | ≤2 hours | Time to master_dataset.csv |
| Memory Usage | ≤4 GB peak | Monitor with `memory_profiler` |
| Disk Usage | ≤5 GB total | Including all CSVs |
| API Requests | ≤10,000 total | Across all sources |
| Error Rate | ≤5% per source | Failed materials / total |

**Monitoring:**
```bash
# Runtime
time python scripts/run_data_collection.py

# Memory
python -m memory_profiler scripts/run_data_collection.py

# Disk usage
du -sh data/
```

***

## Success Criteria Summary

### Phase 2 is successfully integrated if:

✅ **1. Isolation Verified**
- All Phase 2 code in `data_pipeline/` module
- No modifications to Phase 1 files (`dft/`, `ml/`, `screening/`, etc.)
- Phase 2 imports from Phase 1, but Phase 1 doesn't import Phase 2

✅ **2. Tests Pass**
- All 157+ existing Phase 1 tests pass before Phase 2
- All 157+ existing Phase 1 tests pass after Phase 2
- Phase 2 tests (50+ new tests) pass in isolation
- Combined test suite runs without conflicts

✅ **3. Data Collection Works**
- Master dataset created at `data/final/master_dataset.csv`
- 6,000+ materials with properties from 6 sources
- Integration report generated with statistics
- Collection completes in ≤2 hours

✅ **4. Integration Functional**
- Phase 1 ML pipeline can train on Phase 2 data
- Phase 1 screening works with Phase 2 master dataset
- No breaking changes to Phase 1 APIs

✅ **5. Documentation Complete**
- User guide explains Phase 1 vs Phase 2
- Pre-flight checklist available
- Troubleshooting guide included
- Examples provided

***

## Risk Mitigation Strategies

### Risk: Phase 2 accidentally modifies Phase 1

**Mitigation:**
1. Pre-flight checklist requires git commit before Phase 2
2. Main script runs Phase 1 tests before AND after collection
3. Git diff verification in checklist
4. Clear module boundaries (data_pipeline/ separate from rest)

### Risk: Dependency version conflicts

**Mitigation:**
1. Phase 2 uses same versions as Phase 1 where possible
2. Only adds new dependencies (aflow, semanticscholar, beautifulsoup4, lxml)
3. Requirements clearly document Phase 1 vs Phase 2 deps
4. Virtual environment recommended for isolation

### Risk: Data collection failures

**Mitigation:**
1. Graceful degradation (continue if one source fails)
2. Checkpointing (save intermediate CSVs)
3. Retry logic with exponential backoff
4. Clear error messages with actionable steps

### Risk: User confusion about Phase 1 vs Phase 2

**Mitigation:**
1. Clear naming (`data_pipeline/` vs existing modules)
2. Documentation distinguishes components
3. Separate configuration files
4. Examples show both systems working together

***

This completes the **Phase 2 Requirements Document with Strict Isolation**. The specification ensures Phase 2 integrates safely without breaking any of your existing 34 completed tasks. Every requirement emphasizes the DO NOT MODIFY principle and provides clear verification steps.