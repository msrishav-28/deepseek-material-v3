# Implementation Plan: Phase 2 - Isolated Multi-Source Data Collection Pipeline

## ⚠️ CRITICAL SAFETY RULES

**BEFORE STARTING ANY TASK:**
1. Run `pytest tests/ -v` to verify all 157+ Phase 1 tests pass
2. If any tests fail, STOP and report to user
3. Do NOT proceed with Phase 2 implementation if Phase 1 is broken

**DURING IMPLEMENTATION:**
- Create ONLY new files in `src/ceramic_discovery/data_pipeline/`
- Do NOT modify any existing files in `dft/`, `ml/`, `screening/`, `validation/`, `utils/`
- Import FROM existing modules (read-only), never modify them
- Use composition (HAS-A), not inheritance (IS-A)

**AFTER EACH MAJOR MILESTONE:**
- Run `pytest tests/ -v` again to verify Phase 1 still works
- If tests fail, identify what broke and fix immediately

***

## Task Overview

This implementation creates a completely isolated data collection pipeline that:
- Collects data from 6 sources (MP, JARVIS, AFLOW, Semantic Scholar, MatWeb, NIST)
- Integrates all sources into a master dataset
- Does NOT modify any existing Phase 1 code
- Produces `data/final/master_dataset.csv` for Phase 1 ML pipeline to consume

Total estimated materials: 6,000-7,000 with 50+ properties each

***

## Tasks

- [x] 1. Pre-flight verification and project setup





  - Verify all Phase 1 tests pass before starting
  - Create Phase 2 directory structure
  - Set up Phase 2 configuration file
  - Install new dependencies
  - _Requirements: All requirements (safety check)_

- [x] 1.1 Run Phase 1 test suite verification


  - Execute `pytest tests/ -v --tb=short` and capture full output
  - Count total tests and verify all pass (expect 157+ tests)
  - Save test output to `logs/phase1_baseline_tests.log`
  - Check for any warnings or deprecations
  - If ANY tests fail, ABORT immediately and report to user with:
    - Which test(s) failed
    - Error messages
    - Recommendation to fix Phase 1 before proceeding
  - Expected output: "X passed in Y seconds" with X >= 157
  - _Requirements: Requirement 11.2_

- [x] 1.2 Create Phase 2 directory structure



  - Create main module: `src/ceramic_discovery/data_pipeline/`
  - Create subdirectories with __init__.py:
    - `collectors/` - for all 6 data source collectors
    - `integration/` - for formula matching and deduplication
    - `pipelines/` - for orchestrator and integration pipeline
    - `utils/` - for config loader, rate limiter, property extractor
  - Create data directories:
    - `data/processed/` - for individual source CSV files
    - `data/final/` - for master_dataset.csv
    - `data/backups/` - for backup copies
  - Create test directories:
    - `tests/data_pipeline/`
    - `tests/data_pipeline/collectors/`
    - `tests/data_pipeline/integration/`
    - `tests/data_pipeline/pipelines/`
    - `tests/data_pipeline/utils/`
  - Verify structure with `tree src/ceramic_discovery/data_pipeline/`
  - _Requirements: Requirement 1.1_

- [x] 1.3 Create Phase 2 configuration file


  - Create `config/data_pipeline_config.yaml` (NEW file, separate from Phase 1)
  - Add header comment explaining this is Phase 2 config only
  - Configure Materials Project section:
    - api_key: ${MP_API_KEY} (environment variable)
    - target_systems: ["Si-C", "B-C", "W-C", "Ti-C", "Al-O"]
    - batch_size: 1000, max_materials: 5000
  - Configure JARVIS section:
    - carbide_metals: ["Si", "B", "W", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo"]
    - data_source: "dft_3d"
  - Configure AFLOW section:
    - ceramics: {SiC: ["Si", "C"], B4C: ["B", "C"], ...}
    - max_per_system: 500
  - Configure Semantic Scholar section:
    - queries: ["ballistic limit velocity ceramic armor", ...]
    - papers_per_query: 50, min_citations: 5
  - Configure MatWeb section:
    - target_materials: ["Silicon Carbide", "Boron Carbide", ...]
    - rate_limit_delay: 2
  - Configure NIST baseline section:
    - baseline_materials with SiC, B4C, WC, TiC, Al2O3 reference data
  - Configure integration section:
    - deduplication_threshold: 0.95
    - source_priority: ["Literature", "MatWeb", "NIST", "MaterialsProject", "JARVIS", "AFLOW"]
  - Validate YAML syntax with `python -c "import yaml; yaml.safe_load(open('config/data_pipeline_config.yaml'))"`
  - _Requirements: Requirement 10.1, 10.2_

- [x] 1.4 Install Phase 2 dependencies


  - Create `requirements_phase2.txt` with new dependencies:
    - aflow>=0.0.10 (AFLOW API client)
    - semanticscholar>=0.4.0 (literature search)
    - requests>=2.31.0 (HTTP requests)
    - beautifulsoup4>=4.12.0 (HTML parsing)
    - lxml>=4.9.0 (XML/HTML parser)
    - tqdm>=4.65.0 (progress bars)
  - Run `pip install -r requirements_phase2.txt`
  - Verify each import works:
    - `python -c "import aflow; print('✓ aflow')"`
    - `python -c "from semanticscholar import SemanticScholar; print('✓ semanticscholar')"`
    - `python -c "import requests; print('✓ requests')"`
    - `python -c "from bs4 import BeautifulSoup; print('✓ beautifulsoup4')"`
    - `python -c "import lxml; print('✓ lxml')"`
    - `python -c "from tqdm import tqdm; print('✓ tqdm')"`
  - If any import fails, install individually and troubleshoot
  - _Requirements: Requirement 4.2, 5.2, 6.2_

- [x] 2. Implement base collector interface and utilities





  - Create abstract BaseCollector class
  - Create Phase 2 utility modules (config loader, rate limiter, property extractor)
  - _Requirements: Requirement 1_

- [x] 2.1 Implement BaseCollector abstract class


  - Create `src/ceramic_discovery/data_pipeline/collectors/base_collector.py`
  - Import ABC and abstractmethod from abc module
  - Import pandas, Path, logging, Dict, Any from typing
  - Define BaseCollector(ABC) class with:
    - __init__(self, config: Dict[str, Any]) - stores config, calls validate_config()
    - @abstractmethod collect(self) -> pd.DataFrame - must return DataFrame with 'source', 'formula' columns
    - @abstractmethod get_source_name(self) -> str - returns source identifier
    - @abstractmethod validate_config(self) -> bool - validates required config keys
    - save_to_csv(self, df, output_dir) -> Path - saves DataFrame to {source}_data.csv
    - log_stats(self, df) - logs collection statistics (count, property coverage)
  - Add comprehensive docstrings explaining:
    - This is Phase 2 wrapper around Phase 1 clients
    - Do NOT modify existing client classes
    - Use composition (HAS-A) not inheritance (IS-A)
  - Test import: `python -c "from ceramic_discovery.data_pipeline.collectors.base_collector import BaseCollector; print('✓')"`
  - _Requirements: Requirement 1.1, 1.2, 1.3, 1.5_

- [x] 2.2 Write unit tests for BaseCollector


  - Create `tests/data_pipeline/collectors/test_base_collector.py`
  - Import pytest, BaseCollector, pandas, tempfile
  - Test 1: test_base_collector_cannot_instantiate()
    - Verify TypeError raised when trying to instantiate BaseCollector directly
    - Assert error message mentions abstract methods
  - Test 2: test_abstract_methods_required()
    - Create minimal concrete subclass missing one abstract method
    - Verify TypeError raised
  - Test 3: test_concrete_subclass_works()
    - Create complete concrete subclass implementing all abstract methods
    - Verify instantiation succeeds
    - Verify collect() returns DataFrame
  - Test 4: test_save_to_csv()
    - Create test DataFrame with source and formula columns
    - Call save_to_csv() with temp directory
    - Verify CSV file created with correct name
    - Verify CSV content matches DataFrame
  - Test 5: test_log_stats()
    - Create test DataFrame with various properties
    - Capture log output
    - Verify statistics logged (count, property coverage)
  - Run tests: `pytest tests/data_pipeline/collectors/test_base_collector.py -v`
  - All tests must pass before proceeding
  - _Requirements: Requirement 1_

- [x] 2.3 Implement ConfigLoader utility


  - Create `src/ceramic_discovery/data_pipeline/utils/config_loader.py`
  - Implement YAML loading with environment variable substitution
  - Add configuration validation
  - Handle missing config file errors gracefully
  - _Requirements: Requirement 10.2, 10.4, 10.5_

- [x] 2.4 Write unit tests for ConfigLoader


  - Create `tests/data_pipeline/utils/test_config_loader.py`
  - Test 1: test_load_valid_config()
    - Create temp YAML file with valid Phase 2 config
    - Call ConfigLoader.load()
    - Verify all sections present
  - Test 2: test_environment_variable_substitution()
    - Create config with ${TEST_VAR}
    - Set os.environ['TEST_VAR'] = 'test_value'
    - Verify substitution works
  - Test 3: test_missing_config_file()
    - Call load() with non-existent path
    - Verify FileNotFoundError raised with helpful message
  - Test 4: test_validation_missing_section()
    - Create config missing required section
    - Verify ValueError raised
  - Test 5: test_validation_missing_api_key()
    - Create config with empty MP api_key
    - Verify ValueError raised
  - Run tests: `pytest tests/data_pipeline/utils/test_config_loader.py -v`
  - _Requirements: Requirement 10_

- [x] 2.5 Implement RateLimiter utility


  - Create `src/ceramic_discovery/data_pipeline/utils/rate_limiter.py`
  - Import time, functools, Callable, Any, logging
  - Define RateLimiter class with:
    - __init__(self, delay_seconds: float = 1.0) - stores delay
    - wait(self) - calculates elapsed time, sleeps if needed
    - __call__(self, func) - decorator that wraps function with rate limiting
  - Track last_request_time to calculate delays
  - Add debug logging for wait times
  - Test: `python -c "from ceramic_discovery.data_pipeline.utils.rate_limiter import RateLimiter; print('✓')"`
  - _Requirements: Requirement 6.2, 6.6_

- [x] 2.6 Write unit tests for RateLimiter


  - Create `tests/data_pipeline/utils/test_rate_limiter.py`
  - Test 1: test_rate_limiter_delays()
    - Create RateLimiter(delay_seconds=0.1)
    - Call wait() twice in succession
    - Measure time between calls
    - Verify delay >= 0.1 seconds
  - Test 2: test_decorator_functionality()
    - Create test function decorated with @RateLimiter(0.1)
    - Call function multiple times
    - Verify rate limiting applied
  - Test 3: test_no_delay_on_first_call()
    - Create new RateLimiter
    - First wait() should return immediately
  - Run tests: `pytest tests/data_pipeline/utils/test_rate_limiter.py -v`
  - _Requirements: Requirement 6_

- [x] 2.7 Implement PropertyExtractor utility


  - Create `src/ceramic_discovery/data_pipeline/utils/property_extractor.py`
  - Import re, Optional, Dict, Any, logging
  - Define PropertyExtractor class with class methods:
    - V50_PATTERNS: List of regex patterns for ballistic velocity
      - r'V[_\s]?50[:\s=]+(\d+\.?\d*)\s*(m/s|ms)'
      - r'ballistic\s+limit[:\s]+(\d+\.?\d*)\s*(m/s|ms)'
      - r'penetration\s+velocity[:\s]+(\d+\.?\d*)\s*(m/s|ms)'
    - HARDNESS_PATTERNS: List of regex patterns for hardness
      - r'hardness[:\s]+(\d+\.?\d*)\s*(GPa)'
      - r'Vickers\s+hardness[:\s]+(\d+\.?\d*)\s*(HV|GPa)'
      - r'HV[:\s]+(\d+\.?\d*)'
    - KIC_PATTERNS: List of regex patterns for fracture toughness
      - r'fracture\s+toughness[:\s]+(\d+\.?\d*)\s*(MPa[·\s]?m)'
      - r'K[_\s]?IC[:\s]+(\d+\.?\d*)\s*(MPa)'
    - MATERIAL_KEYWORDS: Dict mapping materials to keywords
      - 'SiC': ['sic', 'silicon carbide']
      - 'B4C': ['b4c', 'b₄c', 'boron carbide']
      - etc.
    - extract_v50(text) -> Optional[float] - extracts V50 value
    - extract_hardness(text) -> Optional[float] - extracts hardness, converts HV to GPa (÷100)
    - extract_fracture_toughness(text) -> Optional[float] - extracts K_IC
    - identify_material(text) -> Optional[str] - identifies ceramic type
    - extract_all(text) -> Dict - extracts all properties at once
  - Test: `python -c "from ceramic_discovery.data_pipeline.utils.property_extractor import PropertyExtractor; print('✓')"`
  - _Requirements: Requirement 5.2, 5.4, 5.5, 5.6, 5.7_

- [x] 2.8 Write unit tests for PropertyExtractor


  - Create `tests/data_pipeline/utils/test_property_extractor.py`
  - Test 1: test_extract_v50_standard_format()
    - Text: "V50 = 850 m/s"
    - Verify extracts 850.0
  - Test 2: test_extract_v50_alternative_formats()
    - Test "V_50: 850 m/s", "ballistic limit 850 m/s", etc.
  - Test 3: test_extract_hardness_gpa()
    - Text: "hardness: 26.5 GPa"
    - Verify extracts 26.5
  - Test 4: test_extract_hardness_hv_conversion()
    - Text: "Vickers hardness: 2650 HV"
    - Verify extracts 26.5 (2650 ÷ 100)
  - Test 5: test_extract_kic()
    - Text: "fracture toughness K_IC = 4.0 MPa·m^0.5"
    - Verify extracts 4.0
  - Test 6: test_identify_material_sic()
    - Text: "Silicon carbide (SiC) properties..."
    - Verify returns "SiC"
  - Test 7: test_identify_material_b4c()
    - Text: "Boron carbide armor..."
    - Verify returns "B4C"
  - Test 8: test_extract_all()
    - Text with multiple properties
    - Verify all extracted correctly
  - Test 9: test_no_match_returns_none()
    - Text without properties
    - Verify returns None
  - Run tests: `pytest tests/data_pipeline/utils/test_property_extractor.py -v`
  - _Requirements: Requirement 5_

- [x] 3. Implement wrapper collectors (Phase 1 integration)





  - Create collectors that wrap existing Phase 1 clients
  - Use composition pattern (HAS-A, not IS-A)
  - _Requirements: Requirement 2, 3_

- [x] 3.1 Implement MaterialsProjectCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/materials_project_collector.py`
  - Import existing MaterialsProjectClient (read-only)
  - Create client instance via composition (self.client = MaterialsProjectClient(...))
  - Implement batch collection logic
  - Add error handling with clear ImportError messages
  - _Requirements: Requirement 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3.2 Verify Phase 1 MaterialsProjectClient still works


  - Run `python -c "from ceramic_discovery.dft.materials_project_client import MaterialsProjectClient; print('✓')"`
  - Run existing tests: `pytest tests/test_materials_project_client.py -v` (if exists)
  - Verify no modifications to Phase 1 code
  - _Requirements: Requirement 2 (verification)_

- [x] 3.3 Write unit tests for MaterialsProjectCollector


  - Create `tests/data_pipeline/collectors/test_materials_project_collector.py`
  - Import pytest, MaterialsProjectCollector, mock/patch
  - Test 1: test_collector_uses_composition_not_inheritance()
    - Mock MaterialsProjectClient
    - Create collector with test config
    - Verify MaterialsProjectClient.__init__ was called (composition)
    - Verify collector is NOT instance of MaterialsProjectClient (not inheritance)
  - Test 2: test_collect_returns_dataframe()
    - Mock client.get_materials() to return test data
    - Call collector.collect()
    - Verify returns pandas DataFrame
    - Verify DataFrame has required columns: source, material_id, formula
  - Test 3: test_batch_processing()
    - Mock client to return large dataset
    - Verify batch_size and max_materials limits respected
  - Test 4: test_error_handling_api_failure()
    - Mock client to raise exception
    - Verify collector logs error and continues
  - Test 5: test_missing_api_key()
    - Create collector with config missing api_key
    - Verify ValueError raised with clear message
  - Test 6: test_import_error_handling()
    - Verify ImportError has helpful message if Phase 1 client missing
  - Run tests: `pytest tests/data_pipeline/collectors/test_materials_project_collector.py -v`
  - _Requirements: Requirement 2_

- [x] 3.4 Implement JarvisCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/jarvis_collector.py`
  - Import existing JarvisClient and safe_float utility (read-only)
  - Create client instance via composition
  - Check if existing client has carbide methods, use them if available
  - Add carbide filtering logic in collector (not in existing client)
  - _Requirements: Requirement 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.5 Verify Phase 1 JarvisClient still works


  - Run `pytest tests/test_jarvis_client.py -v`
  - Verify no modifications to Phase 1 JARVIS integration
  - _Requirements: Requirement 3 (verification)_

- [x] 3.6 Write unit tests for JarvisCollector


  - Create `tests/data_pipeline/collectors/test_jarvis_collector.py`
  - Test 1: test_collector_uses_composition()
    - Mock JarvisClient
    - Verify composition pattern (HAS-A not IS-A)
  - Test 2: test_carbide_filtering()
    - Mock client to return mixed materials (carbides and non-carbides)
    - Verify only carbides returned
    - Verify filtering logic: 'C' in elements AND metal in carbide_metals
  - Test 3: test_safe_float_usage()
    - Mock data with various numeric formats (strings, None, floats)
    - Verify safe_float utility used for all numeric conversions
    - Verify no exceptions raised for invalid data
  - Test 4: test_existing_client_method_detection()
    - Mock client with load_carbides method
    - Verify collector uses existing method if available
    - Mock client without load_carbides
    - Verify collector uses fallback filtering logic
  - Test 5: test_dataframe_structure()
    - Verify DataFrame has columns: source, jid, formula, formation_energy_jarvis, bulk_modulus_jarvis
    - Verify source column = 'JARVIS'
  - Run tests: `pytest tests/data_pipeline/collectors/test_jarvis_collector.py -v`
  - _Requirements: Requirement 3_

- [x] 4. Implement new collectors (no Phase 1 dependencies)




  - Create collectors for new data sources
  - No existing code dependencies
  - _Requirements: Requirement 4, 5, 6, 7_

- [x] 4.1 Implement AFLOWCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/aflow_collector.py`
  - Use aflow Python library directly
  - Query thermal properties (thermal_conductivity_300K, thermal_expansion_300K, debye_temperature, heat_capacity)
  - Handle AFLOWAPIError gracefully
  - Return DataFrame with _aflow suffix columns
  - _Requirements: Requirement 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.2 Write unit tests for AFLOWCollector


  - Create `tests/data_pipeline/collectors/test_aflow_collector.py`
  - Test 1: test_aflow_query_construction()
    - Mock aflow.search()
    - Verify query built with correct species
    - Verify LDAU exclusion if configured
  - Test 2: test_thermal_property_extraction()
    - Mock aflow entries with thermal data
    - Verify thermal_conductivity_300K, thermal_expansion_300K, debye_temperature, heat_capacity extracted
  - Test 3: test_error_handling_api_error()
    - Mock aflow.AFLOWAPIError
    - Verify collector logs error and continues with next system
  - Test 4: test_dataframe_structure()
    - Verify columns: source, formula, ceramic_type, density_aflow, thermal_conductivity_300K, etc.
    - Verify source = 'AFLOW'
  - Test 5: test_max_per_system_limit()
    - Mock large result set
    - Verify limit respected
  - Run tests: `pytest tests/data_pipeline/collectors/test_aflow_collector.py -v`
  - _Requirements: Requirement 4_

- [x] 4.3 Implement SemanticScholarCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/semantic_scholar_collector.py`
  - Import SemanticScholar, PropertyExtractor, time, logging
  - Define SemanticScholarCollector(BaseCollector):
    - __init__: Initialize SemanticScholar client, PropertyExtractor, rate limiter
    - collect(): 
      - Loop through configured queries
      - Search papers with semanticscholar library
      - Filter by min_citations threshold
      - Extract properties from title + abstract using PropertyExtractor
      - Only save if extracted useful data (v50, hardness, or K_IC) AND identified material
      - Include provenance: paper_id, title, year, citations, URL
      - Apply rate limiting between queries
      - Return DataFrame with columns: source, paper_id, title, year, citations, url, ceramic_type, v50_literature, hardness_literature, K_IC_literature
  - Test: `python -c "from ceramic_discovery.data_pipeline.collectors.semantic_scholar_collector import SemanticScholarCollector; print('✓')"`
  - _Requirements: Requirement 5.1, 5.2, 5.3, 5.8_

- [x] 4.4 Write unit tests for SemanticScholarCollector


  - Create `tests/data_pipeline/collectors/test_semantic_scholar_collector.py`
  - Test 1: test_paper_search()
    - Mock SemanticScholar.search_paper()
    - Verify queries executed
  - Test 2: test_property_extraction_integration()
    - Mock papers with properties in abstract
    - Verify PropertyExtractor.extract_all() called
    - Verify extracted properties in DataFrame
  - Test 3: test_citation_filtering()
    - Mock papers with various citation counts
    - Verify only papers >= min_citations included
  - Test 4: test_provenance_tracking()
    - Verify paper_id, title, year, citations, URL included in output
  - Test 5: test_rate_limiting()
    - Verify delay between queries
  - Test 6: test_skip_papers_without_properties()
    - Mock papers with no extractable properties
    - Verify not included in output
  - Run tests: `pytest tests/data_pipeline/collectors/test_semantic_scholar_collector.py -v`
  - _Requirements: Requirement 5_

- [x] 4.5 Implement MatWebCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/matweb_collector.py`
  - Import requests, BeautifulSoup, RateLimiter, re, logging
  - Define MatWebCollector(BaseCollector):
    - BASE_URL = "http://www.matweb.com"
    - __init__: Initialize session with User-Agent, create RateLimiter(2 seconds)
    - collect():
      - Loop through target_materials
      - Search MatWeb with rate limiting
      - Parse search results for material links
      - Scrape each material page (up to results_per_material)
      - Extract properties from tables: hardness (Vickers), K_IC, density, compressive strength
      - Convert units: Vickers HV to GPa (÷100), MPa to GPa (÷1000)
      - Return DataFrame with columns: source, material_name, url, hardness_GPa, K_IC_MPa_m05, density_matweb, compressive_strength_GPa
    - _scrape_material_page(path): Scrape individual material page
    - _parse_numeric(text): Extract numeric value from text
  - Add ethical notice in docstring: respects robots.txt, rate limiting, research use only
  - Test: `python -c "from ceramic_discovery.data_pipeline.collectors.matweb_collector import MatWebCollector; print('✓')"`
  - _Requirements: Requirement 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 4.6 Write unit tests for MatWebCollector


  - Create `tests/data_pipeline/collectors/test_matweb_collector.py`
  - Test 1: test_http_request_with_user_agent()
    - Mock requests.get()
    - Verify User-Agent header set
  - Test 2: test_html_parsing()
    - Mock HTML response with property tables
    - Verify properties extracted correctly
  - Test 3: test_unit_conversion_vickers_to_gpa()
    - Mock data: "Vickers hardness: 2650 HV"
    - Verify converts to 26.5 GPa
  - Test 4: test_unit_conversion_mpa_to_gpa()
    - Mock data: "Compressive strength: 3500 MPa"
    - Verify converts to 3.5 GPa
  - Test 5: test_rate_limiting()
    - Verify RateLimiter used between requests
  - Test 6: test_error_handling_http_failure()
    - Mock HTTP error
    - Verify collector logs error and continues
  - Run tests: `pytest tests/data_pipeline/collectors/test_matweb_collector.py -v`
  - _Requirements: Requirement 6_

- [x] 4.7 Implement NISTBaselineCollector


  - Create `src/ceramic_discovery/data_pipeline/collectors/nist_baseline_collector.py`
  - Import BaseCollector, pandas, logging
  - Define NISTBaselineCollector(BaseCollector):
    - __init__: Load baseline_materials from config
    - collect():
      - Convert baseline_materials dict to list of records
      - Each record: {source: 'NIST', formula: X, ceramic_type: X, **props}
      - Return DataFrame
    - validate_config():
      - Verify baseline_materials exists
      - Verify each material has required properties: density, hardness, K_IC, source
      - Return False if any missing
  - Add docstring explaining:
    - This is Phase 2 baseline data (separate from Phase 1 NISTClient)
    - Phase 1 NISTClient: temperature-dependent thermochemical properties
    - Phase 2 NISTBaselineCollector: reference baseline values for validation
  - Test: `python -c "from ceramic_discovery.data_pipeline.collectors.nist_baseline_collector import NISTBaselineCollector; print('✓')"`
  - _Requirements: Requirement 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 4.8 Write unit tests for NISTBaselineCollector


  - Create `tests/data_pipeline/collectors/test_nist_baseline_collector.py`
  - Test 1: test_baseline_data_loading()
    - Create config with 5 baseline materials
    - Call collect()
    - Verify DataFrame has 5 rows
  - Test 2: test_validation_complete_data()
    - Config with all required properties
    - Verify validate_config() returns True
  - Test 3: test_validation_missing_property()
    - Config missing hardness for one material
    - Verify validate_config() returns False
  - Test 4: test_no_conflict_with_phase1_nist()
    - Verify can import both:
      - from ceramic_discovery.dft.nist_client import NISTClient
      - from ceramic_discovery.data_pipeline.collectors.nist_baseline_collector import NISTBaselineCollector
    - Verify they are different classes
  - Test 5: test_dataframe_structure()
    - Verify columns: source, formula, ceramic_type, density, hardness, K_IC, source (provenance)
  - Run tests: `pytest tests/data_pipeline/collectors/test_nist_baseline_collector.py -v`
  - _Requirements: Requirement 7_

- [x] 4.9 Checkpoint: Verify all collectors work


  - Run all collector tests: `pytest tests/data_pipeline/collectors/ -v`
  - Verify all tests pass
  - Run Phase 1 tests: `pytest tests/ -v`
  - Verify Phase 1 still works (no regressions)
  - If any tests fail, fix before proceeding
  - _Requirements: All collector requirements_

- [x] 5. Implement integration pipeline






  - Create multi-source integration logic
  - Implement fuzzy matching and deduplication
  - _Requirements: Requirement 8_

- [x] 5.1 Implement FormulaMatcher utility


  - Create `src/ceramic_discovery/data_pipeline/integration/formula_matcher.py`
  - Use SequenceMatcher for formula similarity
  - Implement find_best_match with threshold 0.95
  - _Requirements: Requirement 8.2, 8.6_

- [x] 5.2 Write unit tests for FormulaMatcher




  - Create `tests/data_pipeline/integration/test_formula_matcher.py`
  - Test 1: test_similarity_identical()
    - Compare "SiC" with "SiC"
    - Verify similarity = 1.0
  - Test 2: test_similarity_different()
    - Compare "SiC" with "B4C"
    - Verify similarity < 1.0
  - Test 3: test_similarity_case_sensitive()
    - Compare "SiC" with "sic"
    - Verify similarity < 1.0 (case matters)
  - Test 4: test_find_best_match_exact()
    - Formula: "SiC", candidates: ["SiC", "B4C", "WC"]
    - Verify returns ("SiC", 1.0)
  - Test 5: test_find_best_match_similar()
    - Formula: "Si1C1", candidates: ["SiC", "B4C"]
    - Verify returns ("SiC", score > 0.95)
  - Test 6: test_find_best_match_below_threshold()
    - Formula: "SiC", candidates: ["B4C", "WC"], threshold=0.95
    - Verify returns (None, 0)
  - Run tests: `pytest tests/data_pipeline/integration/test_formula_matcher.py -v`
  - _Requirements: Requirement 8_

- [x] 5.3 Implement IntegrationPipeline


  - Create `src/ceramic_discovery/data_pipeline/pipelines/integration_pipeline.py`
  - Load all CSV files from data/processed/
  - Implement fuzzy formula matching for deduplication
  - Apply source priority for conflict resolution (Literature > MatWeb > NIST > MP > JARVIS > AFLOW)
  - Create combined property columns (hardness_combined, K_IC_combined, density_combined)
  - Calculate derived properties (specific_hardness, ballistic_efficacy, dop_resistance)
  - Save to data/final/master_dataset.csv
  - Generate integration report
  - _Requirements: Requirement 8.1, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [x] 5.4 Write unit tests for IntegrationPipeline


  - Create `tests/data_pipeline/integration/test_integration_pipeline.py`
  - Test 1: test_load_all_sources()
    - Create temp CSV files for each source
    - Call _load_all_sources()
    - Verify all sources loaded into dict
  - Test 2: test_deduplication_exact_match()
    - Create two sources with same formula
    - Verify merged into single row
  - Test 3: test_deduplication_fuzzy_match()
    - Create sources with similar formulas (SiC vs Si1C1)
    - Verify matched and merged
  - Test 4: test_conflict_resolution_source_priority()
    - Two sources with different hardness values
    - Verify higher priority source value used
  - Test 5: test_create_combined_columns()
    - Multiple sources with hardness values
    - Verify hardness_combined = mean of all sources
  - Test 6: test_calculate_derived_properties()
    - Input: hardness_combined=26, density_combined=3.2, K_IC_combined=4.0
    - Verify specific_hardness = 26/3.2 = 8.125
    - Verify ballistic_efficacy = 26/3.2 = 8.125
    - Verify dop_resistance = √26 × 4.0 / 3.2 ≈ 6.37
  - Test 7: test_master_dataset_saved()
    - Run integration
    - Verify data/final/master_dataset.csv created
  - Test 8: test_integration_report_generated()
    - Verify data/final/integration_report.txt created
    - Verify contains source counts and property coverage
  - Run tests: `pytest tests/data_pipeline/integration/test_integration_pipeline.py -v`
  - _Requirements: Requirement 8_

- [x] 6. Implement collection orchestrator




  - Create main pipeline orchestrator
  - Run all collectors in sequence
  - _Requirements: Requirement 9_

- [x] 6.1 Implement CollectionOrchestrator


  - Create `src/ceramic_discovery/data_pipeline/pipelines/collection_orchestrator.py`
  - Initialize all 6 collectors from config
  - Execute collectors in order: MP → JARVIS → AFLOW → Scholar → MatWeb → NIST → Integration
  - Handle collector failures gracefully (log error, continue with next)
  - Use tqdm progress bars
  - Save intermediate CSV checkpoints
  - Generate execution summary report
  - _Requirements: Requirement 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 6.2 Write unit tests for CollectionOrchestrator


  - Create `tests/data_pipeline/pipelines/test_collection_orchestrator.py`
  - Test 1: test_initialize_collectors()
    - Create orchestrator with config
    - Verify all 6 collectors initialized
  - Test 2: test_sequential_execution()
    - Mock all collectors
    - Call run_all()
    - Verify collectors called in order: MP → JARVIS → AFLOW → Scholar → MatWeb → NIST → Integration
  - Test 3: test_collector_failure_continues()
    - Mock one collector to raise exception
    - Verify error logged
    - Verify next collector still runs
  - Test 4: test_checkpoint_saving()
    - Mock collectors to return test DataFrames
    - Verify each saves CSV to data/processed/
  - Test 5: test_integration_failure_aborts()
    - Mock integration to raise exception
    - Verify pipeline aborts (integration is critical)
  - Test 6: test_progress_tracking()
    - Verify tqdm progress bar used
  - Test 7: test_summary_report_generated()
    - Run orchestrator
    - Verify data/final/pipeline_summary.txt created
    - Verify contains per-source runtime and status
  - Run tests: `pytest tests/data_pipeline/pipelines/test_collection_orchestrator.py -v`
  - _Requirements: Requirement 9_

- [x] 7. Create main execution script




  - Create user-facing script to run entire pipeline
  - Add Phase 1 verification before and after
  - _Requirements: Requirement 11_

- [x] 7.1 Implement run_data_collection.py script


  - Create `scripts/run_data_collection.py`
  - Import sys, subprocess, argparse, logging, Path, datetime
  - Import CollectionOrchestrator, ConfigLoader
  - Define verify_phase1_tests() function:
    - Run `pytest tests/ -v --tb=short -x` with subprocess
    - Capture output
    - Return True if all pass, False otherwise
    - Log detailed error if any fail
  - Define verify_dependencies() function:
    - Check all required packages importable
    - Return list of missing packages
  - Define parse_arguments() function:
    - --config: path to config file (default: config/data_pipeline_config.yaml)
    - --skip-verification: skip Phase 1 test verification (NOT RECOMMENDED)
    - --sources: comma-separated list of sources to run (e.g., mp,jarvis,aflow)
  - Define filter_sources() function:
    - Disable sources not in --sources list
  - Define main() function:
    - Step 1: Verify dependencies
    - Step 2: Verify Phase 1 tests BEFORE (unless --skip-verification)
    - Step 3: Load Phase 2 config
    - Step 4: Filter sources if --sources specified
    - Step 5: Run CollectionOrchestrator
    - Step 6: Verify Phase 1 tests AFTER (unless --skip-verification)
    - Step 7: Report success or failure
  - Add comprehensive logging to logs/collection_{timestamp}.log
  - Add error handling with clear messages
  - Test: `python scripts/run_data_collection.py --help`
  - _Requirements: Requirement 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 7.2 Create validation script


  - Create `scripts/validate_collection_output.py`
  - Verify master_dataset.csv exists and has expected structure
  - Check property coverage statistics
  - Validate data quality (no NaN in critical columns, physical plausibility)
  - _Requirements: Requirement 11 (validation)_

- [x] 8. Update Phase 2 module exports




  - Configure __init__.py files for clean imports
  - _Requirements: All requirements (module structure)_

- [x] 8.1 Update data_pipeline/__init__.py


  - Export main classes: CollectionOrchestrator, IntegrationPipeline, ConfigLoader
  - Export all collector classes
  - Add module docstring explaining Phase 2 isolation
  - _Requirements: All requirements_

- [x] 8.2 Update collectors/__init__.py


  - Export all collector classes
  - Add module docstring
  - _Requirements: Requirement 1-7_

- [x] 8.3 Update integration/__init__.py


  - Export FormulaMatcher and IntegrationPipeline
  - _Requirements: Requirement 8_

- [x] 8.4 Update pipelines/__init__.py


  - Export CollectionOrchestrator and IntegrationPipeline
  - _Requirements: Requirement 8, 9_

- [x] 8.5 Update utils/__init__.py


  - Export ConfigLoader, RateLimiter, PropertyExtractor
  - _Requirements: Requirement 2, 5, 6, 10_

- [x] 9. End-to-end integration testing





  - Test complete pipeline with small dataset
  - Verify Phase 1 compatibility
  - _Requirements: All requirements_

- [x] 9.1 Write end-to-end integration test


  - Create `tests/data_pipeline/test_end_to_end.py`
  - Test 1: test_complete_pipeline_with_mocked_apis()
    - Mock all external APIs (MP, JARVIS, AFLOW, Semantic Scholar, MatWeb)
    - Create test config
    - Run CollectionOrchestrator.run_all()
    - Verify all collectors execute
    - Verify integration runs
    - Verify master_dataset.csv created
  - Test 2: test_master_dataset_structure()
    - Load master_dataset.csv
    - Verify required columns present: formula, ceramic_type, density_combined, hardness_combined, K_IC_combined
    - Verify derived properties present: specific_hardness, ballistic_efficacy, dop_resistance
  - Test 3: test_phase1_can_consume_output()
    - Load master_dataset.csv with pandas
    - Verify can be used as input to Phase 1 ML pipeline
    - Check data types compatible
  - Test 4: test_integration_report_completeness()
    - Verify integration_report.txt contains:
      - Total materials count
      - Per-source counts
      - Property coverage statistics
  - Test 5: test_pipeline_summary_completeness()
    - Verify pipeline_summary.txt contains:
      - Per-collector status (SUCCESS/FAILED)
      - Runtime for each collector
      - Total pipeline runtime
  - Run tests: `pytest tests/data_pipeline/test_end_to_end.py -v`
  - _Requirements: All requirements_

- [x] 9.2 Run complete pipeline with real APIs (manual test)


  - Set up API keys in environment variables
  - Run `python scripts/run_data_collection.py`
  - Verify all collectors execute successfully
  - Check master_dataset.csv has 6,000+ materials
  - Verify integration report shows all sources
  - _Requirements: All requirements_

- [x] 10. Final verification checkpoint





  - Ensure all Phase 1 tests still pass
  - Verify no Phase 1 files were modified
  - _Requirements: All requirements (safety)_

- [x] 10.1 Run final Phase 1 test verification



  - Execute `pytest tests/ -v`
  - Verify all 157+ tests still pass
  - If any fail, identify what broke and fix
  - _Requirements: Requirement 11.3_

- [x] 10.2 Verify no Phase 1 files modified


  - Run `git status` to check for modifications
  - Verify only new files in data_pipeline/ were created
  - Verify no changes to dft/, ml/, screening/, validation/, utils/
  - _Requirements: All requirements (isolation principle)_

- [x] 10.3 Generate Phase 2 documentation


  - Create `docs/data_pipeline_guide.md` with usage instructions
  - Document configuration options
  - Add troubleshooting section
  - Include example commands
  - _Requirements: All requirements (documentation)_

***

## Task Execution Notes

### Dependency Order
- Tasks 1.x must complete before any other tasks (setup)
- Tasks 2.x must complete before 3.x and 4.x (base classes and utilities)
- Tasks 3.x and 4.x can be done in parallel (collectors are independent)
- Task 5.x requires 3.x and 4.x complete (integration needs collectors)
- Task 6.x requires 5.x complete (orchestrator needs integration)
- Task 7.x requires 6.x complete (script needs orchestrator)
- Tasks 8.x can be done anytime after their respective modules exist
- Task 9.x requires all previous tasks (end-to-end testing)
- Task 10.x is final verification

### Testing Strategy
- ALL unit tests are REQUIRED (comprehensive testing approach)
- Each component must have corresponding unit tests that pass
- Property-based tests are NOT required for this phase (focus on unit and integration tests)
- Integration test (9.1) is CRITICAL for verifying Phase 1 compatibility
- Manual testing (9.2) with real APIs must be done before considering complete
- Test coverage should aim for >80% of new Phase 2 code
- All tests must pass before moving to next major milestone

### Checkpoints
After completing these milestones, verify Phase 1 tests still pass:
1. After Task 2 (base infrastructure)
2. After Task 4 (all collectors)
3. After Task 6 (orchestrator)
4. After Task 9 (end-to-end)
5. Final verification (Task 10)

### Success Criteria
- All Phase 1 tests pass (157+ tests)
- master_dataset.csv created with 6,000+ materials
- No modifications to any Phase 1 files
- All 6 data sources successfully collected
- Integration report shows successful merging

***

## Implementation Principles

1. **Isolation**: New code in data_pipeline/ only, never modify existing files
2. **Composition**: Use HAS-A (composition), not IS-A (inheritance)
3. **One-way imports**: Phase 2 imports FROM Phase 1, never the reverse
4. **Graceful degradation**: Individual collector failures don't stop pipeline
5. **Checkpointing**: Save intermediate results for resumability
6. **Verification**: Test Phase 1 integrity before and after implementation
