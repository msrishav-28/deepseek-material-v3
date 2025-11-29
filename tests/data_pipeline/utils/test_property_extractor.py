"""
Unit tests for PropertyExtractor utility.

Tests verify:
1. V50 extraction in standard and alternative formats
2. Hardness extraction in GPa
3. Hardness extraction with HV to GPa conversion
4. K_IC extraction
5. Material identification for various ceramics
6. Extract all properties at once
7. Handling of text without properties
"""

import pytest
from ceramic_discovery.data_pipeline.utils.property_extractor import PropertyExtractor


class TestExtractV50:
    """Test V50 (ballistic limit velocity) extraction."""
    
    def test_extract_v50_standard_format(self):
        """Text: 'V50 = 850 m/s' should extract 850.0."""
        text = "V50 = 850 m/s"
        result = PropertyExtractor.extract_v50(text)
        assert result == 850.0
    
    def test_extract_v50_alternative_formats(self):
        """Test various V50 formats."""
        test_cases = [
            ("V_50: 850 m/s", 850.0),
            ("V 50 = 900 m/s", 900.0),
            ("ballistic limit 750 m/s", 750.0),
            ("ballistic limit: 800 m/s", 800.0),
            ("penetration velocity 920 m/s", 920.0),
            ("penetration velocity: 880 m/s", 880.0),
        ]
        
        for text, expected in test_cases:
            result = PropertyExtractor.extract_v50(text)
            assert result == expected, f"Failed for: {text}"
    
    def test_extract_v50_with_decimal(self):
        """Test V50 with decimal values."""
        text = "V50 = 850.5 m/s"
        result = PropertyExtractor.extract_v50(text)
        assert result == 850.5
    
    def test_extract_v50_case_insensitive(self):
        """Test case insensitivity."""
        test_cases = [
            "V50 = 850 M/S",
            "v50 = 850 m/s",
            "V50 = 850 ms",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.extract_v50(text)
            assert result == 850.0


class TestExtractHardness:
    """Test hardness extraction."""
    
    def test_extract_hardness_gpa(self):
        """Text: 'hardness: 26.5 GPa' should extract 26.5."""
        text = "hardness: 26.5 GPa"
        result = PropertyExtractor.extract_hardness(text)
        assert result == 26.5
    
    def test_extract_hardness_hv_conversion(self):
        """Text: 'Vickers hardness: 2650 HV' should extract 26.5 (2650 ÷ 100)."""
        text = "Vickers hardness: 2650 HV"
        result = PropertyExtractor.extract_hardness(text)
        assert result == 26.5
    
    def test_extract_hardness_various_formats(self):
        """Test various hardness formats."""
        test_cases = [
            ("hardness 28.0 GPa", 28.0),
            ("hardness: 30 GPa", 30.0),
            ("Vickers hardness: 3000 HV", 30.0),
            ("HV: 2500", 25.0),
        ]
        
        for text, expected in test_cases:
            result = PropertyExtractor.extract_hardness(text)
            assert result == expected, f"Failed for: {text}"
    
    def test_extract_hardness_case_insensitive(self):
        """Test case insensitivity."""
        test_cases = [
            "HARDNESS: 26.5 GPA",
            "Hardness: 26.5 gpa",
            "hardness: 26.5 GPa",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.extract_hardness(text)
            assert result == 26.5


class TestExtractFractureToughness:
    """Test fracture toughness (K_IC) extraction."""
    
    def test_extract_kic(self):
        """Text: 'fracture toughness K_IC = 4.0 MPa·m^0.5' should extract 4.0."""
        text = "fracture toughness K_IC = 4.0 MPa·m^0.5"
        result = PropertyExtractor.extract_fracture_toughness(text)
        assert result == 4.0
    
    def test_extract_kic_various_formats(self):
        """Test various K_IC formats."""
        test_cases = [
            ("fracture toughness: 4.5 MPa·m", 4.5),
            ("K_IC: 3.8 MPa", 3.8),
            ("KIC: 4.2 MPa", 4.2),
            ("K IC: 5.0 MPa", 5.0),
            ("toughness: 3.5 MPa·m", 3.5),
        ]
        
        for text, expected in test_cases:
            result = PropertyExtractor.extract_fracture_toughness(text)
            assert result == expected, f"Failed for: {text}"
    
    def test_extract_kic_with_decimal(self):
        """Test K_IC with decimal values."""
        text = "K_IC = 4.25 MPa·m^0.5"
        result = PropertyExtractor.extract_fracture_toughness(text)
        assert result == 4.25


class TestIdentifyMaterial:
    """Test material identification."""
    
    def test_identify_material_sic(self):
        """Text: 'Silicon carbide (SiC) properties...' should return 'SiC'."""
        test_cases = [
            "Silicon carbide (SiC) properties",
            "SiC ceramic armor",
            "silicon-carbide material",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "SiC", f"Failed for: {text}"
    
    def test_identify_material_b4c(self):
        """Text: 'Boron carbide armor...' should return 'B4C'."""
        test_cases = [
            "Boron carbide armor",
            "B4C ceramic",
            "boron-carbide properties",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "B4C", f"Failed for: {text}"
    
    def test_identify_material_wc(self):
        """Test tungsten carbide identification."""
        test_cases = [
            "Tungsten carbide coating",
            "WC properties",
            "tungsten-carbide hardness",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "WC", f"Failed for: {text}"
    
    def test_identify_material_tic(self):
        """Test titanium carbide identification."""
        test_cases = [
            "Titanium carbide coating",
            "TiC properties",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "TiC", f"Failed for: {text}"
    
    def test_identify_material_al2o3(self):
        """Test alumina identification."""
        test_cases = [
            "Alumina ceramic",
            "Al2O3 properties",
            "aluminum oxide",
            "aluminium oxide",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "Al2O3", f"Failed for: {text}"
    
    def test_identify_material_case_insensitive(self):
        """Test case insensitivity."""
        test_cases = [
            "SILICON CARBIDE",
            "Silicon Carbide",
            "silicon carbide",
        ]
        
        for text in test_cases:
            result = PropertyExtractor.identify_material(text)
            assert result == "SiC"


class TestExtractAll:
    """Test extracting all properties at once."""
    
    def test_extract_all(self):
        """Text with multiple properties should extract all correctly."""
        text = """
        Silicon carbide (SiC) is a ceramic material with excellent properties.
        The V50 ballistic limit is 850 m/s.
        Hardness: 26.5 GPa
        Fracture toughness K_IC = 4.0 MPa·m^0.5
        """
        
        result = PropertyExtractor.extract_all(text)
        
        assert result['v50'] == 850.0
        assert result['hardness'] == 26.5
        assert result['K_IC'] == 4.0
        assert result['material'] == 'SiC'
    
    def test_extract_all_partial(self):
        """Text with some properties should extract available ones."""
        text = "Boron carbide has hardness of 28 GPa"
        
        result = PropertyExtractor.extract_all(text)
        
        assert result['v50'] is None
        assert result['hardness'] == 28.0
        assert result['K_IC'] is None
        assert result['material'] == 'B4C'
    
    def test_extract_all_with_hv_conversion(self):
        """Test extract_all with HV to GPa conversion."""
        text = "SiC ceramic with Vickers hardness: 2650 HV and V50 = 850 m/s"
        
        result = PropertyExtractor.extract_all(text)
        
        assert result['v50'] == 850.0
        assert result['hardness'] == 26.5  # Converted from HV
        assert result['material'] == 'SiC'


class TestNoMatch:
    """Test handling of text without properties."""
    
    def test_no_match_returns_none(self):
        """Text without properties should return None for each property."""
        text = "This is some random text without any material properties."
        
        result = PropertyExtractor.extract_all(text)
        
        assert result['v50'] is None
        assert result['hardness'] is None
        assert result['K_IC'] is None
        assert result['material'] is None
    
    def test_empty_text(self):
        """Empty text should return None for all properties."""
        result = PropertyExtractor.extract_all("")
        
        assert result['v50'] is None
        assert result['hardness'] is None
        assert result['K_IC'] is None
        assert result['material'] is None
    
    def test_none_text(self):
        """None text should return None for all properties."""
        result = PropertyExtractor.extract_all(None)
        
        assert result['v50'] is None
        assert result['hardness'] is None
        assert result['K_IC'] is None
        assert result['material'] is None
    
    def test_individual_extractors_with_none(self):
        """Individual extractors should handle None gracefully."""
        assert PropertyExtractor.extract_v50(None) is None
        assert PropertyExtractor.extract_hardness(None) is None
        assert PropertyExtractor.extract_fracture_toughness(None) is None
        assert PropertyExtractor.identify_material(None) is None
