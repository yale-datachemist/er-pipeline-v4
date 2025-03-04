#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for enhanced date processing module integration.
This script verifies that the enhanced date processing features
are correctly integrated into the entity resolution pipeline.
"""

import os
import sys
import logging
import json
from pprint import pprint

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integration module if available
try:
    from integration import (
        check_enhanced_date_processing_available,
        update_config_for_enhanced_dates,
        extract_years_from_text,
        get_life_dates
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Integration module not available. Install it first.")
    INTEGRATION_AVAILABLE = False

# Try to import enhanced date processing
try:
    from enhanced_date_processing import (
        extract_life_dates_enhanced,
        extract_dates_from_provision,
        analyze_temporal_relationship,
        detect_publication_roles,
        is_historical_figure_pattern,
        integrate_enhanced_dates_with_base_features
    )
    ENHANCED_DATE_PROCESSING_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced date processing module not available.")
    ENHANCED_DATE_PROCESSING_AVAILABLE = False

# Try to import enhanced features
try:
    from enhanced_features import (
        extract_life_dates,
        analyze_robust_temporal_features,
        enhance_feature_vector
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced features module not available.")
    ENHANCED_FEATURES_AVAILABLE = False

def test_basic_date_extraction():
    """Test basic date extraction functions."""
    print("\n=== Testing Basic Date Extraction ===\n")
    
    if not ENHANCED_FEATURES_AVAILABLE:
        print("Enhanced features module not available. Skipping basic date tests.")
        return
    
    # Test cases
    test_cases = [
        "Smith, John (1856-1920)",
        "Franklin, Benjamin, 1706-1790",
        "Mozart, Wolfgang Amadeus, 1756-1791",
        "Plato (428 BCE-348 BCE)",
        "Shakespeare, William (1564-1616)"
    ]
    
    for case in test_cases:
        birth, death = extract_life_dates(case)
        print(f"{case} -> Birth: {birth}, Death: {death}")
    
    print("\nTest completed.")

def test_enhanced_date_extraction():
    """Test enhanced date extraction functions."""
    print("\n=== Testing Enhanced Date Extraction ===\n")
    
    if not ENHANCED_DATE_PROCESSING_AVAILABLE:
        print("Enhanced date processing module not available. Skipping enhanced date tests.")
        return
    
    # Test cases
    test_cases = [
        "Smith, John (1856-1920)",
        "Franklin, Benjamin, 1706-1790",
        "Mozart, Wolfgang Amadeus, 1756-1791",
        "Plato (428 BCE-348 BCE)",
        "Shakespeare, William (1564-1616)",
        "Beethoven, active 1798-1827",
        "Anonymous, ca. 1450-1510",
        "Johnson, Samuel, approximately 1709-1784",
        "Smith, fl. 1776",
        "Newton, Isaac, b. 1642",
        "Darwin, Charles, d. 1882"
    ]
    
    for case in test_cases:
        birth, death, active, conf = extract_life_dates_enhanced(case)
        print(f"{case} -> Birth: {birth}, Death: {death}, Active: {active}, Confidence: {conf:.2f}")
    
    print("\nTest completed.")

def test_provision_date_extraction():
    """Test provision date extraction functions."""
    print("\n=== Testing Provision Date Extraction ===\n")
    
    if not ENHANCED_DATE_PROCESSING_AVAILABLE:
        print("Enhanced date processing module not available. Skipping provision date tests.")
        return
    
    # Test cases
    test_cases = [
        "London: Printed for the author, 1732.",
        "New York: Dover Publications, [1967, c1954]",
        "Paris, 1844-1846.",
        "Boston: Little, Brown and Company, between 1920 and 1925.",
        "Cambridge, Massachusetts: Harvard University Press, ca. 1980.",
        "Oxford: Clarendon Press, 18th century.",
        "Chicago: University of Chicago Press, 1950s.",
        "Berlin: Springer-Verlag, approximately 1995.",
        "Amsterdam: Elsevier, [2002?]",
        "London: J. Murray, 1839; 2nd ed."
    ]
    
    for case in test_cases:
        years, ranges, conf = extract_dates_from_provision(case)
        print(f"{case} -> Years: {years}, Ranges: {ranges}, Confidence: {conf:.2f}")
    
    print("\nTest completed.")

def test_temporal_analysis():
    """Test temporal relationship analysis."""
    print("\n=== Testing Temporal Relationship Analysis ===\n")
    
    if not ENHANCED_DATE_PROCESSING_AVAILABLE:
        print("Enhanced date processing module not available. Skipping temporal analysis tests.")
        return
    
    # Test cases - (person1, provision1, person2, provision2)
    test_cases = [
        # Contemporary authors case
        (
            "Smith, John (1856-1920)", 
            "London: J. Murray, 1880; 1899; 1910.",
            "Smith, John (1856-1920)",
            "New York: Harper, 1885; 1905; 1915."
        ),
        # Different people with same name
        (
            "Smith, John (1856-1920)", 
            "London: J. Murray, 1880; 1899; 1910.",
            "Smith, John (1920-1990)",
            "Chicago: University Press, 1950; 1960; 1970."
        ),
        # Historical figure with modern publications
        (
            "Shakespeare, William (1564-1616)",
            "London: First Folio, 1623.",
            "Shakespeare, William (1564-1616)",
            "New York: Penguin Classics, 2001; 2008; 2015."
        ),
        # Active period vs. life dates
        (
            "Beethoven, active 1798-1827",
            "Vienna: 1800; 1810; 1823.",
            "Beethoven, Ludwig van (1770-1827)",
            "Leipzig: Breitkopf & Härtel, 1801; 1812; 1824."
        )
    ]
    
    for case in test_cases:
        person1, prov1, person2, prov2 = case
        
        # Extract life dates and provision dates
        life_dates1 = extract_life_dates_enhanced(person1)
        life_dates2 = extract_life_dates_enhanced(person2)
        
        provision_dates1 = extract_dates_from_provision(prov1)
        provision_dates2 = extract_dates_from_provision(prov2)
        
        # Analyze temporal relationships
        features1 = analyze_temporal_relationship(life_dates1, provision_dates1)
        features2 = analyze_temporal_relationship(life_dates2, provision_dates2)
        
        print(f"\n=== Case: {person1} vs. {person2} ===")
        print(f"Person 1: {person1}")
        print(f"Life dates 1: {life_dates1}")
        print(f"Provision 1: {prov1}")
        print(f"Provision dates 1: {provision_dates1}")
        
        print(f"\nPerson 2: {person2}")
        print(f"Life dates 2: {life_dates2}")
        print(f"Provision 2: {prov2}")
        print(f"Provision dates 2: {provision_dates2}")
        
        print("\nTemporal Features Person 1:")
        for key, value in features1.items():
            print(f"  {key}: {value:.2f}")
        
        print("\nTemporal Features Person 2:")
        for key, value in features2.items():
            print(f"  {key}: {value:.2f}")
        
        # Test integration function
        integrated = integrate_enhanced_dates_with_base_features(
            {},  # Empty base features for this test
            person1, person2, prov1, prov2
        )
        
        print("\nIntegrated Features:")
        pprint({k: round(v, 2) for k, v in integrated.items()})
    
    print("\nTest completed.")

def test_integration():
    """Test integration module functions."""
    print("\n=== Testing Integration Module ===\n")
    
    if not INTEGRATION_AVAILABLE:
        print("Integration module not available. Skipping integration tests.")
        return
    
    # Check if enhanced date processing is available
    available = check_enhanced_date_processing_available()
    print(f"Enhanced date processing available: {available}")
    
    # Test config update
    original_config = {
        "use_enhanced_features": False,
        "enhanced_temporal_features": False
    }
    
    updated_config = update_config_for_enhanced_dates(original_config)
    print("\nOriginal config:")
    pprint(original_config)
    print("\nUpdated config:")
    pprint(updated_config)
    
    # Test year extraction
    test_texts = [
        "London: Printed for the author, 1732.",
        "New York: Dover Publications, [1967, c1954]",
        "Paris, 1844-1846."
    ]
    
    print("\nYear extraction comparison:")
    for text in test_texts:
        enhanced_years = extract_years_from_text(text, True)
        basic_years = extract_years_from_text(text, False)
        print(f"{text} -> Enhanced: {enhanced_years}, Basic: {basic_years}")
    
    # Test life date extraction
    test_persons = [
        "Smith, John (1856-1920)",
        "Beethoven, active 1798-1827",
        "Shakespeare, William (1564-1616)"
    ]
    
    print("\nLife date extraction comparison:")
    for person in test_persons:
        birth, death, conf = get_life_dates(person, True)
        print(f"{person} -> Birth: {birth}, Death: {death}, Confidence: {conf:.2f}")
    
    print("\nTest completed.")

def test_feature_vector_enhancement():
    """Test feature vector enhancement with enhanced dates."""
    print("\n=== Testing Feature Vector Enhancement ===\n")
    
    if not ENHANCED_FEATURES_AVAILABLE:
        print("Enhanced features module not available. Skipping feature enhancement tests.")
        return
    
    # Create mock data
    base_features = [0.8, 0.7, 0.6, 0.5, 0.4]  # Dummy base features
    feature_names = ['person_sim', 'record_sim', 'title_sim', 'roles_sim', 'provision_sim']
    
    unique_strings = {
        "hash1": "Smith, John (1856-1920)",
        "hash2": "London: J. Murray, 1880; 1899; 1910.",
        "hash3": "Contributor",
        "hash4": "Smith, John (1856-1920)",
        "hash5": "New York: Harper, 1885; 1905; 1915.",
        "hash6": "Author"
    }
    
    record_field_hashes = {
        "record1": {
            "person": "hash1",
            "provision": "hash2",
            "roles": "hash3"
        },
        "record2": {
            "person": "hash4",
            "provision": "hash5",
            "roles": "hash6"
        }
    }
    
    # Test the enhance_feature_vector function
    enhanced_features, enhanced_names = enhance_feature_vector(
        base_features,
        feature_names,
        unique_strings,
        record_field_hashes,
        "record1",
        "record2"
    )
    
    print("Base features:")
    print(base_features)
    print("\nBase feature names:")
    print(feature_names)
    
    print("\nEnhanced features:")
    print([round(f, 2) for f in enhanced_features[:20]])  # Show only first 20 for brevity
    print(f"... and {len(enhanced_features) - 20} more")
    
    print("\nEnhanced feature names:")
    print(enhanced_names[:20])  # Show only first 20 for brevity
    print(f"... and {len(enhanced_names) - 20} more")
    
    print("\nTest completed.")

def main():
    """Run all tests."""
    print("\n=== Enhanced Date Processing Integration Tests ===\n")
    
    # Run tests
    test_basic_date_extraction()
    test_enhanced_date_extraction()
    test_provision_date_extraction()
    test_temporal_analysis()
    test_integration()
    test_feature_vector_enhancement()
    
    print("\n=== All Tests Completed ===\n")

if __name__ == "__main__":
    main()
