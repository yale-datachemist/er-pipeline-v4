#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update to the integration.py module to better handle cases with asymmetric life dates.
Add these functions to the integration.py file.
"""

import logging
import importlib.util
import sys
from typing import Dict, List, Any, Tuple, Optional, Set

def compare_year_compatibility(
    years: List[int],
    birth_year: Optional[int] = None,
    death_year: Optional[int] = None,
    active_years: Optional[List[int]] = None
) -> float:
    """
    Compare a set of years (usually from provision dates) with life dates
    to determine compatibility.
    
    Args:
        years: List of years to check (e.g. from provision dates)
        birth_year: Birth year if available
        death_year: Death year if available
        active_years: List of active years if available
        
    Returns:
        Compatibility score (0.0 to 1.0)
    """
    if not years:
        return 0.5  # Neutral if no years to compare
    
    # Determine potential lifetime years
    lifetime_years = set()
    
    if birth_year and death_year:
        # Full lifetime
        lifetime_years = set(range(birth_year, death_year + 1))
    elif birth_year:
        # Assume career from birth+18 to birth+68 (typical 50-year career)
        lifetime_years = set(range(birth_year + 18, birth_year + 68))
    elif death_year:
        # Assume last 50 years of life
        lifetime_years = set(range(death_year - 50, death_year + 1))
    elif active_years:
        # Use active years directly
        lifetime_years = set(active_years)
    
    # If we don't have any lifetime years, return neutral score
    if not lifetime_years:
        return 0.5
    
    # Expand lifetime years to account for posthumous publishing
    expanded_lifetime = lifetime_years.copy()
    if death_year:
        # Add 100 years after death for posthumous works
        posthumous = set(range(death_year + 1, death_year + 101))
        expanded_lifetime = expanded_lifetime.union(posthumous)
    
    # Check overlap with expanded lifetime
    overlap = len([y for y in years if y in expanded_lifetime])
    
    # Calculate compatibility score
    if overlap > 0:
        # More overlap = higher compatibility
        compatibility = min(1.0, 0.5 + (overlap / len(years)) * 0.5)
    else:
        # No overlap = lower compatibility
        compatibility = 0.3
    
    # Special case: check for long posthumous publishing pattern
    if death_year and all(y > death_year + 100 for y in years):
        # This is typical for historical/classical figures
        # Adjust compatibility based on era
        if death_year < 1800:
            # Ancient/classical era - more likely to be republished
            compatibility = 0.7
        elif death_year < 1900:
            # Historical era - somewhat likely to be republished
            compatibility = 0.6
        else:
            # Modern era - less likely to be republished so long after death
            compatibility = 0.4
    
    return compatibility

def extract_active_period(active_period: str) -> List[int]:
    """
    Extract years from an active period string.
    
    Args:
        active_period: Active period string (e.g. "1856-1878")
        
    Returns:
        List of years in the active period
    """
    active_years = []
    
    if active_period and '-' in active_period:
        try:
            parts = active_period.split('-')
            start = parts[0].strip()
            end = parts[1].strip()
            
            if start and start.isdigit():
                start_year = int(start)
                if end and end.isdigit():
                    end_year = int(end)
                    active_years = list(range(start_year, end_year + 1))
                else:
                    # If only start is available, estimate 20 year period
                    active_years = list(range(start_year, start_year + 20))
        except (ValueError, IndexError):
            pass
    elif active_period and active_period.isdigit():
        # Single year active period
        year = int(active_period)
        # Create a 5-year window centered on the specified year
        active_years = list(range(year - 2, year + 3))
    
    return active_years

def analyze_asymmetric_dates(
    person1: str,
    person2: str,
    provision1: str,
    provision2: str
) -> Dict[str, float]:
    """
    Analyze cases with asymmetric life dates (one has dates, the other doesn't).
    Use enhanced date processing for the analysis.
    
    Args:
        person1: First person string
        person2: Second person string
        provision1: First provision string
        provision2: Second provision string
        
    Returns:
        Dictionary of analysis results
    """
    # Check if enhanced date processing is available
    try:
        from enhanced_date_processing import (
            extract_life_dates_enhanced,
            extract_dates_from_provision
        )
        
        # Extract life dates
        life_dates1 = extract_life_dates_enhanced(person1)
        life_dates2 = extract_life_dates_enhanced(person2)
        
        # Extract provision dates
        provision_dates1 = extract_dates_from_provision(provision1)
        provision_dates2 = extract_dates_from_provision(provision2)
        
        # Check if we have asymmetric life dates (one has dates, the other doesn't)
        birth_year1, death_year1, active_period1, life_conf1 = life_dates1
        birth_year2, death_year2, active_period2, life_conf2 = life_dates2
        
        has_life_dates1 = birth_year1 is not None or death_year1 is not None or active_period1 is not None
        has_life_dates2 = birth_year2 is not None or death_year2 is not None or active_period2 is not None
        
        # If both have or both lack life dates, this function doesn't apply
        if has_life_dates1 == has_life_dates2:
            return {'asymmetric_dates': 0.0, 'asymmetry_compatibility': 0.5}
        
        # Get specific years from provision dates
        years1, _, _ = provision_dates1
        years2, _, _ = provision_dates2
        
        # Extract active years if available
        active_years1 = extract_active_period(active_period1) if active_period1 else []
        active_years2 = extract_active_period(active_period2) if active_period2 else []
        
        # Calculate compatibility based on which entity has life dates
        if has_life_dates1:
            compatibility = compare_year_compatibility(
                years2, birth_year1, death_year1, active_years1
            )
        else:
            compatibility = compare_year_compatibility(
                years1, birth_year2, death_year2, active_years2
            )
        
        return {
            'asymmetric_dates': 1.0,
            'asymmetry_compatibility': compatibility
        }
    
    except ImportError:
        # Fall back to a simple version if enhanced date processing isn't available
        from utils import extract_years
        import re
        
        # Simple life date extraction
        has_life_dates1 = bool(re.search(r'\d{4}-\d{4}', person1))
        has_life_dates2 = bool(re.search(r'\d{4}-\d{4}', person2))
        
        # If both have or both lack life dates, this function doesn't apply
        if has_life_dates1 == has_life_dates2:
            return {'asymmetric_dates': 0.0, 'asymmetry_compatibility': 0.5}
        
        # Extract years from provision strings
        years1 = extract_years(provision1)
        years2 = extract_years(provision2)
        
        # Basic compatibility check - do the years overlap at all?
        if years1 and years2:
            intersection = years1.intersection(years2)
            union = years1.union(years2)
            compatibility = len(intersection) / len(union) if union else 0.5
        else:
            compatibility = 0.5
        
        return {
            'asymmetric_dates': 1.0,
            'asymmetry_compatibility': compatibility
        }
    
def check_enhanced_date_processing_available() -> bool:
    """
    Check if the enhanced date processing module is available.
    
    Returns:
        True if enhanced date processing is available, False otherwise
    """
    try:
        spec = importlib.util.find_spec('enhanced_date_processing')
        if spec is None:
            logger.warning("Enhanced date processing module not found.")
            return False
        
        # Try to import main functions to verify they exist
        from enhanced_date_processing import (
            extract_life_dates_enhanced,
            extract_dates_from_provision,
            analyze_temporal_relationship,
            integrate_enhanced_dates_with_base_features
        )
        
        logger.info("Enhanced date processing module is available.")
        return True
    except ImportError as e:
        logger.warning(f"Error importing enhanced date processing: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error checking enhanced date processing: {e}")
        return False

def update_config_for_enhanced_dates(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration to enable enhanced date processing features.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Make a copy to avoid modifying the original
    updated_config = config.copy()
    
    # Check if enhanced date processing is available
    enhanced_available = check_enhanced_date_processing_available()
    
    # Update configuration
    updated_config["use_enhanced_date_processing"] = enhanced_available
    
    # If enhanced date processing is available, enable related features
    if enhanced_available:
        updated_config["use_enhanced_features"] = True
        updated_config["enhanced_temporal_features"] = True
        
        # Set date handling options if not already set
        if "robust_date_handling" not in updated_config:
            updated_config["robust_date_handling"] = True
        
        if "fuzzy_year_threshold" not in updated_config:
            updated_config["fuzzy_year_threshold"] = 5
            
        logger.info("Configuration updated to use enhanced date processing.")
    else:
        logger.info("Using basic date processing - enhanced module not available.")
    
    return updated_config

def extract_years_from_text(
    text: str,
    use_enhanced: bool = True
) -> Set[int]:
    """
    Extract years from text, using enhanced extraction if available.
    
    Args:
        text: Text to extract years from
        use_enhanced: Whether to try using enhanced extraction
        
    Returns:
        Set of years extracted from text
    """
    if use_enhanced and check_enhanced_date_processing_available():
        try:
            from enhanced_date_processing import extract_dates_from_provision
            years, _, _ = extract_dates_from_provision(text)
            return set(years)
        except Exception as e:
            logger.warning(f"Error in enhanced year extraction: {e}")
            # Fall back to basic extraction
    
    # Basic extraction as fallback
    from utils import extract_years
    return extract_years(text)

def get_life_dates(
    person_str: str,
    use_enhanced: bool = True
) -> Tuple[Optional[int], Optional[int], float]:
    """
    Extract life dates from a person string.
    
    Args:
        person_str: Person string
        use_enhanced: Whether to try using enhanced extraction
        
    Returns:
        Tuple of (birth_year, death_year, confidence)
    """
    if use_enhanced and check_enhanced_date_processing_available():
        try:
            from enhanced_date_processing import extract_life_dates_enhanced
            birth_year, death_year, active_period, confidence = extract_life_dates_enhanced(person_str)
            return birth_year, death_year, confidence
        except Exception as e:
            logger.warning(f"Error in enhanced life date extraction: {e}")
            # Fall back to basic extraction
    
    # Basic extraction as fallback
    from enhanced_features import extract_life_dates
    birth_year, death_year = extract_life_dates(person_str)
    confidence = 1.0 if (birth_year is not None and death_year is not None) else 0.5
    
    return birth_year, death_year, confidence