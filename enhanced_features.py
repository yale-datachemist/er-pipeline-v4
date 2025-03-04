#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced feature engineering for the entity resolution pipeline.
Implements advanced temporal analysis and feature interaction terms.
Integrates with enhanced date processing capabilities.
"""

import logging
import re
import statistics
from typing import Dict, List, Any, Set, Tuple, Optional

import numpy as np
from collections import Counter

# Import enhanced date processing if available
try:
    from enhanced_date_processing import (
        extract_life_dates_enhanced, extract_dates_from_provision,
        analyze_temporal_relationship, detect_publication_roles,
        is_historical_figure_pattern, integrate_enhanced_dates_with_base_features
    )
    ENHANCED_DATE_PROCESSING_AVAILABLE = True
except ImportError:
    ENHANCED_DATE_PROCESSING_AVAILABLE = False
    logging.getLogger(__name__).warning("Enhanced date processing module not available. Using basic date processing.")

# Configure logging
logger = logging.getLogger(__name__)

ENHANCED_FEATURE_DESCRIPTIONS = {
    "birth_year_similarity": "Similarity between birth years",
    "death_year_similarity": "Similarity between death years",
    "active_period_similarity": "Similarity between active periods",
    "provision_year_overlap": "Overlap between publication years",
    "date_range_overlap": "Overlap between publication date ranges",
    "temporal_compatibility": "Overall temporal compatibility score",
    # Add descriptions for all enhanced features
}

def get_enhanced_feature_description(name: str) -> str:
    """Get a descriptive name for an enhanced feature."""
    return ENHANCED_FEATURE_DESCRIPTIONS.get(name, name)

def get_descriptive_feature_names(enhanced_features: List[float], base_names: List[str]) -> List[str]:
    """
    Generate descriptive names for enhanced features.
    
    Args:
        enhanced_features: Enhanced feature vector
        base_names: Base feature names before enhancement
        
    Returns:
        List of descriptive feature names
    """
    # Descriptive names for common enhanced features
    name_map = {
        # Vector similarities
        "person_sim": "Person Name Vector Similarity",
        "record_sim": "Record Vector Similarity",
        "title_sim": "Title Vector Similarity",
        "roles_sim": "Roles Vector Similarity",
        "attribution_sim": "Attribution Vector Similarity",
        "provision_sim": "Provision Vector Similarity",
        "subjects_sim": "Subjects Vector Similarity",
        "genres_sim": "Genres Vector Similarity",
        "relatedWork_sim": "Related Work Vector Similarity",
        
        # Name features
        "person_lev_sim": "Person Name Levenshtein Similarity",
        "has_life_dates": "Has Life Dates",
        "life_dates_match_score": "Life Dates Match Score",
        
        # Temporal features
        "temporal_overlap": "Temporal Overlap",
        "birth_year_similarity": "Birth Year Similarity",
        "death_year_similarity": "Death Year Similarity",
        "active_period_similarity": "Active Period Similarity",
        "provision_year_overlap": "Publication Year Overlap",
        "date_range_overlap": "Date Range Overlap",
        "publication_during_lifetime_similarity": "Publication During Lifetime Similarity",
        "posthumous_publication_similarity": "Posthumous Publication Similarity",
        "historical_republication_similarity": "Historical Republication Similarity",
        "temporal_plausibility_similarity": "Temporal Plausibility Similarity",
        "temporal_compatibility": "Temporal Compatibility Score",
        
        # Role features
        "is_author_consistency": "Author Role Consistency",
        "is_subject_consistency": "Subject Role Consistency",
        "is_editor_consistency": "Editor Role Consistency",
        "is_translator_consistency": "Translator Role Consistency",
        "is_contributor_consistency": "Contributor Role Consistency",
        
        # Historical pattern features
        "historical_figure_pattern1": "Historical Figure Pattern (Person 1)",
        "historical_figure_pattern2": "Historical Figure Pattern (Person 2)",
        "historical_pattern_match": "Historical Pattern Match Score",
        
        # Confidence features
        "life_dates_confidence": "Life Dates Confidence",
        "provision_dates_confidence": "Provision Dates Confidence",
        
        # Asymmetry features
        "has_asymmetric_life_dates": "Has Asymmetric Life Dates",
        "asymmetry_compatibility": "Asymmetry Compatibility Score",
        
        # Interaction features
        "person_sim_person_lev_sim_interaction": "Person Vector x Levenshtein Interaction",
        "temporal_overlap_person_sim_interaction": "Temporal Overlap x Person Sim Interaction", 
        "exact_name_temporal_match": "Exact Name + Temporal Match",
        "person_sim_squared": "Person Similarity Squared",
        "person_title_harmonic": "Person-Title Harmonic Mean"
    }
    
    # Create descriptive names for all features
    descriptive_names = []
    for i, base_name in enumerate(base_names):
        if i < len(enhanced_features):
            # Use mapped name if available
            descriptive_names.append(name_map.get(base_name, base_name))
    
    # Add generic names for any additional features
    for i in range(len(descriptive_names), len(enhanced_features)):
        descriptive_names.append(f"Enhanced Feature {i}")
    
    return descriptive_names

def extract_life_dates(person_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract birth and death years from a person name string.
    
    Args:
        person_str: Person name string
        
    Returns:
        Tuple of (birth_year, death_year), both may be None
    """
    if not person_str:
        return None, None
    
    # Look for patterns like "Smith, John (1856-1920)" or "Smith, John (1856-)"
    date_match = re.search(r'\((\d{4})-(\d{4}|\s*)\)', person_str)
    if date_match:
        birth_str = date_match.group(1).strip()
        death_str = date_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        
        # Basic validation
        if birth_year and (birth_year < 1400 or birth_year > 2025):
            birth_year = None  # Likely not a valid year
        
        if death_year and (death_year < 1400 or death_year > 2025):
            death_year = None  # Likely not a valid year
            
        # Check if death is after birth
        if birth_year and death_year and death_year < birth_year:
            # Likely a data error, but keep both values
            pass
        
        return birth_year, death_year
    
    # Alternative pattern: "Smith, John, 1856-1920"
    date_match = re.search(r',\s*(\d{4})-(\d{4}|\s*)', person_str)
    if date_match:
        birth_str = date_match.group(1).strip()
        death_str = date_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        
        # Validation
        if birth_year and (birth_year < 1400 or birth_year > 2025):
            birth_year = None
        
        if death_year and (death_year < 1400 or death_year > 2025):
            death_year = None
            
        return birth_year, death_year
    
    return None, None


def calculate_fuzzy_year_match(year1: int, year2: int, threshold: int = 5) -> float:
    """
    Calculate fuzzy match score between two years, robust against data quality issues.
    
    Args:
        year1: First year
        year2: Second year
        threshold: Maximum difference for scaled matching
        
    Returns:
        Match score (0.0 to 1.0)
    """
    if year1 == year2:
        return 1.0  # Exact match
    
    # Check for small differences (typos)
    diff = abs(year1 - year2)
    if diff <= threshold:
        return 1.0 - (diff / (threshold + 1))
    
    # Check for transposed digits
    year1_str = str(year1)
    year2_str = str(year2)
    if len(year1_str) == len(year2_str) == 4:
        # Check if digits are the same but in different order
        if sorted(year1_str) == sorted(year2_str):
            return 0.8  # High confidence for transposed digits
        
        # Check for specific transposition patterns (adjacent digits)
        for i in range(3):
            transposed = year1_str[:i] + year1_str[i+1] + year1_str[i] + year1_str[i+2:]
            if transposed == year2_str:
                return 0.8  # Adjacent transposition
    
    # Check for century errors (e.g., 1789 vs 1889 - same last two digits)
    if len(year1_str) == len(year2_str) == 4:
        if year1_str[2:] == year2_str[2:] and abs(int(year1_str[:2]) - int(year2_str[:2])) == 1:
            return 0.6  # Medium confidence for century errors
    
    # Check for single-digit errors
    if len(year1_str) == len(year2_str) == 4:
        diff_count = sum(1 for a, b in zip(year1_str, year2_str) if a != b)
        if diff_count == 1:
            return 0.7  # Good confidence for single-digit error
    
    return 0.0  # No match


def analyze_robust_temporal_features(
    years1: List[int],
    years2: List[int],
    birth_year1: Optional[int],
    death_year1: Optional[int],
    birth_year2: Optional[int],
    death_year2: Optional[int]
) -> Dict[str, float]:
    """
    Extract temporal features with robustness against data quality issues.
    Takes into account both life dates and publication years.
    
    Args:
        years1: Publication years from first record
        years2: Publication years from second record
        birth_year1: Birth year from first person
        death_year1: Death year from first person
        birth_year2: Birth year from second person
        death_year2: Death year from second person
        
    Returns:
        Dictionary of robust temporal features
    """
    features = {}
    
    # Handle empty data with neutral values
    if not years1 and not years2 and not birth_year1 and not birth_year2:
        return {
            'temporal_overlap': 0.5,
            'temporal_span_similarity': 0.5,
            'publication_era_match': 0.5,
            'life_dates_match': 0.0,
            'biographical_plausibility': 0.5,
            'posthumous_publication': 0.0,
            'lifespan_publication_alignment': 0.5
        }
    
    # Life dates matching (with fuzzy comparison)
    if birth_year1 and birth_year2:
        birth_match = calculate_fuzzy_year_match(birth_year1, birth_year2)
        features['birth_year_match'] = birth_match
    else:
        features['birth_year_match'] = 0.0
        
    if death_year1 and death_year2:
        death_match = calculate_fuzzy_year_match(death_year1, death_year2)
        features['death_year_match'] = death_match
    else:
        features['death_year_match'] = 0.0
    
    # Overall life dates match
    if birth_year1 and birth_year2:
        if death_year1 and death_year2:
            features['life_dates_match'] = (features['birth_year_match'] + features['death_year_match']) / 2
        else:
            features['life_dates_match'] = features['birth_year_match']
    elif death_year1 and death_year2:
        features['life_dates_match'] = features['death_year_match']
    else:
        features['life_dates_match'] = 0.0
    
    # Fuzzy temporal overlap between publication years
    fuzzy_overlaps = 0
    total_comparisons = 0
    
    for y1 in years1:
        for y2 in years2:
            match_score = calculate_fuzzy_year_match(y1, y2)
            if match_score > 0:
                fuzzy_overlaps += match_score
            total_comparisons += 1
    
    if total_comparisons > 0:
        features['temporal_overlap'] = fuzzy_overlaps / total_comparisons
    elif years1 or years2:  # One has years but not the other
        features['temporal_overlap'] = 0.0
    else:  # Both have no years
        features['temporal_overlap'] = 0.5  # Neutral
    
    # Biographical plausibility checks
    # Check if publication years fall within or slightly after life span
    lifetime_plausibility1 = 1.0
    lifetime_plausibility2 = 1.0
    
    # Function to calculate plausibility of publication years given life dates
    def calculate_plausibility(pub_years, birth_year, death_year, roles=None):
        """
        Calculate plausibility of publication timeline given life dates.
        Handles both contemporary publications and posthumous/historical works.
        
        Args:
            pub_years: List of publication years
            birth_year: Birth year or None
            death_year: Death year or None
            roles: List of roles (author, editor, subject, etc.) or None
            
        Returns:
            Plausibility score (0.0-1.0)
        """
        if not pub_years or (birth_year is None and death_year is None):
            return 0.5  # Neutral when missing data
        
        # IMPORTANT: No arbitrary limit on posthumous publications
        # Works can be published centuries after an author's death
        
        # Different patterns to check:
        # 1. Is the first publication during lifetime? (Strong signal for actual authorship)
        # 2. Is there a publication pattern consistent with an active career?
        # 3. Are publications primarily posthumous? (Common for collected works, translations)
        
        # First, identify if this is primarily lifetime or posthumous publishing
        if death_year:
            lifetime_pubs = [y for y in pub_years if y <= death_year]
            posthumous_pubs = [y for y in pub_years if y > death_year]
            
            # Calculate percentage of posthumous publications
            posthumous_ratio = len(posthumous_pubs) / len(pub_years) if pub_years else 0
            
            # If ≥50% are posthumous, this is likely a posthumous publication pattern
            primarily_posthumous = posthumous_ratio >= 0.5
        else:
            # Without death year, assume contemporary publishing
            primarily_posthumous = False
            lifetime_pubs = pub_years
        
        # Check for contemporary publishing pattern
        if not primarily_posthumous and birth_year:
            # Expect most publications to start around age 20-25
            expected_start = birth_year + 20
            
            # Check if first publication is plausibly timed
            first_pub = min(pub_years) if pub_years else 0
            
            # Allow publications slightly before recorded birth (data errors)
            if first_pub < birth_year - 10:
                # First publication significantly before birth is implausible
                # Unless person is subject rather than author
                return 0.3
            
            # Check for a reasonable active career pattern
            if death_year:
                career_length = death_year - expected_start
                pub_span = max(pub_years) - min(pub_years) if len(pub_years) > 1 else 0
                
                # Is publishing span reasonably aligned with career?
                # Note: Don't penalize short careers or publishing spans
                if career_length > 10 and pub_span > 0:
                    span_ratio = min(pub_span / career_length, 1.0)
                    return 0.5 + (span_ratio * 0.5)  # Higher score for alignment
            
            # Without death year, simply check if publications start at plausible age
            return 0.8 if first_pub >= expected_start - 10 else 0.4
        
        # For posthumous publishing pattern
        elif primarily_posthumous and death_year:
            # Check if earliest posthumous publication is within 50 years of death
            # (common for collected works and immediate posthumous publishing)
            first_posthumous = min(posthumous_pubs) if posthumous_pubs else 0
            recently_posthumous = (first_posthumous - death_year) <= 50
            
            # If some publications were during lifetime and some posthumous,
            # this is a common pattern for important authors
            if lifetime_pubs and posthumous_pubs:
                return 0.7 if recently_posthumous else 0.5
            
            # If all publications are posthumous but started within 50 years of death
            if not lifetime_pubs and recently_posthumous:
                return 0.6
            
            # All publications are much later than death - less certain but still possible
            # (classics, historical figures, etc.)
            return 0.4
        
        # When we have limited information, be neutral
        return 0.5
    
    # Calculate plausibility in both directions
    if years2:
        lifetime_plausibility1 = calculate_plausibility(years2, birth_year1, death_year1)
    
    if years1:
        lifetime_plausibility2 = calculate_plausibility(years1, birth_year2, death_year2)
    
    # Average the plausibility in both directions
    features['biographical_plausibility'] = (lifetime_plausibility1 + lifetime_plausibility2) / 2
    
    # Posthumous publication detection
    # Instead of looking for arbitrary timeframes, look for publishing patterns
    posthumous_pattern = False
    
    if death_year1 and years2:
        # Check if most publications are after death
        posthumous_count = sum(1 for year in years2 if year > death_year1)
        posthumous_pattern = posthumous_count > len(years2) / 2
    
    if death_year2 and years1:
        # Check if most publications are after death
        posthumous_count = sum(1 for year in years1 if year > death_year2)
        posthumous_pattern = posthumous_pattern or (posthumous_count > len(years1) / 2)
    
    features['posthumous_publication_pattern'] = 1.0 if posthumous_pattern else 0.0
    
    # Instead of simple posthumous flag, provide more details about the pattern
    if death_year1 and years2:
        # Calculate time ranges after death
        years_after_death = [year - death_year1 for year in years2 if year > death_year1]
        if years_after_death:
            min_years_after = min(years_after_death)
            max_years_after = max(years_after_death)
            features['min_years_after_death1'] = min_years_after / 100  # Normalize
            features['max_years_after_death1'] = max_years_after / 1000  # Normalize for centuries
            features['long_posthumous_publishing1'] = 1.0 if max_years_after > 50 else 0.0
    
    if death_year2 and years1:
        # Calculate time ranges after death
        years_after_death = [year - death_year2 for year in years1 if year > death_year2]
        if years_after_death:
            min_years_after = min(years_after_death)
            max_years_after = max(years_after_death)
            features['min_years_after_death2'] = min_years_after / 100  # Normalize
            features['max_years_after_death2'] = max_years_after / 1000  # Normalize for centuries
            features['long_posthumous_publishing2'] = 1.0 if max_years_after > 50 else 0.0
    
    # Publication span similarity (with error tolerance)
    if len(years1) > 1 and len(years2) > 1:
        span1 = max(years1) - min(years1)
        span2 = max(years2) - min(years2)
        
        # Normalize the difference
        max_span = max(span1, span2, 1)  # Avoid division by zero
        features['temporal_span_similarity'] = 1.0 - (abs(span1 - span2) / max_span)
    else:
        features['temporal_span_similarity'] = 0.5  # Neutral for insufficient data
    
    # Era categorization with fuzzy matching for errors
    era_ranges = [(0, 1800), (1800, 1850), (1850, 1900), (1900, 1925), 
                 (1925, 1950), (1950, 1975), (1975, 2000), (2000, 2025), (2025, 2100)]
    
    # Function to get eras with fuzzy boundary matching
    def get_eras_with_fuzzy_boundaries(years):
        eras = set()
        for year in years:
            # Check each era range
            for i, (start, end) in enumerate(era_ranges):
                # Exact match
                if start <= year < end:
                    eras.add(i)
                    continue
                
                # Fuzzy match for years near boundaries
                if abs(year - start) <= 3:  # Within 3 years of start
                    eras.add(i)  # Current era
                    if i > 0:
                        eras.add(i-1)  # Previous era
                
                if abs(year - end) <= 3:  # Within 3 years of end
                    eras.add(i)  # Current era
                    if i < len(era_ranges) - 1:
                        eras.add(i+1)  # Next era
        
        return eras
    
    era1 = get_eras_with_fuzzy_boundaries(years1)
    era2 = get_eras_with_fuzzy_boundaries(years2)
    
    # Calculate era match with Jaccard similarity for flexibility
    if era1 and era2:
        intersection = len(era1 & era2)
        union = len(era1 | era2)
        features['publication_era_match'] = intersection / union if union > 0 else 0.0
    else:
        features['publication_era_match'] = 0.5  # Neutral
    
    # Publication vs. life dates timeline alignment
    # Check how publication timeline aligns with life dates
    def align_pub_with_life(pub_years, birth_year, death_year):
        if not pub_years or (birth_year is None and death_year is None):
            return 0.5  # Neutral
        
        # Handle both regular publishing and posthumous classics
        
        # Are all publications long after death?
        # (Classical authors, historical figures being republished)
        if death_year and all(year > death_year + 100 for year in pub_years):
            # This is a pattern for Plato, Shakespeare, etc. - all publications very posthumous
            # This should not be treated as a signal against identity match
            # In fact, it could be a signal FOR a match for famous historical figures
            return 0.6
        
        # Are all publications very recent but birth/death dates are historical?
        # (Historical figures with modern editions)
        current_year = 2025  # Use fixed reference point
        if birth_year and birth_year < current_year - 150:  # Historical birth date
            if all(year > current_year - 50 for year in pub_years):  # Recent publications
                # Pattern of republishing historical authors in modern editions
                return 0.7
                
        # Regular case for contemporary authors
        if birth_year and death_year:
            expected_career_start = birth_year + 20  # Typical earliest publication age
            expected_career_end = death_year
            
            # Calculate active career years
            career_years = range(expected_career_start, expected_career_end + 1)
            
            # Check publication distribution across career
            pub_in_early = any(expected_career_start <= y <= expected_career_start + 15 for y in pub_years)
            pub_in_late = any(expected_career_end - 15 <= y <= expected_career_end for y in pub_years)
            
            # Calculate percentage of career covered by publications
            if expected_career_end > expected_career_start:
                career_span = expected_career_end - expected_career_start
                min_pub = min(pub_years) if pub_years else 0
                max_pub = max(pub_years) if pub_years else 0
                
                if min_pub <= max_pub:  # Valid publication span
                    if min_pub < expected_career_start:
                        min_pub = expected_career_start
                    if max_pub > expected_career_end:
                        max_pub = expected_career_end
                    
                    pub_span = max(0, max_pub - min_pub)
                    coverage = pub_span / career_span if career_span > 0 else 0
                    
                    # Balance of career coverage and publication at key career points
                    score = (coverage * 0.6) + (0.2 if pub_in_early else 0) + (0.2 if pub_in_late else 0)
                    return score
        
        return 0.5  # Neutral for other cases
    
    # Calculate alignment in both directions
    align1 = align_pub_with_life(years2, birth_year1, death_year1)
    align2 = align_pub_with_life(years1, birth_year2, death_year2)
    
    # Average the alignment scores
    features['lifespan_publication_alignment'] = (align1 + align2) / 2
    
    return features


def handle_life_date_asymmetry(
    life_dates1: Tuple[Optional[int], Optional[int], Optional[str], float],
    life_dates2: Tuple[Optional[int], Optional[int], Optional[str], float],
    provision_dates1: Tuple[List[int], List[str], float],
    provision_dates2: Tuple[List[int], List[str], float]
) -> Dict[str, float]:
    """
    Handle cases where one entity has life dates but the other doesn't.
    Use provision dates to determine if the entities are likely the same person.
    
    Args:
        life_dates1: Life dates for first entity (birth_year, death_year, active_period, confidence)
        life_dates2: Life dates for second entity (birth_year, death_year, active_period, confidence)
        provision_dates1: Provision dates for first entity (years, ranges, confidence)
        provision_dates2: Provision dates for second entity (years, ranges, confidence)
        
    Returns:
        Dictionary of features related to life date asymmetry
    """
    birth_year1, death_year1, active_period1, life_conf1 = life_dates1
    birth_year2, death_year2, active_period2, life_conf2 = life_dates2
    specific_years1, date_ranges1, prov_conf1 = provision_dates1
    specific_years2, date_ranges2, prov_conf2 = provision_dates2
    
    features = {}
    
    # Check if we have asymmetric life dates (one has dates, the other doesn't)
    has_life_dates1 = birth_year1 is not None or death_year1 is not None or active_period1 is not None
    has_life_dates2 = birth_year2 is not None or death_year2 is not None or active_period2 is not None
    asymmetric = has_life_dates1 != has_life_dates2
    
    features['has_asymmetric_life_dates'] = 1.0 if asymmetric else 0.0
    
    # If not asymmetric, no need for further processing
    if not asymmetric:
        features['asymmetry_compatibility'] = 0.5  # Neutral
        return features
    
    # Get the entity with life dates and the one without
    if has_life_dates1:
        life_dates_entity = life_dates1
        life_dates_provision = provision_dates1
        no_dates_provision = provision_dates2
    else:
        life_dates_entity = life_dates2
        life_dates_provision = provision_dates2
        no_dates_provision = provision_dates1
    
    birth, death, active, _ = life_dates_entity
    no_dates_years, _, _ = no_dates_provision
    
    # Calculate timeline compatibility
    compatibility = 0.5  # Start with neutral value
    
    if no_dates_years and (birth or death or active):
        # Convert active period to years if available
        active_years = []
        if active and '-' in active:
            try:
                parts = active.split('-')
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] and parts[1].isdigit() else None
                
                if start and end:
                    active_years = list(range(start, end + 1))
                elif start:
                    # Estimate 20 year career if only start year is available
                    active_years = list(range(start, start + 20))
            except (ValueError, IndexError):
                pass
        
        # Determine potential lifetime years
        lifetime_years = set()
        
        if birth and death:
            # Full lifetime
            lifetime_years = set(range(birth, death + 1))
        elif birth:
            # Assume career from birth+18 to birth+68 (typical 50-year career)
            lifetime_years = set(range(birth + 18, birth + 68))
        elif death:
            # Assume last 50 years of life
            lifetime_years = set(range(death - 50, death + 1))
        elif active_years:
            # Use active years directly
            lifetime_years = set(active_years)
        
        # Expand lifetime years to account for posthumous publishing
        expanded_lifetime = lifetime_years.copy()
        if death:
            # Add 100 years after death for posthumous works
            posthumous = set(range(death + 1, death + 101))
            expanded_lifetime = expanded_lifetime.union(posthumous)
        
        # Check overlap of no_dates_years with expanded lifetime
        overlap = len([y for y in no_dates_years if y in expanded_lifetime])
        
        # Calculate compatibility score
        if overlap > 0:
            # More overlap = higher compatibility
            compatibility = min(1.0, 0.5 + (overlap / len(no_dates_years)) * 0.5)
        else:
            # No overlap = lower compatibility
            compatibility = 0.3
        
        # Special case: check for long posthumous publishing pattern
        if death and all(y > death + 100 for y in no_dates_years):
            # This is typical for historical/classical figures
            # Adjust compatibility based on era
            if death < 1800:
                # Ancient/classical era - more likely to be republished
                compatibility = 0.7
            elif death < 1900:
                # Historical era - somewhat likely to be republished
                compatibility = 0.6
            else:
                # Modern era - less likely to be republished so long after death
                compatibility = 0.4
    
    features['asymmetry_compatibility'] = compatibility
    
    return features


def compare_provision_dates_without_life_dates(
    provision_dates1: Tuple[List[int], List[str], float],
    provision_dates2: Tuple[List[int], List[str], float]
) -> Dict[str, float]:
    """
    Enhanced comparison of provision dates when life dates are missing,
    using more sophisticated timeline analysis.
    
    Args:
        provision_dates1: First provision dates (years, ranges, confidence)
        provision_dates2: Second provision dates (years, ranges, confidence)
        
    Returns:
        Dictionary of enhanced provision comparison features
    """
    specific_years1, date_ranges1, conf1 = provision_dates1
    specific_years2, date_ranges2, conf2 = provision_dates2
    
    features = {}
    
    # Start with confidence in the data
    features['provision_data_confidence'] = conf1 * conf2
    
    # If no provision years, return neutral features
    if not specific_years1 or not specific_years2:
        return {
            'provision_data_confidence': features['provision_data_confidence'],
            'temporal_span_similarity': 0.5,
            'era_overlap': 0.5,
            'publication_density_similarity': 0.5,
            'publication_pattern_similarity': 0.5
        }
    
    # 1. Temporal span similarity (publication career length)
    if len(specific_years1) > 1 and len(specific_years2) > 1:
        span1 = max(specific_years1) - min(specific_years1)
        span2 = max(specific_years2) - min(specific_years2)
        
        # Normalize the difference
        max_span = max(span1, span2, 1)  # Avoid division by zero
        features['temporal_span_similarity'] = 1.0 - (abs(span1 - span2) / max_span)
    else:
        features['temporal_span_similarity'] = 0.5  # Neutral for insufficient data
    
    # 2. Era overlap (historical periods)
    # Define historical eras
    eras = [
        (0, 1600, 'Early Modern'),
        (1600, 1700, '17th Century'),
        (1700, 1800, '18th Century'),
        (1800, 1850, 'Early 19th Century'),
        (1850, 1900, 'Late 19th Century'),
        (1900, 1925, 'Early 20th Century'),
        (1925, 1950, 'Mid 20th Century'),
        (1950, 1975, 'Post-War'),
        (1975, 2000, 'Late 20th Century'),
        (2000, 2025, '21st Century')
    ]
    
    # Function to get eras for years
    def get_eras_for_years(years):
        result = set()
        for year in years:
            for start, end, name in eras:
                if start <= year < end:
                    result.add(name)
        return result
    
    eras1 = get_eras_for_years(specific_years1)
    eras2 = get_eras_for_years(specific_years2)
    
    # Calculate Jaccard similarity of eras
    if eras1 and eras2:
        intersection = len(eras1.intersection(eras2))
        union = len(eras1.union(eras2))
        features['era_overlap'] = intersection / union if union > 0 else 0.0
    else:
        features['era_overlap'] = 0.5
    
    # 3. Publication density similarity
    # Check if publication frequency/density is similar
    if len(specific_years1) > 2 and len(specific_years2) > 2:
        # Sort years
        sorted_years1 = sorted(specific_years1)
        sorted_years2 = sorted(specific_years2)
        
        # Calculate gaps between publications
        gaps1 = [sorted_years1[i+1] - sorted_years1[i] for i in range(len(sorted_years1) - 1)]
        gaps2 = [sorted_years2[i+1] - sorted_years2[i] for i in range(len(sorted_years2) - 1)]
        
        # Calculate average gap
        avg_gap1 = sum(gaps1) / len(gaps1) if gaps1 else 0
        avg_gap2 = sum(gaps2) / len(gaps2) if gaps2 else 0
        
        # Calculate median gap
        median_gap1 = sorted(gaps1)[len(gaps1) // 2] if gaps1 else 0
        median_gap2 = sorted(gaps2)[len(gaps2) // 2] if gaps2 else 0
        
        # Compare density patterns
        avg_gap_similarity = 1.0 - min(1.0, abs(avg_gap1 - avg_gap2) / max(avg_gap1, avg_gap2, 1))
        median_gap_similarity = 1.0 - min(1.0, abs(median_gap1 - median_gap2) / max(median_gap1, median_gap2, 1))
        
        features['publication_density_similarity'] = (avg_gap_similarity + median_gap_similarity) / 2
    else:
        features['publication_density_similarity'] = 0.5
    
    # 4. Publication pattern similarity
    # Compare timeline patterns
    if len(specific_years1) >= 3 and len(specific_years2) >= 3:
        # Identify if publication pattern is:
        # - Regular (steady publications)
        # - Burst (clusters of publications)
        # - Sporadic (random publications)
        
        # Calculate coefficient of variation (CV) of gaps 
        # (standard deviation / mean - shows regularity)
        import statistics
        
        sorted_years1 = sorted(specific_years1)
        sorted_years2 = sorted(specific_years2)
        
        gaps1 = [sorted_years1[i+1] - sorted_years1[i] for i in range(len(sorted_years1) - 1)]
        gaps2 = [sorted_years2[i+1] - sorted_years2[i] for i in range(len(sorted_years2) - 1)]
        
        try:
            cv1 = statistics.stdev(gaps1) / statistics.mean(gaps1) if gaps1 else 0
        except statistics.StatisticsError:
            cv1 = 0
            
        try:
            cv2 = statistics.stdev(gaps2) / statistics.mean(gaps2) if gaps2 else 0
        except statistics.StatisticsError:
            cv2 = 0
        
        # Determine pattern type
        def get_pattern_type(cv):
            if cv < 0.3:
                return "Regular"
            elif cv < 0.7:
                return "Mixed"
            else:
                return "Sporadic"
        
        pattern1 = get_pattern_type(cv1)
        pattern2 = get_pattern_type(cv2)
        
        # Compare patterns
        features['publication_pattern_similarity'] = 1.0 if pattern1 == pattern2 else 0.3
    else:
        features['publication_pattern_similarity'] = 0.5
    
    # 5. Overall temporal compatibility score
    features['provision_temporal_compatibility'] = (
        features.get('temporal_span_similarity', 0.5) * 0.3 +
        features.get('era_overlap', 0.5) * 0.3 +
        features.get('publication_density_similarity', 0.5) * 0.2 +
        features.get('publication_pattern_similarity', 0.5) * 0.2
    )
    
    return features


def generate_interaction_features(
    base_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Generate interaction terms between important features.
    
    Args:
        base_features: Dictionary of base features
        
    Returns:
        Dictionary of interaction features
    """
    interaction_features = {}
    
    # List of core features to consider for interactions
    semantic_features = [
        'person_sim', 'title_sim', 'subjects_sim', 'roles_sim'
    ]
    
    temporal_features = [
        'temporal_overlap', 'temporal_span_similarity', 'publication_era_match',
        'min_year_difference', 'max_year_difference', 'median_year_difference'
    ]
    
    string_features = ['person_lev_sim']
    
    domain_features = ['subjects_sim', 'genres_sim', 'roles_sim']
    
    # Only use features that exist in the base_features
    semantic_features = [f for f in semantic_features if f in base_features]
    temporal_features = [f for f in temporal_features if f in base_features]
    string_features = [f for f in string_features if f in base_features]
    domain_features = [f for f in domain_features if f in base_features]
    
    # 1. Interactions between semantic similarity and string similarity
    for sem_feat in semantic_features:
        for str_feat in string_features:
            if sem_feat != str_feat:  # Avoid self-interaction
                key = f"{sem_feat}_{str_feat}_interaction"
                interaction_features[key] = base_features[sem_feat] * base_features[str_feat]
    
    # 2. Interactions between temporal features and other features
    for temp_feat in temporal_features:
        # Temporal x Person Name
        if 'person_sim' in base_features:
            key = f"{temp_feat}_person_sim_interaction"
            interaction_features[key] = base_features[temp_feat] * base_features['person_sim']
        
        if 'person_lev_sim' in base_features:
            key = f"{temp_feat}_person_lev_interaction"
            interaction_features[key] = base_features[temp_feat] * base_features['person_lev_sim']
        
        # Temporal x Role (especially important for works published posthumously)
        if 'roles_sim' in base_features:
            key = f"{temp_feat}_roles_interaction" 
            interaction_features[key] = base_features[temp_feat] * base_features['roles_sim']
    
    # 3. Interactions between subject/genre and other features
    for domain_feat in domain_features:
        # Extra weight when both subject and person match
        if 'person_sim' in base_features:
            key = f"{domain_feat}_person_sim_interaction"
            interaction_features[key] = base_features[domain_feat] * base_features['person_sim']
        
        # Subject/genre and temporal features interaction
        for temp_feat in temporal_features[:2]:  # Only use main temporal features
            key = f"{domain_feat}_{temp_feat}_interaction"
            interaction_features[key] = base_features[domain_feat] * base_features[temp_feat]
    
    # 4. Special interactions for strong signals
    # When person names match exactly AND temporal overlap
    if all(f in base_features for f in ['person_lev_sim', 'temporal_overlap']):
        # Higher weight for exact name match with temporal overlap
        exact_match_temporal = (base_features['person_lev_sim'] > 0.9) and (base_features['temporal_overlap'] > 0.5)
        interaction_features['exact_name_temporal_match'] = 1.0 if exact_match_temporal else 0.0
    
    # 5. Life dates interaction (if available)
    if 'has_life_dates' in base_features and 'temporal_overlap' in base_features:
        interaction_features['life_dates_temporal_interaction'] = (
            base_features['has_life_dates'] * base_features['temporal_overlap']
        )
    
    # 6. Non-linear transformations
    # Square of person similarity - emphasizes high similarity
    if 'person_sim' in base_features:
        interaction_features['person_sim_squared'] = base_features['person_sim'] ** 2
    
    # Product of all temporal features - strong signal when all align
    if len(temporal_features) >= 3:
        temp_product = 1.0
        for feat in temporal_features[:3]:  # Use first 3 temporal features
            temp_product *= base_features[feat]
        interaction_features['temporal_consensus'] = temp_product
    
    # Harmonic mean of key features
    if all(f in base_features for f in ['person_sim', 'title_sim']):
        if base_features['person_sim'] > 0 and base_features['title_sim'] > 0:
            harmonic_mean = 2 * (base_features['person_sim'] * base_features['title_sim']) / (
                base_features['person_sim'] + base_features['title_sim']
            )
            interaction_features['person_title_harmonic'] = harmonic_mean
        else:
            interaction_features['person_title_harmonic'] = 0.0
    
    # 7. Asymmetric life dates interactions (if available)
    if 'has_asymmetric_life_dates' in base_features and base_features['has_asymmetric_life_dates'] > 0:
        # Increase importance of asymmetry compatibility feature
        if 'asymmetry_compatibility' in base_features:
            if 'person_lev_sim' in base_features:
                # If names are very similar, boost the compatibility signal
                interaction_features['asymmetry_name_interaction'] = (
                    base_features['asymmetry_compatibility'] * 
                    (0.5 + 0.5 * base_features['person_lev_sim'])
                )
            
            # If roles match well, boost compatibility
            if 'roles_sim' in base_features:
                interaction_features['asymmetry_roles_interaction'] = (
                    base_features['asymmetry_compatibility'] * 
                    base_features['roles_sim']
                )
    
    return interaction_features


def enhance_feature_vector(
    base_features: List[float],
    feature_names: List[str],
    unique_strings: Dict[str, str],
    record_field_hashes: Dict[str, Dict[str, str]],
    record_id1: str,
    record_id2: str
) -> Tuple[List[float], List[str]]:
    """
    Enhance an existing feature vector with advanced features.
    
    Args:
        base_features: Base feature vector
        feature_names: Names of base features
        unique_strings: Dictionary of hash → string value
        record_field_hashes: Dictionary of record ID → field → hash
        record_id1: First record ID
        record_id2: Second record ID
        
    Returns:
        Tuple of (enhanced_features, enhanced_feature_names)
    """
    # Convert base features to dictionary
    base_features_dict = {name: value for name, value in zip(feature_names, base_features)}
    
    # Get field hashes for both records
    fields1 = record_field_hashes.get(record_id1, {})
    fields2 = record_field_hashes.get(record_id2, {})
    
    # Get person strings for life dates extraction
    person_hash1 = fields1.get('person', "NULL")
    person_hash2 = fields2.get('person', "NULL")
    
    person_str1 = unique_strings.get(person_hash1, "")
    person_str2 = unique_strings.get(person_hash2, "")
    
    # Get provision strings for temporal analysis
    prov_hash1 = fields1.get('provision', "NULL")
    prov_hash2 = fields2.get('provision', "NULL")
    
    prov_str1 = unique_strings.get(prov_hash1, "")
    prov_str2 = unique_strings.get(prov_hash2, "")
    
    # Get roles strings if available
    roles_hash1 = fields1.get('roles', "NULL")
    roles_hash2 = fields2.get('roles', "NULL")
    
    roles_str1 = unique_strings.get(roles_hash1, "")
    roles_str2 = unique_strings.get(roles_hash2, "")
    
    # Use enhanced date processing if available
    if ENHANCED_DATE_PROCESSING_AVAILABLE:
        logger.debug("Using enhanced date processing")
        
        # Extract enhanced life dates
        life_dates1 = extract_life_dates_enhanced(person_str1)
        life_dates2 = extract_life_dates_enhanced(person_str2)
        
        # Extract dates from provision strings
        provision_dates1 = extract_dates_from_provision(prov_str1)
        provision_dates2 = extract_dates_from_provision(prov_str2)
        
        # Generate integrated features using enhanced date processing
        enhanced_date_features = integrate_enhanced_dates_with_base_features(
            base_features_dict,
            person_str1,
            person_str2,
            prov_str1,
            prov_str2,
            roles_str1,
            roles_str2
        )
        
        # Update base features with enhanced date features
        base_features_dict.update(enhanced_date_features)
        
        # Add asymmetry handling for life dates
        asymmetry_features = handle_life_date_asymmetry(
            life_dates1, 
            life_dates2,
            provision_dates1,
            provision_dates2
        )
        base_features_dict.update(asymmetry_features)
        
        # If both lack life dates, use enhanced provision date comparison
        if not (life_dates1[0] or life_dates1[1] or life_dates1[2]) and not (life_dates2[0] or life_dates2[1] or life_dates2[2]):
            provision_features = compare_provision_dates_without_life_dates(
                provision_dates1,
                provision_dates2
            )
            base_features_dict.update(provision_features)
        
    else:
        # Fall back to basic date processing
        logger.debug("Using basic date processing")
        
        # Extract basic life dates
        birth_year1, death_year1 = extract_life_dates(person_str1)
        birth_year2, death_year2 = extract_life_dates(person_str2)
        
        # Extract years from provision strings
        from utils import extract_years
        years1 = extract_years(prov_str1)
        years2 = extract_years(prov_str2)
        
        # Generate robust temporal features with basic approach
        temporal_features = analyze_robust_temporal_features(
            list(years1), list(years2), birth_year1, death_year1, birth_year2, death_year2
        )
        
        # Update base features with basic temporal features
        base_features_dict.update(temporal_features)
        
        # Add basic life dates features
        has_life_dates1 = birth_year1 is not None or death_year1 is not None
        has_life_dates2 = birth_year2 is not None or death_year2 is not None
        base_features_dict['has_life_dates_either'] = 1.0 if (has_life_dates1 or has_life_dates2) else 0.0
        base_features_dict['has_life_dates_both'] = 1.0 if (has_life_dates1 and has_life_dates2) else 0.0
        
        # Life dates exact match features
        if birth_year1 and birth_year2 and birth_year1 == birth_year2:
            if death_year1 and death_year2 and death_year1 == death_year2:
                base_features_dict['exact_life_dates_match'] = 1.0
            else:
                base_features_dict['exact_birth_only_match'] = 1.0
        else:
            base_features_dict['exact_life_dates_match'] = 0.0
            base_features_dict['exact_birth_only_match'] = 0.0
            
        # Add basic asymmetry features
        asymmetric = has_life_dates1 != has_life_dates2
        base_features_dict['has_asymmetric_life_dates'] = 1.0 if asymmetric else 0.0
        
        if asymmetric:
            # Basic compatibility check for asymmetric dates
            if has_life_dates1:
                if years2 and death_year1:
                    # Check if works published after death
                    posthumous = [y > death_year1 for y in years2]
                    if all(posthumous) and death_year1 < 1900:
                        # Historical figure pattern
                        base_features_dict['asymmetry_compatibility'] = 0.7
                    elif any(posthumous) and any(not p for p in posthumous):
                        # Mix of posthumous and lifetime works
                        base_features_dict['asymmetry_compatibility'] = 0.6
                    else:
                        base_features_dict['asymmetry_compatibility'] = 0.5
                else:
                    base_features_dict['asymmetry_compatibility'] = 0.5
            else:
                if years1 and death_year2:
                    # Check if works published after death
                    posthumous = [y > death_year2 for y in years1]
                    if all(posthumous) and death_year2 < 1900:
                        # Historical figure pattern
                        base_features_dict['asymmetry_compatibility'] = 0.7
                    elif any(posthumous) and any(not p for p in posthumous):
                        # Mix of posthumous and lifetime works
                        base_features_dict['asymmetry_compatibility'] = 0.6
                    else:
                        base_features_dict['asymmetry_compatibility'] = 0.5
                else:
                    base_features_dict['asymmetry_compatibility'] = 0.5
        else:
            base_features_dict['asymmetry_compatibility'] = 0.5
    
    # Generate interaction features
    interaction_features = generate_interaction_features(base_features_dict)
    
    # Combine all features
    enhanced_features_dict = {**base_features_dict, **interaction_features}
    
    # Convert back to list format
    enhanced_feature_names = list(enhanced_features_dict.keys())
    enhanced_feature_values = [enhanced_features_dict[name] for name in enhanced_feature_names]
    
    return enhanced_feature_values, enhanced_feature_names
    #enhanced_feature_names = [get_enhanced_feature_description(name) for name in all_feature_names]
    #return final_features, enhanced_feature_names
