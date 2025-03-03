#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced date processing for library catalog entity resolution.
Provides sophisticated extraction and analysis of life dates and provision dates.
"""

import logging
import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime
import statistics

# Configure logging
logger = logging.getLogger(__name__)


def extract_life_dates_enhanced(person_str: str) -> Tuple[
    Optional[int], Optional[int], Optional[str], float
]:
    """
    Extract birth and death years from a person name string with enhanced pattern recognition.
    Handles multiple formats including "active", "circa", approximate dates, etc.
    
    Args:
        person_str: Person name string
        
    Returns:
        Tuple of (birth_year, death_year, active_period, confidence)
    """
    if not person_str:
        return None, None, None, 0.0
    
    # Initialize variables
    birth_year = None
    death_year = None
    active_period = None
    confidence = 0.0
    
    # ----- Standard Life Date Patterns -----
    
    # Pattern 1: "Name (1856-1920)" or "Name (1856-)"
    date_match = re.search(r'\((\d{4})-(\d{4}|\s*)\)', person_str)
    if date_match:
        birth_str = date_match.group(1).strip()
        death_str = date_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.9
        return birth_year, death_year, active_period, confidence
    
    # Pattern 2: "Name, 1856-1920" (after a comma)
    date_match = re.search(r',\s*(\d{4})-(\d{4}|\s*)', person_str)
    if date_match:
        birth_str = date_match.group(1).strip()
        death_str = date_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.85
        return birth_year, death_year, active_period, confidence
    
    # Pattern 3: "Name, -1920." (death only, with period and comma)
    death_only_match = re.search(r',\s*-(\d{4})\.', person_str)
    if death_only_match:
        death_str = death_only_match.group(1).strip()
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.8
        return birth_year, death_year, active_period, confidence
    
    # Pattern 4: "-1920." (death only, with period)
    death_only_match = re.search(r'-(\d{4})\.', person_str)
    if death_only_match:
        death_str = death_only_match.group(1).strip()
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.75
        return birth_year, death_year, active_period, confidence
    
    # Pattern 5: "-1920" (death only, without period)
    death_only_match = re.search(r'-(\d{4})', person_str)
    if death_only_match:
        death_str = death_only_match.group(1).strip()
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.7
        return birth_year, death_year, active_period, confidence
    
    # ----- Approximate Life Date Patterns -----
    
    # Pattern 6: "1856 or 1857-1920 or 1921" (uncertain years)
    approx_match = re.search(r'(\d{4})\s+or\s+\d{4}-(\d{4})(?:\s+or\s+\d{4})?', person_str)
    if approx_match:
        birth_str = approx_match.group(1).strip()
        death_str = approx_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.6  # Lower confidence due to uncertainty
        return birth_year, death_year, active_period, confidence
    
    # Pattern 7: "approximately 1856-1920" (approximate dates)
    approx_match = re.search(r'approximately\s+(\d{4})-(\d{4}|\s*)', person_str)
    if approx_match:
        birth_str = approx_match.group(1).strip()
        death_str = approx_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.5  # Lower confidence due to approximation
        return birth_year, death_year, active_period, confidence
    
    # Pattern 8: "ca. 1856-1920" (circa dates)
    circa_match = re.search(r'ca\.\s*(\d{4})-(\d{4}|\s*)', person_str)
    if circa_match:
        birth_str = circa_match.group(1).strip()
        death_str = circa_match.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.5  # Lower confidence for circa dates
        return birth_year, death_year, active_period, confidence
    
    # ----- Active Period Patterns -----
    
    # Pattern 9: "active 1856-1920" (active period instead of life dates)
    active_match = re.search(r'active\s+(\d{4})-(\d{4}|\s*)', person_str)
    if active_match:
        start_str = active_match.group(1).strip()
        end_str = active_match.group(2).strip()
        
        start_year = int(start_str) if start_str and start_str.isdigit() else None
        end_year = int(end_str) if end_str and end_str.isdigit() else None
        
        if start_year and end_year:
            active_period = f"{start_year}-{end_year}"
        elif start_year:
            active_period = f"{start_year}-"
        
        confidence = 0.4  # Active period is less reliable for life dates
        return birth_year, death_year, active_period, confidence
    
    # Pattern 10: "active approximately 1856" (approximate active period)
    active_approx_match = re.search(r'active\s+approximately\s+(\d{4})', person_str)
    if active_approx_match:
        year_str = active_approx_match.group(1).strip()
        year = int(year_str) if year_str and year_str.isdigit() else None
        
        if year:
            active_period = f"{year}"
        
        confidence = 0.3  # Approximate active period is less reliable
        return birth_year, death_year, active_period, confidence
    
    # Pattern 11: "fl. 1856" (floruit notation - flourished in)
    floruit_match = re.search(r'fl\.\s+(\d{4})', person_str)
    if floruit_match:
        year_str = floruit_match.group(1).strip()
        year = int(year_str) if year_str and year_str.isdigit() else None
        
        if year:
            active_period = f"{year}"
        
        confidence = 0.3  # Floruit is less reliable for life dates
        return birth_year, death_year, active_period, confidence
    
    # Pattern 12: "active 16th century" (century-level active period)
    century_match = re.search(r'active\s+(\d+)(?:st|nd|rd|th)\s+century', person_str)
    if century_match:
        century_str = century_match.group(1).strip()
        century = int(century_str) if century_str and century_str.isdigit() else None
        
        if century:
            # Convert century to approximate year range
            start_year = (century - 1) * 100 + 1
            end_year = century * 100
            active_period = f"{start_year}-{end_year}"
        
        confidence = 0.2  # Century-level information is very approximate
        return birth_year, death_year, active_period, confidence
    
    # Pattern 13: "b. 1856" (birth only)
    birth_only_match = re.search(r'b\.\s+(\d{4})', person_str)
    if birth_only_match:
        birth_str = birth_only_match.group(1).strip()
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        confidence = 0.7
        return birth_year, death_year, active_period, confidence
    
    # Pattern 14: "d. 1920" (death only)
    death_only_match = re.search(r'd\.\s+(\d{4})', person_str)
    if death_only_match:
        death_str = death_only_match.group(1).strip()
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.7
        return birth_year, death_year, active_period, confidence
    
    # Pattern 15: "1856-1920," (comma after date)
    date_match_comma = re.search(r'(\d{4})-(\d{4}),', person_str)
    if date_match_comma:
        birth_str = date_match_comma.group(1).strip()
        death_str = date_match_comma.group(2).strip()
        
        birth_year = int(birth_str) if birth_str and birth_str.isdigit() else None
        death_year = int(death_str) if death_str and death_str.isdigit() else None
        confidence = 0.85
        return birth_year, death_year, active_period, confidence
    
    # If we didn't find any specific pattern, return None with zero confidence
    return None, None, None, 0.0


def extract_dates_from_provision(provision_str: str) -> Tuple[List[int], List[str], float]:
    """
    Extract years and date ranges from a provision string with enhanced pattern recognition.
    
    Args:
        provision_str: Provision string
        
    Returns:
        Tuple of (specific_years, date_ranges, confidence)
    """
    if not provision_str:
        return [], [], 0.0
    
    specific_years = []
    date_ranges = []
    confidence = 0.0
    
    # ----- Specific Year Patterns -----
    
    # Pattern 1: Basic 4-digit years between 1400-2025
    year_pattern = r'\b(1[4-9]\d\d|20[0-2]\d)\b'
    years = [int(year) for year in re.findall(year_pattern, provision_str)]
    
    # Pattern 2: Years with uncertainty "[1800?]" or "1800?"
    uncertain_years = re.findall(r'\[(\d{4})\?\]|\b(\d{4})\?', provision_str)
    for match in uncertain_years:
        for group in match:
            if group and group.isdigit():
                years.append(int(group))
    
    # Pattern 3: Circa years "ca. 1800" or "circa 1800"
    circa_years = re.findall(r'(?:ca\.|circa)\s*(\d{4})', provision_str)
    years.extend([int(year) for year in circa_years if year.isdigit()])
    
    # Pattern 4: Approximate years "approximately 1800"
    approx_years = re.findall(r'approximately\s*(\d{4})', provision_str)
    years.extend([int(year) for year in approx_years if year.isdigit()])
    
    # Pattern 5: Bracketed editorial years "[1800]"
    bracketed_years = re.findall(r'\[(\d{4})\]', provision_str)
    years.extend([int(year) for year in bracketed_years if year.isdigit()])
    
    # ----- Date Range Patterns -----
    
    # Pattern 6: Simple year range "1800-1850"
    simple_ranges = re.findall(r'(\d{4})-(\d{4})', provision_str)
    for start, end in simple_ranges:
        if start.isdigit() and end.isdigit():
            start_year = int(start)
            end_year = int(end)
            if 1400 <= start_year <= 2025 and 1400 <= end_year <= 2025:
                date_ranges.append(f"{start_year}-{end_year}")
                # Also add individual years to the specific years list
                years.append(start_year)
                years.append(end_year)
    
    # Pattern 7: "between X and Y"
    between_ranges = re.findall(r'between\s+(\d{4})\s+and\s+(\d{4})', provision_str)
    for start, end in between_ranges:
        if start.isdigit() and end.isdigit():
            start_year = int(start)
            end_year = int(end)
            if 1400 <= start_year <= 2025 and 1400 <= end_year <= 2025:
                date_ranges.append(f"{start_year}-{end_year}")
                years.append(start_year)
                years.append(end_year)
    
    # Pattern 8: Century notation "18th century"
    century_matches = re.findall(r'(\d+)(?:st|nd|rd|th)\s+century', provision_str)
    for century in century_matches:
        if century.isdigit():
            century_num = int(century)
            start_year = (century_num - 1) * 100 + 1
            end_year = century_num * 100
            date_ranges.append(f"{start_year}-{end_year}")
            # Add midpoint year to specific years for century references
            years.append(start_year + 50)
    
    # Pattern 9: Decade notation "1850s"
    decade_matches = re.findall(r'(\d{3})0s', provision_str)
    for decade in decade_matches:
        if decade.isdigit():
            decade_start = int(decade + '0')
            if 1400 <= decade_start <= 2020:
                date_ranges.append(f"{decade_start}-{decade_start+9}")
                # Add midpoint year to specific years for decade references
                years.append(decade_start + 5)
    
    # Filter out duplicates and sort
    specific_years = sorted(list(set(years)))
    date_ranges = sorted(list(set(date_ranges)))
    
    # Calculate confidence based on the specificity and number of dates found
    if specific_years:
        # Higher confidence for specific years
        confidence = min(0.9, 0.5 + (len(specific_years) * 0.1))
    elif date_ranges:
        # Medium confidence for date ranges only
        confidence = min(0.7, 0.3 + (len(date_ranges) * 0.1))
    else:
        # No dates found
        confidence = 0.0
    
    return specific_years, date_ranges, confidence


def analyze_temporal_relationship(
    life_dates: Tuple[Optional[int], Optional[int], Optional[str], float],
    provision_dates: Tuple[List[int], List[str], float]
) -> Dict[str, Any]:
    """
    Analyze the temporal relationship between life dates and provision dates.
    
    Args:
        life_dates: Tuple of (birth_year, death_year, active_period, confidence)
        provision_dates: Tuple of (specific_years, date_ranges, confidence)
        
    Returns:
        Dictionary of temporal relationship features
    """
    birth_year, death_year, active_period, life_confidence = life_dates
    specific_years, date_ranges, provision_confidence = provision_dates
    
    features = {}
    
    # Basic confidence in the temporal data
    features['temporal_data_confidence'] = life_confidence * provision_confidence
    
    # If we don't have any dates, return neutral features
    if (not birth_year and not death_year and not active_period) or not specific_years:
        return {
            'temporal_data_confidence': features['temporal_data_confidence'],
            'publication_during_lifetime': 0.5,
            'posthumous_publication': 0.5,
            'historical_republication': 0.5,
            'temporal_plausibility': 0.5,
            'publication_sequence': 0.5,
            'has_active_career_pattern': 0.5,
            'has_posthumous_pattern': 0.5,
            'has_historical_republication_pattern': 0.5,
            'temporal_consistency': 0.5
        }
    
    # Extract active years from active period if available
    active_years = []
    if active_period:
        if '-' in active_period:
            parts = active_period.split('-')
            if len(parts) == 2:
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
        else:
            # Single year active period
            if active_period.isdigit():
                year = int(active_period)
                active_years = [year]
    
    # If we have birth/death years, estimate active years as adult life
    if birth_year and not active_years:
        career_start = birth_year + 18  # Assume career starts around age 18
        
        if death_year:
            career_end = death_year
        else:
            # If no death year, estimate 50 year career
            career_end = career_start + 50
            
        active_years = list(range(career_start, career_end + 1))
    
    # Calculate temporal features
    
    # 1. Publication during lifetime
    if birth_year and death_year:
        lifetime_pubs = [y for y in specific_years if birth_year <= y <= death_year]
        features['publication_during_lifetime'] = len(lifetime_pubs) / len(specific_years) if specific_years else 0.5
    elif active_years:
        # Use active years as proxy for lifetime
        lifetime_pubs = [y for y in specific_years if y in active_years]
        features['publication_during_lifetime'] = len(lifetime_pubs) / len(specific_years) if specific_years else 0.5
    else:
        features['publication_during_lifetime'] = 0.5  # Neutral value
    
    # 2. Posthumous publication
    if death_year:
        posthumous_pubs = [y for y in specific_years if y > death_year]
        features['posthumous_publication'] = len(posthumous_pubs) / len(specific_years) if specific_years else 0.0
        
        # Check for long posthumous gap (historical republication)
        if posthumous_pubs:
            max_gap = max(y - death_year for y in posthumous_pubs)
            # Normalize to range 0-1 with 100 years being 1.0
            features['historical_republication'] = min(1.0, max_gap / 100)
        else:
            features['historical_republication'] = 0.0
    else:
        features['posthumous_publication'] = 0.0
        features['historical_republication'] = 0.0
    
    # 3. Publication before birth (implausible unless person is subject)
    if birth_year:
        prebirth_pubs = [y for y in specific_years if y < birth_year]
        features['prebirth_publication'] = len(prebirth_pubs) / len(specific_years) if specific_years else 0.0
    else:
        features['prebirth_publication'] = 0.0
    
    # 4. Temporal plausibility score
    # Higher score means more plausible temporal relationship
    plausibility = 1.0
    
    # Penalize for excessive posthumous publishing (but don't penalize classics)
    if death_year and features['posthumous_publication'] > 0.8:
        # Check if this looks like a classical author pattern
        # (death year before 1900, publications spread over centuries)
        if death_year < 1900 and features['historical_republication'] > 0.5:
            # This is likely a classical author - don't penalize
            pass
        else:
            # Too many posthumous publications, somewhat suspicious
            plausibility *= 0.8
    
    # Penalize for prebirth publications (unless very old author)
    if features['prebirth_publication'] > 0.0:
        if birth_year and birth_year < 1800:
            # Historical figure - prebirth pubs might be about them rather than by them
            plausibility *= 0.9
        else:
            # Modern person - prebirth pubs are suspicious
            plausibility *= (1.0 - features['prebirth_publication'])
    
    features['temporal_plausibility'] = plausibility
    
    # 5. Publication sequence (coherence of publication timeline)
    if len(specific_years) > 1:
        # Calculate average gap between publications
        sorted_years = sorted(specific_years)
        gaps = [sorted_years[i+1] - sorted_years[i] for i in range(len(sorted_years)-1)]
        
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            # Normalize to range 0-1 with 10 year average gap being 0.5
            features['publication_sequence'] = 1.0 / (1.0 + avg_gap / 10)
        else:
            features['publication_sequence'] = 0.5
    else:
        features['publication_sequence'] = 0.5
    
    # 6. Publication patterns
    
    # Active career pattern - steady publications during expected lifetime
    if active_years and specific_years:
        career_pubs = [y for y in specific_years if y in active_years]
        career_coverage = len(career_pubs) / len(active_years) if active_years else 0
        features['has_active_career_pattern'] = min(1.0, career_coverage * 10)  # Scale up to reward even sparse coverage
    else:
        features['has_active_career_pattern'] = 0.0
    
    # Posthumous pattern - publications continue after death
    if death_year and specific_years:
        # Check for steady posthumous publishing pattern
        posthumous_years = [y for y in specific_years if y > death_year]
        if posthumous_years and len(posthumous_years) > 1:
            # Look for sustained posthumous publishing (multiple years)
            sorted_post = sorted(posthumous_years)
            post_span = sorted_post[-1] - sorted_post[0]
            
            if post_span > 10 and len(posthumous_years) >= 3:
                features['has_posthumous_pattern'] = 0.8
            else:
                features['has_posthumous_pattern'] = 0.5
        else:
            features['has_posthumous_pattern'] = 0.0
    else:
        features['has_posthumous_pattern'] = 0.0
    
    # Historical republication pattern - gap then renewed interest
    if death_year and specific_years:
        sorted_years = sorted(specific_years)
        if sorted_years and sorted_years[-1] > death_year + 100:
            # Last publication is more than 100 years after death
            # Check if there's a gap followed by new publications
            posthumous_years = [y for y in sorted_years if y > death_year]
            if len(posthumous_years) >= 2:
                gaps = [posthumous_years[i+1] - posthumous_years[i] for i in range(len(posthumous_years)-1)]
                max_gap = max(gaps) if gaps else 0
                
                if max_gap > 50:  # Big gap suggests historical republication
                    features['has_historical_republication_pattern'] = 0.9
                else:
                    features['has_historical_republication_pattern'] = 0.3
            else:
                features['has_historical_republication_pattern'] = 0.1
        else:
            features['has_historical_republication_pattern'] = 0.0
    else:
        features['has_historical_republication_pattern'] = 0.0
    
    # 7. Overall temporal consistency
    # Weighted average of other features
    temporal_consistency = (
        features['temporal_plausibility'] * 0.4 +
        features['publication_sequence'] * 0.2 +
        (1.0 - features['prebirth_publication']) * 0.2 +
        features['publication_during_lifetime'] * 0.2
    )
    
    features['temporal_consistency'] = temporal_consistency
    
    return features


def generate_enhanced_temporal_features(
    person_str1: str,
    person_str2: str,
    provision_str1: str,
    provision_str2: str
) -> Dict[str, float]:
    """
    Generate enhanced temporal features by comparing two person records.
    
    Args:
        person_str1: First person string
        person_str2: Second person string
        provision_str1: First provision string
        provision_str2: Second provision string
        
    Returns:
        Dictionary of enhanced temporal features
    """
    # Extract life dates
    life_dates1 = extract_life_dates_enhanced(person_str1)
    life_dates2 = extract_life_dates_enhanced(person_str2)
    
    # Extract provision dates
    provision_dates1 = extract_dates_from_provision(provision_str1)
    provision_dates2 = extract_dates_from_provision(provision_str2)
    
    # Calculate individual temporal features
    temporal_features1 = analyze_temporal_relationship(life_dates1, provision_dates1)
    temporal_features2 = analyze_temporal_relationship(life_dates2, provision_dates2)
    
    # Compare temporal features to derive similarity scores
    features = {}
    
    # 1. Life dates similarity
    birth_year1, death_year1, active_period1, life_conf1 = life_dates1
    birth_year2, death_year2, active_period2, life_conf2 = life_dates2
    
    # Calculate birth year similarity
    if birth_year1 and birth_year2:
        birth_diff = abs(birth_year1 - birth_year2)
        # Scale: 0 years diff = 1.0, 5 years diff = 0.5, 10+ years diff = 0.0
        features['birth_year_similarity'] = max(0.0, 1.0 - (birth_diff / 10))
    else:
        features['birth_year_similarity'] = 0.5  # Neutral value
    
    # Calculate death year similarity
    if death_year1 and death_year2:
        death_diff = abs(death_year1 - death_year2)
        # Scale: 0 years diff = 1.0, 5 years diff = 0.5, 10+ years diff = 0.0
        features['death_year_similarity'] = max(0.0, 1.0 - (death_diff / 10))
    else:
        features['death_year_similarity'] = 0.5  # Neutral value
    
    # Calculate active period similarity
    if active_period1 and active_period2:
        # Simple string equality for now
        # Could be enhanced to compare the actual year ranges
        features['active_period_similarity'] = 1.0 if active_period1 == active_period2 else 0.0
    else:
        features['active_period_similarity'] = 0.5  # Neutral value
    
    # 2. Provision years overlap
    specific_years1, date_ranges1, prov_conf1 = provision_dates1
    specific_years2, date_ranges2, prov_conf2 = provision_dates2
    
    # Calculate year overlap (Jaccard similarity)
    if specific_years1 and specific_years2:
        set1 = set(specific_years1)
        set2 = set(specific_years2)
        
        overlap = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        features['provision_year_overlap'] = overlap / union if union > 0 else 0.0
    else:
        features['provision_year_overlap'] = 0.5  # Neutral value
    
    # Calculate date range overlap
    if date_ranges1 and date_ranges2:
        # Count how many ranges overlap
        overlap_count = 0
        
        for range1 in date_ranges1:
            for range2 in date_ranges2:
                # Parse ranges
                if '-' in range1 and '-' in range2:
                    r1_start, r1_end = map(int, range1.split('-'))
                    r2_start, r2_end = map(int, range2.split('-'))
                    
                    # Check for overlap
                    if (r1_start <= r2_end and r2_start <= r1_end):
                        overlap_count += 1
        
        total_ranges = len(date_ranges1) + len(date_ranges2)
        features['date_range_overlap'] = (2 * overlap_count) / total_ranges if total_ranges > 0 else 0.0
    else:
        features['date_range_overlap'] = 0.5  # Neutral value
    
    # 3. Temporal pattern similarity
    # Compare the temporal features from each record
    for key in ['publication_during_lifetime', 'posthumous_publication', 
                'historical_republication', 'temporal_plausibility',
                'publication_sequence']:
        if key in temporal_features1 and key in temporal_features2:
            similarity = 1.0 - abs(temporal_features1[key] - temporal_features2[key])
            features[f'{key}_similarity'] = similarity
    
    # 4. Pattern consistency checks
    # High score = both records show same pattern
    # Low score = records show contradictory patterns
    
    # Check if both have similar active career patterns
    if ('has_active_career_pattern' in temporal_features1 and 
        'has_active_career_pattern' in temporal_features2):
        pattern1 = temporal_features1['has_active_career_pattern'] > 0.5
        pattern2 = temporal_features2['has_active_career_pattern'] > 0.5
        features['active_career_pattern_consistency'] = 1.0 if pattern1 == pattern2 else 0.0
    else:
        features['active_career_pattern_consistency'] = 0.5
    
    # Check if both have similar posthumous publishing patterns
    if ('has_posthumous_pattern' in temporal_features1 and 
        'has_posthumous_pattern' in temporal_features2):
        pattern1 = temporal_features1['has_posthumous_pattern'] > 0.5
        pattern2 = temporal_features2['has_posthumous_pattern'] > 0.5
        features['posthumous_pattern_consistency'] = 1.0 if pattern1 == pattern2 else 0.0
    else:
        features['posthumous_pattern_consistency'] = 0.5
    
    # Check if both have similar historical republication patterns
    if ('has_historical_republication_pattern' in temporal_features1 and 
        'has_historical_republication_pattern' in temporal_features2):
        pattern1 = temporal_features1['has_historical_republication_pattern'] > 0.5
        pattern2 = temporal_features2['has_historical_republication_pattern'] > 0.5
        features['historical_republication_consistency'] = 1.0 if pattern1 == pattern2 else 0.0
    else:
        features['historical_republication_consistency'] = 0.5
    
    # 5. Calculate overall temporal compatibility score
    # Weighted sum of features
    
    features['temporal_compatibility'] = (
        (features.get('birth_year_similarity', 0.5) * 0.2) +
        (features.get('death_year_similarity', 0.5) * 0.2) +
        (features.get('provision_year_overlap', 0.5) * 0.2) +
        (features.get('temporal_plausibility_similarity', 0.5) * 0.2) +
        (features.get('active_career_pattern_consistency', 0.5) * 0.1) +
        (features.get('posthumous_pattern_consistency', 0.5) * 0.05) +
        (features.get('historical_republication_consistency', 0.5) * 0.05)
    )
    
    return features


def detect_publication_roles(
    roles_str: str
) -> Dict[str, float]:
    """
    Detect and categorize a person's roles in relation to the publication.
    This helps distinguish between authors, subjects, editors, etc.
    
    Args:
        roles_str: String containing role information
        
    Returns:
        Dictionary of role probabilities
    """
    roles = {}
    
    # Lowercase and normalize
    if not roles_str:
        return {
            'is_author': 0.5,
            'is_subject': 0.0,
            'is_editor': 0.0,
            'is_translator': 0.0,
            'is_contributor': 0.5
        }
    
    roles_lower = roles_str.lower()
    
    # Check for author role
    author_patterns = ['author', 'written by', 'composed by', 'created by']
    author_score = sum(1.0 for p in author_patterns if p in roles_lower)
    roles['is_author'] = min(1.0, author_score / 2)
    
    # Check for subject role
    subject_patterns = ['subject', 'about', 'biography of', 'life of', 'study of']
    subject_score = sum(1.0 for p in subject_patterns if p in roles_lower)
    roles['is_subject'] = min(1.0, subject_score / 2)
    
    # Check for editor role
    editor_patterns = ['editor', 'edited by', 'compiled by']
    editor_score = sum(1.0 for p in editor_patterns if p in roles_lower)
    roles['is_editor'] = min(1.0, editor_score / 2)
    
    # Check for translator role
    translator_patterns = ['translator', 'translated by', 'translation by']
    translator_score = sum(1.0 for p in translator_patterns if p in roles_lower)
    roles['is_translator'] = min(1.0, translator_score / 2)
    
    # Generic contributor
    if 'contributor' in roles_lower:
        roles['is_contributor'] = 1.0
    else:
        roles['is_contributor'] = 0.5
    
    # If no specific roles detected, default to generic contributor
    if all(v == 0.0 for k, v in roles.items() if k != 'is_contributor'):
        roles['is_contributor'] = 0.8
    
    return roles


def is_historical_figure_pattern(
    life_dates: Tuple[Optional[int], Optional[int], Optional[str], float],
    provision_dates: Tuple[List[int], List[str], float]
) -> float:
    """
    Detect if the pattern matches a historical figure (e.g., classical author, 
    historical person) with publications spread across centuries.
    
    Args:
        life_dates: Tuple of (birth_year, death_year, active_period, confidence)
        provision_dates: Tuple of (specific_years, date_ranges, confidence)
        
    Returns:
        Probability of being a historical figure pattern (0.0-1.0)
    """
    birth_year, death_year, active_period, life_confidence = life_dates
    specific_years, date_ranges, provision_confidence = provision_dates
    
    # If we don't have enough information, return low probability
    if not specific_years:
        return 0.1
    
    # Check if person lived/died before 1900
    historical_period = False
    
    if birth_year and birth_year < 1800:
        historical_period = True
    elif death_year and death_year < 1900:
        historical_period = True
    elif active_period and '-' in active_period:
        try:
            active_start = int(active_period.split('-')[0])
            if active_start < 1800:
                historical_period = True
        except:
            pass
    
    if not historical_period:
        return 0.1  # Not a historical figure pattern
    
    # Check for publications spanning multiple centuries
    if specific_years:
        min_year = min(specific_years)
        max_year = max(specific_years)
        
        # Publications span more than 100 years
        if max_year - min_year > 100:
            return 0.9
        
        # Death/active period was centuries ago, but still being published
        if (death_year and max_year > death_year + 200) or \
           (active_period and '-' in active_period and 
            max_year > int(active_period.split('-')[1]) + 200):
            return 0.8
    
    # Check for modern publications of very old works
    current_year = 2025
    if specific_years and any(y > current_year - 50 for y in specific_years):
        # Recent publications
        if death_year and death_year < 1800:
            # Very old author with recent publications
            return 0.9
        elif active_period and '-' in active_period:
            try:
                active_end = int(active_period.split('-')[1])
                if active_end < 1800:
                    # Very old active period with recent publications
                    return 0.8
            except:
                pass
    
    # Default - moderate probability if historical but not strong pattern
    return 0.4 if historical_period else 0.1


def integrate_enhanced_dates_with_base_features(
    base_features: Dict[str, float],
    person_str1: str,
    person_str2: str,
    provision_str1: str,
    provision_str2: str,
    roles_str1: str = "",
    roles_str2: str = ""
) -> Dict[str, float]:
    """
    Integrate enhanced date processing features with the base feature set.
    
    Args:
        base_features: Dictionary of base features
        person_str1: First person string
        person_str2: Second person string
        provision_str1: First provision string
        provision_str2: Second provision string
        roles_str1: First roles string (optional)
        roles_str2: Second roles string (optional)
        
    Returns:
        Dictionary of combined features
    """
    # Generate enhanced temporal features
    temporal_features = generate_enhanced_temporal_features(
        person_str1, 
        person_str2, 
        provision_str1, 
        provision_str2
    )
    
    # Parse life dates
    life_dates1 = extract_life_dates_enhanced(person_str1)
    life_dates2 = extract_life_dates_enhanced(person_str2)
    
    # Parse provision dates
    provision_dates1 = extract_dates_from_provision(provision_str1)
    provision_dates2 = extract_dates_from_provision(provision_str2)
    
    # Detect roles
    roles1 = detect_publication_roles(roles_str1)
    roles2 = detect_publication_roles(roles_str2)
    
    # Check for historical figure pattern
    historical_pattern1 = is_historical_figure_pattern(life_dates1, provision_dates1)
    historical_pattern2 = is_historical_figure_pattern(life_dates2, provision_dates2)
    historical_pattern_match = 1.0 if (historical_pattern1 > 0.5 and historical_pattern2 > 0.5) else 0.0
    
    # Combine roles
    role_consistency = {}
    for role_key in ['is_author', 'is_subject', 'is_editor', 'is_translator', 'is_contributor']:
        if role_key in roles1 and role_key in roles2:
            # High score when roles match, low when they contradict
            role_consistency[f'{role_key}_consistency'] = 1.0 - abs(roles1[role_key] - roles2[role_key])
    
    # Setup integration result, starting with base features
    integrated_features = base_features.copy()
    
    # Add temporal features
    integrated_features.update(temporal_features)
    
    # Add role consistency features
    integrated_features.update(role_consistency)
    
    # Add historical pattern features
    integrated_features['historical_figure_pattern1'] = historical_pattern1
    integrated_features['historical_figure_pattern2'] = historical_pattern2
    integrated_features['historical_pattern_match'] = historical_pattern_match
    
    # Add confidence features
    _, _, _, life_conf1 = life_dates1
    _, _, _, life_conf2 = life_dates2
    _, _, prov_conf1 = provision_dates1
    _, _, prov_conf2 = provision_dates2
    
    integrated_features['life_dates_confidence'] = (life_conf1 + life_conf2) / 2
    integrated_features['provision_dates_confidence'] = (prov_conf1 + prov_conf2) / 2
    
    # Calculate overall temporal compatibility score (already in temporal_features)
    # This can be used directly for entity resolution decisions
    
    return integrated_features


if __name__ == "__main__":
    # Test the module
    person1 = "Shakespeare, William, 1564-1616"
    person2 = "Shakespeare, William (1564-1616)"
    
    provision1 = "1800-1850, 19th century."
    provision2 = "circa 1920, between 1900 and 1950."
    
    print("Testing enhanced date processing...")
    
    life_dates1 = extract_life_dates_enhanced(person1)
    print(f"Life dates 1: {life_dates1}")
    
    provision_dates1 = extract_dates_from_provision(provision1)
    print(f"Provision dates 1: {provision_dates1}")
    
    temporal_features = generate_enhanced_temporal_features(person1, person2, provision1, provision2)
    print(f"Temporal features: {temporal_features}")
    
    print("Tests completed successfully.")
