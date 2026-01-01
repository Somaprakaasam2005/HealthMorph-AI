"""
Syndrome & Genetic Disorder Phenotype Matching for Phase 2 (v0.4)
Matches facial features against known genetic syndrome patterns.
Includes: Down syndrome, Marfan syndrome, Turner syndrome, FXS, Williams, etc.
"""

import numpy as np
from typing import Dict, List


# Syndrome phenotype databases (facial feature thresholds)
SYNDROME_PROFILES = {
    "down_syndrome": {
        "display_name": "Down Syndrome (Trisomy 21)",
        "features": {
            "eye_distance_wide": True,  # Increased intercanthal distance
            "eye_slant_upward": True,   # Upward palpebral fissures
            "nose_small": True,         # Small/flat nasal bridge
            "mouth_open": True,         # Open mouth posture
            "face_round": True,         # Round/flat facial contour
            "skin_loose": True,         # Loose skin folds
        },
        "sensitivity": 0.65,
    },
    "marfan_syndrome": {
        "display_name": "Marfan Syndrome",
        "features": {
            "face_long": True,          # Dolichocephalic (long) face
            "jaw_prominent": True,      # Prominent jaw
            "eye_distance_wide": True,  # Widely spaced eyes
            "palate_high": True,        # High-arched palate
            "teeth_crowded": True,      # Dental crowding
        },
        "sensitivity": 0.60,
    },
    "turner_syndrome": {
        "display_name": "Turner Syndrome (45,X)",
        "features": {
            "neck_short": True,         # Short/webbed neck
            "face_wide": True,          # Broad face
            "jaw_small": True,          # Small/underdeveloped jaw
            "ear_position_low": True,   # Low-set ears
            "mouth_small": True,        # Small mouth
        },
        "sensitivity": 0.58,
    },
    "williams_syndrome": {
        "display_name": "Williams Syndrome",
        "features": {
            "eyes_large": True,         # Large, prominent eyes
            "iris_blue": True,          # Blue iris with stellate pattern
            "nose_small": True,         # Small upturned nose
            "lips_full": True,          # Full lips
            "face_elfin": True,         # Elfin/pixie facies
            "voice_hoarse": True,       # Characteristic hoarse voice
        },
        "sensitivity": 0.62,
    },
    "fragile_x_syndrome": {
        "display_name": "Fragile X Syndrome (FXS)",
        "features": {
            "face_long": True,          # Long face
            "jaw_prominent": True,      # Prominent mandible
            "ear_large": True,          # Large ears
            "forehead_prominent": True, # Prominent forehead
            "postauricular_fullness": True,  # Fullness behind ears
        },
        "sensitivity": 0.60,
    },
    "fetal_alcohol_syndrome": {
        "display_name": "Fetal Alcohol Spectrum Disorder",
        "features": {
            "palpebral_short": True,    # Short palpebral fissures
            "nose_small": True,         # Small upturned nose
            "philtrum_smooth": True,    # Smooth philtrum
            "lips_thin": True,          # Thin upper lip
            "face_small": True,         # Microcephaly/small face
        },
        "sensitivity": 0.61,
    },
    "noonan_syndrome": {
        "display_name": "Noonan Syndrome",
        "features": {
            "face_triangular": True,    # Triangular face shape
            "forehead_prominent": True, # Prominent forehead
            "eyes_wide_spaced": True,   # Wide-set eyes with ptosis
            "jaw_small": True,          # Small jaw
            "ears_low": True,           # Low-set ears
        },
        "sensitivity": 0.59,
    },
    "treacher_collins_syndrome": {
        "display_name": "Treacher Collins Syndrome",
        "features": {
            "cheekbone_hypoplastic": True,  # Underdeveloped cheekbones
            "jaw_underdeveloped": True,     # Micrognathia
            "ear_malformed": True,          # Malformed/missing ears
            "cleft_palate_risk": True,      # Increased cleft palate risk
            "eye_downslant": True,          # Downward-slanting eyes
        },
        "sensitivity": 0.57,
    },
}


def match_syndrome_phenotype(
    facial_features: Dict,
    voice_features: Dict = None,
    symptom_keywords: str = None,
) -> List[Dict]:
    """
    Match observed facial features against syndrome phenotype patterns.
    
    Args:
        facial_features: dict with measurements (eye_distance, jaw_size, face_shape, etc.)
        voice_features: optional dict with voice characteristics
        symptom_keywords: optional symptom text for cross-reference
    
    Returns:
        List of dicts with:
        - syndrome: name (str)
        - display_name: human-readable name
        - match_score: 0-100 (confidence of match)
        - present_features: list of matching features
        - differential_diagnoses: list of other similar syndromes
    """
    
    matches = []
    
    for syndrome_key, syndrome_data in SYNDROME_PROFILES.items():
        match_score = _compute_phenotype_match(
            facial_features,
            syndrome_data,
            voice_features,
            symptom_keywords
        )
        
        if match_score > 30:  # Only include > 30% confidence
            present_features = _identify_present_features(
                facial_features,
                syndrome_data["features"]
            )
            
            matches.append({
                "syndrome": syndrome_key,
                "display_name": syndrome_data["display_name"],
                "match_score": round(match_score, 2),
                "present_features": present_features,
                "confidence": "high" if match_score > 70 else "medium" if match_score > 50 else "low",
            })
    
    # Sort by match score descending
    matches = sorted(matches, key=lambda x: x["match_score"], reverse=True)
    
    return matches[:5]  # Top 5 matches


def _compute_phenotype_match(
    facial_features: Dict,
    syndrome_profile: Dict,
    voice_features: Dict = None,
    symptoms: str = None,
) -> float:
    """
    Compute percentage match between observed features and syndrome profile.
    Heuristic: simulated based on feature overlap.
    """
    
    if not facial_features:
        return 0.0
    
    # Simple heuristic: count matching features
    syndrome_features = syndrome_profile.get("features", {})
    
    match_count = 0
    total_count = len(syndrome_features)
    
    # This is simplified; in production, use ML/morphometric analysis
    for feature_name, feature_expected in syndrome_features.items():
        if feature_name in facial_features:
            feature_value = facial_features.get(feature_name, 0)
            # If feature is expected and observed, count as match
            if feature_expected and feature_value > 0.5:
                match_count += 1
    
    # Base match percentage
    base_match = (match_count / total_count * 100) if total_count > 0 else 0
    
    # Voice/symptom cross-reference boost
    voice_boost = 0
    if voice_features and voice_features.get("emotion_indicator") == "distressed":
        voice_boost = 5
    
    symptom_boost = 0
    if symptoms:
        keywords = ["delayed development", "intellectual", "genetic", "facial features", "birth defect"]
        for kw in keywords:
            if kw.lower() in (symptoms or "").lower():
                symptom_boost += 5
    
    return float(np.clip(base_match + voice_boost + symptom_boost, 0, 100))


def _identify_present_features(
    facial_features: Dict,
    syndrome_features: Dict,
) -> List[str]:
    """
    Identify which expected features are present.
    """
    
    present = []
    for feature_name, is_expected in syndrome_features.items():
        if is_expected and feature_name in facial_features:
            if facial_features[feature_name] > 0.5:
                # Convert feature name to readable format
                readable = feature_name.replace("_", " ").title()
                present.append(readable)
    
    return present


def syndrome_match_to_risk_score(matches: List[Dict]) -> float:
    """
    Convert syndrome matches to health risk score.
    High-confidence genetic syndrome match may warrant genetic counseling.
    
    Args:
        matches: list from match_syndrome_phenotype()
    
    Returns:
        float: risk score 0-100
    """
    
    if not matches:
        return 0.0
    
    top_match = matches[0] if matches else {}
    match_score = top_match.get("match_score", 0)
    confidence = top_match.get("confidence", "low")
    
    score = match_score * 0.5  # Match score contributes up to 50 points
    
    # Confidence weighting
    if confidence == "high":
        score += 25
    elif confidence == "medium":
        score += 12
    
    return float(np.clip(score, 0, 100))
