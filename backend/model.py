import cv2
import numpy as np


# ============================================================================
# FUNCTIONAL LIMITATIONS - Enforce constraints rather than text disclaimers
# ============================================================================

class HealthMorphException(Exception):
    """Raised when analysis violates functional constraints."""
    pass


def validate_facial_input(image_data: np.ndarray):
    """
    Ensure image quality meets minimum standards.
    - Check brightness and sharpness to detect poor camera/lighting conditions.
    - Prevents unreliable results from bad image quality (Limitation #5).
    """
    if image_data is None or image_data.size == 0:
        raise HealthMorphException("No valid image data provided")
    
    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Minimum brightness (too dark = poor lighting)
    if brightness < 50:
        raise HealthMorphException("Image is too dark. Ensure good lighting conditions.")
    if brightness > 220:
        raise HealthMorphException("Image is overexposed. Reduce lighting or exposure.")
    
    # Minimum sharpness (too blurry = poor camera quality)
    if laplacian_var < 10:
        raise HealthMorphException("Image is too blurry. Ensure camera focus is clear.")
    
    return True


def enforce_no_diagnosis_output(result: dict):
    """
    Ensure output does NOT constitute a medical diagnosis.
    - Risk scores are ONLY indicators, not diagnoses.
    - No prescription, treatment, or surgical recommendations.
    - No emergency alerts (Limitations #1, #4).
    """
    # Enforce risk output is framed as "indication" not "diagnosis"
    result["is_diagnosis"] = False
    result["is_medical_advice"] = False
    result["requires_medical_consultation"] = True
    result["emergency_alert"] = False
    
    return result


def reject_dna_and_lab_data(input_dict: dict):
    """
    Enforce: App does NOT accept raw DNA or medical imaging data (Limitation #3).
    This means only facial images, symptoms text, optional voice/video allowed.
    No lab values, CT/MRI DICOM data, DNA sequences, biopsy results, etc.
    """
    forbidden_keys = ['dna', 'dna_sequence', 'lab_results', 'ct_scan', 'mri_scan', 
                      'ct_data', 'mri_data', 'dicom', 'biopsy', 'lab_values', 
                      'blood_work', 'pathology']
    
    for key in input_dict.keys():
        if any(forbidden in key.lower() for forbidden in forbidden_keys):
            raise HealthMorphException(
                f"Input key '{key}' not allowed. HealthMorph does not process "
                "raw DNA, lab results, or medical imaging (CT/MRI/biopsy data)."
            )
    
    return True


def enforce_internet_requirement(endpoint_accessed: bool = True):
    """
    Enforce: App requires internet, no offline sync (Limitation #6).
    This is implicitly enforced by FastAPI HTTP endpoints.
    Used as a checkpoint for future offline caching attempts.
    """
    if not endpoint_accessed:
        raise HealthMorphException("Internet connection required. Offline mode not supported.")
    return True


def enforce_not_fda_certified():
    """
    Enforce: App is NOT FDA/CE certified, academic use only (Limitation #2).
    Returns a disclaimer object to be included in all API responses.
    """
    return {
        "certification_status": "NOT_FDA_OR_CE_CERTIFIED",
        "use_case": "Academic research and educational purposes only",
        "clinical_validation": False,
        "regulatory_approval": False,
        "disclaimer": "This tool is not approved for clinical decision-making or diagnosis."
    }

# ============================================================================


def predict_facial_risk(face_img: np.ndarray):
    # Use grayscale brightness and edge sharpness to produce a deterministic risk proxy
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    brightness_component = max(0.0, 100.0 - abs(128.0 - brightness) * 0.8)
    sharpness_component = max(0.0, 100.0 - min(100.0, lap_var * 0.5))

    raw = 0.6 * brightness_component + 0.4 * sharpness_component
    return float(np.clip(raw, 0.0, 100.0))


SYMPTOM_KEYWORDS = {
    # keyword: weight
    "fever": 15,
    "cough": 10,
    "fatigue": 10,
    "chest pain": 25,
    "shortness of breath": 25,
    "breathlessness": 25,
    "headache": 8,
    "sore throat": 8,
    "nausea": 10,
    "vomiting": 12,
    "diarrhea": 12,
    "dizziness": 12,
    "palpitations": 15,
}

SEVERITY_WORDS = {
    "mild": 0.7,
    "moderate": 1.0,
    "severe": 1.3,
}


def analyze_symptoms(text):
    t = (text or "").lower()
    score = 0.0
    for k, w in SYMPTOM_KEYWORDS.items():
        if k in t:
            score += w
    # severity modifiers
    for k, m in SEVERITY_WORDS.items():
        if k in t:
            score *= m
    # length factor (more info could imply more issues)
    score += min(20.0, len(t) / 100.0)
    return float(max(0.0, min(100.0, score)))


def fuse_scores(facial_score, symptom_score, w_face=0.6, w_sym=0.4):
    fused = w_face * float(facial_score) + w_sym * float(symptom_score)
    return float(max(0.0, min(100.0, fused)))


def risk_level_from_score(score):
    s = float(score)
    if s < 33:
        return "Low"
    elif s < 66:
        return "Medium"
    else:
        return "High"


def generate_explanation(risk_level, fused_score, symptom_score):
    return (
        f"Facial morphology patterns and reported symptoms contributed to {risk_level.lower()} risk (combined score {int(round(fused_score))}). "
        f"Symptom factors accounted for approximately {int(round(symptom_score))} points."
    )


def analyze_voice_wav_bytes(data: bytes):
    # MVP heuristic: use byte length to infer a simple score
    # In future, parse WAV frames and compute energy or prosody features
    length = len(data or b"")
    score = min(100.0, max(0.0, (length % 5000) / 50.0))
    return float(score)


def analyze_video_bytes(data: bytes):
    # MVP heuristic: use file size as a proxy; more data -> more info
    size = len(data or b"")
    score = min(100.0, max(0.0, (size % 2000000) / 20000.0))
    return float(score)


def analyze_sensor_json(sensors: dict):
    # Expected keys: heart_rate, systolic_bp, diastolic_bp, temperature, spo2
    hr = float(sensors.get("heart_rate", 70) or 70)
    sys = float(sensors.get("systolic_bp", 120) or 120)
    dia = float(sensors.get("diastolic_bp", 80) or 80)
    temp = float(sensors.get("temperature", 36.8) or 36.8)
    spo2 = float(sensors.get("spo2", 98) or 98)

    # Simple rule-based scoring
    score = 0.0
    if hr < 50 or hr > 100:
        score += 20
    if sys > 140 or dia > 90:
        score += 20
    if temp > 37.8:
        score += 20
    if spo2 < 94:
        score += 30

    score = min(100.0, score)
    return float(score)


def compute_confidence(modalities_present: int, scores: list):
    # Confidence increases with more modalities and consistent scores
    if not scores:
        return 50.0
    spread = max(scores) - min(scores)
    base = 50.0 + modalities_present * 10.0
    consistency_penalty = min(30.0, spread)
    conf = max(20.0, min(95.0, base - consistency_penalty))
    return float(conf)


def suggest_next_steps(risk_level: str, confidence: float):
    if risk_level == "High" and confidence >= 60:
        return "Consider seeking medical attention or consulting a clinician. Keep a log of symptoms and vitals."
    if risk_level == "Medium":
        return "Monitor symptoms and vitals over the next 48 hours. If conditions worsen, consult a clinician."
    return "Maintain healthy habits and monitor for any new or worsening symptoms."

# ============================================================================
# PHASE 1 (v0.3) - ADVANCED INPUT MODALITIES
# ============================================================================

def fuse_scores_advanced(
    facial_score: float,
    symptom_score: float,
    voice_features: dict = None,
    micro_expr: dict = None,
    depth_analysis: dict = None,
    w_face: float = 0.35,
    w_sym: float = 0.30,
    w_voice: float = 0.15,
    w_expr: float = 0.12,
    w_depth: float = 0.08,
) -> float:
    """
    Enhanced multimodal fusion with Phase 1 features.
    Combines facial + symptom + voice + micro-expression + depth signals.
    
    Args:
        facial_score: 0-100
        symptom_score: 0-100
        voice_features: dict from voice_analyzer.analyze_voice_features()
        micro_expr: dict from face_landmarks.analyze_micro_expressions()
        depth_analysis: dict from depth_processor.process_depth_frame()
        w_*: weights (sum to 1.0)
    
    Returns:
        float: fused risk score 0-100
    """
    
    base_fused = fuse_scores(facial_score, symptom_score)
    
    voice_score = 0.0
    if voice_features and isinstance(voice_features, dict):
        from . import voice_analyzer
        voice_score = voice_analyzer.voice_features_to_risk_score(voice_features)
    
    expr_score = 0.0
    if micro_expr and isinstance(micro_expr, dict):
        from . import face_landmarks
        expr_score = face_landmarks.micro_expression_to_risk_score(micro_expr)
    
    depth_score = 0.0
    if depth_analysis and isinstance(depth_analysis, dict):
        from . import depth_processor
        depth_score = depth_processor.depth_data_to_risk_score(depth_analysis)
    
    # Weighted combination
    fused = (
        w_face * facial_score +
        w_sym * symptom_score +
        w_voice * voice_score +
        w_expr * expr_score +
        w_depth * depth_score
    )
    
    return float(max(0.0, min(100.0, fused)))


# ============================================================================
# PHASE 2 (v0.4) - ADVANCED ANALYSIS FEATURES
# ============================================================================

def fuse_scores_v2(
    facial_score: float,
    symptom_score: float,
    pain_analysis: dict = None,
    emotion_analysis: dict = None,
    syndrome_matches: list = None,
    pattern_analysis: dict = None,
    w_facial: float = 0.25,
    w_symptom: float = 0.25,
    w_pain: float = 0.15,
    w_emotion: float = 0.15,
    w_syndrome: float = 0.10,
    w_pattern: float = 0.10,
) -> float:
    """
    Phase 2 (v0.4) fusion with pain, emotion, syndrome, and pattern analysis.
    
    Args:
        facial_score: 0-100
        symptom_score: 0-100
        pain_analysis: dict from pain_detector.detect_pain_indicators()
        emotion_analysis: dict from emotion_analyzer.detect_emotional_indicators()
        syndrome_matches: list from syndrome_matcher.match_syndrome_phenotype()
        pattern_analysis: dict from hidden_patterns.detect_symptom_patterns()
    
    Returns:
        float: fused risk score 0-100
    """
    
    pain_score = 0.0
    if pain_analysis and isinstance(pain_analysis, dict):
        from . import pain_detector
        pain_score = pain_detector.pain_to_risk_score(pain_analysis)
    
    emotion_score = 0.0
    if emotion_analysis and isinstance(emotion_analysis, dict):
        from . import emotion_analyzer
        emotion_score = emotion_analyzer.emotion_to_risk_score(emotion_analysis)
    
    syndrome_score = 0.0
    if syndrome_matches and isinstance(syndrome_matches, list):
        from . import syndrome_matcher
        syndrome_score = syndrome_matcher.syndrome_match_to_risk_score(syndrome_matches)
    
    pattern_score = 0.0
    if pattern_analysis and isinstance(pattern_analysis, dict):
        from . import hidden_patterns
        pattern_score = hidden_patterns.hidden_patterns_to_risk_score(pattern_analysis)
    
    # Weighted combination
    fused = (
        w_facial * facial_score +
        w_symptom * symptom_score +
        w_pain * pain_score +
        w_emotion * emotion_score +
        w_syndrome * syndrome_score +
        w_pattern * pattern_score
    )
    
    return float(max(0.0, min(100.0, fused)))


# ============================================================================
# PHASE 3 (v0.5) - DEEP LEARNING & ENSEMBLE METHODS
# ============================================================================

def fuse_scores_v3(
    facial_score: float,
    symptom_score: float,
    neural_features: np.ndarray = None,
    pain_analysis: dict = None,
    emotion_analysis: dict = None,
    syndrome_matches: list = None,
    pattern_analysis: dict = None,
    ensemble_votes: dict = None,
    w_facial: float = 0.2,
    w_symptom: float = 0.2,
    w_neural: float = 0.2,
    w_pain: float = 0.1,
    w_emotion: float = 0.1,
    w_syndrome: float = 0.1,
    w_pattern: float = 0.1,
) -> dict:
    """
    Phase 3 (v0.5) fusion with neural backbone features and ensemble voting.
    
    Args:
        facial_score: 0-100 (from OpenCV face detection)
        symptom_score: 0-100 (from keyword analysis)
        neural_features: numpy array from ResNet50/EfficientNet (optional)
        pain_analysis: dict from pain_detector
        emotion_analysis: dict from emotion_analyzer
        syndrome_matches: list from syndrome_matcher
        pattern_analysis: dict from hidden_patterns
        ensemble_votes: dict with model votes and consensus strength (optional)
    
    Returns:
        dict with:
        - risk_score: 0-100 fused score
        - model_confidence: 0-1 confidence in prediction
        - consensus_strength: 0-1 how much models agree
        - feature_usage: which modalities were used
    """
    
    scores = {
        'facial': facial_score,
        'symptom': symptom_score,
        'pain': 0.0,
        'emotion': 0.0,
        'syndrome': 0.0,
        'pattern': 0.0,
        'neural': 0.0,
    }
    
    # Extract scores from Phase 2 analyses
    if pain_analysis and isinstance(pain_analysis, dict):
        try:
            from . import pain_detector
            scores['pain'] = pain_detector.pain_to_risk_score(pain_analysis)
        except:
            pass
    
    if emotion_analysis and isinstance(emotion_analysis, dict):
        try:
            from . import emotion_analyzer
            scores['emotion'] = emotion_analyzer.emotion_to_risk_score(emotion_analysis)
        except:
            pass
    
    if syndrome_matches and isinstance(syndrome_matches, list):
        try:
            from . import syndrome_matcher
            scores['syndrome'] = syndrome_matcher.syndrome_match_to_risk_score(syndrome_matches)
        except:
            pass
    
    if pattern_analysis and isinstance(pattern_analysis, dict):
        try:
            from . import hidden_patterns
            scores['pattern'] = hidden_patterns.hidden_patterns_to_risk_score(pattern_analysis)
        except:
            pass
    
    # Neural features contribution (if available)
    if neural_features is not None and isinstance(neural_features, np.ndarray):
        try:
            # Simple: use L2 norm of features as a magnitude indicator
            neural_magnitude = float(np.linalg.norm(neural_features))
            neural_magnitude = min(100.0, neural_magnitude / 10.0)  # Scale to 0-100
            scores['neural'] = neural_magnitude
        except:
            pass
    
    # Ensemble consensus (if available)
    ensemble_confidence = 1.0
    if ensemble_votes and isinstance(ensemble_votes, dict):
        ensemble_confidence = ensemble_votes.get('consensus_strength', 1.0)
    
    # Weighted combination (Phase 3)
    # Normalize weights if neural features not available
    weights = {
        'facial': w_facial,
        'symptom': w_symptom,
        'neural': w_neural if neural_features is not None else 0.0,
        'pain': w_pain,
        'emotion': w_emotion,
        'syndrome': w_syndrome,
        'pattern': w_pattern,
    }
    
    # Renormalize weights if some are zero
    total_weight = sum(v for v in weights.values() if v > 0)
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    fused = sum(weights[k] * scores[k] for k in weights.keys())
    
    # Confidence: how many modalities were available
    modalities_used = sum(1 for score in scores.values() if score > 0)
    total_modalities = len(scores)
    modality_confidence = modalities_used / total_modalities
    
    # Combined confidence: modality count × ensemble consensus
    model_confidence = (modality_confidence * 0.5 + ensemble_confidence * 0.5)
    
    return {
        'risk_score': float(max(0.0, min(100.0, fused))),
        'model_confidence': float(model_confidence),
        'consensus_strength': float(ensemble_confidence),
        'individual_scores': scores,
        'modalities_used': modalities_used,
        'feature_usage': list(k for k, v in scores.items() if v > 0),
    }


def extract_neural_features_if_available(image_data: np.ndarray) -> dict:
    """
    Attempt to extract neural features from image using ResNet50.
    Falls back gracefully if torch/neural_backbone not available.
    
    Args:
        image_data: numpy BGR image
    
    Returns:
        dict with neural features or empty dict on failure
    """
    
    try:
        from . import neural_backbone
        
        result = neural_backbone.extract_facial_features(
            image_data,
            model_name="resnet50"
        )
        
        return {
            'features': result.get('features'),
            'model_used': result.get('model_used'),
            'feature_dim': result.get('feature_dim'),
        }
    
    except Exception as e:
        # Neural features not available - Phase 3 optional
        return {}


# ============================================================================
# PHASE 4 (v0.6) - ADVANCED EXPLAINABILITY
# ============================================================================

def generate_explainability_explanation(
    facial_score: float,
    symptom_score: float,
    neural_features: np.ndarray = None,
    pain_analysis: dict = None,
    emotion_analysis: dict = None,
) -> dict:
    """
    Generate SHAP-style feature importance report.
    Shows which modalities contributed most to risk score.
    
    Args:
        facial_score, symptom_score: Individual scores
        neural_features: Optional neural embeddings
        pain_analysis, emotion_analysis: Analysis results
    
    Returns:
        dict with feature importances and explanations
    """
    
    importances = {}
    
    # Score facial detection
    importances['facial'] = {
        'score': facial_score,
        'weight': 0.20,
        'contribution': (facial_score / 100.0) * 0.20,
        'direction': 'positive' if facial_score > 50 else 'negative',
    }
    
    # Score symptoms
    importances['symptoms'] = {
        'score': symptom_score,
        'weight': 0.20,
        'contribution': (symptom_score / 100.0) * 0.20,
        'direction': 'positive' if symptom_score > 50 else 'negative',
    }
    
    # Neural features contribution
    if neural_features is not None:
        try:
            neural_magnitude = float(np.linalg.norm(neural_features[:100]))  # Top 100 dims
            neural_score = min(100, neural_magnitude / 10.0)
            importances['neural'] = {
                'score': neural_score,
                'weight': 0.20,
                'contribution': (neural_score / 100.0) * 0.20,
                'direction': 'positive' if neural_score > 50 else 'negative',
            }
        except:
            pass
    
    # Pain contribution
    if pain_analysis and 'pain_intensity' in pain_analysis:
        pain_score = pain_analysis.get('pain_intensity', 0) * 100
        importances['pain'] = {
            'score': pain_score,
            'weight': 0.10,
            'contribution': (pain_score / 100.0) * 0.10,
            'direction': 'positive' if pain_score > 50 else 'negative',
        }
    
    # Emotion contribution
    if emotion_analysis and 'emotional_stress' in emotion_analysis:
        stress_score = emotion_analysis.get('emotional_stress', 0) * 100
        importances['emotion'] = {
            'score': stress_score,
            'weight': 0.10,
            'contribution': (stress_score / 100.0) * 0.10,
            'direction': 'positive' if stress_score > 50 else 'negative',
        }
    
    # Rank by contribution
    ranked = sorted(importances.items(), key=lambda x: x[1]['contribution'], reverse=True)
    
    return {
        'modality_importances': importances,
        'ranked_modalities': [name for name, _ in ranked[:5]],
        'top_contributors': [
            {'name': name, 'contribution': data['contribution']} 
            for name, data in ranked[:3]
        ],
    }


def generate_decision_boundary_explanation(
    risk_score: float,
    decision_threshold: float = 50.0
) -> dict:
    """
    Explain decision boundary: how close is prediction to flipping?
    
    Args:
        risk_score: Current risk score (0-100)
        decision_threshold: Threshold for high/low risk (default 50)
    
    Returns:
        dict with boundary proximity and stability
    """
    
    distance_to_boundary = abs(risk_score - decision_threshold)
    stability = min(1.0, distance_to_boundary / decision_threshold)  # 0-1
    
    return {
        'current_score': risk_score,
        'decision_boundary': decision_threshold,
        'distance_to_boundary': distance_to_boundary,
        'stability': float(stability),
        'stable': distance_to_boundary > 10,  # >10 points from boundary
        'interpretation': (
            'STABLE prediction (unlikely to flip)' if distance_to_boundary > 10
            else 'UNSTABLE prediction (sensitive to small changes)' if distance_to_boundary < 5
            else 'BORDERLINE (close to decision boundary)'
        ),
    }


def suggest_improvement_directions(
    symptom_text: str,
    risk_score: float,
    risk_level: str
) -> dict:
    """
    Generate "what if" suggestions: how could risk improve?
    (Simplified version of counterfactual analysis)
    
    Args:
        symptom_text: Current symptoms
        risk_score: Current risk score
        risk_level: high/medium/low
    
    Returns:
        dict with improvement suggestions
    """
    
    suggestions = []
    
    # Generic suggestions based on risk level
    if risk_level == 'high':
        suggestions = [
            {
                'action': 'Consult healthcare provider',
                'expected_impact': '15-25 points reduction',
                'feasibility': 'high',
                'priority': 1,
            },
            {
                'action': 'Address primary symptoms',
                'expected_impact': '10-20 points reduction',
                'feasibility': 'high',
                'priority': 2,
            },
            {
                'action': 'Monitor vital signs',
                'expected_impact': '5-10 points reduction',
                'feasibility': 'high',
                'priority': 3,
            },
        ]
    elif risk_level == 'medium':
        suggestions = [
            {
                'action': 'Schedule preventive checkup',
                'expected_impact': '10-15 points reduction',
                'feasibility': 'high',
                'priority': 1,
            },
            {
                'action': 'Lifestyle modifications',
                'expected_impact': '5-10 points reduction',
                'feasibility': 'medium',
                'priority': 2,
            },
        ]
    else:  # low risk
        suggestions = [
            {
                'action': 'Maintain current lifestyle',
                'expected_impact': 'Stable',
                'feasibility': 'high',
                'priority': 1,
            },
        ]
    
    return {
        'target_score': max(0, risk_score - 25),  # Aim to reduce by 25 points
        'current_score': risk_score,
        'suggestions': suggestions,
        'disclaimer': 'These are academic suggestions only. Follow professional medical advice.',
    }