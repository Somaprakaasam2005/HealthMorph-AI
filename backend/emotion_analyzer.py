"""
Emotional & Behavioral Indicators for Phase 2 (v0.4)
Detects affect and emotional states based on facial features.
Uses valence (positive/negative) and arousal (calm/excited) dimensions.
"""

import cv2
import numpy as np
from typing import Dict


def detect_emotional_indicators(image: np.ndarray) -> Dict:
    """
    Detect emotional state indicators from facial expression.
    
    Returns valence (positivity) and arousal (activation) dimensions.
    
    Args:
        image: BGR numpy array
    
    Returns:
        dict with:
        - valence: -100 (very negative) to +100 (very positive)
        - arousal: 0 (calm/sleepy) to 100 (excited/anxious)
        - dominant_emotion: str (joy, anger, fear, sadness, disgust, surprise, neutral)
        - emotional_stress: 0-100 (overall stress/anxiety level)
        - behavioral_state: str (normal, distressed, fatigued, hyperactive)
    """
    
    if image is None or image.size == 0:
        return _empty_emotion_result()
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Extract key facial regions
        
        # 1. MOUTH REGION (smile detection)
        mouth_region = gray[2*h//3:h, w//4:3*w//4]
        mouth_corners_intensity = np.mean(cv2.Canny(mouth_region, 50, 150))
        smile_score = float(mouth_corners_intensity)  # More edges = smile
        
        # 2. EYES REGION (eye openness, sparkle)
        eye_region = gray[h//4:h//3, :]
        eye_openness = 100 - np.var(eye_region) / 5  # More uniform = closed
        eye_sparkle = np.percentile(eye_region, 95) - np.percentile(eye_region, 5)
        
        # 3. BROW REGION (intensity, furrow)
        brow_region = gray[0:h//6, :]
        brow_edges = np.sum(cv2.Canny(brow_region, 50, 150))
        brow_intensity = brow_edges / (brow_region.size / 100)
        
        # 4. FOREHEAD/SKIN SMOOTHNESS (tension)
        forehead = gray[0:h//5, :]
        forehead_texture = np.std(cv2.Laplacian(forehead, cv2.CV_64F))
        skin_tension = forehead_texture * 2  # Wrinkles = tension
        
        # COMPUTE EMOTIONAL DIMENSIONS
        
        # Valence (positivity)
        valence = (
            smile_score * 0.5 +
            eye_sparkle * 0.3 -
            brow_intensity * 0.2
        )
        valence = float(np.clip(valence - 50, -100, 100))  # Range: -100 to +100
        
        # Arousal (activation)
        arousal = (
            eye_openness * 0.4 +
            brow_intensity * 0.3 +
            skin_tension * 0.3
        )
        arousal = float(np.clip(arousal, 0, 100))
        
        # DERIVE DOMINANT EMOTION from valence + arousal
        if valence > 30 and arousal > 50:
            emotion = "joy"
        elif valence < -30 and arousal > 60:
            emotion = "anger"
        elif valence < -40 and arousal < 40:
            emotion = "sadness"
        elif valence > 0 and arousal > 70:
            emotion = "surprise"
        elif valence < -50 and arousal < 30:
            emotion = "disgust"
        elif valence < -30 and arousal > 70:
            emotion = "fear"
        else:
            emotion = "neutral"
        
        # EMOTIONAL STRESS (combination of negative arousal)
        stress = max(0, -valence * 0.5 + arousal * 0.5)
        stress = float(np.clip(stress, 0, 100))
        
        # BEHAVIORAL STATE
        if stress > 70:
            behavioral = "distressed"
        elif arousal < 30 and -valence > 20:
            behavioral = "fatigued"
        elif arousal > 70 and valence > 20:
            behavioral = "hyperactive"
        else:
            behavioral = "normal"
        
        return {
            "valence": round(valence, 2),
            "arousal": round(arousal, 2),
            "dominant_emotion": emotion,
            "emotional_stress": round(stress, 2),
            "behavioral_state": behavioral,
            "smile_intensity": round(smile_score, 2),
            "eye_openness": round(eye_openness, 2),
        }
    
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return _empty_emotion_result()


def _empty_emotion_result() -> Dict:
    return {
        "valence": 0.0,
        "arousal": 0.0,
        "dominant_emotion": "unknown",
        "emotional_stress": 0.0,
        "behavioral_state": "unknown",
        "smile_intensity": 0.0,
        "eye_openness": 0.0,
    }


def emotion_to_risk_score(emotion_analysis: Dict) -> float:
    """
    Convert emotional analysis to health risk score.
    Negative emotions + high stress may indicate mental health concerns or acute illness.
    
    Args:
        emotion_analysis: dict from detect_emotional_indicators()
    
    Returns:
        float: risk score 0-100
    """
    
    score = 0.0
    
    # Emotional stress is a direct risk indicator
    stress = emotion_analysis.get("emotional_stress", 0)
    score += stress * 0.7
    
    # Negative valence indicates distress
    valence = emotion_analysis.get("valence", 0)
    if valence < -30:
        score += 20
    elif valence < -15:
        score += 10
    
    # High arousal with negative valence = anxiety/fear
    arousal = emotion_analysis.get("arousal", 0)
    if arousal > 70 and valence < -20:
        score += 15
    
    # Behavioral state risk contribution
    behavioral = emotion_analysis.get("behavioral_state", "normal")
    if behavioral == "distressed":
        score += 25
    elif behavioral == "fatigued":
        score += 15
    
    return float(np.clip(score, 0, 100))
