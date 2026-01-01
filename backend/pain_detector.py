"""
Pain & Distress Detection for Phase 2 (v0.4)
Detects facial pain indicators based on UNBC McMaster Shoulder Pain dataset heuristics.
Analyzes: eye closure, brow lowering, nose wrinkling, lip tightening, jaw clenching.
"""

import cv2
import numpy as np
from typing import Dict


def detect_pain_indicators(image: np.ndarray, landmarks: Dict = None) -> Dict:
    """
    Detect pain/distress indicators from facial expression.
    
    Based on UNBC Pain Database features:
    - Brow lowering (AU 4)
    - Eye closure (AU 43)
    - Nose wrinkling (AU 9)
    - Lip corner pulling (AU 12, 20)
    - Jaw clenching (AU 25)
    
    Args:
        image: BGR numpy array
        landmarks: optional dict of facial landmarks from MediaPipe
    
    Returns:
        dict with:
        - eye_closure: 0-100 (degree of eye tightness)
        - brow_lowering: 0-100 (furrow intensity)
        - nose_wrinkling: 0-100 (nasal creases)
        - lip_tightening: 0-100 (mouth tension)
        - jaw_clenching: 0-100 (masseter contraction)
        - pain_detected: bool
        - pain_intensity: 0-100
    """
    
    if image is None or image.size == 0:
        return _empty_pain_result()
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. EYE CLOSURE (AU 43) - detect closed eyelids
        # Upper face region (eyes)
        eye_region = gray[h//4:h//2, :]
        eye_variance = np.var(eye_region)
        # Low variance = more uniform (closed eyes), high = more open
        eye_closure = float(np.clip(100 - (eye_variance / 100), 0, 100))
        
        # 2. BROW LOWERING (AU 4) - detect furrowed brows
        # Brow region (upper forehead)
        brow_region = gray[0:h//6, :]
        brow_edges = cv2.Canny(brow_region, 50, 150)
        brow_density = np.sum(brow_edges) / brow_edges.size
        brow_lowering = float(brow_density * 100)
        
        # 3. NOSE WRINKLING (AU 9) - detect nasal creases
        # Nose region
        nose_region = gray[h//3:h//2, w//3:2*w//3]
        nose_laplacian = cv2.Laplacian(nose_region, cv2.CV_64F)
        nose_variance = np.var(nose_laplacian)
        nose_wrinkling = float(np.clip(nose_variance / 50, 0, 100))
        
        # 4. LIP TIGHTENING (AU 23, 24) - detect lip corner tension
        # Mouth region
        mouth_region = gray[2*h//3:h, :]
        mouth_edges = cv2.Canny(mouth_region, 50, 150)
        mouth_edge_density = np.sum(mouth_edges) / mouth_edges.size
        lip_tightening = float(mouth_edge_density * 150)
        
        # 5. JAW CLENCHING (AU 25) - detect masseter contraction (jaw closure)
        # Lower face/jaw region
        jaw_region = gray[3*h//5:h, :]
        jaw_variance = np.var(jaw_region)
        jaw_clenching = float(np.clip(100 - (jaw_variance / 50), 0, 100))
        
        # COMPOSITE PAIN INTENSITY
        # Weighted combination of pain indicators
        pain_intensity = (
            0.25 * eye_closure +
            0.25 * brow_lowering +
            0.15 * nose_wrinkling +
            0.20 * lip_tightening +
            0.15 * jaw_clenching
        )
        pain_intensity = float(np.clip(pain_intensity, 0, 100))
        
        # Pain detected if any indicator > threshold
        pain_detected = (eye_closure > 40 or brow_lowering > 35 or 
                        lip_tightening > 40 or jaw_clenching > 45)
        
        return {
            "eye_closure": round(eye_closure, 2),
            "brow_lowering": round(brow_lowering, 2),
            "nose_wrinkling": round(nose_wrinkling, 2),
            "lip_tightening": round(lip_tightening, 2),
            "jaw_clenching": round(jaw_clenching, 2),
            "pain_detected": pain_detected,
            "pain_intensity": round(pain_intensity, 2),
        }
    
    except Exception as e:
        print(f"Pain detection error: {e}")
        return _empty_pain_result()


def _empty_pain_result() -> Dict:
    return {
        "eye_closure": 0.0,
        "brow_lowering": 0.0,
        "nose_wrinkling": 0.0,
        "lip_tightening": 0.0,
        "jaw_clenching": 0.0,
        "pain_detected": False,
        "pain_intensity": 0.0,
    }


def pain_to_risk_score(pain_analysis: Dict) -> float:
    """
    Convert pain analysis to health risk score.
    High pain/distress may indicate acute illness, injury, or serious condition.
    
    Args:
        pain_analysis: dict from detect_pain_indicators()
    
    Returns:
        float: risk score 0-100
    """
    
    intensity = pain_analysis.get("pain_intensity", 0)
    score = intensity * 0.8  # Pain contributes up to 80 points
    
    # Detected pain adds baseline risk
    if pain_analysis.get("pain_detected", False):
        score += 10
    
    return float(np.clip(score, 0, 100))
