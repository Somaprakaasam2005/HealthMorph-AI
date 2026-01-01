"""
Facial Micro-Expression Detection for Phase 1 (v0.3)
Uses MediaPipe Facemesh to detect subtle facial movements and muscle contractions.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List

# MediaPipe import with fallback
try:
    import mediapipe as mp
    FaceMesh = mp.solutions.face_mesh.FaceMesh if hasattr(mp.solutions, 'face_mesh') else None
except (ImportError, AttributeError):
    FaceMesh = None


def detect_face_landmarks(image: np.ndarray) -> Tuple[List[Dict], bool]:
    """
    Detect 468 facial landmarks using MediaPipe Facemesh.
    
    Args:
        image: BGR numpy array (from cv2.imread or camera)
    
    Returns:
        (landmarks_list, success) where:
        - landmarks_list: list of dicts with 'x', 'y', 'z' normalized coordinates
        - success: bool indicating if face was detected
    """
    
    if FaceMesh is None:
        print("MediaPipe not available, returning empty landmarks")
        return [], False
    
    try:
        with FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape
            
            results = face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return [], False
            
            # Extract landmarks from first face
            landmarks = results.multi_face_landmarks[0].landmark
            landmarks_list = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in landmarks
            ]
            
            return landmarks_list, True
    
    except Exception as e:
        print(f"Face mesh error: {e}")
        return [], False


def analyze_micro_expressions(landmarks_prev: List[Dict], landmarks_curr: List[Dict]) -> Dict:
    """
    Detect micro-expressions by comparing consecutive face landmark frames.
    
    Args:
        landmarks_prev: previous frame landmarks (468 points)
        landmarks_curr: current frame landmarks (468 points)
    
    Returns:
        dict with:
        - mouth_movement: 0-100 (lip tightness, smile)
        - eye_movement: 0-100 (eye openness, blink speed)
        - brow_movement: 0-100 (eyebrow raise/furrow)
        - cheek_movement: 0-100 (cheek contraction)
        - micro_expression_detected: bool
        - expression_type: str (smile, fear, disgust, surprise, anger, sadness, neutral)
    """
    
    if not landmarks_prev or not landmarks_curr or len(landmarks_prev) != len(landmarks_curr):
        return {
            "mouth_movement": 0.0,
            "eye_movement": 0.0,
            "brow_movement": 0.0,
            "cheek_movement": 0.0,
            "micro_expression_detected": False,
            "expression_type": "unknown",
        }
    
    # Convert to numpy for easier computation
    prev = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks_prev])
    curr = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks_curr])
    
    # Calculate Euclidean distances between corresponding landmarks
    distances = np.linalg.norm(curr - prev, axis=1)
    
    try:
        # Key landmark indices for different facial regions
        # Mouth: 61-100
        mouth_indices = list(range(61, 101))
        mouth_dist = np.mean(distances[mouth_indices]) if mouth_indices else 0.0
        mouth_movement = float(np.clip(mouth_dist * 1000, 0, 100))
        
        # Eyes: 33-133 (left), 362-481 (right)
        eye_indices = list(range(33, 134)) + list(range(362, 382))
        eye_dist = np.mean(distances[eye_indices]) if eye_indices else 0.0
        eye_movement = float(np.clip(eye_dist * 1000, 0, 100))
        
        # Eyebrows: 70-132 (left), 300-330 (right)
        brow_indices = list(range(70, 133)) + list(range(300, 330))
        brow_dist = np.mean(distances[brow_indices]) if brow_indices else 0.0
        brow_movement = float(np.clip(brow_dist * 1000, 0, 100))
        
        # Cheeks: 50-230
        cheek_indices = list(range(50, 230))
        cheek_dist = np.mean(distances[cheek_indices]) if cheek_indices else 0.0
        cheek_movement = float(np.clip(cheek_dist * 1000, 0, 100))
        
        # Detect micro-expression type based on movement patterns
        expression_type = "neutral"
        if mouth_movement > 5 and cheek_movement > 3:
            expression_type = "smile"
        elif brow_movement > 8 and eye_movement > 6:
            expression_type = "surprise"
        elif mouth_movement > 8 and brow_movement > 5:
            expression_type = "anger"
        elif eye_movement > 10 and mouth_movement < 3:
            expression_type = "fear"
        elif mouth_movement > 7 and cheek_movement < 2:
            expression_type = "disgust"
        elif brow_movement > 7 and mouth_movement < 2:
            expression_type = "sadness"
        
        # Micro-expression detected if any major movement
        micro_expr_detected = (mouth_movement > 5 or eye_movement > 5 or 
                               brow_movement > 5 or cheek_movement > 5)
        
        return {
            "mouth_movement": round(mouth_movement, 2),
            "eye_movement": round(eye_movement, 2),
            "brow_movement": round(brow_movement, 2),
            "cheek_movement": round(cheek_movement, 2),
            "micro_expression_detected": micro_expr_detected,
            "expression_type": expression_type,
        }
    
    except Exception as e:
        print(f"Micro-expression analysis error: {e}")
        return {
            "mouth_movement": 0.0,
            "eye_movement": 0.0,
            "brow_movement": 0.0,
            "cheek_movement": 0.0,
            "micro_expression_detected": False,
            "expression_type": "error",
        }


def micro_expression_to_risk_score(micro_expr: Dict) -> float:
    """
    Convert micro-expression analysis to health risk score.
    Heuristic: fear, disgust, anger, or sadness may indicate distress or health concerns.
    
    Args:
        micro_expr: dict from analyze_micro_expressions()
    
    Returns:
        float: risk score 0-100
    """
    
    score = 0.0
    
    # Expression type weighting
    expr_weights = {
        "fear": 25,
        "disgust": 20,
        "anger": 15,
        "sadness": 20,
        "surprise": 5,
        "smile": 0,
        "neutral": 0,
        "unknown": 0,
        "error": 0,
    }
    score += expr_weights.get(micro_expr.get("expression_type", "error"), 0)
    
    # Movement intensity contributions
    mouth = micro_expr.get("mouth_movement", 0)
    eye = micro_expr.get("eye_movement", 0)
    brow = micro_expr.get("brow_movement", 0)
    
    # High mouth movement + low smile = tension
    if mouth > 10 and micro_expr.get("expression_type") not in ["smile", "surprise"]:
        score += 10
    
    # High eye movement = stress/attention
    if eye > 12:
        score += 10
    
    # High brow movement = concern
    if brow > 10:
        score += 8
    
    return float(np.clip(score, 0, 100))
