"""
Hidden Symptom Pattern Detection for Phase 2 (v0.4)
Detects subtle symptom patterns and anomalies using clustering and statistical analysis.
Identifies correlations user may not have explicitly stated.
"""

import numpy as np
from typing import Dict, List
from collections import Counter


SYMPTOM_CLUSTER_KEYWORDS = {
    "respiratory": ["cough", "breathe", "shortness", "wheeze", "asthma", "bronch"],
    "gastrointestinal": ["nausea", "vomit", "diarrhea", "constipation", "abdom", "gas"],
    "cardiovascular": ["palpitation", "chest", "heart", "pressure", "syncope", "dizzy"],
    "neurological": ["headache", "migraine", "seizure", "tremor", "parkinson", "stroke"],
    "infectious": ["fever", "chill", "flu", "cold", "infection", "sepsis"],
    "musculoskeletal": ["pain", "ache", "stiff", "joint", "arthritis", "fracture"],
    "endocrine": ["diabetes", "thyroid", "hormone", "glucose", "insulin"],
    "psychiatric": ["anxiety", "depression", "stress", "mood", "panic", "ptsd"],
    "dermatological": ["rash", "eczema", "hives", "psoriasis", "lesion", "itching"],
    "ophthalmological": ["vision", "blind", "glaucoma", "cataracts", "diplopia"],
}


def detect_symptom_patterns(symptom_text: str) -> Dict:
    """
    Analyze symptom text for hidden patterns and clusters.
    
    Args:
        symptom_text: raw symptom description
    
    Returns:
        dict with:
        - symptom_clusters: list of detected organ system clusters
        - cluster_strength: dict with confidence for each cluster
        - anomalies: list of unusual symptom combinations
        - comorbidity_risk: 0-100 (likelihood of multiple conditions)
        - symptom_duration_inferred: str (acute, subacute, chronic)
        - urgency_indicator: str (routine, urgent, emergency)
    """
    
    if not symptom_text or not isinstance(symptom_text, str):
        return _empty_pattern_result()
    
    text_lower = symptom_text.lower()
    
    # 1. DETECT SYMPTOM CLUSTERS
    cluster_scores = {}
    for cluster_name, keywords in SYMPTOM_CLUSTER_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in text_lower)
        if match_count > 0:
            cluster_scores[cluster_name] = match_count / len(keywords)
    
    symptom_clusters = list(cluster_scores.keys())
    
    # 2. DETECT ANOMALIES (unusual combinations)
    anomalies = []
    if "respiratory" in cluster_scores and "gastrointestinal" in cluster_scores:
        if cluster_scores["respiratory"] > 0.5 and cluster_scores["gastrointestinal"] > 0.5:
            anomalies.append("Concurrent respiratory and GI symptoms (possible viral infection or sepsis)")
    
    if "cardiovascular" in cluster_scores and "neurological" in cluster_scores:
        anomalies.append("Combined cardiac and neurological symptoms (possible TIA, stroke risk)")
    
    if "psychiatric" in cluster_scores and "infectious" in cluster_scores:
        anomalies.append("Psychiatric symptoms with fever (possible CNS infection)")
    
    # 3. COMORBIDITY RISK (multiple system involvement)
    comorbidity_risk = min(100, len(symptom_clusters) * 20)
    
    # 4. SYMPTOM DURATION INFERENCE
    duration_keywords = {
        "acute": ["sudden", "acute", "onset", "started", "began"],
        "chronic": ["long", "chronic", "persistent", "ongoing", "years"],
        "subacute": ["gradual", "progressive", "worsening", "days", "weeks"],
    }
    
    duration = "unknown"
    for dur_type, keywords in duration_keywords.items():
        if any(kw in text_lower for kw in keywords):
            duration = dur_type
            break
    
    # 5. URGENCY INDICATOR
    urgent_keywords = ["severe", "emergency", "critical", "dying", "can't breathe", "chest pain", "stroke"]
    routine_keywords = ["mild", "slight", "minor", "manageable", "tolerable"]
    
    urgency = "routine"
    if any(kw in text_lower for kw in urgent_keywords):
        urgency = "emergency"
    elif duration == "acute" and len(cluster_scores) > 1:
        urgency = "urgent"
    elif any(kw in text_lower for kw in routine_keywords):
        urgency = "routine"
    else:
        urgency = "standard"
    
    return {
        "symptom_clusters": symptom_clusters,
        "cluster_strength": {k: round(v, 2) for k, v in cluster_scores.items()},
        "anomalies": anomalies,
        "comorbidity_risk": round(comorbidity_risk, 2),
        "symptom_duration_inferred": duration,
        "urgency_indicator": urgency,
        "unique_symptoms_count": len(text_lower.split()),
    }


def _empty_pattern_result() -> Dict:
    return {
        "symptom_clusters": [],
        "cluster_strength": {},
        "anomalies": [],
        "comorbidity_risk": 0.0,
        "symptom_duration_inferred": "unknown",
        "urgency_indicator": "routine",
        "unique_symptoms_count": 0,
    }


def hidden_patterns_to_risk_score(pattern_analysis: Dict) -> float:
    """
    Convert hidden pattern analysis to health risk score.
    Multiple clusters, anomalies, and high comorbidity = higher risk.
    
    Args:
        pattern_analysis: dict from detect_symptom_patterns()
    
    Returns:
        float: risk score 0-100
    """
    
    score = 0.0
    
    # Cluster diversity contribution
    num_clusters = len(pattern_analysis.get("symptom_clusters", []))
    score += num_clusters * 10  # Each additional cluster = +10 points
    
    # Comorbidity risk
    comorbidity = pattern_analysis.get("comorbidity_risk", 0)
    score += comorbidity * 0.5
    
    # Anomaly presence (unusual combinations)
    num_anomalies = len(pattern_analysis.get("anomalies", []))
    score += num_anomalies * 15
    
    # Duration weighting
    duration = pattern_analysis.get("symptom_duration_inferred", "unknown")
    if duration == "acute":
        score += 15
    elif duration == "chronic":
        score += 10
    
    # Urgency weighting
    urgency = pattern_analysis.get("urgency_indicator", "routine")
    if urgency == "emergency":
        score += 40
    elif urgency == "urgent":
        score += 25
    
    return float(np.clip(score, 0, 100))
