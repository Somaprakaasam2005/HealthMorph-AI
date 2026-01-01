"""
Continuous Learning Module for Phase 3 (v0.5)
Online model adaptation based on user feedback and real-world performance.
Detects concept drift and manages model versioning.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path


@dataclass
class UserFeedback:
    """Feedback on model prediction for continuous learning."""
    prediction_id: str
    image_hash: str
    model_version: str
    original_prediction: str
    original_confidence: float
    actual_diagnosis: str  # Ground truth from clinician
    feedback_type: str  # "correct", "incorrect", "partially_correct"
    severity_correction: Optional[float] = None  # If risk score was wrong
    notes: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelVersion:
    """Metadata for a model version."""
    version_id: str
    version_name: str
    creation_date: str
    parent_version: Optional[str]
    accuracy_on_feedback: float = 0.0
    num_samples_trained: int = 0
    is_active: bool = False
    drift_detected: bool = False
    notes: str = ""


class FeedbackCollector:
    """
    Collects and stores user feedback for model improvement.
    Enables clinicians to correct predictions and provide ground truth.
    """
    
    def __init__(self, feedback_dir: str = "feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.feedback_log = self.feedback_dir / "feedback_log.jsonl"
        self.summary_file = self.feedback_dir / "summary.json"
    
    def submit_feedback(self, feedback: UserFeedback) -> bool:
        """Record feedback from clinician."""
        
        try:
            # Add timestamp if not present
            if not feedback.timestamp:
                feedback.timestamp = datetime.now().isoformat()
            
            # Append to log
            with open(self.feedback_log, 'a') as f:
                f.write(json.dumps(feedback.to_dict()) + '\n')
            
            return True
        
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict:
        """Get summary statistics of feedback."""
        
        if not self.feedback_log.exists():
            return {
                "total_feedback": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0.0,
            }
        
        feedbacks = []
        with open(self.feedback_log, 'r') as f:
            for line in f:
                feedbacks.append(json.loads(line))
        
        if not feedbacks:
            return {
                "total_feedback": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0.0,
            }
        
        total = len(feedbacks)
        correct = sum(1 for fb in feedbacks if fb['feedback_type'] == 'correct')
        incorrect = sum(1 for fb in feedbacks if fb['feedback_type'] == 'incorrect')
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": accuracy,
            "first_feedback": feedbacks[0]['timestamp'] if feedbacks else None,
            "last_feedback": feedbacks[-1]['timestamp'] if feedbacks else None,
        }
    
    def get_feedback_for_retraining(
        self,
        since_model_version: Optional[str] = None,
        min_feedback_size: int = 10
    ) -> List[UserFeedback]:
        """
        Get feedback suitable for retraining.
        
        Args:
            since_model_version: only include feedback after this version
            min_feedback_size: require at least this many feedback entries
        
        Returns:
            List of UserFeedback objects ready for model update
        """
        
        if not self.feedback_log.exists():
            return []
        
        feedbacks = []
        with open(self.feedback_log, 'r') as f:
            for line in f:
                fb_dict = json.loads(line)
                feedbacks.append(UserFeedback(**fb_dict))
        
        # Filter by version if specified
        if since_model_version:
            # Get feedback only after this version was created
            feedbacks = [
                fb for fb in feedbacks
                if fb.model_version >= since_model_version
            ]
        
        # Require minimum feedback size
        if len(feedbacks) < min_feedback_size:
            print(f"Insufficient feedback: {len(feedbacks)} < {min_feedback_size}")
            return []
        
        return feedbacks


class ConceptDriftDetector:
    """
    Detects concept drift in model performance.
    Alerts when the data distribution changes significantly.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: number of recent predictions to track
        """
        self.window_size = window_size
        self.prediction_history = []
    
    def add_prediction(self, prediction: str, confidence: float, is_correct: bool):
        """Record prediction outcome."""
        
        self.prediction_history.append({
            'prediction': prediction,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat(),
        })
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.window_size:
            self.prediction_history = self.prediction_history[-self.window_size:]
    
    def detect_drift(self, threshold: float = 0.2) -> Tuple[bool, Dict]:
        """
        Detect if model accuracy is degrading (concept drift).
        
        Args:
            threshold: if accuracy drop > threshold, flag as drift
        
        Returns:
            (drift_detected, metrics)
        """
        
        if len(self.prediction_history) < 2:
            return False, {}
        
        # Split into two halves
        mid = len(self.prediction_history) // 2
        
        early = self.prediction_history[:mid]
        recent = self.prediction_history[mid:]
        
        # Calculate accuracy in each half
        early_acc = sum(1 for p in early if p['is_correct']) / len(early) if early else 1.0
        recent_acc = sum(1 for p in recent if p['is_correct']) / len(recent) if recent else 1.0
        
        # Calculate average confidence
        early_conf = np.mean([p['confidence'] for p in early])
        recent_conf = np.mean([p['confidence'] for p in recent])
        
        # Drift metrics
        accuracy_drop = early_acc - recent_acc
        confidence_change = recent_conf - early_conf
        
        drift_detected = accuracy_drop > threshold
        
        return drift_detected, {
            'early_accuracy': early_acc,
            'recent_accuracy': recent_acc,
            'accuracy_drop': accuracy_drop,
            'early_confidence': early_conf,
            'recent_confidence': recent_conf,
            'confidence_change': confidence_change,
        }


class ModelVersionManager:
    """Manages model versions and lineage."""
    
    def __init__(self, versions_dir: str = "model_versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        
        self.versions_file = self.versions_dir / "versions.jsonl"
    
    def create_version(
        self,
        version_name: str,
        parent_version: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """Create new model version."""
        
        # Generate version ID
        timestamp = datetime.now().isoformat()
        version_id = hashlib.md5(f"{version_name}{timestamp}".encode()).hexdigest()[:8]
        
        version = ModelVersion(
            version_id=version_id,
            version_name=version_name,
            creation_date=timestamp,
            parent_version=parent_version,
            notes=notes,
        )
        
        # Save to log
        with open(self.versions_file, 'a') as f:
            f.write(json.dumps(asdict(version)) + '\n')
        
        return version_id
    
    def list_versions(self) -> List[ModelVersion]:
        """List all model versions."""
        
        if not self.versions_file.exists():
            return []
        
        versions = []
        with open(self.versions_file, 'r') as f:
            for line in f:
                versions.append(ModelVersion(**json.loads(line)))
        
        return versions
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific version metadata."""
        
        for version in self.list_versions():
            if version.version_id == version_id:
                return version
        
        return None
    
    def set_active_version(self, version_id: str) -> bool:
        """Mark version as active."""
        
        versions = self.list_versions()
        updated = False
        
        for v in versions:
            if v.version_id == version_id:
                v.is_active = True
                updated = True
            else:
                v.is_active = False
        
        # Rewrite file
        with open(self.versions_file, 'w') as f:
            for v in versions:
                f.write(json.dumps(asdict(v)) + '\n')
        
        return updated
    
    def update_version_accuracy(self, version_id: str, accuracy: float):
        """Update version's accuracy on validation set."""
        
        versions = self.list_versions()
        
        for v in versions:
            if v.version_id == version_id:
                v.accuracy_on_feedback = accuracy
        
        # Rewrite file
        with open(self.versions_file, 'w') as f:
            for v in versions:
                f.write(json.dumps(asdict(v)) + '\n')


class RetrainingPipeline:
    """Orchestrates periodic retraining based on accumulated feedback."""
    
    def __init__(
        self,
        feedback_dir: str = "feedback",
        versions_dir: str = "model_versions",
        min_feedback_for_retrain: int = 50,
    ):
        
        self.feedback_collector = FeedbackCollector(feedback_dir)
        self.version_manager = ModelVersionManager(versions_dir)
        self.drift_detector = ConceptDriftDetector()
        
        self.min_feedback_for_retrain = min_feedback_for_retrain
        self.last_retrain_date = None
        self.retraining_log = []
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained.
        
        Returns:
            (should_retrain, reason)
        """
        
        stats = self.feedback_collector.get_feedback_stats()
        
        if stats['total_feedback'] < self.min_feedback_for_retrain:
            return False, f"Insufficient feedback: {stats['total_feedback']} < {self.min_feedback_for_retrain}"
        
        # Check for concept drift
        drift_detected, drift_metrics = self.drift_detector.detect_drift()
        
        if drift_detected:
            return True, f"Concept drift detected: {drift_metrics['accuracy_drop']:.2%} accuracy drop"
        
        # Check for accuracy degradation
        if stats['accuracy'] < 0.80:  # Below 80%
            return True, f"Low accuracy: {stats['accuracy']:.2%}"
        
        return False, "Model performing well, no retrain needed"
    
    def prepare_retraining_data(self) -> Dict:
        """Prepare data for retraining from feedback."""
        
        # Get current active version
        versions = self.version_manager.list_versions()
        active_version = next((v.version_id for v in versions if v.is_active), None)
        
        # Get feedback since last version
        feedbacks = self.feedback_collector.get_feedback_for_retraining(
            since_model_version=active_version,
            min_feedback_size=self.min_feedback_for_retrain,
        )
        
        if not feedbacks:
            return {}
        
        # Organize feedback by class
        feedback_by_class = {}
        
        for fb in feedbacks:
            diagnosis = fb.actual_diagnosis
            
            if diagnosis not in feedback_by_class:
                feedback_by_class[diagnosis] = []
            
            feedback_by_class[diagnosis].append(fb)
        
        return {
            "num_samples": len(feedbacks),
            "num_classes": len(feedback_by_class),
            "class_distribution": {cls: len(fbs) for cls, fbs in feedback_by_class.items()},
            "feedbacks": feedbacks,
        }
    
    def log_retraining(self, version_id: str, metrics: Dict, success: bool):
        """Log retraining event."""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'version_id': version_id,
            'metrics': metrics,
            'success': success,
        }
        
        self.retraining_log.append(log_entry)
        self.last_retrain_date = datetime.now()


def create_feedback(
    prediction_id: str,
    image_hash: str,
    model_version: str,
    original_prediction: str,
    original_confidence: float,
    actual_diagnosis: str,
    feedback_type: str = "correct",
) -> UserFeedback:
    """Factory function to create feedback."""
    
    return UserFeedback(
        prediction_id=prediction_id,
        image_hash=image_hash,
        model_version=model_version,
        original_prediction=original_prediction,
        original_confidence=original_confidence,
        actual_diagnosis=actual_diagnosis,
        feedback_type=feedback_type,
        timestamp=datetime.now().isoformat(),
    )
