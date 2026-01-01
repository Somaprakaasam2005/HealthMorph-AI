"""
Model Ensemble for Phase 3 (v0.5)
Combines multiple models via voting and weighted averaging for robust predictions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnsembleVote:
    """Single model prediction for ensemble voting."""
    prediction: str
    confidence: float
    model_name: str
    features: Optional[np.ndarray] = None


@dataclass
class EnsembleResult:
    """Result from ensemble voting."""
    prediction: str
    confidence: float
    consensus_strength: float  # 0-1, how confident the ensemble is
    votes: Dict[str, int]
    model_predictions: List[str]
    method: str


class EnsembleClassifier:
    """
    Combines multiple classifiers via voting.
    Supports:
    - Hard voting (majority)
    - Soft voting (average confidence)
    - Weighted voting (custom weights per model)
    """
    
    def __init__(self, voting_method: str = "soft", weights: Optional[Dict[str, float]] = None):
        """
        Args:
            voting_method: "hard" (majority), "soft" (average), "weighted"
            weights: dict mapping model name -> weight (used for weighted voting)
        """
        self.voting_method = voting_method
        self.weights = weights or {}
        self.model_history = []
    
    def vote_classification(self, votes: List[EnsembleVote]) -> EnsembleResult:
        """
        Combine multiple model votes on classification task.
        
        Args:
            votes: list of EnsembleVote objects from different models
        
        Returns:
            EnsembleResult with consensus prediction
        """
        
        if not votes:
            return EnsembleResult(
                prediction="unknown",
                confidence=0.0,
                consensus_strength=0.0,
                votes={},
                model_predictions=[],
                method=self.voting_method
            )
        
        if self.voting_method == "hard":
            return self._hard_vote(votes)
        elif self.voting_method == "soft":
            return self._soft_vote(votes)
        elif self.voting_method == "weighted":
            return self._weighted_vote(votes)
        else:
            return self._soft_vote(votes)  # Default fallback
    
    def _hard_vote(self, votes: List[EnsembleVote]) -> EnsembleResult:
        """Majority voting."""
        
        predictions = [v.prediction for v in votes]
        vote_counts = {}
        
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        # Winner by majority
        winner = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        winner_votes = vote_counts[winner]
        total_votes = len(votes)
        
        # Consensus strength = how dominant the winner is
        consensus_strength = winner_votes / total_votes
        
        # Find max confidence for winner
        winner_confidences = [v.confidence for v in votes if v.prediction == winner]
        confidence = float(np.mean(winner_confidences)) if winner_confidences else 0.0
        
        return EnsembleResult(
            prediction=winner,
            confidence=confidence,
            consensus_strength=consensus_strength,
            votes=vote_counts,
            model_predictions=predictions,
            method="hard"
        )
    
    def _soft_vote(self, votes: List[EnsembleVote]) -> EnsembleResult:
        """Average confidence across models."""
        
        predictions = [v.prediction for v in votes]
        confidences = [v.confidence for v in votes]
        
        vote_counts = {}
        confidence_sums = {}
        
        for pred, conf in zip(predictions, confidences):
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
            confidence_sums[pred] = confidence_sums.get(pred, 0) + conf
        
        # Winner: class with highest average confidence
        winner = max(confidence_sums.keys(), key=lambda k: confidence_sums[k] / vote_counts[k])
        avg_confidence = confidence_sums[winner] / vote_counts[winner]
        
        # Consensus strength: how much higher is winner vs second-place
        sorted_classes = sorted(
            [(k, confidence_sums[k] / vote_counts[k]) for k in confidence_sums.keys()],
            key=lambda x: x[1],
            reverse=True
        )
        
        consensus_strength = (
            (sorted_classes[0][1] - sorted_classes[1][1]) / sorted_classes[0][1]
            if len(sorted_classes) > 1
            else 1.0
        )
        
        return EnsembleResult(
            prediction=winner,
            confidence=float(avg_confidence),
            consensus_strength=float(np.clip(consensus_strength, 0, 1)),
            votes=vote_counts,
            model_predictions=predictions,
            method="soft"
        )
    
    def _weighted_vote(self, votes: List[EnsembleVote]) -> EnsembleResult:
        """Weighted voting using pre-defined weights."""
        
        predictions = [v.prediction for v in votes]
        
        vote_counts = {}
        weighted_confidence = {}
        
        for vote in votes:
            model_weight = self.weights.get(vote.model_name, 1.0)
            
            vote_counts[vote.prediction] = vote_counts.get(vote.prediction, 0) + model_weight
            weighted_confidence[vote.prediction] = (
                weighted_confidence.get(vote.prediction, 0) + 
                vote.confidence * model_weight
            )
        
        # Normalize weights
        total_weight = sum(self.weights.values()) if self.weights else len(votes)
        
        winner = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        avg_confidence = weighted_confidence[winner] / vote_counts[winner]
        consensus_strength = vote_counts[winner] / total_weight
        
        return EnsembleResult(
            prediction=winner,
            confidence=float(avg_confidence),
            consensus_strength=float(np.clip(consensus_strength, 0, 1)),
            votes=vote_counts,
            model_predictions=predictions,
            method="weighted"
        )
    
    def vote_regression(self, votes: List[Tuple[str, float]]) -> Tuple[float, float]:
        """
        Combine multiple regression predictions (e.g., risk scores 0-1).
        
        Args:
            votes: list of (model_name, score) tuples
        
        Returns:
            (ensemble_score, consensus_strength)
        """
        
        if not votes:
            return 0.5, 0.0
        
        scores = [score for _, score in votes]
        
        # Average score
        ensemble_score = float(np.mean(scores))
        
        # Consensus strength: inverse of standard deviation
        std_dev = float(np.std(scores))
        # Low std_dev = high agreement = high consensus
        consensus_strength = float(np.exp(-std_dev))  # Exponential decay
        
        return ensemble_score, consensus_strength


class RiskScoreEnsemble:
    """Specialized ensemble for risk score regression."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: model name -> weight (e.g., {"neural_model": 0.6, "rule_based": 0.4})
        """
        self.weights = weights or {}
    
    def fuse_scores(self, model_scores: Dict[str, float]) -> Dict:
        """
        Fuse multiple model risk scores with optional weighting.
        
        Args:
            model_scores: {model_name: score (0-1)}
        
        Returns:
            {
                "ensemble_score": float (0-1),
                "score_std": standard deviation of input scores,
                "score_range": (min, max) of input scores,
                "agreement_level": "high" | "medium" | "low",
                "individual_scores": input dict,
            }
        """
        
        scores = list(model_scores.values())
        
        if not scores:
            return {
                "ensemble_score": 0.5,
                "score_std": 0.0,
                "score_range": (0.5, 0.5),
                "agreement_level": "none",
                "individual_scores": {},
            }
        
        # Weighted average
        if self.weights:
            total_weight = 0
            weighted_sum = 0
            for model_name, score in model_scores.items():
                weight = self.weights.get(model_name, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            ensemble_score = weighted_sum / total_weight
        else:
            ensemble_score = float(np.mean(scores))
        
        # Agreement metrics
        std_dev = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        score_range = max_score - min_score
        
        # Agreement classification
        if score_range < 0.1:
            agreement_level = "high"
        elif score_range < 0.25:
            agreement_level = "medium"
        else:
            agreement_level = "low"
        
        return {
            "ensemble_score": float(np.clip(ensemble_score, 0, 1)),
            "score_std": std_dev,
            "score_range": (min_score, max_score),
            "agreement_level": agreement_level,
            "individual_scores": model_scores,
        }


def create_ensemble_vote(model_name: str, prediction: str, confidence: float) -> EnsembleVote:
    """Factory function to create ensemble vote."""
    return EnsembleVote(
        prediction=prediction,
        confidence=float(np.clip(confidence, 0, 1)),
        model_name=model_name,
    )
