"""
Counterfactual Scenario Analysis for Phase 4 (v0.6)
"What if" analysis: shows how input changes affect predictions.
Interactive decision boundary exploration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class ScenarioResult:
    """Result of a counterfactual scenario."""
    scenario_name: str
    original_score: float
    new_score: float
    score_change: float
    percentage_change: float
    prediction_flipped: bool
    required_changes: Dict[str, float]
    feasibility: str  # "easy", "moderate", "difficult"


class CounterfactualScenarioAnalyzer:
    """
    Interactive "what if" analysis for medical risk predictions.
    Explores decision boundaries and sensitivity to input changes.
    """
    
    def __init__(self, model_func):
        """
        Args:
            model_func: Function that takes features and returns risk score (0-100)
        """
        self.model_func = model_func
        self.scenario_history = []
    
    def _compute_sensitivity(self, features: np.ndarray, feature_idx: int, delta: float = 1.0) -> float:
        """
        Compute sensitivity: how much does score change per unit feature change?
        
        Args:
            features: Input features
            feature_idx: Which feature to perturb
            delta: Perturbation size
        
        Returns:
            Sensitivity (score change per unit input change)
        """
        
        original_score = self.model_func(features)
        
        perturbed = np.copy(features)
        perturbed[feature_idx] += delta
        new_score = self.model_func(perturbed)
        
        sensitivity = (new_score - original_score) / (delta + 1e-8)
        return float(sensitivity)
    
    def analyze_symptom_change(
        self,
        features: np.ndarray,
        symptom_idx: int,
        change_amount: float
    ) -> ScenarioResult:
        """
        Scenario: What if a symptom severity changed?
        
        Args:
            features: Original features
            symptom_idx: Which symptom (feature index)
            change_amount: How much to change (positive = worse, negative = better)
        
        Returns:
            ScenarioResult with prediction impact
        """
        
        original_score = self.model_func(features)
        
        scenario_features = np.copy(features)
        scenario_features[symptom_idx] = np.clip(
            scenario_features[symptom_idx] + change_amount,
            0, 100  # Assume 0-100 scale
        )
        new_score = self.model_func(scenario_features)
        
        score_change = new_score - original_score
        percentage_change = (score_change / (original_score + 1e-8)) * 100
        prediction_flipped = (original_score > 50) != (new_score > 50)
        
        # Feasibility based on magnitude of change
        abs_change = abs(change_amount)
        if abs_change < 10:
            feasibility = "easy"
        elif abs_change < 25:
            feasibility = "moderate"
        else:
            feasibility = "difficult"
        
        return ScenarioResult(
            scenario_name=f"Symptom {symptom_idx} change {change_amount:+.0f}",
            original_score=float(original_score),
            new_score=float(new_score),
            score_change=float(score_change),
            percentage_change=float(percentage_change),
            prediction_flipped=prediction_flipped,
            required_changes={f"symptom_{symptom_idx}": float(change_amount)},
            feasibility=feasibility,
        )
    
    def find_decision_boundary(
        self,
        features: np.ndarray,
        target_score: float = 50.0,
        tolerance: float = 1.0,
        max_iterations: int = 10
    ) -> Dict:
        """
        Find minimal changes to cross decision boundary (50 = high/low threshold).
        
        Args:
            features: Original features
            target_score: Target to reach (default: 50 = decision boundary)
            tolerance: Acceptable error
            max_iterations: Max optimization steps
        
        Returns:
            Dict with boundary and required changes
        """
        
        original_score = self.model_func(features)
        current_features = np.copy(features)
        
        for iteration in range(max_iterations):
            current_score = self.model_func(current_features)
            
            if abs(current_score - target_score) < tolerance:
                break
            
            # Gradient-free optimization: test each feature
            best_feature = 0
            best_delta = 0
            best_improvement = float('inf')
            
            for i in range(len(current_features)):
                # Test positive and negative deltas
                for delta in [1.0, -1.0, 5.0, -5.0]:
                    test_features = np.copy(current_features)
                    test_features[i] = np.clip(test_features[i] + delta, 0, 100)
                    
                    test_score = self.model_func(test_features)
                    error = abs(test_score - target_score)
                    
                    if error < best_improvement:
                        best_improvement = error
                        best_feature = i
                        best_delta = delta
            
            # Apply best move
            current_features[best_feature] = np.clip(
                current_features[best_feature] + best_delta,
                0, 100
            )
        
        final_score = self.model_func(current_features)
        feature_changes = {i: (float(features[i]), float(current_features[i])) 
                          for i in range(len(features))}
        
        return {
            'original_score': float(original_score),
            'boundary_score': float(final_score),
            'target_score': float(target_score),
            'iterations': iteration + 1,
            'feature_changes': {f"feature_{i}": {
                'original': v[0],
                'new': v[1],
                'change': v[1] - v[0]
            } for i, v in feature_changes.items() if v[0] != v[1]},
            'prediction_flipped': (original_score > 50) != (final_score > 50),
        }
    
    def create_recovery_plan(
        self,
        features: np.ndarray,
        target_risk_reduction: float = 25.0
    ) -> Dict:
        """
        Generate "recovery plan" showing how to improve health indicators.
        
        Args:
            features: Current health indicators (risk scores)
            target_risk_reduction: Desired risk score reduction (points)
        
        Returns:
            Dict with recommendations and expected outcomes
        """
        
        original_score = self.model_func(features)
        target_score = max(0, original_score - target_risk_reduction)
        
        # Find which features have highest sensitivity (most impactful)
        sensitivities = []
        for i in range(len(features)):
            sens = abs(self._compute_sensitivity(features, i))
            sensitivities.append((i, sens))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations for top features
        recommendations = []
        recovery_features = np.copy(features)
        
        for feature_idx, sensitivity in sensitivities[:3]:
            if original_score <= target_score:
                break
            
            # How much should this feature change?
            needed_reduction = original_score - target_score
            feature_improvement = needed_reduction / (sensitivity + 1e-8)
            
            # Cap improvement at reasonable bounds
            max_improvement = min(20, features[feature_idx] * 0.5)
            feature_improvement = min(max_improvement, abs(feature_improvement))
            
            recovery_features[feature_idx] = max(0, features[feature_idx] - feature_improvement)
            
            recommendations.append({
                'feature': f"health_indicator_{feature_idx}",
                'current_value': float(features[feature_idx]),
                'target_value': float(recovery_features[feature_idx]),
                'improvement': float(feature_improvement),
                'impact_per_unit': float(sensitivity),
                'priority': 'high' if sensitivity > np.median([s for _, s in sensitivities]) else 'medium',
            })
        
        recovery_score = self.model_func(recovery_features)
        
        return {
            'current_risk_score': float(original_score),
            'target_risk_score': float(target_score),
            'achievable_score': float(recovery_score),
            'recommendations': recommendations,
            'estimated_improvement': float(original_score - recovery_score),
            'feasible': recovery_score <= target_score,
        }
    
    def compare_scenarios(
        self,
        features: np.ndarray,
        scenarios: List[Dict]
    ) -> List[ScenarioResult]:
        """
        Compare multiple "what if" scenarios.
        
        Args:
            features: Original features
            scenarios: List of dicts {feature_idx: change_amount, ...}
        
        Returns:
            List of ScenarioResults
        """
        
        results = []
        
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_features = np.copy(features)
            
            for feature_idx, change_amount in scenario.items():
                scenario_features[feature_idx] += change_amount
            
            original_score = self.model_func(features)
            new_score = self.model_func(scenario_features)
            
            result = ScenarioResult(
                scenario_name=f"Scenario {scenario_idx + 1}",
                original_score=float(original_score),
                new_score=float(new_score),
                score_change=float(new_score - original_score),
                percentage_change=float(((new_score - original_score) / original_score) * 100),
                prediction_flipped=(original_score > 50) != (new_score > 50),
                required_changes=scenario,
                feasibility="moderate",
            )
            results.append(result)
        
        return results
    
    def sensitivity_analysis(
        self,
        features: np.ndarray,
        perturbation_range: float = 10.0
    ) -> Dict:
        """
        Analyze sensitivity of prediction to each input feature.
        Shows "importance" from a perturbation perspective.
        
        Args:
            features: Original features
            perturbation_range: How much to perturb each feature
        
        Returns:
            Dict with sensitivity for each feature
        """
        
        sensitivities = {}
        original_score = self.model_func(features)
        
        for i in range(len(features)):
            # Perturbation: increase by range
            perturbed_up = np.copy(features)
            perturbed_up[i] = np.clip(perturbed_up[i] + perturbation_range, 0, 100)
            
            perturbed_down = np.copy(features)
            perturbed_down[i] = np.clip(perturbed_down[i] - perturbation_range, 0, 100)
            
            score_up = self.model_func(perturbed_up)
            score_down = self.model_func(perturbed_down)
            
            # Average absolute effect
            sensitivity = abs(score_up - original_score) + abs(original_score - score_down)
            sensitivity = sensitivity / 2.0
            
            sensitivities[f"feature_{i}"] = {
                'sensitivity': float(sensitivity),
                'effect_on_increase': float(score_up - original_score),
                'effect_on_decrease': float(score_down - original_score),
            }
        
        # Rank by sensitivity
        ranked = sorted(
            sensitivities.items(),
            key=lambda x: x[1]['sensitivity'],
            reverse=True
        )
        
        return {
            'sensitivities': sensitivities,
            'ranked_features': [(name, data['sensitivity']) for name, data in ranked[:5]],
            'total_model_sensitivity': sum(s['sensitivity'] for s in sensitivities.values()),
        }


def create_counterfactual_analysis(
    features: np.ndarray,
    model_func=None,
    analysis_type: str = "recovery"
) -> Dict:
    """
    Factory function for counterfactual analysis.
    
    Args:
        features: Input features
        model_func: Model prediction function
        analysis_type: "recovery" | "boundary" | "sensitivity"
    
    Returns:
        Counterfactual analysis results
    """
    
    if model_func is None:
        # Dummy model if not provided
        model_func = lambda x: float(np.mean(x))
    
    analyzer = CounterfactualScenarioAnalyzer(model_func)
    
    if analysis_type == "recovery":
        return analyzer.create_recovery_plan(features)
    elif analysis_type == "boundary":
        return analyzer.find_decision_boundary(features)
    elif analysis_type == "sensitivity":
        return analyzer.sensitivity_analysis(features)
    else:
        return {}
