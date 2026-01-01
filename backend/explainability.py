"""
Explainability Module for Phase 4 (v0.6)
SHAP, LIME, attention visualization, and counterfactual analysis.
Provides interpretability for neural model predictions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class FeatureImportance:
    """Feature importance from SHAP analysis."""
    feature_name: str
    importance_score: float  # SHAP value
    contribution_direction: str  # "positive" or "negative"
    impact_on_prediction: float  # 0-100


@dataclass
class LocalExplanation:
    """LIME-style local explanation for single prediction."""
    prediction: str
    confidence: float
    local_features: List[Tuple[str, float]]  # (feature, weight)
    decision_boundary: float
    explanation_text: str


@dataclass
class CounterfactualExample:
    """Counterfactual explanation showing decision boundary."""
    original_features: Dict
    changed_features: Dict
    feature_changes: Dict[str, Tuple[float, float]]  # (original, new)
    new_prediction: str
    steps_to_change: int  # How many features need to change


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for feature importance.
    Approximates Shapley values to explain model predictions.
    """
    
    def __init__(self, model_func, background_data: np.ndarray = None):
        """
        Args:
            model_func: Function that takes features and returns predictions
            background_data: Reference data for baseline (e.g., population stats)
        """
        self.model_func = model_func
        self.background_data = background_data or np.zeros((100, 2048))
        self.base_value = self._compute_base_value()
    
    def _compute_base_value(self) -> float:
        """Compute baseline prediction (average over background data)."""
        try:
            if self.background_data.size > 0:
                predictions = [self.model_func(x) for x in self.background_data[:10]]
                return float(np.mean(predictions))
        except:
            pass
        return 0.5
    
    def explain_prediction(self, features: np.ndarray, num_samples: int = 100) -> Dict:
        """
        Compute SHAP values for a single prediction using kernel SHAP.
        
        Args:
            features: Feature vector (e.g., ResNet50 embeddings)
            num_samples: Number of samples for approximation
        
        Returns:
            Dict with SHAP values and importances
        """
        
        if features is None or len(features) == 0:
            return {}
        
        feature_dim = len(features)
        shap_values = np.zeros(feature_dim)
        
        # Simplified SHAP approximation
        # In production, use shap library: import shap; explainer = shap.KernelExplainer(...)
        
        for i in range(min(num_samples, feature_dim)):
            idx = i % feature_dim
            
            # Marginal contribution: effect of this feature
            baseline = np.copy(features)
            baseline[idx] = 0
            
            with_feature = self.model_func(features)
            without_feature = self.model_func(baseline)
            
            shap_values[idx] = (with_feature - without_feature) / num_samples
        
        # Create importance report
        importances = []
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:5]  # Top 5
        
        for idx in sorted_idx:
            importance = FeatureImportance(
                feature_name=f"dimension_{idx}",
                importance_score=float(shap_values[idx]),
                contribution_direction="positive" if shap_values[idx] > 0 else "negative",
                impact_on_prediction=float(np.abs(shap_values[idx]) * 100),
            )
            importances.append(importance)
        
        return {
            'shap_values': shap_values.tolist(),
            'base_value': self.base_value,
            'importances': [asdict(imp) for imp in importances],
            'total_impact': float(np.sum(np.abs(shap_values))),
        }


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations)
    Explains individual predictions with local linear approximations.
    """
    
    def __init__(self, model_func, feature_names: List[str] = None):
        """
        Args:
            model_func: Black-box model function
            feature_names: Names of features for interpretability
        """
        self.model_func = model_func
        self.feature_names = feature_names or [f"feat_{i}" for i in range(10)]
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_samples: int = 100,
        num_features: int = 5
    ) -> LocalExplanation:
        """
        Explain single prediction using LIME.
        
        Args:
            instance: Feature vector to explain
            num_samples: Local samples around instance
            num_features: Number of important features to return
        
        Returns:
            LocalExplanation with interpretable features
        """
        
        original_prediction = self.model_func(instance)
        
        # Generate local perturbations around instance
        perturbed_samples = []
        perturbed_predictions = []
        
        for _ in range(num_samples):
            # Add Gaussian noise to create local neighborhood
            perturbed = instance + np.random.normal(0, 0.1, instance.shape)
            perturbed_samples.append(perturbed)
            perturbed_predictions.append(self.model_func(perturbed))
        
        # Fit local linear model: predict using top features
        # Simplified: use correlation as feature weights
        correlations = []
        for i in range(min(num_features, len(instance))):
            perturbed_feature = [p[i] for p in perturbed_samples]
            corr = float(np.corrcoef(perturbed_feature, perturbed_predictions)[0, 1])
            correlations.append((f"feature_{i}", corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create explanation text
        top_features = correlations[:num_features]
        explanation = "Model prediction based on:\n"
        for feat, weight in top_features:
            direction = "increases" if weight > 0 else "decreases"
            explanation += f"  • {feat} {direction} prediction (weight: {weight:.3f})\n"
        
        return LocalExplanation(
            prediction="high_risk" if original_prediction > 0.5 else "low_risk",
            confidence=float(abs(original_prediction - 0.5) * 2),
            local_features=top_features,
            decision_boundary=0.5,
            explanation_text=explanation,
        )


class AttentionVisualizer:
    """
    Visualizes attention patterns and layer activations.
    Shows which parts of input influenced the prediction.
    """
    
    def __init__(self):
        self.layer_activations = {}
    
    def compute_attention_weights(
        self,
        input_tensor: np.ndarray,
        activations: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Compute attention heatmap from neural network activations.
        
        Args:
            input_tensor: Original input (e.g., image)
            activations: Dict of layer outputs {layer_name: activation}
        
        Returns:
            Dict with attention weights and visualization
        """
        
        self.layer_activations = activations
        
        attention_map = {}
        
        # For each layer, compute attention
        for layer_name, activation in activations.items():
            if activation.size == 0:
                continue
            
            # Global average pooling + softmax for attention
            if len(activation.shape) == 1:
                # Feature vector
                attention = np.abs(activation) / (np.sum(np.abs(activation)) + 1e-8)
            else:
                # Spatial map
                attention = np.mean(activation, axis=0)
                attention = attention / (np.sum(attention) + 1e-8)
            
            attention_map[layer_name] = attention.tolist()
        
        # Aggregate attention across layers
        if attention_map:
            layer_names = list(attention_map.keys())
            all_attentions = [np.array(att) for att in attention_map.values()]
            
            # Average attention across layers
            aggregated = np.mean([a.flatten() for a in all_attentions], axis=0)
            aggregated = aggregated / (np.sum(aggregated) + 1e-8)
        else:
            aggregated = np.ones(10) / 10
        
        return {
            'layer_attention': attention_map,
            'aggregated_attention': aggregated.tolist(),
            'top_activated_features': np.argsort(aggregated)[::-1][:5].tolist(),
        }
    
    def create_saliency_map(self, gradients: np.ndarray) -> np.ndarray:
        """
        Create saliency map showing gradient-based importance.
        
        Args:
            gradients: Input gradients (d(prediction)/d(input))
        
        Returns:
            Saliency map (same shape as input)
        """
        
        # Saliency = absolute value of gradients
        saliency = np.abs(gradients)
        
        # Normalize to 0-1
        saliency = saliency / (np.max(saliency) + 1e-8)
        
        return saliency


class CounterfactualExplainer:
    """
    Generates counterfactual explanations showing decision boundaries.
    "What would need to change for a different prediction?"
    """
    
    def __init__(self, model_func, feature_names: List[str] = None):
        """
        Args:
            model_func: Prediction function
            feature_names: Names of features
        """
        self.model_func = model_func
        self.feature_names = feature_names or [f"feat_{i}" for i in range(10)]
    
    def find_counterfactual(
        self,
        instance: np.ndarray,
        target_prediction: float = 0.5,
        max_changes: int = 3,
        step_size: float = 0.1
    ) -> CounterfactualExample:
        """
        Find minimal changes to flip prediction across decision boundary.
        
        Args:
            instance: Original features
            target_prediction: Desired prediction value
            max_changes: Max number of features to change
            step_size: Size of feature adjustments
        
        Returns:
            CounterfactualExample with minimal perturbation
        """
        
        original_pred = self.model_func(instance)
        counterfactual = np.copy(instance)
        changes_made = 0
        modified_features = {}
        
        # Greedy approach: change most impactful features
        for iteration in range(max_changes):
            current_pred = self.model_func(counterfactual)
            
            # Stop if target reached
            if abs(current_pred - target_prediction) < 0.1:
                break
            
            # Find feature with highest gradient
            best_idx = 0
            best_improvement = 0
            
            for i in range(len(counterfactual)):
                # Test perturbation
                test = np.copy(counterfactual)
                test[i] += step_size if current_pred < target_prediction else -step_size
                
                new_pred = self.model_func(test)
                improvement = abs(new_pred - target_prediction) - abs(current_pred - target_prediction)
                
                if improvement < best_improvement:  # Move toward target
                    best_improvement = improvement
                    best_idx = i
            
            # Apply best change
            old_val = counterfactual[best_idx]
            counterfactual[best_idx] += step_size if current_pred < target_prediction else -step_size
            modified_features[f"feature_{best_idx}"] = (float(old_val), float(counterfactual[best_idx]))
            changes_made += 1
        
        new_pred = self.model_func(counterfactual)
        
        return CounterfactualExample(
            original_features={f"feat_{i}": float(instance[i]) for i in range(min(5, len(instance)))},
            changed_features={f"feat_{i}": float(counterfactual[i]) for i in range(min(5, len(counterfactual)))},
            feature_changes=modified_features,
            new_prediction="high_risk" if new_pred > 0.5 else "low_risk",
            steps_to_change=changes_made,
        )


class ExplainabilityReport:
    """Comprehensive explainability report combining multiple methods."""
    
    def __init__(
        self,
        prediction: float,
        confidence: float,
        features: np.ndarray,
        feature_names: List[str] = None,
    ):
        
        self.prediction = prediction
        self.confidence = confidence
        self.features = features
        self.feature_names = feature_names or [f"dim_{i}" for i in range(len(features))]
        
        self.shap_explanation = {}
        self.lime_explanation = {}
        self.attention_explanation = {}
        self.counterfactual_explanation = {}
    
    def generate_report(self) -> Dict:
        """Generate comprehensive explainability report."""
        
        return {
            'prediction': "high_risk" if self.prediction > 0.5 else "low_risk",
            'confidence': float(self.confidence),
            'shap': self.shap_explanation,
            'lime': self.lime_explanation,
            'attention': self.attention_explanation,
            'counterfactual': self.counterfactual_explanation,
            'summary': self._generate_summary(),
        }
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary of explanation."""
        
        summary = f"""
Model Prediction: {'HIGH RISK' if self.prediction > 0.5 else 'LOW RISK'} ({self.confidence:.1%} confidence)

Key Contributing Factors:
"""
        
        if self.shap_explanation and 'importances' in self.shap_explanation:
            for imp in self.shap_explanation['importances'][:3]:
                summary += f"  • {imp['feature_name']}: {imp['impact_on_prediction']:.1f}% impact ({imp['contribution_direction']})\n"
        
        summary += "\nDecision Boundary Changes Needed:\n"
        if self.counterfactual_explanation:
            summary += f"  • Minimal changes: {self.counterfactual_explanation.get('steps_to_change', '?')}\n"
        
        return summary.strip()


def create_explainability_report(
    prediction: float,
    confidence: float,
    features: np.ndarray,
    model_func=None,
) -> Dict:
    """
    Factory function to create full explainability report.
    
    Args:
        prediction: Model prediction (0-1)
        confidence: Confidence score
        features: Input features
        model_func: Model function for explanation (optional)
    
    Returns:
        Complete explainability report dict
    """
    
    report = ExplainabilityReport(
        prediction, confidence, features
    )
    
    # Add SHAP if model available
    if model_func:
        try:
            shap_explainer = SHAPExplainer(model_func, background_data=None)
            report.shap_explanation = shap_explainer.explain_prediction(features)
        except Exception as e:
            print(f"SHAP explanation error: {e}")
    
    # Add LIME
    try:
        lime_explainer = LIMEExplainer(model_func) if model_func else None
        if lime_explainer:
            lime_result = lime_explainer.explain_instance(features)
            report.lime_explanation = {
                'prediction': lime_result.prediction,
                'confidence': lime_result.confidence,
                'local_features': lime_result.local_features,
                'explanation': lime_result.explanation_text,
            }
    except Exception as e:
        print(f"LIME explanation error: {e}")
    
    return report.generate_report()


# Helper for dataclass serialization
def asdict(obj):
    """Convert dataclass to dict."""
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict as _asdict
        return _asdict(obj)
    return obj.__dict__
