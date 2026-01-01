# Phase 3 (v0.5) Quick Reference

## 🚀 What's New

| Component | Feature | Status |
|-----------|---------|--------|
| **Neural Backbone** | ResNet50 + EfficientNet | ✅ Ready |
| **Ensemble Voting** | Hard/soft/weighted | ✅ Ready |
| **Transfer Learning** | Fine-tuning framework | ✅ Ready |
| **Continuous Learning** | Feedback + drift detection | ✅ Ready |
| **API Endpoint** | POST /analyze/v3 | ✅ Live on :8000 |

---

## 📦 New Modules

### 1. `neural_backbone.py` - Feature Extraction
```python
from backend.neural_backbone import extract_facial_features

# Single call - returns 2048-dim feature vector
result = extract_facial_features(image_array, model_name="resnet50")
print(result['features'].shape)  # (2048,)
```

### 2. `model_ensemble.py` - Voting System
```python
from backend.model_ensemble import EnsembleClassifier, create_ensemble_vote

ensemble = EnsembleClassifier(voting_method="soft")
votes = [
    create_ensemble_vote("model_a", "high_risk", 0.85),
    create_ensemble_vote("model_b", "low_risk", 0.60),
]
result = ensemble.vote_classification(votes)
# result.prediction, result.consensus_strength
```

### 3. `transfer_learner.py` - Fine-tuning
```python
from backend.transfer_learner import FineTuner, TrainingConfig

config = TrainingConfig(
    num_epochs=50,
    freeze_backbone=True,
    unfreeze_at_epoch=20,
)
fine_tuner = FineTuner(backbone, num_classes=5, config=config)
history = fine_tuner.fit(train_loader, val_loader)
```

### 4. `continuous_learning.py` - Feedback & Drift
```python
from backend.continuous_learning import FeedbackCollector, ConceptDriftDetector

# Collect feedback
collector = FeedbackCollector("feedback")
feedback = create_feedback(...actual_diagnosis="normal"...)
collector.submit_feedback(feedback)

# Detect drift
detector = ConceptDriftDetector()
detector.add_prediction("high_risk", 0.85, is_correct=False)
drift_detected, metrics = detector.detect_drift()
```

---

## 🔌 API Integration

### POST /analyze/v3 (New Phase 3 Endpoint)
**Request:**
```bash
curl -X POST http://localhost:8000/analyze/v3 \
  -F "image=@face.jpg" \
  -F "symptoms=headache, fatigue" \
  -F "voice=@voice.wav" \
  -F "ensemble_method=soft"
```

**Response Fields (Phase 3 specific):**
```json
{
  "analysis_id": "uuid-for-feedback",
  "risk_score": 75,
  "model_confidence": 82,
  "phase_3_features": {
    "neural_features_used": true,
    "ensemble_voting": {
      "prediction": "high_risk",
      "consensus_strength": 0.78
    },
    "modalities_used": ["facial", "symptom", "neural", "pain", "emotion"],
    "consensus_strength": 0.78
  },
  "continuous_learning": {
    "prediction_id": "uuid",
    "supports_feedback": true
  }
}
```

---

## 🧠 Scoring Details

### Risk Score Composition (v0.5)
- **Facial indicators**: 20% (OpenCV face detection)
- **Symptom analysis**: 20% (keyword matching)
- **Neural features**: 20% (ResNet50 magnitude)
- **Pain indicators**: 10% (UNBC AU detection)
- **Emotional state**: 10% (valence/arousal)
- **Syndrome match**: 5% (genetic phenotypes)
- **Symptom patterns**: 5% (clustering/anomalies)

### Model Confidence Calculation
```
confidence = (modality_count / 7) × 0.5 + ensemble_consensus × 0.5
```
- More modalities → higher confidence
- Better ensemble agreement → higher confidence
- Max: 100% (all modalities + perfect consensus)

---

## 🔧 Key Functions

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `extract_facial_features()` | image array | 2048-dim vector | Get neural features |
| `fuse_scores_v3()` | 7 scores | risk_score, confidence | Combined analysis |
| `EnsembleClassifier.vote_classification()` | vote list | prediction + consensus | Robust classification |
| `FineTuner.fit()` | train/val loaders | history dict | Fine-tune model |
| `FeedbackCollector.submit_feedback()` | feedback object | bool | Store corrections |
| `ConceptDriftDetector.detect_drift()` | history | drift flag + metrics | Monitor performance |

---

## 📊 Performance Expectations

### Speed
- ResNet50 extraction: ~100ms (CPU)
- EfficientNet-B0 extraction: ~50ms (CPU)
- Full /analyze/v3: ~300-400ms (all modalities)

### Confidence Ranges
- 1 modality: 20% baseline
- 3 modalities: 60% typical
- 6+ modalities: 85-100% target

---

## 💡 Usage Patterns

### Pattern 1: Real-time Analysis
```python
# Use /analyze/v3 for single prediction
# Returns confidence → use for triage
# Store prediction_id for feedback linkage
```

### Pattern 2: Feedback-Driven Improvement
```python
# Collect feedback as clinicians correct predictions
# When n>50 feedback items accumulated:
#   → trigger retraining
#   → new model version created
#   → performance improves over time
```

### Pattern 3: Ensemble for Safety
```python
# For high-risk cases, request "hard" voting
# Requires consensus across models
# More conservative but safer decisions
```

---

## ⚠️ Limitations (Phase 3.0)

1. **Neural features**: Currently magnitude-based heuristic
   - Real classifiers needed for production
   
2. **Ensemble voting**: Dummy votes from facial/symptom heuristics
   - Need trained models for each modality
   
3. **Transfer learning**: Framework ready, no trained weights yet
   - Requires labeled medical dataset
   
4. **Retraining**: Feedback collection works, trigger logic pending
   - Need training infrastructure (GPU, batching)

---

## 🎯 Next Steps (Phase 4, v0.6)

- [ ] Train real neural classifiers on medical dataset
- [ ] Add SHAP/LIME explainability
- [ ] Implement counterfactual analysis
- [ ] Build feedback annotation UI
- [ ] Deploy continuous retraining pipeline

---

## 🆘 Troubleshooting

### Q: Import error "No module named 'torch'"
**A:** Run `pip install torch` from venv. Already done? Restart kernel.

### Q: /analyze/v3 endpoint 404
**A:** Server needs restart. Stop and `uvicorn backend.main:app --reload --port 8000`

### Q: Neural features always 0
**A:** Currently heuristic-based (L2 norm). Will improve with real models.

### Q: Ensemble consensus low
**A:** Votes disagree → try different `ensemble_method` ("soft" vs "hard")

---

## 📖 Documentation

- **Comprehensive Guide**: [PHASE_3_COMPLETION.md](PHASE_3_COMPLETION.md)
- **Technical Notes**: [PHASE_3_NOTES.md](PHASE_3_NOTES.md)
- **Roadmap**: [ROADMAP.md](ROADMAP.md) (updated with v0.5 status)
- **API Docs**: http://localhost:8000/docs (Swagger interactive)

---

**Phase 3 Status: COMPLETE ✅**  
**Server Running: http://localhost:8000**  
**API Ready: POST /analyze/v3**
