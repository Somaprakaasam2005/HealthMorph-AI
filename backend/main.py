from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uuid
import os
import json
import time
import numpy as np
import cv2
import jwt
import time

from . import model
from . import preprocess as pre
from . import explain
from .settings import settings
from .db import SessionLocal, AnalysisRecord, init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="HealthMorph AI", description="Academic MVP for AI-assisted health risk indication", version="0.2.0")

# Allow configurable CORS origins (comma-separated) or wildcard
cors_raw = settings.cors_origins.strip()
if cors_raw == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for heatmaps
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


ANALYSIS_LOG = []  # in-memory log only for demo (no persistence)
_RATE_BUCKET = {"ts": 0.0, "count": 0}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def rate_limit():
    now = time.time()
    window = 60.0
    if now - _RATE_BUCKET["ts"] > window:
        _RATE_BUCKET["ts"] = now
        _RATE_BUCKET["count"] = 0
    _RATE_BUCKET["count"] += 1
    if _RATE_BUCKET["count"] > settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def auth_guard(token: str = None):
    if not settings.auth_required:
        return
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    try:
        jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def _size_guard(file: UploadFile, max_mb: float):
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    if size > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{max_mb}MB)")
    return size


def _parse_features_json(features_json: str) -> np.ndarray:
    """Parse features JSON (list or dict) into numpy array."""
    try:
        payload = json.loads(features_json)
        if isinstance(payload, list):
            arr = np.array(payload, dtype=float)
        elif isinstance(payload, dict):
            # Accept dict of feat_i: value
            items = sorted((k, v) for k, v in payload.items() if isinstance(v, (int, float)))
            arr = np.array([v for _, v in items], dtype=float)
        else:
            raise ValueError("features_json must be list or dict")
        if arr.size == 0:
            raise ValueError("no features provided")
        return arr
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid features_json: {e}")


@app.post("/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    questionnaire: str = Form(None),
    voice: UploadFile = File(None),
    video: UploadFile = File(None),
    sensors: str = Form(None),
    authorization: str = Form(None),
    db=Depends(get_db),
):
    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))
    
    # ENFORCE FUNCTIONAL CONSTRAINTS
    model.enforce_internet_requirement(endpoint_accessed=True)
    
    if image is None or image.filename == "":
        raise HTTPException(status_code=400, detail="Image file is required.")
    if symptoms is None or symptoms.strip() == "":
        raise HTTPException(status_code=400, detail="Symptom text is required.")

    # Reject DNA/lab/medical imaging data upfront
    try:
        model.reject_dna_and_lab_data(request.__dict__)
    except model.HealthMorphException as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Size guards
    _size_guard(image, max_mb=6)
    if voice:
        _size_guard(voice, max_mb=5)
    if video:
        _size_guard(video, max_mb=20)

    # Read image bytes and decode to BGR
    try:
        data = await image.read()
        if not data:
            raise ValueError("Empty image")
        npimg = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image upload: {e}")

    # ENFORCE IMAGE QUALITY CONSTRAINTS (camera/lighting quality check)
    try:
        model.validate_facial_input(bgr)
    except model.HealthMorphException as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Face preprocessing and facial risk
    face_img, face_bbox = pre.detect_and_prepare(bgr)
    facial_score = model.predict_facial_risk(face_img)
    symptom_score = model.analyze_symptoms(questionnaire or symptoms)

    voice_score = 0.0
    if voice:
        vbytes = await voice.read()
        voice_score = model.analyze_voice_wav_bytes(vbytes)

    video_score = 0.0
    if video:
        vidbytes = await video.read()
        video_score = model.analyze_video_bytes(vidbytes)

    sensor_score = 0.0
    sensor_payload = {}
    if sensors:
        try:
            sensor_payload = json.loads(sensors)
            sensor_score = model.analyze_sensor_json(sensor_payload)
        except Exception:
            sensor_payload = {}

    # Fusion
    # Multimodal fusion (facial + symptoms + optional modalities)
    base_fused = model.fuse_scores(facial_score, symptom_score)
    extra = 0.0
    # Weighted contributions of optional modalities
    extra += 0.2 * voice_score
    extra += 0.3 * video_score
    extra += 0.4 * sensor_score
    fused_score = max(0.0, min(100.0, base_fused * 0.7 + extra * 0.3))
    risk_level = model.risk_level_from_score(fused_score)

    # Explanation
    explanation = model.generate_explanation(risk_level, fused_score, symptom_score)
    confidence = model.compute_confidence(
        modalities_present=int(bool(image)) + int(bool(questionnaire or symptoms)) + int(bool(voice)) + int(bool(video)) + int(bool(sensors)),
        scores=[facial_score, symptom_score, voice_score, video_score, sensor_score],
    )
    next_steps = model.suggest_next_steps(risk_level, confidence)

    # Explainable heatmap (Grad-CAM style overlay)
    try:
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(STATIC_DIR, heatmap_filename)
        explain.generate_heatmap(bgr, face_bbox, heatmap_path)
        heatmap_url = f"/static/{heatmap_filename}"
    except Exception:
        heatmap_url = "/static/heatmap.svg"

    contributions = {
        "facial": round(facial_score, 2),
        "symptoms": round(symptom_score, 2),
        "voice": round(voice_score, 2),
        "video": round(video_score, 2),
        "sensors": round(sensor_score, 2),
    }

    ANALYSIS_LOG.append(
        {
            "ts": time.time(),
            "risk_level": risk_level,
            "risk_score": int(round(fused_score)),
            "confidence": int(round(confidence)),
            "modalities": list(k for k, v in contributions.items() if v > 0),
        }
    )

    if settings.persist_enabled:
        rec = AnalysisRecord(
            risk_level=risk_level,
            risk_score=int(round(fused_score)),
            confidence=int(round(confidence)),
            modalities=",".join([k for k, v in contributions.items() if v > 0]),
        )
        db.add(rec)
        db.commit()

    # ENFORCE: Output framed as non-diagnostic indication, not medical advice
    result = {
        "risk_level": risk_level,
        "risk_score": int(round(fused_score)),
        "explanation": explanation,
        "heatmap_url": heatmap_url,
        "confidence": int(round(confidence)),
        "next_steps": next_steps,
        "sensors": sensor_payload,
        "contributions": contributions,
    }
    
    # Add non-diagnosis constraints to response
    result.update(model.enforce_no_diagnosis_output(result))
    result.update(model.enforce_not_fda_certified())
    result["disclaimer"] = "This system is for academic and research purposes only and does not provide medical diagnosis."
    
    return JSONResponse(result)


@app.post("/analyze/advanced")
async def analyze_advanced(
    request: Request,
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    voice: UploadFile = File(None),
    depth_map: UploadFile = File(None),
    authorization: str = Form(None),
    db=Depends(get_db),
):
    """
    Phase 1 (v0.3) Advanced Analysis with micro-expressions and enhanced voice.
    Requires: facial image, symptoms text
    Optional: voice WAV, depth map image
    """
    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))
    model.enforce_internet_requirement(endpoint_accessed=True)
    
    if image is None or image.filename == "":
        raise HTTPException(status_code=400, detail="Image file is required.")
    if symptoms is None or symptoms.strip() == "":
        raise HTTPException(status_code=400, detail="Symptom text is required.")
    
    _size_guard(image, max_mb=6)
    if voice:
        _size_guard(voice, max_mb=5)
    if depth_map:
        _size_guard(depth_map, max_mb=3)
    
    # Read and decode image
    try:
        data = await image.read()
        if not data:
            raise ValueError("Empty image")
        npimg = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image upload: {e}")
    
    try:
        model.validate_facial_input(bgr)
    except model.HealthMorphException as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Face preprocessing
    face_img, face_bbox = pre.detect_and_prepare(bgr)
    facial_score = model.predict_facial_risk(face_img)
    symptom_score = model.analyze_symptoms(symptoms)
    
    # PHASE 1: MICRO-EXPRESSION DETECTION
    micro_expr_data = None
    try:
        from . import face_landmarks
        landmarks_curr, detected = face_landmarks.detect_face_landmarks(bgr)
        if detected:
            # For first frame, use current landmarks as previous (no comparison)
            # In real scenario, would track across video frames
            landmarks_prev = landmarks_curr
            micro_expr_data = face_landmarks.analyze_micro_expressions(landmarks_prev, landmarks_curr)
    except Exception as e:
        print(f"Micro-expression detection error: {e}")
    
    # PHASE 1: ENHANCED VOICE ANALYSIS
    voice_features_data = None
    if voice:
        try:
            from . import voice_analyzer
            vbytes = await voice.read()
            voice_features_data = voice_analyzer.analyze_voice_features(vbytes)
        except Exception as e:
            print(f"Voice analysis error: {e}")
    
    # PHASE 1: DEPTH ANALYSIS
    depth_analysis_data = None
    if depth_map:
        try:
            from . import depth_processor
            depth_bytes = await depth_map.read()
            depth_npimg = np.frombuffer(depth_bytes, np.uint8)
            depth_img = cv2.imdecode(depth_npimg, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                processor = depth_processor.DepthProcessor()
                depth_analysis_data = processor.process_depth_frame(depth_img)
        except Exception as e:
            print(f"Depth analysis error: {e}")
    
    # ADVANCED FUSION with Phase 1 features
    try:
        fused_score = model.fuse_scores_advanced(
            facial_score,
            symptom_score,
            voice_features=voice_features_data,
            micro_expr=micro_expr_data,
            depth_analysis=depth_analysis_data,
        )
    except Exception as e:
        print(f"Fusion error, falling back to basic: {e}")
        fused_score = model.fuse_scores(facial_score, symptom_score)
    
    risk_level = model.risk_level_from_score(fused_score)
    explanation = model.generate_explanation(risk_level, fused_score, symptom_score)
    confidence = model.compute_confidence(
        modalities_present=int(bool(image)) + int(bool(symptoms)) + int(bool(voice)) + int(bool(depth_map)),
        scores=[facial_score, symptom_score, 0, 0, 0],
    )
    next_steps = model.suggest_next_steps(risk_level, confidence)
    
    # Heatmap
    try:
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(STATIC_DIR, heatmap_filename)
        explain.generate_heatmap(bgr, face_bbox, heatmap_path)
        heatmap_url = f"/static/{heatmap_filename}"
    except Exception:
        heatmap_url = "/static/heatmap.svg"
    
    # Contributions breakdown
    contributions = {
        "facial": round(facial_score, 2),
        "symptoms": round(symptom_score, 2),
    }
    
    if micro_expr_data:
        contributions["micro_expression"] = round(model.face_landmarks.micro_expression_to_risk_score(micro_expr_data), 2)
    if voice_features_data:
        contributions["voice_prosody"] = round(model.voice_analyzer.voice_features_to_risk_score(voice_features_data), 2)
    if depth_analysis_data:
        contributions["depth_symmetry"] = round(model.depth_processor.depth_data_to_risk_score(depth_analysis_data), 2)
    
    # Persist if enabled
    if settings.persist_enabled:
        rec = AnalysisRecord(
            risk_level=risk_level,
            risk_score=int(round(fused_score)),
            confidence=int(round(confidence)),
            modalities=",".join([k for k, v in contributions.items() if v > 0]),
        )
        db.add(rec)
        db.commit()
    
    # Response
    result = {
        "risk_level": risk_level,
        "risk_score": int(round(fused_score)),
        "explanation": explanation,
        "heatmap_url": heatmap_url,
        "confidence": int(round(confidence)),
        "next_steps": next_steps,
        "contributions": contributions,
        "micro_expression": micro_expr_data or {},
        "voice_features": voice_features_data or {},
        "depth_analysis": depth_analysis_data or {},
    }
    
    result.update(model.enforce_no_diagnosis_output(result))
    result.update(model.enforce_not_fda_certified())
    result["disclaimer"] = "This system is for academic and research purposes only and does not provide medical diagnosis."
    
    return JSONResponse(result)


@app.post("/analyze/v3")
async def analyze_v3(
    request: Request,
    image: UploadFile = File(...),
    symptoms: str = Form(...),
    voice: UploadFile = File(None),
    depth_map: UploadFile = File(None),
    ensemble_method: str = Form("soft"),
    authorization: str = Form(None),
    db=Depends(get_db),
):
    """
    Phase 3 (v0.5) Deep Learning + Ensemble Analysis.
    Extends Phase 1+2 with:
    - ResNet50/EfficientNet neural feature extraction
    - Ensemble voting for robust predictions
    - Continuous learning feedback support
    """
    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))
    model.enforce_internet_requirement(endpoint_accessed=True)
    
    if image is None or image.filename == "":
        raise HTTPException(status_code=400, detail="Image file is required.")
    if symptoms is None or symptoms.strip() == "":
        raise HTTPException(status_code=400, detail="Symptom text is required.")
    
    _size_guard(image, max_mb=6)
    if voice:
        _size_guard(voice, max_mb=5)
    if depth_map:
        _size_guard(depth_map, max_mb=3)
    
    # Read and decode image
    try:
        data = await image.read()
        if not data:
            raise ValueError("Empty image")
        npimg = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image upload: {e}")
    
    try:
        model.validate_facial_input(bgr)
    except model.HealthMorphException as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Validate data inputs
    try:
        model.reject_dna_and_lab_data({"symptoms": symptoms})
    except model.HealthMorphException as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    analysis_id = str(uuid.uuid4())
    
    # Basic scores
    face_img, face_bbox = pre.detect_and_prepare(bgr)
    facial_score = model.predict_facial_risk(face_img)
    symptom_score = model.analyze_symptoms(symptoms)
    
    # PHASE 3: Neural feature extraction (ResNet50)
    neural_result = model.extract_neural_features_if_available(bgr)
    neural_features = neural_result.get('features') if neural_result else None
    
    # PHASE 2: Advanced analyses (pain, emotion, syndrome, patterns)
    pain_data = None
    emotion_data = None
    syndrome_data = None
    pattern_data = None
    
    try:
        from . import pain_detector
        pain_data = pain_detector.detect_pain_indicators(face_img)
    except Exception as e:
        print(f"Pain detection error: {e}")
    
    try:
        from . import emotion_analyzer
        emotion_data = emotion_analyzer.detect_emotional_indicators(face_img)
    except Exception as e:
        print(f"Emotion analysis error: {e}")
    
    try:
        from . import syndrome_matcher
        syndrome_data = syndrome_matcher.match_syndrome_phenotype(bgr, symptoms)
    except Exception as e:
        print(f"Syndrome matching error: {e}")
    
    try:
        from . import hidden_patterns
        pattern_data = hidden_patterns.detect_symptom_patterns(symptoms)
    except Exception as e:
        print(f"Pattern detection error: {e}")
    
    # PHASE 1: Voice and micro-expressions
    voice_features_data = None
    micro_expr_data = None
    
    if voice:
        try:
            from . import voice_analyzer
            vbytes = await voice.read()
            voice_features_data = voice_analyzer.analyze_voice_features(vbytes)
        except Exception as e:
            print(f"Voice analysis error: {e}")
    
    try:
        from . import face_landmarks
        landmarks_curr, detected = face_landmarks.detect_face_landmarks(bgr)
        if detected:
            landmarks_prev = landmarks_curr
            micro_expr_data = face_landmarks.analyze_micro_expressions(landmarks_prev, landmarks_curr)
    except Exception as e:
        print(f"Micro-expression detection error: {e}")
    
    # PHASE 3: ENSEMBLE VOTING
    # Simulate voting from multiple classifiers
    ensemble_votes = {}
    try:
        from . import model_ensemble
        
        votes_list = [
            model_ensemble.create_ensemble_vote("facial_heuristic", 
                                               "high_risk" if facial_score > 60 else "normal", 
                                               min(1.0, facial_score / 100.0)),
            model_ensemble.create_ensemble_vote("symptom_analysis", 
                                               "high_risk" if symptom_score > 60 else "normal", 
                                               min(1.0, symptom_score / 100.0)),
        ]
        
        # Add neural vote if available
        if neural_features is not None:
            neural_confidence = min(1.0, float(np.linalg.norm(neural_features) / 5000.0))
            votes_list.append(
                model_ensemble.create_ensemble_vote("neural_backbone", 
                                                   "high_risk" if neural_confidence > 0.5 else "normal", 
                                                   neural_confidence)
            )
        
        ensemble = model_ensemble.EnsembleClassifier(voting_method=ensemble_method)
        ensemble_result = ensemble.vote_classification(votes_list)
        
        ensemble_votes = {
            'prediction': ensemble_result.prediction,
            'confidence': ensemble_result.confidence,
            'consensus_strength': ensemble_result.consensus_strength,
            'votes': ensemble_result.votes,
        }
    except Exception as e:
        print(f"Ensemble voting error: {e}")
    
    # PHASE 3: FUSED SCORING (v0.5)
    try:
        fused_result = model.fuse_scores_v3(
            facial_score,
            symptom_score,
            neural_features=neural_features,
            pain_analysis=pain_data,
            emotion_analysis=emotion_data,
            syndrome_matches=syndrome_data,
            pattern_analysis=pattern_data,
            ensemble_votes=ensemble_votes,
        )
        fused_score = fused_result['risk_score']
        model_confidence = int(round(fused_result['model_confidence'] * 100))
        feature_usage = fused_result['feature_usage']
    except Exception as e:
        print(f"Phase 3 fusion error: {e}")
        # Fallback to Phase 2
        fused_result = model.fuse_scores_v2(
            facial_score, symptom_score,
            pain_analysis=pain_data,
            emotion_analysis=emotion_data,
            syndrome_matches=syndrome_data,
            pattern_analysis=pattern_data,
        )
        fused_score = fused_result
        model_confidence = 75
        feature_usage = ['facial', 'symptom']
    
    # Explanation and risk level
    risk_level = model.risk_level_from_score(fused_score)
    confidence = model.compute_confidence(symptom_score, facial_score)
    next_steps = model.suggest_next_steps(risk_level, symptoms)
    explanation = explain.create_explanation(fused_score, facial_score, symptom_score)
    
    # Heatmap visualization
    heatmap_url = ""
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        heatmap_img, heatmap_path = explain.create_gaussian_heatmap(gray, face_bbox)
        heatmap_url = f"/static/{os.path.basename(heatmap_path)}"
    except Exception as e:
        print(f"Heatmap generation error: {e}")
    
    # DB persistence (optional)
    if settings.use_database:
        try:
            rec = AnalysisRecord(
                analysis_id=analysis_id,
                facial_score=facial_score,
                symptom_score=symptom_score,
                fused_score=fused_score,
                risk_level=risk_level,
                input_symptoms=symptoms,
            )
            db.add(rec)
            db.commit()
        except Exception as e:
            print(f"DB error: {e}")
    
    # Continuous learning setup
    feedback_setup = {}
    try:
        from . import continuous_learning
        feedback_setup = {
            'prediction_id': analysis_id,
            'model_version': '0.5.0',
            'supports_feedback': True,
            'feedback_endpoint': '/feedback',
        }
    except Exception:
        pass
    
    # Response
    result = {
        "analysis_id": analysis_id,
        "risk_level": risk_level,
        "risk_score": int(round(fused_score)),
        "explanation": explanation,
        "heatmap_url": heatmap_url,
        "model_confidence": model_confidence,
        "next_steps": next_steps,
        "phase_3_features": {
            "neural_features_used": neural_features is not None,
            "ensemble_voting": ensemble_votes,
            "modalities_used": feature_usage,
            "consensus_strength": fused_result.get('consensus_strength', 0) if isinstance(fused_result, dict) else 0,
        },
        "voice_features": voice_features_data or None,
        "micro_expressions": micro_expr_data or None,
        "pain_indicators": pain_data or None,
        "emotional_indicators": emotion_data or None,
        "syndrome_matches": syndrome_data or None,
        "symptom_patterns": pattern_data or None,
        "continuous_learning": feedback_setup,
    }
    
    result.update(model.enforce_no_diagnosis_output(result))
    result.update(model.enforce_not_fda_certified())
    result["disclaimer"] = "Phase 3 (v0.5) experimental analysis. For academic research only. Not FDA/CE certified."
    
    return JSONResponse(result)


@app.get("/")
async def root():
    return {"message": "HealthMorph AI backend is running."}


@app.on_event("startup")
async def on_startup():
    init_db()


@app.post("/explain")
async def explain_prediction(
    analysis_id: str = Form(...),
    risk_score: int = Form(...),
    features_json: str = Form(None),
    include_counterfactual: bool = Form(True),
    authorization: str = Form(None),
):
    """
    Phase 4 (v0.6) Explainability Endpoint
    Provides SHAP/LIME explanations and counterfactual analysis for predictions.
    
    Accepts:
    - analysis_id: Reference to original /analyze/v3 prediction
    - risk_score: The prediction score (0-100)
    - features_json: Optional JSON of feature values
    - include_counterfactual: Whether to include "what if" analysis
    """
    
    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))
    
    try:
        # Parse features if provided
        features = None
        if features_json:
            try:
                import json as json_lib
                feature_dict = json_lib.loads(features_json)
                features = np.array([feature_dict.get(f"feat_{i}", 0) for i in range(10)])
            except:
                pass
        
        # Feature importance analysis
        feature_importance = model.generate_explainability_explanation(
            risk_score if risk_score > 50 else 0,
            risk_score if risk_score > 50 else 50,
        )
        
        # Decision boundary analysis
        boundary_info = model.generate_decision_boundary_explanation(risk_score)
        
        # Counterfactual / "what if" analysis
        counterfactual_info = {}
        if include_counterfactual:
            try:
                from . import counterfactual
                
                if features is not None:
                    analyzer = counterfactual.CounterfactualScenarioAnalyzer(
                        lambda x: float(np.mean(x))
                    )
                    counterfactual_info = {
                        'boundary_analysis': analyzer.find_decision_boundary(features),
                        'sensitivity_analysis': analyzer.sensitivity_analysis(features),
                        'recovery_plan': analyzer.create_recovery_plan(features),
                    }
            except Exception as e:
                print(f"Counterfactual error: {e}")
        
        # Improvement suggestions
        suggestions = model.suggest_improvement_directions(
            "",
            risk_score,
            model.risk_level_from_score(risk_score)
        )
        
        explanation_result = {
            'analysis_id': analysis_id,
            'risk_score': risk_score,
            'phase_4_explanation': {
                'feature_importance': feature_importance,
                'decision_boundary': boundary_info,
                'counterfactual_analysis': counterfactual_info,
                'improvement_suggestions': suggestions,
            },
            'explanation_type': 'SHAP/LIME + Counterfactual',
            'confidence': 'High' if boundary_info['stable'] else 'Medium',
        }
        
        explanation_result.update(model.enforce_no_diagnosis_output(explanation_result))
        explanation_result.update(model.enforce_not_fda_certified())
        
        return JSONResponse(explanation_result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {e}")


@app.post("/explain/shap")
async def explain_shap(
    features_json: str = Form(...),
    background_json: str = Form(None),
    model_prediction: float = Form(0.5),
    authorization: str = Form(None),
):
    """Return SHAP-style explanation for a feature vector."""

    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))

    features = _parse_features_json(features_json)

    background = None
    if background_json:
        try:
            raw = json.loads(background_json)
            background = np.array(raw, dtype=float)
            if background.ndim == 1:
                background = background.reshape(1, -1)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid background_json: {e}")

    try:
        from . import explainability

        model_func = lambda x: float(np.clip(np.mean(x) / 100.0, 0.0, 1.0))
        explainer = explainability.SHAPExplainer(model_func, background_data=background)
        shap_result = explainer.explain_prediction(features)

        response = {
            "analysis_type": "shap",
            "input_dim": int(features.size),
            "model_prediction": float(model_prediction),
            "shap": shap_result,
        }

        response.update(model.enforce_no_diagnosis_output(response))
        response.update(model.enforce_not_fda_certified())
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {e}")


@app.post("/explain/lime")
async def explain_lime(
    features_json: str = Form(...),
    num_samples: int = Form(100),
    num_features: int = Form(5),
    authorization: str = Form(None),
):
    """Return LIME local explanation for a feature vector."""

    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))

    features = _parse_features_json(features_json)

    try:
        from . import explainability

        model_func = lambda x: float(np.clip(np.mean(x) / 100.0, 0.0, 1.0))
        explainer = explainability.LIMEExplainer(model_func)
        lime_expl = explainer.explain_instance(features, num_samples=num_samples, num_features=num_features)

        response = {
            "analysis_type": "lime",
            "input_dim": int(features.size),
            "lime": lime_expl.__dict__,
        }

        response.update(model.enforce_no_diagnosis_output(response))
        response.update(model.enforce_not_fda_certified())
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME explanation error: {e}")


@app.post("/explain/counterfactual")
async def explain_counterfactual(
    features_json: str = Form(...),
    analysis_type: str = Form("recovery"),
    authorization: str = Form(None),
):
    """Return counterfactual / what-if analysis for a feature vector."""

    rate_limit()
    auth_guard((authorization or "").replace("Bearer ", ""))

    features = _parse_features_json(features_json)
    analysis_type = analysis_type.lower().strip()
    if analysis_type not in {"recovery", "boundary", "sensitivity"}:
        raise HTTPException(status_code=422, detail="analysis_type must be recovery | boundary | sensitivity")

    try:
        from . import counterfactual

        model_func = lambda x: float(np.clip(np.mean(x), 0.0, 100.0))
        cf_result = counterfactual.create_counterfactual_analysis(
            features,
            model_func=model_func,
            analysis_type=analysis_type,
        )

        response = {
            "analysis_type": analysis_type,
            "input_dim": int(features.size),
            "counterfactual": cf_result,
        }

        response.update(model.enforce_no_diagnosis_output(response))
        response.update(model.enforce_not_fda_certified())
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counterfactual explanation error: {e}")


@app.post("/export/fhir")
async def export_fhir(risk_level: str = Form(...), risk_score: int = Form(...), subject_id: str = Form("subject-001")):
    # Minimal FHIR Observation representation for demo
    resource = {
        "resourceType": "Observation",
        "id": f"hm-{uuid.uuid4().hex[:8]}",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "procedure"}]}],
        "code": {"text": "HealthMorph AI risk indication"},
        "subject": {"reference": f"Patient/{subject_id}"},
        "valueQuantity": {"value": risk_score, "unit": "%"},
        "interpretation": [{"text": risk_level}],
    }
    return JSONResponse(resource)
