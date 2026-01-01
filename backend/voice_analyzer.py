"""
Enhanced Voice Analysis for Phase 1 (v0.3)
Extracts prosody, pitch, energy, and voice quality indicators from WAV/MP3 audio.
"""

import numpy as np
import librosa
import io


def analyze_voice_features(audio_bytes: bytes, sr: int = 22050):
    """
    Extract advanced voice features from audio data.
    
    Returns:
        dict with keys:
        - prosody_score: 0-100 (voice pitch variation + intonation)
        - pitch: float (Hz, fundamental frequency)
        - energy: float (0-100, average RMS energy)
        - voice_quality: 0-100 (clarity, noise ratio)
        - speech_rate: float (words per minute estimate)
        - emotion_indicator: str (happy, neutral, distressed, etc.)
    """
    
    if not audio_bytes or len(audio_bytes) < 1000:
        return {
            "prosody_score": 0.0,
            "pitch": 0.0,
            "energy": 0.0,
            "voice_quality": 0.0,
            "speech_rate": 0.0,
            "emotion_indicator": "insufficient_data"
        }
    
    try:
        # Load audio from bytes
        y, sr_actual = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        
        # 1. ENERGY / LOUDNESS
        S = librosa.feature.melspectrogram(y=y, sr=sr_actual)
        log_S = librosa.power_to_db(S, ref=np.max)
        energy_rms = np.mean(librosa.feature.rms(y=y)[0])
        energy_score = float(np.clip(energy_rms * 100, 0, 100))
        
        # 2. PITCH / FUNDAMENTAL FREQUENCY
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_filtered = f0[f0 > 0]
        pitch_hz = float(np.median(f0_filtered)) if len(f0_filtered) > 0 else 0.0
        
        # 3. PROSODY (pitch variation + intonation)
        if len(f0_filtered) > 10:
            pitch_std = np.std(f0_filtered)
            pitch_range = np.ptp(f0_filtered)
            prosody_score = float(np.clip((pitch_std + pitch_range / 100) * 0.5, 0, 100))
        else:
            prosody_score = 0.0
        
        # 4. VOICE QUALITY (spectral centroid + zero crossing rate)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_actual))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        voice_quality = float(np.clip(spectral_centroid / 50, 0, 100))
        
        # 5. SPEECH RATE (onset detection + frame count)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr_actual)
        num_onsets = len(onset_frames)
        duration_sec = len(y) / sr_actual
        speech_rate = (num_onsets / duration_sec * 60) if duration_sec > 0 else 0.0
        
        # 6. EMOTION INDICATOR (heuristic based on pitch + energy)
        emotion = "neutral"
        if pitch_hz > 180 and energy_score > 60:
            emotion = "happy"
        elif pitch_hz > 200 and pitch_std > 50:
            emotion = "animated"
        elif pitch_hz < 120 and energy_score < 40:
            emotion = "sad"
        elif energy_score > 75 and pitch_std > 60:
            emotion = "distressed"
        
        return {
            "prosody_score": round(prosody_score, 2),
            "pitch": round(pitch_hz, 2),
            "energy": round(energy_score, 2),
            "voice_quality": round(voice_quality, 2),
            "speech_rate": round(speech_rate, 2),
            "emotion_indicator": emotion,
        }
    
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return {
            "prosody_score": 0.0,
            "pitch": 0.0,
            "energy": 0.0,
            "voice_quality": 0.0,
            "speech_rate": 0.0,
            "emotion_indicator": "error",
        }


def voice_features_to_risk_score(features: dict) -> float:
    """
    Convert voice features to a health risk indicator (0-100).
    Heuristic: distressed emotion, high pitch variance, low energy may indicate stress.
    
    Args:
        features: dict from analyze_voice_features()
    
    Returns:
        float: risk score 0-100
    """
    
    score = 0.0
    
    # Emotion-based weighting
    emotion_weights = {
        "distressed": 25,
        "sad": 15,
        "animated": 10,
        "happy": 0,
        "neutral": 5,
        "insufficient_data": 0,
        "error": 0,
    }
    score += emotion_weights.get(features.get("emotion_indicator", "error"), 0)
    
    # Prosody contribution (high variance may indicate anxiety)
    prosody = features.get("prosody_score", 0)
    if prosody > 70:
        score += 15
    elif prosody > 40:
        score += 5
    
    # Energy contribution (very low may indicate fatigue/illness)
    energy = features.get("energy", 0)
    if energy < 20:
        score += 20
    elif energy < 40:
        score += 10
    
    # Pitch contribution (extremes may indicate strain)
    pitch = features.get("pitch", 0)
    if pitch < 80 or pitch > 280:
        score += 10
    
    return float(np.clip(score, 0, 100))
