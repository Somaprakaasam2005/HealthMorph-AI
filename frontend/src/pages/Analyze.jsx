import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Spinner from '../components/Spinner.jsx';
import Disclaimer from '../components/Disclaimer.jsx';

const API_BASE = 'http://localhost:8000';

const doTips = [
  'Use a single face, front-facing, good light, no motion blur.',
  'Keep background quiet; short 5–10s voice/video clips are enough.',
  'State symptoms in plain words and include timing/duration.',
  'If you add vitals, use recent readings from a trusted device.',
];

export default function Analyze() {
  const [imageFile, setImageFile] = useState(null);
  const [symptoms, setSymptoms] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [questionnaire, setQuestionnaire] = useState('');
  const [voiceFile, setVoiceFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [sensors, setSensors] = useState({ heart_rate: '', systolic_bp: '', diastolic_bp: '', temperature: '', spo2: '' });

  const navigate = useNavigate();

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!imageFile) {
      setError('Please upload a face image.');
      return;
    }
    if (!symptoms.trim()) {
      setError('Please enter symptom details.');
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      form.append('image', imageFile);
      form.append('symptoms', symptoms);
      if (questionnaire.trim()) form.append('questionnaire', questionnaire.trim());
      if (voiceFile) form.append('voice', voiceFile);
      if (videoFile) form.append('video', videoFile);
      const sensorPayload = {
        heart_rate: sensors.heart_rate ? Number(sensors.heart_rate) : undefined,
        systolic_bp: sensors.systolic_bp ? Number(sensors.systolic_bp) : undefined,
        diastolic_bp: sensors.diastolic_bp ? Number(sensors.diastolic_bp) : undefined,
        temperature: sensors.temperature ? Number(sensors.temperature) : undefined,
        spo2: sensors.spo2 ? Number(sensors.spo2) : undefined,
      };
      form.append('sensors', JSON.stringify(sensorPayload));

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || 'Request failed');
      }
      const data = await res.json();
      navigate('/results', { state: data });
    } catch (err) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <h2 style={{ color: 'white', margin: '0 0 12px 0' }}>🔬 Health Analysis</h2>
        <p style={{ opacity: 0.95, lineHeight: '1.6', margin: 0 }}>
          Upload a face image and describe your symptoms for AI-powered health risk assessment.
        </p>
      </div>
      
      <div className="card">
      <form onSubmit={onSubmit}>
        {/* Face Image Upload */}
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: 'var(--text)' }}>
            <span style={{ fontSize: '24px' }}>📸</span>
            Face Image
            <span style={{ fontSize: '12px', color: '#ef4444', fontWeight: '700' }}>*</span>
          </label>
          <div style={{ position: 'relative' }}>
            <input 
              id="imageInput"
              className="input" 
              type="file" 
              accept="image/*" 
              capture="user" 
              onChange={(e) => setImageFile(e.target.files?.[0] || null)}
              style={{ display: 'none' }}
            />
            <label 
              htmlFor="imageInput"
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                gap: '12px',
                padding: '24px',
                background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
                border: '2px dashed #93c5fd',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                minHeight: '100px'
              }}
              onMouseOver={(e) => e.currentTarget.style.borderColor = '#3b82f6'}
              onMouseOut={(e) => e.currentTarget.style.borderColor = '#93c5fd'}
            >
              <span style={{ fontSize: '32px' }}>📷</span>
              <div>
                <div style={{ fontSize: '16px', fontWeight: '600', color: 'var(--primary)' }}>
                  {imageFile ? imageFile.name : 'Click to upload face image'}
                </div>
                <div style={{ fontSize: '14px', color: 'var(--muted)', marginTop: '4px' }}>
                  {imageFile ? 'Click to change file' : 'JPG, PNG, or HEIC supported'}
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Symptoms Input */}
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '16px', fontWeight: '600', color: 'var(--text)' }}>
            <span style={{ fontSize: '24px' }}>🩺</span>
            Symptoms Description
            <span style={{ fontSize: '12px', color: '#ef4444', fontWeight: '700' }}>*</span>
          </label>
          <textarea 
            className="input" 
            rows={6} 
            value={symptoms} 
            onChange={(e) => setSymptoms(e.target.value)} 
            placeholder="Describe your symptoms in detail... e.g., mild fever for 2 days, persistent dry cough, fatigue, headache..."
            style={{ 
              minHeight: '140px',
              fontSize: '15px',
              lineHeight: '1.6',
              background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
              border: '2px solid #fcd34d'
            }}
          />
        </div>

        {/* Optional Inputs Card */}
        <div className="card" style={{ marginTop: 24, background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)', border: '2px solid #e2e8f0' }}>
          <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '24px' }}>📋</span>
            Optional Inputs
          </h3>
          
          {/* Additional Details */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: 'var(--text)' }}>
              Additional Medical Information
            </label>
            <textarea 
              className="input" 
              rows={4} 
              value={questionnaire} 
              onChange={(e) => setQuestionnaire(e.target.value)} 
              placeholder="Any relevant medical history, current medications, allergies, or additional concerns..."
              style={{ fontSize: '14px', lineHeight: '1.6' }}
            />
          </div>

          {/* Voice Recording */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '14px', fontWeight: '600', color: 'var(--text)' }}>
              <span style={{ fontSize: '20px' }}>🎤</span>
              Voice Recording
            </label>
            <input 
              id="voiceInput"
              className="input" 
              type="file" 
              accept="audio/wav" 
              onChange={(e) => setVoiceFile(e.target.files?.[0] || null)}
              style={{ display: 'none' }}
            />
            <label 
              htmlFor="voiceInput"
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                padding: '16px',
                background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                border: '2px dashed #86efac',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.borderColor = '#22c55e'}
              onMouseOut={(e) => e.currentTarget.style.borderColor = '#86efac'}
            >
              <span style={{ fontSize: '24px' }}>🎙️</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--success)' }}>
                  {voiceFile ? voiceFile.name : 'Click to upload voice recording'}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
                  WAV format only
                </div>
              </div>
            </label>
          </div>

          {/* Video Recording */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '14px', fontWeight: '600', color: 'var(--text)' }}>
              <span style={{ fontSize: '20px' }}>🎥</span>
              Video Recording
            </label>
            <input 
              id="videoInput"
              className="input" 
              type="file" 
              accept="video/mp4" 
              capture="user" 
              onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
              style={{ display: 'none' }}
            />
            <label 
              htmlFor="videoInput"
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                padding: '16px',
                background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
                border: '2px dashed #fcd34d',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
              onMouseOver={(e) => e.currentTarget.style.borderColor = '#eab308'}
              onMouseOut={(e) => e.currentTarget.style.borderColor = '#fcd34d'}
            >
              <span style={{ fontSize: '24px' }}>📹</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--warning)' }}>
                  {videoFile ? videoFile.name : 'Click to upload video recording'}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
                  MP4 format only
                </div>
              </div>
            </label>
          </div>

          {/* Vital Signs */}
          <div>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', fontSize: '14px', fontWeight: '600', color: 'var(--text)' }}>
              <span style={{ fontSize: '20px' }}>💓</span>
              Vital Signs (Sensor Data)
            </label>
            <div className="grid" style={{ gap: '12px' }}>
              <div>
                <input 
                  className="input" 
                  type="number" 
                  placeholder="❤️ Heart Rate" 
                  value={sensors.heart_rate} 
                  onChange={(e) => setSensors(s => ({ ...s, heart_rate: e.target.value }))}
                  style={{ fontSize: '14px' }}
                />
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px', paddingLeft: '4px' }}>bpm</div>
              </div>
              <div>
                <input 
                  className="input" 
                  type="number" 
                  placeholder="📊 Systolic BP" 
                  value={sensors.systolic_bp} 
                  onChange={(e) => setSensors(s => ({ ...s, systolic_bp: e.target.value }))}
                  style={{ fontSize: '14px' }}
                />
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px', paddingLeft: '4px' }}>mmHg</div>
              </div>
              <div>
                <input 
                  className="input" 
                  type="number" 
                  placeholder="📉 Diastolic BP" 
                  value={sensors.diastolic_bp} 
                  onChange={(e) => setSensors(s => ({ ...s, diastolic_bp: e.target.value }))}
                  style={{ fontSize: '14px' }}
                />
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px', paddingLeft: '4px' }}>mmHg</div>
              </div>
              <div>
                <input 
                  className="input" 
                  type="number" 
                  step="0.1" 
                  placeholder="🌡️ Temperature" 
                  value={sensors.temperature} 
                  onChange={(e) => setSensors(s => ({ ...s, temperature: e.target.value }))}
                  style={{ fontSize: '14px' }}
                />
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px', paddingLeft: '4px' }}>°C</div>
              </div>
              <div>
                <input 
                  className="input" 
                  type="number" 
                  placeholder="💨 SpO2" 
                  value={sensors.spo2} 
                  onChange={(e) => setSensors(s => ({ ...s, spo2: e.target.value }))}
                  style={{ fontSize: '14px' }}
                />
                <div style={{ fontSize: '11px', color: 'var(--muted)', marginTop: '4px', paddingLeft: '4px' }}>%</div>
              </div>
            </div>
          </div>
        </div>

        <div className="card" style={{ marginTop: 24, background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)', border: '2px solid #93c5fd' }}>
          <h3 style={{ marginTop: 0 }}>💡 Data Quality Tips</h3>
          <div>
            <h4 style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              ✅ Best Practices
            </h4>
            <ul className="feature-list">
              {doTips.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginTop: 24 }}>
          <button className="button" type="submit" disabled={loading}>
            {loading ? 'Analyzing...' : '🔬 Analyze Health Risk'}
          </button>
          {loading && <Spinner />}
        </div>
      </form>

      {error && (
        <div className="disclaimer" style={{ borderLeftColor: '#ef4444', background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)', color: '#7f1d1d' }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      
      <Disclaimer />
      </div>
    </div>
  );
}
