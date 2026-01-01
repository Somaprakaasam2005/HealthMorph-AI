import React from 'react';

export default function About() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div className="card" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <h2 style={{ color: 'white', marginBottom: '16px' }}>About HealthMorph AI</h2>
        <p style={{ fontSize: '18px', lineHeight: '1.6', opacity: 0.95 }}>
          HealthMorph AI demonstrates an advanced academic approach to AI-assisted health risk assessment using cutting-edge multimodal analysis techniques.
        </p>
      </div>

      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <span style={{ fontSize: '32px' }}>🎯</span>
          <h3 style={{ margin: 0 }}>Project Objective</h3>
        </div>
        <p style={{ color: 'var(--muted)', lineHeight: '1.6' }}>
          This project showcases an innovative approach to early health risk indication through facial image analysis, voice pattern recognition, and symptom text processing. Built for research and educational purposes.
        </p>
      </div>

      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <span style={{ fontSize: '32px' }}>⚙️</span>
          <h3 style={{ margin: 0 }}>Technologies Used</h3>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          <div style={{ padding: '16px', background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)', borderRadius: '12px', border: '2px solid #93c5fd' }}>
            <strong style={{ color: 'var(--primary)' }}>Frontend</strong>
            <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: 'var(--muted)' }}>React.js, Modern UI/UX</p>
          </div>
          <div style={{ padding: '16px', background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)', borderRadius: '12px', border: '2px solid #86efac' }}>
            <strong style={{ color: 'var(--success)' }}>Backend</strong>
            <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: 'var(--muted)' }}>FastAPI, Python</p>
          </div>
          <div style={{ padding: '16px', background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', borderRadius: '12px', border: '2px solid #fcd34d' }}>
            <strong style={{ color: 'var(--warning)' }}>AI/ML</strong>
            <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: 'var(--muted)' }}>OpenCV, NumPy, NLP</p>
          </div>
        </div>
      </div>

      <div className="card" style={{ background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', border: '2px solid #fcd34d' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <span style={{ fontSize: '32px' }}>⚠️</span>
          <h3 style={{ margin: 0 }}>Ethical Disclaimer</h3>
        </div>
        <p style={{ color: '#92400e', lineHeight: '1.6', fontWeight: 500 }}>
          This demo does not provide medical diagnosis or treatment recommendations. It should not be used for clinical decisions. Always consult qualified healthcare professionals for medical advice.
        </p>
      </div>

      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <span style={{ fontSize: '32px' }}>🎓</span>
          <h3 style={{ margin: 0 }}>Academic Use</h3>
        </div>
        <p style={{ color: 'var(--muted)', lineHeight: '1.6' }}>
          Built exclusively for research and educational purposes. Suitable for project submissions, demonstrations, and academic presentations.
        </p>
      </div>
    </div>
  );
}
