import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [loginType, setLoginType] = useState('user'); // 'user' or 'doctor'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email.trim()) {
      setError('Please enter your email');
      return;
    }
    if (!password.trim()) {
      setError('Please enter your password');
      return;
    }

    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      // Map doctor to admin privileges
      const role = loginType === 'doctor' ? 'admin' : 'user';
      localStorage.setItem('authToken', `${role}-token-` + Date.now());
      localStorage.setItem('userType', role);
      navigate('/');
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%)',
      padding: '48px 16px'
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
        width: '100%',
        maxWidth: '520px'
      }}>
        <div className="card" style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          textAlign: 'center',
          padding: '40px'
        }}>
          <h1 style={{ color: 'white', marginBottom: '12px', fontSize: '28px' }}>🔐 Login</h1>
          <p style={{ opacity: 0.95, lineHeight: '1.6', margin: 0 }}>
            Access HealthMorph AI
          </p>
        </div>

        <div className="card" style={{ padding: '40px' }}>
          <div style={{ marginBottom: '32px', display: 'flex', gap: '12px' }}>
            <button
              onClick={() => setLoginType('user')}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                background: loginType === 'user' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f1f5f9',
                color: loginType === 'user' ? 'white' : 'var(--text)',
                transition: 'all 0.3s ease'
              }}
            >
              👤 User Login
            </button>
            <button
              onClick={() => setLoginType('doctor')}
              style={{
                flex: 1,
                padding: '12px',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '14px',
                background: loginType === 'doctor' ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : '#f1f5f9',
                color: loginType === 'doctor' ? 'white' : 'var(--text)',
                transition: 'all 0.3s ease'
              }}
            >
              🩺 Doctor Login
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '20px' }}>
              <label style={{
                display: 'block',
                marginBottom: '8px',
                fontWeight: '600',
                color: 'var(--text)',
                fontSize: '14px'
              }}>
                Email Address
              </label>
              <input
                className="input"
                type="email"
                placeholder="your.email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{ fontSize: '14px' }}
              />
            </div>

            <div style={{ marginBottom: '28px' }}>
              <label style={{
                display: 'block',
                marginBottom: '8px',
                fontWeight: '600',
                color: 'var(--text)',
                fontSize: '14px'
              }}>
                Password
              </label>
              <input
                className="input"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ fontSize: '14px' }}
              />
            </div>

            <button
              className="button"
              type="submit"
              disabled={loading}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '16px',
                fontWeight: '600'
              }}
            >
              {loading ? '🔄 Logging in...' : loginType === 'doctor' ? '🔓 Doctor Login' : '🔓 User Login'}
            </button>
          </form>

          {loginType === 'user' && (
            <div style={{ marginTop: '24px', paddingTop: '24px', borderTop: '1px solid #e2e8f0' }}>
              <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '13px', marginBottom: '16px', marginTop: 0 }}>
                Or continue with
              </p>
              <button
                type="button"
                onClick={() => {
                  localStorage.setItem('authToken', `google-token-` + Date.now());
                  localStorage.setItem('userType', 'user');
                  navigate('/');
                }}
                style={{
                  width: '100%',
                  padding: '12px',
                  borderRadius: '8px',
                  border: '2px solid #e2e8f0',
                  background: 'white',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px',
                  transition: 'all 0.3s ease',
                  color: '#333'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.borderColor = '#667eea';
                  e.currentTarget.style.background = '#f8f9ff';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.borderColor = '#e2e8f0';
                  e.currentTarget.style.background = 'white';
                }}
              >
                <span style={{ fontSize: '20px' }}>🔵</span>
                Continue with Google
              </button>
            </div>
          )}

          {error && (
            <div style={{
              marginTop: '20px',
              padding: '12px',
              background: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
              color: '#7f1d1d',
              borderRadius: '8px',
              border: '1px solid #fca5a5',
              fontSize: '14px'
            }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          <div style={{
            marginTop: '24px',
            padding: '16px',
            background: '#f8fafc',
            borderRadius: '8px',
            border: '1px solid #e2e8f0'
          }}>
            <p style={{ margin: '0 0 12px 0', fontWeight: '600', color: 'var(--text)', fontSize: '13px' }}>Demo Credentials:</p>
            {loginType === 'user' && (
              <div style={{ fontSize: '12px', color: 'var(--muted)', lineHeight: '1.8' }}>
                <p style={{ margin: '4px 0' }}>📧 Email: user@example.com</p>
                <p style={{ margin: '4px 0' }}>🔑 Password: user123</p>
              </div>
            )}
            {loginType === 'doctor' && (
              <div style={{ fontSize: '12px', color: 'var(--muted)', lineHeight: '1.8' }}>
                <p style={{ margin: '4px 0' }}>📧 Email: doctor@example.com</p>
                <p style={{ margin: '4px 0' }}>🔑 Password: doctor123</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
