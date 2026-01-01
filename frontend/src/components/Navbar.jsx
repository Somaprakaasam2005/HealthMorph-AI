import React from 'react';
import { Link } from 'react-router-dom';
import logoImage from '/logo.png';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="brand" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <Link to="/" style={{ textDecoration: 'none', display: 'flex' }}>
          <img 
            src={logoImage} 
            alt="HealthMorph AI Logo" 
            style={{ 
              height: '100px', 
              width: 'auto',
              objectFit: 'contain',
              cursor: 'pointer'
            }} 
            onError={(e) => console.error('Logo failed to load', e)}
          />
        </Link>
      </div>
      <nav>
        <Link to="/">Home</Link>
        <Link to="/analyze">Analyze</Link>
      </nav>
    </header>
  );
}
