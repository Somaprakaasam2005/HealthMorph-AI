import React from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar.jsx';
import Home from './pages/Home.jsx';
import Analyze from './pages/Analyze.jsx';
import Results from './pages/Results.jsx';

export default function App() {
  const navigate = useNavigate();
  return (
    <div className="app-container">
      <Navbar onStart={() => navigate('/analyze')} />
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </main>
    </div>
  );
}
