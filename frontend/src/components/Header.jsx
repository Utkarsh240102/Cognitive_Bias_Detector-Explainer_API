export default function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">🧠</span>
          <div>
            <h1 className="logo-title">Cognitive Bias Detector</h1>
            <p className="logo-subtitle">AI-Powered Bias Analysis & Neutral Rewriting</p>
          </div>
        </div>
        <div className="header-badge">
          <span className="badge-dot" />
          <span>v1.0</span>
        </div>
      </div>
    </header>
  );
}
