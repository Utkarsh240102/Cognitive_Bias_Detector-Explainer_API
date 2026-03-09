import { useEffect, useState } from 'react';

const BIAS_COLORS = {
  'Stereotyping':           { bar: '#ef4444', bg: 'rgba(239,68,68,0.1)' },
  'Overgeneralization':     { bar: '#f97316', bg: 'rgba(249,115,22,0.1)' },
  'Hasty Generalization':   { bar: '#eab308', bg: 'rgba(234,179,8,0.1)' },
  'Confirmation Bias':      { bar: '#8b5cf6', bg: 'rgba(139,92,246,0.1)' },
  'Emotional Reasoning':    { bar: '#ec4899', bg: 'rgba(236,72,153,0.1)' },
  'False Dilemma':          { bar: '#14b8a6', bg: 'rgba(20,184,166,0.1)' },
  'Catastrophizing':        { bar: '#f43f5e', bg: 'rgba(244,63,94,0.1)' },
  'Black-and-White Thinking': { bar: '#6366f1', bg: 'rgba(99,102,241,0.1)' },
};

const BIAS_ICONS = {
  'Stereotyping':           '🏷️',
  'Overgeneralization':     '🌐',
  'Hasty Generalization':   '⚡',
  'Confirmation Bias':      '🔍',
  'Emotional Reasoning':    '💭',
  'False Dilemma':          '⚖️',
  'Catastrophizing':        '🌪️',
  'Black-and-White Thinking': '◐',
};

export default function BiasCard({ bias, index }) {
  const [animated, setAnimated] = useState(false);
  const pct = Math.round(bias.confidence * 100);
  const colors = BIAS_COLORS[bias.type] || { bar: '#6366f1', bg: 'rgba(99,102,241,0.1)' };
  const icon = BIAS_ICONS[bias.type] || '🔹';

  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), 80 * index);
    return () => clearTimeout(timer);
  }, [index]);

  const severity = pct >= 85 ? 'HIGH' : pct >= 65 ? 'MED' : 'LOW';
  const severityClass = severity === 'HIGH' ? 'sev-high' : severity === 'MED' ? 'sev-med' : 'sev-low';

  return (
    <div
      className={`bias-card ${animated ? 'visible' : ''}`}
      style={{ '--delay': `${index * 80}ms`, background: colors.bg }}
    >
      <div className="bias-card-top">
        <div className="bias-label">
          <span className="bias-icon">{icon}</span>
          <span className="bias-name">{bias.type}</span>
        </div>
        <div className={`severity-badge ${severityClass}`}>{severity}</div>
      </div>

      <div className="bias-bar-track">
        <div
          className="bias-bar-fill"
          style={{
            width: animated ? `${pct}%` : '0%',
            background: colors.bar,
          }}
        />
      </div>

      <div className="bias-card-bottom">
        <span className="confidence-value" style={{ color: colors.bar }}>
          {pct}%
        </span>
        <span className="confidence-label">confidence</span>
      </div>
    </div>
  );
}
