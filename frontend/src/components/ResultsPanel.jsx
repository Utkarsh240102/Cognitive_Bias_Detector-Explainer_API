import { useState } from 'react';
import BiasCard from './BiasCard';

export default function ResultsPanel({ results }) {
  const { biases, explanation, neutral_rewrite } = results;
  const [copied, setCopied] = useState(null);

  const copyText = (text, field) => {
    navigator.clipboard.writeText(text);
    setCopied(field);
    setTimeout(() => setCopied(null), 2000);
  };

  const noBias = !biases || biases.length === 0;

  return (
    <section className="results-section fade-in">
      {/* ── Bias Cards ─────────────────────────── */}
      <div className="results-block">
        <div className="results-block-header">
          <h3>
            <span className="section-icon">📊</span>
            Detected Biases
          </h3>
          {!noBias && (
            <span className="results-count">{biases.length} found</span>
          )}
        </div>

        {noBias ? (
          <div className="no-bias">
            <span className="no-bias-icon">✅</span>
            <p className="no-bias-text">No significant cognitive biases detected.</p>
            <p className="no-bias-sub">This statement appears to be relatively neutral.</p>
          </div>
        ) : (
          <div className="bias-grid">
            {biases.map((b, i) => (
              <BiasCard key={b.type} bias={b} index={i} />
            ))}
          </div>
        )}
      </div>

      {/* ── Explanation ─────────────────────────── */}
      <div className="results-block">
        <div className="results-block-header">
          <h3>
            <span className="section-icon">💡</span>
            Explanation
          </h3>
          <button
            className={`btn-copy ${copied === 'explanation' ? 'copied' : ''}`}
            onClick={() => copyText(explanation, 'explanation')}
          >
            {copied === 'explanation' ? '✓ Copied' : '📋 Copy'}
          </button>
        </div>
        <div className="text-block explanation-block">
          <p>{explanation}</p>
        </div>
      </div>

      {/* ── Neutral Rewrite ────────────────────── */}
      <div className="results-block">
        <div className="results-block-header">
          <h3>
            <span className="section-icon">✏️</span>
            Neutral Rewrite
          </h3>
          <button
            className={`btn-copy ${copied === 'rewrite' ? 'copied' : ''}`}
            onClick={() => copyText(neutral_rewrite, 'rewrite')}
          >
            {copied === 'rewrite' ? '✓ Copied' : '📋 Copy'}
          </button>
        </div>
        <div className="text-block rewrite-block">
          <p>{neutral_rewrite}</p>
        </div>
      </div>
    </section>
  );
}
