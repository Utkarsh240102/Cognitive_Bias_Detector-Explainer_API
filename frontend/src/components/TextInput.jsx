export default function TextInput({ text, setText, onAnalyze, onExample, loading }) {
  const charCount = text.length;
  const maxChars = 5000;

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      onAnalyze();
    }
  };

  return (
    <section className="input-section">
      <div className="input-card">
        <div className="input-header">
          <h2 className="input-title">
            <span className="input-icon">📝</span>
            Enter text to analyze
          </h2>
          <button className="btn-example" onClick={onExample} disabled={loading}>
            🎲 Try an example
          </button>
        </div>

        <div className="textarea-wrapper">
          <textarea
            className="textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type or paste a statement you want to check for cognitive biases..."
            maxLength={maxChars}
            rows={5}
            disabled={loading}
          />
          <div className="textarea-footer">
            <span className="char-hint">Ctrl + Enter to analyze</span>
            <span className={`char-count ${charCount > maxChars * 0.9 ? 'warn' : ''}`}>
              {charCount.toLocaleString()} / {maxChars.toLocaleString()}
            </span>
          </div>
        </div>

        <button
          className={`btn-analyze ${loading ? 'loading' : ''}`}
          onClick={onAnalyze}
          disabled={loading || charCount < 10}
        >
          {loading ? (
            <>
              <span className="spinner" />
              Analyzing...
            </>
          ) : (
            <>
              <span className="btn-icon">⚡</span>
              Analyze for Biases
            </>
          )}
        </button>
      </div>
    </section>
  );
}
