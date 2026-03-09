import { useState, useRef } from 'react';
import Header from './components/Header';
import TextInput from './components/TextInput';
import ResultsPanel from './components/ResultsPanel';
import ErrorToast from './components/ErrorToast';

const EXAMPLES = [
  "All politicians are corrupt and only care about themselves.",
  "Women are too emotional to be good leaders.",
  "If we allow any immigration, the country will be destroyed.",
  "Young people these days are lazy and entitled.",
  "I feel like this is wrong, so it must be wrong.",
  "Either we ban all cars or the planet will die.",
];

export default function App() {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const resultsRef = useRef(null);

  const analyze = async () => {
    if (text.trim().length < 10) {
      setError('Text must be at least 10 characters.');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const res = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResults(data);

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = () => {
    const example = EXAMPLES[Math.floor(Math.random() * EXAMPLES.length)];
    setText(example);
    setResults(null);
  };

  return (
    <div className="app">
      <div className="bg-grid" />
      <div className="glow glow-1" />
      <div className="glow glow-2" />

      <Header />

      <main className="main">
        <TextInput
          text={text}
          setText={setText}
          onAnalyze={analyze}
          onExample={loadExample}
          loading={loading}
        />

        <div ref={resultsRef}>
          {results && <ResultsPanel results={results} />}
        </div>
      </main>

      <footer className="footer">
        <p>Powered by BART Zero-Shot Classification & Google Gemini 2.5 Flash</p>
      </footer>

      {error && <ErrorToast message={error} onClose={() => setError(null)} />}
    </div>
  );
}
