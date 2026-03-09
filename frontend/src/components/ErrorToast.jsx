import { useEffect, useState } from 'react';

export default function ErrorToast({ message, onClose }) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setExiting(true);
      setTimeout(onClose, 300);
    }, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const dismiss = () => {
    setExiting(true);
    setTimeout(onClose, 300);
  };

  return (
    <div className={`toast ${exiting ? 'toast-exit' : 'toast-enter'}`} onClick={dismiss}>
      <span className="toast-icon">⚠️</span>
      <span className="toast-msg">{message}</span>
      <button className="toast-close" onClick={dismiss}>✕</button>
    </div>
  );
}
