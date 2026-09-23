import React, { useState, useEffect } from 'react';
import './App.css';

// Preset examples as required: English spam, English ham, Hindi spam, Hindi ham, Hinglish
const EXAMPLES = [
  {
    category: 'English Spam',
    type: 'spam',
    tagClass: 'category-spam',
    text: 'WINNER! You have won a guaranteed £1,000 cash prize or a holiday voucher! Call 09061701461 now to claim your code. T&Cs apply.'
  },
  {
    category: 'English Ham',
    type: 'ham',
    tagClass: 'category-ham',
    text: 'Hey, are you free this afternoon? Let us meet at the library around 4 PM to work on our semester project.'
  },
  {
    category: 'Hindi Spam',
    type: 'spam',
    tagClass: 'category-spam',
    text: 'बधाई हो! आपने ₹50,000 का नकद इनाम जीता है। अपनी पुरस्कार राशि तुरंत पाने के लिए लिंक पर क्लिक करें: https://prize-claim.in'
  },
  {
    category: 'Hindi Ham',
    type: 'ham',
    tagClass: 'category-ham',
    text: 'नमस्ते भाई, कल कॉलेज कितने बजे आना है? क्या तुम मुझे क्लास नोट्स और असाइनमेंट की पीडीएफ भेज सकते हो?'
  },
  {
    category: 'Hinglish',
    type: 'hinglish',
    tagClass: 'category-hinglish',
    text: 'Bhai congratulations! You have won ₹25,000 Paytm cashback reward. Claim immediately click url: http://paytm-reward.xyz'
  }
];

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [modelType, setModelType] = useState('Linear SVM');

  // Verify backend health on load
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      // Try local proxy or direct backend
      const res = await fetch('/health', {
        headers: { 'Accept': 'application/json' }
      });
      if (res.ok) {
        const data = await res.json();
        setBackendStatus('online');
        if (data.model_type) {
          setModelType(data.model_type);
        }
      } else {
        setBackendStatus('offline');
      }
    } catch {
      // Fallback try direct url if proxy is bypassed
      try {
        const directRes = await fetch('http://127.0.0.1:5000/health');
        if (directRes.ok) {
          const d = await directRes.json();
          setBackendStatus('online');
          if (d.model_type) setModelType(d.model_type);
          return;
        }
      } catch {
        // Ignored
      }
      setBackendStatus('offline');
    }
  };

  const handleCheckMessage = async () => {
    const trimmed = message.trim();
    if (!trimmed) {
      setError('Please enter an SMS message to check.');
      return;
    }

    setError('');
    setLoading(true);
    setResult(null);

    try {
      // Try Vite proxy first, fallback to direct 127.0.0.1:5000
      let response;
      try {
        response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json; charset=utf-8'
          },
          body: JSON.stringify({ text: trimmed })
        });
      } catch (err) {
        response = await fetch('http://127.0.0.1:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json; charset=utf-8'
          },
          body: JSON.stringify({ text: trimmed })
        });
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with status: ${response.status}`);
      }

      const data = await response.json();
      setResult({
        prediction: data.prediction ? data.prediction.toUpperCase() : 'UNKNOWN',
        confidence: typeof data.confidence === 'number' ? Math.round(data.confidence * 100) : 0,
        language: data.language || 'English'
      });
      setBackendStatus('online');
    } catch (err) {
      console.error('Prediction request error:', err);
      setError('Backend unavailable or request failed. Make sure the Flask server is running.');
      setBackendStatus('offline');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessage('');
    setResult(null);
    setError('');
  };

  const handleSelectExample = (exampleText) => {
    setMessage(exampleText);
    setError('');
    // Clear previous result so user clicks Check Message
    setResult(null);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">Bilingual SMS Spam Detector</h1>
        <p className="app-subtitle">Check whether an SMS is Spam or Ham</p>
      </header>

      {/* Main Card */}
      <main className="main-card">
        {/* Error / Alert Banner */}
        {error && (
          <div className="alert-banner" role="alert">
            <span>⚠️ {error}</span>
            <button className="alert-close" onClick={() => setError('')} title="Dismiss">
              ×
            </button>
          </div>
        )}

        {/* Text Input Box */}
        <div className="textarea-wrapper">
          <textarea
            id="sms-input"
            className="sms-textarea"
            rows="5"
            placeholder="Enter your SMS message..."
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              if (error) setError('');
            }}
          />
          <div className="textarea-footer">
            <span>Supports English, Hindi (Devanagari), and Hinglish</span>
            <span>{message.length} characters</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="action-buttons">
          <button
            id="check-button"
            className="btn-primary"
            onClick={handleCheckMessage}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>Checking...</span>
              </>
            ) : (
              <span>Check Message</span>
            )}
          </button>
          <button
            id="clear-button"
            className="btn-secondary"
            onClick={handleClear}
            disabled={loading}
          >
            Clear
          </button>
        </div>

        {/* Visual Result Card */}
        {result && (
          <section
            id="result-card"
            className={`result-card ${result.prediction === 'SPAM' ? 'spam' : 'ham'}`}
          >
            <div className="result-header">
              <span className="result-title">Result</span>
              <span className="lang-pill">
                Language: <strong>{result.language}</strong>
              </span>
            </div>

            <div className="prediction-badge">
              <span>{result.prediction === 'SPAM' ? '🚨 SPAM' : '✅ HAM'}</span>
            </div>

            <div className="result-details-grid">
              <div className="result-item">
                <span className="result-label">Confidence</span>
                <span className="result-value">{result.confidence}%</span>
                <div className="confidence-bar-bg">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
              </div>

              <div className="result-item">
                <span className="result-label">Detected Language</span>
                <span className="result-value">
                  {result.language === 'Hindi' && '🇮🇳 Hindi (हिंदी)'}
                  {result.language === 'English' && '🌐 English'}
                  {result.language === 'Hinglish' && '🇮🇳/🌐 Hinglish'}
                </span>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Clickable Examples Section */}
      <section className="examples-section">
        <h2 className="examples-title">Clickable Example Messages</h2>
        <div className="examples-grid">
          {EXAMPLES.map((ex, index) => (
            <button
              key={index}
              id={`example-${index}`}
              className="example-chip"
              onClick={() => handleSelectExample(ex.text)}
              title="Click to insert into message box"
            >
              <div className="example-chip-header">
                <span className={`example-category ${ex.tagClass}`}>
                  {ex.category}
                </span>
              </div>
              <p className="example-text">{ex.text}</p>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;
