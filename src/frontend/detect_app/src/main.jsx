import { Activity, AlertCircle, CheckCircle, Loader2, Upload } from 'lucide-react';
import React, { useState } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

function LungPneumoniaDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setPrediction(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError('Failed to get prediction. Make sure the API is running at http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <div className="card-container">
        <div className="card">
          <div className="header">
            <div className="header-content">
              <Activity className="header-icon" />
              <div>
                <h1 className="title">Lung Pneumonia Detection</h1>
                <p className="subtitle">Upload a chest X-ray image for AI-powered analysis</p>
              </div>
            </div>
          </div>

          <div className="content">
            {!preview ? (
              <div className="upload-area">
                <label htmlFor="file-upload" className="upload-label">
                  <Upload className="upload-icon" />
                  <p className="upload-text">Click to upload X-ray image</p>
                  <p className="upload-subtext">Supports JPG, PNG formats</p>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="file-input"
                  />
                </label>
              </div>
            ) : (
              <div className="preview-section">
                <div className="image-container">
                  <img src={preview} alt="X-ray preview" className="preview-image" />
                </div>

                <div className="button-group">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="btn btn-primary"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="spinner" />
                        Analyzing...
                      </>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                  <button onClick={handleReset} className="btn btn-secondary">
                    Reset
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="alert alert-error">
                <AlertCircle className="alert-icon" />
                <div>
                  <p className="alert-title">Error</p>
                  <p className="alert-message">{error}</p>
                </div>
              </div>
            )}

            {prediction && (
              <div className={`result ${prediction.prediction === 'Normal' ? 'result-normal' : 'result-pneumonia'}`}>
                <div className="result-content">
                  {prediction.prediction === 'Normal' ? (
                    <CheckCircle className="result-icon result-icon-normal" />
                  ) : (
                    <AlertCircle className="result-icon result-icon-pneumonia" />
                  )}
                  <div className="result-details">
                    <h3 className="result-title">Prediction: {prediction.prediction}</h3>
                    <div className="confidence-section">
                      <div className="confidence-row">
                        <span className="confidence-label">Confidence:</span>
                        <span className="confidence-value">
                          {(prediction.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="progress-bar">
                        <div
                          className={`progress-fill ${prediction.prediction === 'Normal' ? 'progress-normal' : 'progress-pneumonia'}`}
                          style={{ width: `${prediction.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    <p className="result-description">
                      {prediction.prediction === 'Normal' 
                        ? 'The X-ray appears normal with no signs of pneumonia detected.'
                        : 'Pneumonia indicators detected. Please consult with a healthcare professional for proper diagnosis and treatment.'}
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="disclaimer">
              <p>
                <strong>Disclaimer:</strong> This is an AI-assisted tool for educational purposes. 
                Always consult with qualified healthcare professionals for medical diagnosis and treatment.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <LungPneumoniaDetection />
  </React.StrictMode>,
);