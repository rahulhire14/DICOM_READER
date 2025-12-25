import { Activity, AlertCircle, CheckCircle, Loader2, Upload } from 'lucide-react';
import { useState } from 'react';

export default function LungPneumoniaDetection() {
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-white" />
              <h1 className="text-3xl font-bold text-white">Lung Pneumonia Detection</h1>
            </div>
            <p className="text-blue-100 mt-2">Upload a chest X-ray image for AI-powered analysis</p>
          </div>

          <div className="p-8">
            {!preview ? (
              <div className="border-3 border-dashed border-blue-300 rounded-xl p-12 text-center hover:border-blue-500 transition-colors">
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="w-16 h-16 mx-auto text-blue-500 mb-4" />
                  <p className="text-lg font-semibold text-gray-700 mb-2">
                    Click to upload X-ray image
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports JPG, PNG formats
                  </p>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="relative">
                  <img
                    src={preview}
                    alt="X-ray preview"
                    className="w-full max-h-96 object-contain rounded-lg border-2 border-gray-200"
                  />
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                  <button
                    onClick={handleReset}
                    className="px-6 py-3 rounded-lg font-semibold border-2 border-gray-300 hover:border-gray-400 transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-red-800">Error</p>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {prediction && (
              <div className={`mt-6 border-l-4 p-6 rounded-lg ${
                prediction.prediction === 'Normal' 
                  ? 'bg-green-50 border-green-500' 
                  : 'bg-yellow-50 border-yellow-500'
              }`}>
                <div className="flex items-start gap-4">
                  {prediction.prediction === 'Normal' ? (
                    <CheckCircle className="w-8 h-8 text-green-600 flex-shrink-0" />
                  ) : (
                    <AlertCircle className="w-8 h-8 text-yellow-600 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-2">
                      Prediction: {prediction.prediction}
                    </h3>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">Confidence:</span>
                        <span className="text-lg font-bold">
                          {(prediction.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full ${
                            prediction.prediction === 'Normal' 
                              ? 'bg-green-500' 
                              : 'bg-yellow-500'
                          }`}
                          style={{ width: `${prediction.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    <p className="text-sm mt-4 text-gray-600">
                      {prediction.prediction === 'Normal' 
                        ? 'The X-ray appears normal with no signs of pneumonia detected.'
                        : 'Pneumonia indicators detected. Please consult with a healthcare professional for proper diagnosis and treatment.'}
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-8 p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-gray-600">
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