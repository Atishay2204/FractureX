import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Capacitor } from '@capacitor/core';
import { Camera } from '@capacitor/camera';

const SENSITIVITY = {
    "Strict: fewer alerts": 20,
    "Balanced (Recommended)": 15,
    "Sensitive: catches faint signals": 8,
    "Very sensitive: shows all noise": 2,
};

function boostConfidence(rawPct) {
    if (rawPct <= 0) return 0;
    // Less aggressive boost: stops 5% hallucinations from becoming 80%.
    // raw 5 -> 12%
    // raw 15 -> 37%
    // raw 30 -> 75%
    return Math.min(99.9, rawPct * 2.5);
}

function confidenceBand(boostedPct) {
    if (boostedPct >= 70) return ["hi", "High Confidence"];
    if (boostedPct >= 40) return ["mid", "Moderate Confidence"];
    return ["lo", "Low Confidence"];
}

function App() {
    const [imageSource, setImageSource] = useState("file");
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [sensitivity, setSensitivity] = useState(15); // Default to Balanced
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    
    const fileInputRef = useRef(null);
    const webcamRef = useRef(null);
    const resultsRef = useRef(null); // Added for auto-scrolling
    
    const apiUrl = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

    const selectImageFile = (selectedFile) => {
        if (!selectedFile) return;
        if (selectedFile.size > 10 * 1024 * 1024) {
            setError("Choose an image that is 10 MB or smaller.");
            return;
        }
        setFile(selectedFile);
        setPreviewUrl(URL.createObjectURL(selectedFile));
        setResult(null);
        setError(null);
        handleUpload(selectedFile, sensitivity);
    };

    const handleFileChange = (e) => {
        selectImageFile(e.target.files[0]);
        e.target.value = '';
    };

    const captureWebcam = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
            fetch(imageSrc)
                .then(res => res.blob())
                .then(blob => {
                    const capturedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                    selectImageFile(capturedFile);
                });
        }
    }, [webcamRef, sensitivity]);

    const handleNativeCamera = async () => {
        try {
            const photo = await Camera.takePhoto({ quality: 90 });
            if (!photo.webPath) throw new Error("The camera did not return an image.");
            const photoResponse = await fetch(photo.webPath);
            const blob = await photoResponse.blob();
            const photoFile = new File([blob], "xray.jpg", {
                type: blob.type || "image/jpeg",
            });
            selectImageFile(photoFile);
        } catch (cameraError) {
            if (!String(cameraError?.message || cameraError).toLowerCase().includes("cancel")) {
                setError(cameraError?.message || "Could not open the camera.");
            }
        }
    };

    const handleSensitivityChange = (e) => {
        const val = parseInt(e.target.value);
        setSensitivity(val);
        if (file) {
            handleUpload(file, val);
        }
    };

    const handleUpload = async (selectedFile, currentSensitivity) => {
        if (!selectedFile) return;
        
        setLoading(true);
        setError(null);
        
        // Auto-scroll to results on mobile
        setTimeout(() => {
            resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
        
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("confidence", currentSensitivity);

        try {
            const response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Failed to process image.");
            }
            setResult({
                image: `data:image/png;base64,${data.annotated_image}`,
                rows: data.rows,
                is_binary: data.is_binary,
                fileName: selectedFile.name
            });
        } catch (err) {
            setError(err.message || "Something went wrong.");
        } finally {
            setLoading(false);
        }
    };

    const isNative = Capacitor.isNativePlatform();

    return (
        <div className="block-container">
            <div className="topnav">
                <div className="topnav-left">
                    <div className="topnav-logo">🩻</div>
                    <div className="topnav-brand">Fracture<span>X</span></div>
                </div>
                <div className="topnav-right">
                    <span className="topnav-link">Fracture Screening</span>
                    <span className="topnav-link">YOLO v8 Model</span>
                    <div className="topnav-badge"><div className="pulse"></div>Online</div>
                </div>
            </div>

            <div className="hero">
                <div className="hero-label">Deep Learning · Computer Vision · Medical Imaging</div>
                <h1>Bone Fracture<br/><em>Detection System</em></h1>
                <p className="hero-desc">
                    Take a photo of an X-ray or choose one from your device. Our model highlights
                    suspected fracture sites to help you prepare for a clinical consultation.
                </p>
                <div className="hero-features">
                    <div className="hero-feat"><span className="fi">⚡</span>Real-time inference</div>
                    <div className="hero-feat"><span className="fi">🔒</span>Uploads not persisted</div>
                    <div className="hero-feat"><span className="fi">📱</span>Works on any device</div>
                    <div className="hero-feat"><span className="fi">🩻</span>YOLO v8 backbone</div>
                </div>
            </div>

            <div className="steps">
                <div className="step-card">
                    <div className="step-num">1</div>
                    <div className="step-title">Capture or choose</div>
                    <div className="step-desc">Take a photo or select a clear JPG or PNG X-ray.</div>
                </div>
                <div className="step-card">
                    <div className="step-num">2</div>
                    <div className="step-title">Auto-analyse</div>
                    <div className="step-desc">The model runs instantly — no button needed.</div>
                </div>
                <div className="step-card">
                    <div className="step-num">3</div>
                    <div className="step-title">Review & Share</div>
                    <div className="step-desc">See annotated results and download for your doctor.</div>
                </div>
            </div>

            <div className="main-layout">
                <div className="panel">
                    <div className="panel-hdr">
                        <div className="panel-ico">📤</div>
                        <div>
                            <div className="panel-ttl">Your X-ray</div>
                            <div className="panel-sub">Take a photo or choose a JPG or PNG · max 10 MB</div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', background: 'var(--bg-card)', padding: '0.5rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', color: 'var(--ink)' }}>
                            <input type="radio" name="source" value="file" checked={imageSource === "file"} onChange={() => setImageSource("file")} />
                            Choose a file
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', color: 'var(--ink)' }}>
                            <input type="radio" name="source" value="camera" checked={imageSource === "camera"} onChange={() => setImageSource("camera")} />
                            Take a photo
                        </label>
                    </div>

                    {imageSource === "file" && (
                        <div className="file-uploader" onClick={() => fileInputRef.current?.click()}>
                            <input 
                                type="file" 
                                accept="image/jpeg, image/png" 
                                ref={fileInputRef} 
                                onChange={handleFileChange} 
                                style={{ display: 'none' }}
                            />
                            <p>Drag and drop file here</p>
                            <p>Limit 10MB per file • JPG, PNG</p>
                            <div className="btn">Browse files</div>
                        </div>
                    )}

                    {imageSource === "camera" && (
                        <div style={{ marginBottom: '1rem' }}>
                            {isNative ? (
                                <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                                    <p style={{ marginBottom: '1rem', color: 'var(--ink-muted)' }}>Tap below to open your device camera.</p>
                                    <button type="button" className="btn-primary" onClick={handleNativeCamera}>📷 Open Camera</button>
                                </div>
                            ) : (
                                <div style={{ background: '#030710', padding: '0.5rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                                    <Webcam
                                        audio={false}
                                        ref={webcamRef}
                                        screenshotFormat="image/jpeg"
                                        videoConstraints={{ facingMode: "environment" }}
                                        style={{ width: '100%', borderRadius: 'var(--radius-xs)', display: 'block' }}
                                    />
                                    <button type="button" className="btn-primary" onClick={captureWebcam} style={{ marginTop: '0.5rem' }}>
                                        📸 Capture Photo
                                    </button>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="select-wrapper">
                        <label>⚙️ Detection sensitivity</label>
                        <select value={sensitivity} onChange={handleSensitivityChange}>
                            {Object.entries(SENSITIVITY).map(([label, val]) => (
                                <option key={val} value={val}>{label}</option>
                            ))}
                        </select>
                    </div>
                    
                    <div className="privacy-tag">🔒 Images are processed for analysis; this demo does not persist uploads.</div>

                    {previewUrl && !result && !loading && (
                        <div className="img-display">
                            <img src={previewUrl} alt="Uploaded" />
                        </div>
                    )}
                </div>

                <div className="panel" ref={resultsRef}>
                    <div className="panel-hdr">
                        <div className="panel-ico">🔬</div>
                        <div>
                            <div className="panel-ttl">Analysis Results</div>
                            <div className="panel-sub">AI-generated — always verify with a clinician</div>
                        </div>
                    </div>

                    {loading && (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '4rem 2rem', background: 'var(--bg-card)', borderRadius: 'var(--radius-lg)', border: '1px dashed var(--border-accent)', marginTop: '1rem' }}>
                            <div className="spinner" style={{
                                width: '40px', height: '40px', border: '3px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 1s linear infinite', marginBottom: '1rem'
                            }}></div>
                            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                            <h4 style={{ color: 'var(--ink)', margin: '0 0 0.5rem', fontFamily: "'Space Grotesk', sans-serif" }}>Analysing Image...</h4>
                            <p style={{ color: 'var(--ink-muted)', margin: 0, fontSize: '0.9rem' }}>Running YOLOv8 inference. This takes a few seconds.</p>
                        </div>
                    )}
                    
                    {error && <div style={{ color: "var(--danger)", padding: "1rem", border: "1px solid var(--danger)", borderRadius: "8px", marginTop: "1rem" }}>{error}</div>}

                    {!loading && !result && !error && (
                        <div className="empty-box">
                            <div className="empty-anim">🩻</div>
                            <h4>Results will appear here</h4>
                            <p>Take a photo or choose an X-ray image — analysis starts automatically.</p>
                        </div>
                    )}

                    {!loading && result && (
                        <div>
                            {result.rows.length === 0 ? (
                                <div className="verdict is-ok">
                                    <div className="verdict-ico">✅</div>
                                    <div>
                                        <p className="verdict-title">No High-Confidence Finding</p>
                                        <p className="verdict-msg">The model detected no strong fracture signal above the current threshold.
                                        This does <strong>not</strong> rule out a fracture. If you are in pain, please see a doctor.</p>
                                    </div>
                                </div>
                            ) : (
                                result.rows[0].Confidence < 10 ? (
                                    <div className="verdict is-warn">
                                        <div className="verdict-ico">⚠️</div>
                                        <div>
                                            <p className="verdict-title">Low-Confidence Signal</p>
                                            <p className="verdict-msg">Possible area of concern near: <strong>{result.is_binary ? 'a possible fracture' : result.rows.map(r => r.Region.replace(' fracture', '').toLowerCase()).join(', ')}</strong>.
                                            This is a screening hint, not a diagnosis. A clinician must review the X-ray.</p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="verdict is-warn">
                                        <div className="verdict-ico">🚨</div>
                                        <div>
                                            <p className="verdict-title">Possible Fracture Detected</p>
                                            <p className="verdict-msg">Suspected region: <strong>{result.is_binary ? 'a possible fracture' : result.rows.map(r => r.Region.replace(' fracture', '').toLowerCase()).join(', ')}</strong>.
                                            Review the annotated image below and show it to your doctor.</p>
                                        </div>
                                    </div>
                                )
                            )}

                            <div className="img-display">
                                <img src={result.image} alt="Annotated Result" />
                            </div>

                            {result.rows.length > 0 && (
                                <div style={{ marginTop: "1rem" }}>
                                    <div className="panel-hdr" style={{ borderBottom: "none", paddingBottom: 0 }}>
                                        <div className="panel-ico">📊</div>
                                        <div><div className="panel-ttl">Confidence Breakdown</div></div>
                                    </div>
                                    
                                    {result.rows.map((r, i) => {
                                        const boosted = boostConfidence(r.Confidence);
                                        const [band, label] = confidenceBand(boosted);
                                        return (
                                            <div key={i} className={`finding ${band}`}>
                                                <div className="f-top">
                                                    <span className="f-name">{r.Region}</span>
                                                    <div className="f-meta">
                                                        <span className="f-pct">{boosted.toFixed(1)}%</span>
                                                        <span className={`f-tag ${band}`}>{label}</span>
                                                    </div>
                                                </div>
                                                <div className="cbar"><div className={`cfill ${band}`} style={{ width: `${boosted}%` }}></div></div>
                                            </div>
                                        )
                                    })}
                                </div>
                            )}

                            <div className="info-card">
                                <div className="ic-icon">🏥</div>
                                <div>
                                    <div className="ic-title">What to do next</div>
                                    <p>Show this result and your original X-ray to a doctor or radiologist.
                                    Do <strong>not</strong> use it alone to decide on any treatment.</p>
                                </div>
                            </div>
                            
                            <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                                <a className="btn-primary" href={result.image} download={`${result.fileName.split('.')[0]}_marked.png`}>
                                    ⬇️ Save Annotated Image
                                </a>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <div className="app-footer">
                <strong>FractureX</strong> · Fracture Detection System · For educational & demonstration purposes only<br/>
                Built with React, FastAPI & YOLOv8 · Uploads are not persisted
            </div>
        </div>
    );
}

export default App;
