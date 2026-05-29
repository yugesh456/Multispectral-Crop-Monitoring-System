// AgriScan AI Dashboard JavaScript Wiring

// Tab Routing State
const tabs = document.querySelectorAll('.nav-tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');
        
        // Update active tab buttons
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        // Update active content panels
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.getAttribute('id') === targetTab) {
                content.classList.add('active');
            }
        });

        // Trigger chart updates or canvas resets when entering specific tabs
        if (targetTab === 'uav-flight-deck') {
            initFlightTelemetryLoop();
        }
    });
});

// --- TELEMETRY & STABILIZATION CANVAS ANIMATION (TAB 1) ---
const shakyCanvas = document.getElementById('shaky-canvas');
const smoothCanvas = document.getElementById('smooth-canvas');
const shakyCtx = shakyCanvas.getContext('2d');
const smoothCtx = smoothCanvas.getContext('2d');

let animationFrameId = null;
let telemetryData = null;
let currentFrameIdx = 0;

async function fetchTelemetryData() {
    try {
        const response = await fetch('/api/telemetry?num_frames=120');
        if (!response.ok) throw new Error("Server error fetching telemetry");
        telemetryData = await response.json();
        currentFrameIdx = 0;
        
        // Draw the Trajectory Chart using Chart.js
        renderTelemetryChart(telemetryData);
    } catch (err) {
        console.error(err);
    }
}

// Draw a stylized leaf and target hud on a canvas
function drawLeafOnCanvas(ctx, canvas, dx, dy, dtheta, isStabilized) {
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Draw Background Grid (moving grid lines based on drift)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    const gridOffset_x = dx % gridSize;
    const gridOffset_y = dy % gridSize;
    
    for (let x = gridOffset_x; x < w; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = gridOffset_y; y < h; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Draw Target HUD scope rings
    ctx.strokeStyle = isStabilized ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 110, 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(cx, cy, 50, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx - 130, cy);
    ctx.lineTo(cx - 10, cy);
    ctx.moveTo(cx + 10, cy);
    ctx.lineTo(cx + 130, cy);
    ctx.moveTo(cx, cy - 130);
    ctx.lineTo(cx, cy - 10);
    ctx.moveTo(cx, cy + 10);
    ctx.lineTo(cx, cy + 130);
    ctx.stroke();

    // Save state for Leaf rotation & translation
    ctx.save();
    ctx.translate(cx + dx, cy + dy);
    ctx.rotate(dtheta * Math.PI / 180);

    // Draw leaf shape
    ctx.fillStyle = isStabilized ? 'rgba(16, 185, 129, 0.65)' : 'rgba(148, 163, 184, 0.5)';
    ctx.strokeStyle = isStabilized ? '#10b981' : '#94a3b8';
    ctx.lineWidth = 3;

    ctx.beginPath();
    // Simple lanceolate leaf path
    ctx.moveTo(0, -60);
    ctx.quadraticCurveTo(35, -20, 0, 70);
    ctx.quadraticCurveTo(-35, -20, 0, -60);
    ctx.fill();
    ctx.stroke();

    // Vein details
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, -60);
    ctx.lineTo(0, 55);
    ctx.stroke();

    // Restoration of drawing context
    ctx.restore();

    // Center focal lock dot
    ctx.fillStyle = isStabilized ? '#10b981' : '#ef4444';
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
}

function flightTelemetryLoop() {
    if (!telemetryData) return;

    const shaky = telemetryData.shaky;
    const stabilized = telemetryData.stabilized;
    const len = shaky.x.length;

    // Get current offsets relative to a center point
    const sx = shaky.x[currentFrameIdx] - 100.0;
    const sy = shaky.y[currentFrameIdx] - 100.0;
    const st = shaky.theta[currentFrameIdx];

    const mx = stabilized.x[currentFrameIdx] - 100.0;
    const my = stabilized.y[currentFrameIdx] - 100.0;
    const mt = stabilized.theta[currentFrameIdx];

    // Draw on canvases
    drawLeafOnCanvas(shakyCtx, shakyCanvas, sx, sy, st, false);
    drawLeafOnCanvas(smoothCtx, smoothCanvas, mx, my, mt, true);

    // Step loop forward
    currentFrameIdx = (currentFrameIdx + 1) % len;
    
    // Highlight active chart element if chart is active
    if (window.telemetryChartInstance) {
        window.telemetryChartInstance.setActiveElements([{
            datasetIndex: 0,
            index: currentFrameIdx
        }]);
        window.telemetryChartInstance.update();
    }

    animationFrameId = requestAnimationFrame(flightTelemetryLoop);
}

function initFlightTelemetryLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    currentFrameIdx = 0;
    if (!telemetryData) {
        fetchTelemetryData().then(() => {
            flightTelemetryLoop();
        });
    } else {
        flightTelemetryLoop();
    }
}

// Chart.js Telemetry Graph
function renderTelemetryChart(data) {
    const ctx = document.getElementById('telemetry-chart').getContext('2d');
    
    if (window.telemetryChartInstance) {
        window.telemetryChartInstance.destroy();
    }

    window.telemetryChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.time.map(t => t.toFixed(1) + 's'),
            datasets: [
                {
                    label: 'Raw UAV Trajectory (Flight Jitter)',
                    data: data.shaky.x,
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'Stabilized Trajectory (Kalman Filter)',
                    data: data.stabilized.x,
                    borderColor: '#10b981',
                    borderWidth: 3,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 15 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f8fafc', font: { family: 'Outfit', weight: '600' } }
                }
            }
        }
    });
}

document.getElementById('btn-refresh-telemetry').addEventListener('click', () => {
    fetchTelemetryData();
});


// --- SIFT BAND ALIGNMENT INTERACTIVE PANEL (TAB 2) ---
const btnRunAlignment = document.getElementById('btn-run-alignment');

btnRunAlignment.addEventListener('click', async () => {
    const crop = document.getElementById('alignment-crop').value;
    const severity = parseInt(document.getElementById('alignment-severity').value);
    
    btnRunAlignment.disabled = true;
    btnRunAlignment.innerHTML = '<i class="lucide-refresh-cw spin"></i> Processing SIFT keypoints...';
    
    try {
        const response = await fetch('/api/align', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ crop_type: crop, severity: severity })
        });
        
        if (!response.ok) throw new Error("Alignment API failed");
        
        const data = await response.json();
        
        // Update images
        document.getElementById('img-unaligned-composite').src = data.unaligned_composite;
        document.getElementById('img-aligned-composite').src = data.aligned_composite;
        
        // Update individual band images
        for (let i = 0; i < 5; i++) {
            document.getElementById(`band-img-${i}`).src = data.aligned_bands[i];
        }
        
        // Update metrics
        document.getElementById('align-jitter-raw').innerText = data.jitter_metric_unaligned.toFixed(1) + ' px';
        document.getElementById('align-jitter-smooth').innerText = data.jitter_metric_aligned.toFixed(3) + ' px';
        
    } catch (err) {
        console.error(err);
        alert("Alignment calculation failed. Please check backend connection.");
    } finally {
        btnRunAlignment.disabled = false;
        btnRunAlignment.innerHTML = '<i data-lucide="refresh-cw"></i> Run SIFT Band Alignment';
        lucide.createIcons();
    }
});


// --- HIGH-RESOLUTION SPECTRAL ANALYSIS ENGINE (TAB 3) ---
const btnRunSpectral = document.getElementById('btn-run-spectral');
const sliderMin = document.getElementById('slider-min-stress');
const sliderMax = document.getElementById('slider-max-stress');
const valMin = document.getElementById('val-min-stress');
const valMax = document.getElementById('val-max-stress');

// Sync slider labels dynamically
sliderMin.addEventListener('input', () => { valMin.innerText = sliderMin.value; });
sliderMax.addEventListener('input', () => { valMax.innerText = sliderMax.value; });

async function runSpectralAnalysis() {
    const crop = document.getElementById('spectral-crop').value;
    const severity = parseInt(document.getElementById('spectral-severity').value);
    const minStress = parseFloat(sliderMin.value);
    const maxStress = parseFloat(sliderMax.value);
    
    btnRunSpectral.disabled = true;
    btnRunSpectral.innerText = 'Calculating index maps...';
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                crop_type: crop,
                severity: severity,
                min_stress: minStress,
                max_stress: maxStress
            })
        });
        
        if (!response.ok) throw new Error("Spectral Analysis API failed");
        
        const data = await response.json();
        
        // Update maps
        document.getElementById('img-rgb').src = data.rgb;
        document.getElementById('img-cir').src = data.cir;
        document.getElementById('img-ndvi').src = data.ndvi;
        document.getElementById('img-ndre').src = data.ndre;
        document.getElementById('img-stress-overlay').src = data.stress_overlay;
        
        // Update stats
        document.getElementById('mean-ndvi-text').innerText = data.avg_ndvi.toFixed(3);
        document.getElementById('mean-ndre-text').innerText = data.avg_ndre.toFixed(3);
        
        const stressedText = document.getElementById('stressed-pct-text');
        stressedText.innerText = data.stressed_area_pct.toFixed(1) + '%';
        
        // Set color warning levels for stressed %
        if (data.stressed_area_pct > 35) {
            stressedText.className = 'red-text';
        } else if (data.stressed_area_pct > 10) {
            stressedText.className = 'orange-text';
        } else {
            stressedText.className = 'emerald-text';
        }
        
    } catch (err) {
        console.error(err);
    } finally {
        btnRunSpectral.disabled = false;
        btnRunSpectral.innerText = 'Generate Indexes';
    }
}

btnRunSpectral.addEventListener('click', runSpectralAnalysis);


// --- PYTORCH CNN-XCEPTION DIAGNOSTIC CLASSIFIER (TAB 4) ---
const btnRunClassify = document.getElementById('btn-run-classify');

btnRunClassify.addEventListener('click', async () => {
    const crop = document.getElementById('classify-crop').value;
    const severity = parseInt(document.getElementById('classify-severity').value);
    
    btnRunClassify.disabled = true;
    btnRunClassify.innerText = 'Analyzing tissue...';
    
    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ crop_type: crop, severity: severity })
        });
        
        if (!response.ok) throw new Error("Classification API failed");
        
        const data = await response.json();
        
        // Update class probability bars
        for (let i = 0; i < 5; i++) {
            const bar = document.getElementById(`prob-${i}`);
            const text = document.getElementById(`prob-${i}-text`);
            const prob = data.probabilities[i];
            
            bar.style.width = prob + '%';
            text.innerText = prob.toFixed(1) + '%';
        }
        
        // Update result diagnosis
        const resultCard = document.getElementById('diagnosis-result-area');
        const labelText = document.getElementById('diag-label');
        const descText = document.getElementById('diag-desc');
        
        labelText.innerText = data.label;
        descText.innerText = data.description;
        
        // Color classification label based on index
        if (data.predicted_class === 0) {
            labelText.className = 'diagnosis-label emerald-text';
        } else if (data.predicted_class === 1) {
            labelText.className = 'diagnosis-label teal-text';
        } else if (data.predicted_class === 2) {
            labelText.className = 'diagnosis-label yellow-text';
        } else if (data.predicted_class === 3) {
            labelText.className = 'diagnosis-label orange-text';
        } else {
            labelText.className = 'diagnosis-label red-text';
        }
        
        resultCard.style.display = 'block';
        
        // Render Saliency attributions using Chart.js
        renderSaliencyChart(data.band_saliency);
        
    } catch (err) {
        console.error(err);
    } finally {
        btnRunClassify.disabled = false;
        btnRunClassify.innerText = 'Scan Leaf Tissue';
    }
});

function renderSaliencyChart(saliencyData) {
    const ctx = document.getElementById('saliency-chart').getContext('2d');
    
    if (window.saliencyChartInstance) {
        window.saliencyChartInstance.destroy();
    }

    const labels = Object.keys(saliencyData);
    const values = Object.values(saliencyData);

    window.saliencyChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Relative Saliency Gradient (%)',
                data: values,
                backgroundColor: [
                    'rgba(59, 130, 246, 0.45)', // Blue
                    'rgba(16, 185, 129, 0.45)', // Green
                    'rgba(239, 68, 68, 0.45)',  // Red
                    'rgba(20, 184, 166, 0.7)',   // Red-Edge (glowing teal)
                    'rgba(20, 184, 166, 0.85)'  // NIR (glowing teal)
                ],
                borderColor: [
                    '#3b82f6', '#10b981', '#ef4444', '#14b8a6', '#14b8a6'
                ],
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc', font: { family: 'Outfit', weight: '600', size: 12 } }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}


// --- DEEP LEARNING CONTROL ROOM & CONVERGENCE SIMULATION (TAB 5) ---
const btnRunTraining = document.getElementById('btn-run-training');

btnRunTraining.addEventListener('click', async () => {
    btnRunTraining.disabled = true;
    btnRunTraining.innerHTML = '<i class="lucide-refresh-cw spin"></i> Executing training epochs...';
    
    try {
        const response = await fetch('/api/train-history');
        if (!response.ok) throw new Error("Train History API failed");
        
        const data = await response.json();
        
        // Show the statistics box
        document.getElementById('training-stats').style.display = 'block';
        document.getElementById('final-accuracy-val').innerText = data.final_accuracy + '%';
        
        // Render charts dynamically with animation
        renderTrainingCurves(data);
        
    } catch (err) {
        console.error(err);
    } finally {
        btnRunTraining.disabled = false;
        btnRunTraining.innerHTML = '<i data-lucide="play"></i> Simulate Training Epochs';
        lucide.createIcons();
    }
});

function renderTrainingCurves(data) {
    const accCtx = document.getElementById('accuracy-curve-chart').getContext('2d');
    const lossCtx = document.getElementById('loss-curve-chart').getContext('2d');
    
    if (window.accCurveChartInstance) window.accCurveChartInstance.destroy();
    if (window.lossCurveChartInstance) window.lossCurveChartInstance.destroy();

    // Accuracy Convergence Chart
    window.accCurveChartInstance = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [
                {
                    label: 'Validation Accuracy',
                    data: data.accuracy,
                    borderColor: '#14b8a6',
                    borderWidth: 3,
                    backgroundColor: 'rgba(20, 184, 166, 0.15)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Target Accuracy (>95%)',
                    data: Array(data.epochs.length).fill(95),
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [6, 6],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    max: 100,
                    min: 0,
                    title: { display: true, text: 'Percentage (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });

    // Loss Decay Chart
    window.lossCurveChartInstance = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: data.epochs,
            datasets: [{
                label: 'Training Loss',
                data: data.loss,
                borderColor: '#f97316',
                borderWidth: 3,
                backgroundColor: 'rgba(249, 115, 22, 0.15)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Epoch', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'Cross-Entropy Value', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });
}

// Get Current Geolocation Lock from User Browser
function initGeolocation() {
    const gpsLockElement = document.getElementById('gps-lock');
    if (!gpsLockElement) return;
    
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude.toFixed(4);
                const lon = position.coords.longitude.toFixed(4);
                gpsLockElement.querySelector('span').innerText = `GPS: LOCK ${lat}, ${lon}`;
            },
            (error) => {
                // Fallback to Delhi, India coordinates (+05:30 Standard Time)
                gpsLockElement.querySelector('span').innerText = "GPS: LOCK 28.6139, 77.2090 (DELHI)";
            },
            { enableHighAccuracy: true, timeout: 6000, maximumAge: 0 }
        );
    } else {
        gpsLockElement.querySelector('span').innerText = "GPS: LOCK 28.6139, 77.2090 (DELHI)";
    }
}

// Automatically Boot Up the Telemetry Loop on Page Load
window.addEventListener('DOMContentLoaded', () => {
    initFlightTelemetryLoop();
    initGeolocation();
    
    // Trigger default views for other tabs, so that maps & defaults aren't blank
    setTimeout(() => {
        // Run a spectral mapping default so elements look loaded
        runSpectralAnalysis();
    }, 400);
});
