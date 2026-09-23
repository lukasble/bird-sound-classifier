let session = null;
let classMapping = {};
let chartInstance = null;

// DOM Elements
const statusEl = document.getElementById('status');
const audioInput = document.getElementById('audio-input');
const audioPlayer = document.getElementById('audio-player');
const resultsSection = document.getElementById('results');
const topSpeciesEl = document.getElementById('top-species');
const topConfidenceEl = document.getElementById('top-confidence');

// 1. Initialize ONNX Session and Load Species Mapping JSON
async function init() {
    try {
        statusEl.innerText = "Loading ONNX model...";
        
        // Configure ONNX Runtime to use WebGL or Wasm
        ort.env.wasm.numThreads = 2;
        session = await ort.InferenceSession.create('./models/bird_classifier_efficientnet.onnx', {
            executionProviders: ['webgl', 'wasm']
        });

        statusEl.innerText = "Loading species class mapping...";
        const response = await fetch('./models/species_class_mapping.json');
        classMapping = await response.json();

        statusEl.innerText = "Ready! Upload an audio recording to classify.";
        audioInput.disabled = false;
    } catch (err) {
        console.error("Initialization failed:", err);
        statusEl.innerText = `Error loading model/mapping: ${err.message}`;
        statusEl.style.borderLeftColor = "#e74c3c";
        statusEl.style.backgroundColor = "#fdf2f2";
    }
}

// 2. Audio Processing via Web Audio API
audioInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Set audio player source
    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.style.display = 'block';

    statusEl.innerText = "Processing audio spectrogram...";

    try {
        const audioBuffer = await decodeAudioFile(file);
        const spectrogramTensor = extractSpectrogramTensor(audioBuffer);
        
        statusEl.innerText = "Running ONNX inference in browser...";
        await runInference(spectrogramTensor);
        
        statusEl.innerText = "Classification complete!";
    } catch (err) {
        console.error("Processing failed:", err);
        statusEl.innerText = `Error processing audio: ${err.message}`;
    }
});

// Helper: Decode Audio File into AudioBuffer
async function decodeAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    return await audioCtx.decodeAudioData(arrayBuffer);
}

// Helper: Format raw PCM audio into [1, 1, 128, 313] Tensor shape expected by EfficientNet
function extractSpectrogramTensor(audioBuffer) {
    const pcmData = audioBuffer.getChannelData(0); // Mono channel
    
    // Create a dummy spectrogram buffer matching input dimensions [1, 1, 128, 313]
    // Note: For exact feature extraction parity, you can link a Web Audio Analyser node 
    // or a lightweight JS FFT/Librosa spectrogram package.
    const numBins = 128;
    const timeFrames = 313;
    const float32Data = new Float32Array(1 * 1 * numBins * timeFrames);

    // Populate tensor buffer with normalized signal amplitude frames
    const step = Math.floor(pcmData.length / (numBins * timeFrames)) || 1;
    for (let i = 0; i < float32Data.length; i++) {
        const pcmIndex = (i * step) % pcmData.length;
        float32Data[i] = pcmData[pcmIndex] || 0.0;
    }

    return new ort.Tensor('float32', float32Data, [1, 1, numBins, timeFrames]);
}

// 3. Execute Inference & Render Results
async function runInference(inputTensor) {
    const feeds = { input_spectrogram: inputTensor };
    const results = await session.run(feeds);
    const logits = results.species_logits.data;

    // Apply Softmax to Logits
    const probabilities = softmax(Array.from(logits));

    // Get Top 5 Predictions
    const indexedProbs = probabilities.map((prob, idx) => ({ prob, idx }));
    indexedProbs.sort((a, b) => b.prob - a.prob);
    const top5 = indexedProbs.slice(0, 5);

    // Display Top Match
    const topMatch = top5[0];
    const topSpeciesName = classMapping[topMatch.idx] || `Species #${topMatch.idx}`;
    
    topSpeciesEl.innerText = topSpeciesName;
    topConfidenceEl.innerText = `${(topMatch.prob * 100).toFixed(2)}%`;

    // Render Chart
    renderChart(top5);
    resultsSection.style.display = 'block';
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
}

function renderChart(top5) {
    const labels = top5.map(item => classMapping[item.idx] || `Species #${item.idx}`);
    const data = top5.map(item => (item.prob * 100).toFixed(2));

    const ctx = document.getElementById('confidence-chart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });
}

// Start app
init();