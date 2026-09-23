let session = null;
let classMapping = {};
let chartInstance = null;

const statusEl = document.getElementById('status');
const audioInput = document.getElementById('audio-input');
const audioPlayer = document.getElementById('audio-player');
const resultsSection = document.getElementById('results');
const topSpeciesEl = document.getElementById('top-species');
const topConfidenceEl = document.getElementById('top-confidence');

// 1. Initialize ONNX Runtime Session and Fetch Class Mappings
async function init() {
    try {
        statusEl.innerText = "Loading ONNX model...";
        
        ort.env.wasm.numThreads = 2;
        session = await ort.InferenceSession.create('./models/bird_classifier_efficientnet.onnx', {
            executionProviders: ['webgl', 'wasm']
        });

        statusEl.innerText = "Loading species class mapping...";
        const response = await fetch('./models/species_class_mapping.json');
        classMapping = await response.json();

        statusEl.innerText = "Ready! Upload an audio file to classify.";
        audioInput.disabled = false;
    } catch (err) {
        console.error("Initialization failed:", err);
        statusEl.innerText = `Error loading model/mapping: ${err.message}`;
        statusEl.style.borderLeftColor = "#e74c3c";
        statusEl.style.backgroundColor = "#fdf2f2";
    }
}

// 2. Helper Functions for Mel Filterbank Calculation (PyTorch / torchaudio Parity)
function hzToMel(hz) {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
}

function melToHz(mel) {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
}

// Generates a [numMels, fftSize / 2 + 1] filterbank matrix matching torchaudio
function createMelFilterbank(numMels, fftSize, sampleRate, fMin = 0, fMax = null) {
    if (!fMax) fMax = sampleRate / 2;
    const numFftBins = Math.floor(fftSize / 2) + 1;
    
    const minMel = hzToMel(fMin);
    const maxMel = hzToMel(fMax);
    
    const melPoints = new Float32Array(numMels + 2);
    for (let i = 0; i < numMels + 2; i++) {
        melPoints[i] = minMel + (i / (numMels + 1)) * (maxMel - minMel);
    }
    
    const hzPoints = melPoints.map(melToHz);
    const binPoints = hzPoints.map(hz => Math.floor(((fftSize + 1) * hz) / sampleRate));
    
    const filterbank = Array.from({ length: numMels }, () => new Float32Array(numFftBins));
    
    for (let m = 1; m <= numMels; m++) {
        const fPrev = binPoints[m - 1];
        const fCurr = binPoints[m];
        const fNext = binPoints[m + 1];
        
        for (let k = fPrev; k < fCurr; k++) {
            if (k < numFftBins) {
                filterbank[m - 1][k] = (k - fPrev) / (fCurr - fPrev || 1);
            }
        }
        for (let k = fCurr; k < fNext; k++) {
            if (k < numFftBins) {
                filterbank[m - 1][k] = (fNext - k) / (fNext - fCurr || 1);
            }
        }
    }
    return filterbank;
}

// 3. Audio File Upload Listener
audioInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.style.display = 'block';

    statusEl.innerText = "Extracting Mel-Spectrogram...";

    try {
        const audioBuffer = await decodeAudioFile(file);
        const spectrogramTensor = extractMelSpectrogramTensor(audioBuffer);
        
        statusEl.innerText = "Running ONNX model inference...";
        await runInference(spectrogramTensor);
        
        statusEl.innerText = "Classification complete!";
    } catch (err) {
        console.error("Processing failed:", err);
        statusEl.innerText = `Error processing audio: ${err.message}`;
    }
});

// Decodes audio file and resamples to 32 kHz (Nyquist-Shannon target rate)
async function decodeAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 32000 });
    return await audioCtx.decodeAudioData(arrayBuffer);
}

// Extract exact [1, 1, 128, 313] Log-Mel-Spectrogram Tensor with Z-Score Normalization
function extractMelSpectrogramTensor(audioBuffer) {
    const pcmData = audioBuffer.getChannelData(0); // Mono channel mix-down
    const sampleRate = audioBuffer.sampleRate;
    
    // Select highest-energy 5-second window (RMS search)
    const windowSamples = sampleRate * 5;
    let startSample = 0;

    if (pcmData.length > windowSamples) {
        let maxEnergy = 0;
        let step = Math.floor(sampleRate / 2);
        for (let i = 0; i <= pcmData.length - windowSamples; i += step) {
            let energy = 0;
            for (let j = i; j < i + windowSamples; j += 100) {
                energy += pcmData[j] * pcmData[j];
            }
            if (energy > maxEnergy) {
                maxEnergy = energy;
                startSample = i;
            }
        }
    }

    const segment = pcmData.slice(startSample, startSample + windowSamples);

    const fftSize = 1024;
    const timeFrames = 313;
    const numMels = 128;
    const hopSize = 512; // Matches HOP_LENGTH_SAMPLES = 512 in Cell 3
    const numFftBins = Math.floor(fftSize / 2) + 1;
    
    const melFilterbank = createMelFilterbank(numMels, fftSize, sampleRate);
    const rawSpectrogram = new Float32Array(numMels * timeFrames);

    Meyda.bufferSize = fftSize;
    Meyda.sampleRate = sampleRate;

    for (let frame = 0; frame < timeFrames; frame++) {
        const frameOffset = frame * hopSize;
        const frameBuffer = segment.slice(frameOffset, frameOffset + fftSize);
        
        if (frameBuffer.length === fftSize) {
            const powerSpec = Meyda.extract('powerSpectrum', frameBuffer);
            
            if (powerSpec) {
                for (let mel = 0; mel < numMels; mel++) {
                    let melEnergy = 0.0;
                    for (let k = 0; k < numFftBins; k++) {
                        melEnergy += (powerSpec[k] || 0.0) * melFilterbank[mel][k];
                    }
                    
                    // AmplitudeToDB transform matching torchaudio
                    const db = 10.0 * Math.log10(Math.max(1e-10, melEnergy));
                    rawSpectrogram[mel * timeFrames + frame] = db;
                }
            }
        }
    }

    // Step 4 from Cell 3: Z-Score Normalization (mean = 0, std = 1)
    let sum = 0;
    for (let i = 0; i < rawSpectrogram.length; i++) {
        sum += rawSpectrogram[i];
    }
    const mean = sum / rawSpectrogram.length;

    let squareSum = 0;
    for (let i = 0; i < rawSpectrogram.length; i++) {
        const diff = rawSpectrogram[i] - mean;
        squareSum += diff * diff;
    }
    const std = Math.sqrt(squareSum / rawSpectrogram.length) + 1e-6;

    const float32Data = new Float32Array(1 * 1 * numMels * timeFrames);
    for (let i = 0; i < rawSpectrogram.length; i++) {
        float32Data[i] = (rawSpectrogram[i] - mean) / std;
    }

    return new ort.Tensor('float32', float32Data, [1, 1, numMels, timeFrames]);
}

// 4. Run Model Inference & Render Results
async function runInference(inputTensor) {
    const feeds = { input_spectrogram: inputTensor };
    const results = await session.run(feeds);
    const logits = results.species_logits.data;

    const probabilities = softmax(Array.from(logits));

    const indexedProbs = probabilities.map((prob, idx) => ({ prob, idx }));
    indexedProbs.sort((a, b) => b.prob - a.prob);
    const top5 = indexedProbs.slice(0, 5);

    const topMatch = top5[0];
    const rawMapping = classMapping[topMatch.idx];
    
    const topSpeciesName = typeof rawMapping === 'object' ? (rawMapping.sv || rawMapping.en || rawMapping.code) : (rawMapping || `Species #${topMatch.idx}`);
    
    topSpeciesEl.innerText = topSpeciesName;
    topConfidenceEl.innerText = `${(topMatch.prob * 100).toFixed(2)}%`;

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
    const labels = top5.map(item => {
        const m = classMapping[item.idx];
        return typeof m === 'object' ? (m.sv || m.en || m.code) : (m || `#${item.idx}`);
    });
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

init();