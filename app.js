let session = null;
let classMapping = {};
let chartInstance = null;

const statusEl = document.getElementById('status');
const audioInput = document.getElementById('audio-input');
const audioPlayer = document.getElementById('audio-player');
const resultsSection = document.getElementById('results');
const topSpeciesEl = document.getElementById('top-species');
const topConfidenceEl = document.getElementById('top-confidence');

// 1. Initiera ONNX och Hämta Class Mapping
async function init() {
    try {
        statusEl.innerText = "Laddar ONNX-modell...";
        
        ort.env.wasm.numThreads = 2;
        session = await ort.InferenceSession.create('./models/bird_classifier_efficientnet.onnx', {
            executionProviders: ['webgl', 'wasm']
        });

        statusEl.innerText = "Laddar artmappning...";
        const response = await fetch('./models/species_class_mapping.json');
        classMapping = await response.json();

        statusEl.innerText = "Redo! Ladda upp en ljudfil för att klassificera.";
        audioInput.disabled = false;
    } catch (err) {
        console.error("Initialization failed:", err);
        statusEl.innerText = `Fel vid laddning: ${err.message}`;
        statusEl.style.borderLeftColor = "#e74c3c";
        statusEl.style.backgroundColor = "#fdf2f2";
    }
}

// 2. Lyssna på filuppladdning
audioInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.style.display = 'block';

    statusEl.innerText = "Extraherar Mel-spektrogram från ljudfilen...";

    try {
        const audioBuffer = await decodeAudioFile(file);
        const spectrogramTensor = extractMelSpectrogramTensor(audioBuffer);
        
        statusEl.innerText = "Kör AI-modell i webbläsaren...";
        await runInference(spectrogramTensor);
        
        statusEl.innerText = "Klassificering klar!";
    } catch (err) {
        console.error("Processing failed:", err);
        statusEl.innerText = `Fel vid ljudbearbetning: ${err.message}`;
    }
});

// Avkoda ljudfil via Web Audio API
async function decodeAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 32000 });
    return await audioCtx.decodeAudioData(arrayBuffer);
}

// Extrahera exakt 1x128x313 Log-Mel-Spektrogram
function extractMelSpectrogramTensor(audioBuffer) {
    const pcmData = audioBuffer.getChannelData(0); // Mono
    const sampleRate = audioBuffer.sampleRate;
    
    // Välj det mest ljudintensiva 5-sekunderssegmentet (RMS)
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

    // Parametrar anpassade för 128 mel-band och 313 tidssteg
    const fftSize = 1024;
    const timeFrames = 313;
    const numBins = 128;
    const hopSize = Math.floor((segment.length - fftSize) / (timeFrames - 1));
    
    const float32Data = new Float32Array(1 * 1 * numBins * timeFrames);

    Meyda.bufferSize = fftSize;
    Meyda.sampleRate = sampleRate;

    for (let frame = 0; frame < timeFrames; frame++) {
        const frameOffset = frame * hopSize;
        const frameBuffer = segment.slice(frameOffset, frameOffset + fftSize);
        
        if (frameBuffer.length === fftSize) {
            const spec = Meyda.extract('powerSpectrum', frameBuffer);
            
            if (spec) {
                for (let mel = 0; mel < numBins; mel++) {
                    const binIdx = Math.floor((mel / numBins) * (spec.length / 2));
                    const val = spec[binIdx] || 1e-6;
                    // AmplitudeToDB / Log-Mel scaling
                    const logMel = Math.log10(Math.max(1e-6, val));
                    float32Data[mel * timeFrames + frame] = logMel;
                }
            }
        }
    }

    return new ort.Tensor('float32', float32Data, [1, 1, numBins, timeFrames]);
}

// 3. Kör inferens & Visa resultat
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
    
    // Stöd för antingen enkel sträng eller struktur med koder/namn
    const topSpeciesName = typeof rawMapping === 'object' ? (rawMapping.sv || rawMapping.en || rawMapping.code) : (rawMapping || `Art #${topMatch.idx}`);
    
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
                label: 'Konfidensgrad (%)',
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