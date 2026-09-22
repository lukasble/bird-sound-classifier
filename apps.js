let session = null;
let classMapping = null;

async function initModel() {
  const statusEl = document.getElementById('status');
  try {
    statusEl.innerText = "Laddar klassmappning...";
    const mappingRes = await fetch('./models/species_class_mapping.json');
    classMapping = await mappingRes.json();

    statusEl.innerText = "Laddar ONNX-modell...";
    session = await ort.InferenceSession.create('./models/bird_classifier_efficientnet.onnx', {
      executionProviders: ['wasm']
    });

    statusEl.innerText = " Modellen är redo! Välj en ljudfil ovan.";
  } catch (err) {
    console.error(err);
    statusEl.innerText = " Väntar på att modellfilerna ska läggas till i /models/...";
  }
}

initModel();