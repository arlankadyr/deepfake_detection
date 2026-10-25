(() => {
  const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
  const modelSelect = document.getElementById('model-select');
  const sizeSelect = document.getElementById('size-select');
  const seedSelect = document.getElementById('seed-select');
  const fileInput = document.getElementById('image-file');
  const dropzone = document.getElementById('dropzone');
  const formError = document.getElementById('form-error');
  const analyseButton = document.getElementById('analyse-button');
  const emptyResult = document.getElementById('empty-result');
  const predictionGrid = document.getElementById('prediction-grid');
  const resultState = document.getElementById('result-state');
  const checkpointStatus = document.getElementById('checkpoint-status');
  let modelData = [];
  let selectedFile = null;
  let previewUrl = null;

  const modelLabels = { resnet50: 'ResNet-50', deit_small: 'DeiT-S' };

  function combinationsForSelection() {
    const chosen = modelSelect.value === 'both'
      ? modelData
      : modelData.filter((item) => item.model === modelSelect.value);
    if (!chosen.length) return [];
    const allRuns = chosen.map((item) => new Set(item.available_runs.map((run) => `${run.train_size}:${run.seed}`)));
    return chosen[0].available_runs
      .filter((run) => allRuns.every((runs) => runs.has(`${run.train_size}:${run.seed}`)))
      .sort((left, right) => right.train_size - left.train_size || left.seed - right.seed);
  }

  function fillSelect(select, entries, placeholder, selectedValue) {
    select.replaceChildren();
    if (!entries.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = placeholder;
      select.append(option);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    entries.forEach(({ value, label }) => {
      const option = document.createElement('option');
      option.value = String(value);
      option.textContent = label;
      select.append(option);
    });
    if (selectedValue != null && entries.some((entry) => String(entry.value) === String(selectedValue))) {
      select.value = String(selectedValue);
    }
  }

  function updateRunOptions() {
    const oldSize = sizeSelect.value;
    const oldSeed = seedSelect.value;
    const runs = combinationsForSelection();
    const sizes = [...new Set(runs.map((run) => run.train_size))].sort((a, b) => b - a);
    fillSelect(sizeSelect, sizes.map((value) => ({ value, label: value.toLocaleString('en-US') })), 'No runs available', oldSize);
    updateSeedOptions(runs, oldSeed);
    const hasRuns = runs.length > 0;
    analyseButton.disabled = !hasRuns;
    if (!hasRuns) {
      checkpointStatus.textContent = 'Model not trained yet — no checkpoint is available for this model selection.';
    } else {
      const count = runs.length;
      checkpointStatus.textContent = `${count} available model run${count === 1 ? '' : 's'} for this selection`;
    }
  }

  function updateSeedOptions(runs, preferredSeed) {
    if (!runs) runs = combinationsForSelection();
    const size = Number(sizeSelect.value);
    const seeds = [...new Set(runs.filter((run) => run.train_size === size).map((run) => run.seed))].sort((a, b) => a - b);
    fillSelect(seedSelect, seeds.map((value) => ({ value, label: String(value) })), '—', preferredSeed);
  }

  async function loadModels() {
    try {
      const response = await fetch('/api/models');
      if (!response.ok) throw new Error('Model availability could not be read.');
      const data = await response.json();
      modelData = data.models || [];
      const allRuns = modelData.flatMap((item) => item.available_runs);
      checkpointStatus.textContent = allRuns.length
        ? `${allRuns.length} trained checkpoint${allRuns.length === 1 ? '' : 's'} found`
        : 'Model not trained yet — no checkpoints found.';
      updateRunOptions();
      if (allRuns.length) checkpointStatus.textContent = `${combinationsForSelection().length} available model run${combinationsForSelection().length === 1 ? '' : 's'} for this selection`;
    } catch (error) {
      checkpointStatus.textContent = 'Checkpoint availability could not be loaded.';
      showError(error.message || 'Checkpoint availability could not be loaded.');
    }
  }

  function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function clearError() {
    formError.textContent = '';
    formError.hidden = true;
  }

  function displayFile(file) {
    clearError();
    if (!file) return;
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      showError('Choose a JPEG, PNG, or WebP image.');
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      showError('Image uploads must be 10 MB or smaller.');
      return;
    }
    if (file.size === 0) {
      showError('The selected image is empty or corrupted.');
      return;
    }

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    const preview = document.getElementById('image-preview');
    preview.src = previewUrl;
    preview.onload = () => {
      document.getElementById('preview-wrap').hidden = false;
      dropzone.classList.add('has-preview');
      document.getElementById('dropzone-title').textContent = 'Image selected';
      document.getElementById('upload-help').textContent = 'Choose another image to replace it';
    };
    preview.onerror = () => {
      clearFile();
      showError('The selected image is corrupted or cannot be previewed.');
    };
    document.getElementById('selected-file').hidden = false;
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  }

  function clearFile() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    selectedFile = null;
    fileInput.value = '';
    document.getElementById('image-preview').removeAttribute('src');
    document.getElementById('preview-wrap').hidden = true;
    document.getElementById('selected-file').hidden = true;
    document.getElementById('dropzone-title').textContent = 'Drop an image here';
    document.getElementById('upload-help').innerHTML = 'or <span class="inline-link">browse files</span>';
    dropzone.classList.remove('has-preview');
  }

  function errorMessage(detail) {
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return 'Check the selected model, training-set size, and seed, then try again.';
    return 'The analysis could not be completed. Please try again.';
  }

  function makePredictionCard(prediction) {
    const card = document.createElement('article');
    card.className = 'prediction-card';

    const header = document.createElement('div');
    header.className = 'prediction-card-header';
    const name = document.createElement('h3');
    name.textContent = modelLabels[prediction.model] || prediction.model;
    const badge = document.createElement('span');
    badge.className = `label-pill ${prediction.label === 'Real' ? 'real' : 'fake'}`;
    badge.textContent = prediction.label;
    header.append(name, badge);

    const probabilityRow = document.createElement('div');
    probabilityRow.className = 'probability-row';
    const value = document.createElement('strong');
    value.className = 'probability-value';
    value.textContent = `${(prediction.fake_probability * 100).toFixed(1)}%`;
    const caption = document.createElement('span');
    caption.className = 'probability-caption';
    caption.textContent = 'fake probability';
    probabilityRow.append(value, caption);

    const track = document.createElement('div');
    track.className = 'progress-track';
    track.setAttribute('role', 'progressbar');
    track.setAttribute('aria-label', `${name.textContent} fake probability`);
    track.setAttribute('aria-valuemin', '0');
    track.setAttribute('aria-valuemax', '100');
    track.setAttribute('aria-valuenow', String((prediction.fake_probability * 100).toFixed(1)));
    const fill = document.createElement('div');
    fill.className = 'progress-fill';
    fill.style.width = `${Math.max(0, Math.min(100, prediction.fake_probability * 100))}%`;
    track.append(fill);

    const timing = document.createElement('div');
    timing.className = 'inference-time';
    const timeLabel = document.createElement('span');
    timeLabel.textContent = 'Inference time';
    const timeValue = document.createElement('strong');
    timeValue.textContent = `${prediction.inference_ms.toFixed(1)} ms`;
    timing.append(timeLabel, timeValue);
    card.append(header, probabilityRow, track, timing);
    return card;
  }

  async function analyse() {
    clearError();
    if (!selectedFile) {
      showError('Choose an image before starting the analysis.');
      return;
    }
    const payload = new FormData();
    payload.append('file', selectedFile, selectedFile.name);
    payload.append('model', modelSelect.value);
    if (sizeSelect.value) payload.append('train_size', sizeSelect.value);
    if (seedSelect.value) payload.append('seed', seedSelect.value);

    analyseButton.disabled = true;
    analyseButton.classList.add('is-loading');
    resultState.textContent = 'Analysing…';
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: payload });
      const data = await response.json();
      if (!response.ok) throw new Error(errorMessage(data.detail));
      predictionGrid.replaceChildren(...data.predictions.map(makePredictionCard));
      emptyResult.hidden = true;
      predictionGrid.hidden = false;
      document.getElementById('disagreement-banner').hidden = data.models_agree || data.predictions.length < 2;
      resultState.textContent = 'Analysis complete';
    } catch (error) {
      resultState.textContent = 'Analysis unavailable';
      showError(error.message || 'The analysis could not be completed. Please try again.');
    } finally {
      analyseButton.classList.remove('is-loading');
      analyseButton.disabled = combinationsForSelection().length === 0;
    }
  }

  modelSelect.addEventListener('change', updateRunOptions);
  sizeSelect.addEventListener('change', () => updateSeedOptions(combinationsForSelection()));
  fileInput.addEventListener('change', (event) => displayFile(event.target.files[0]));
  document.getElementById('clear-file').addEventListener('click', () => { clearFile(); clearError(); });
  analyseButton.addEventListener('click', analyse);

  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('drag-active'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-active'));
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('drag-active');
    displayFile(event.dataTransfer.files[0]);
  });
  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); }
  });

  loadModels();
})();
