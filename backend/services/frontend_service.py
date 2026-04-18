import re
from pathlib import Path


RUNTIME_SCRIPT = """<script>
  const CLASS_ORDER = ['Normal', 'Osteopenia', 'Osteoporosis'];
  const CLASS_INFO = {
    Normal: {
      cls: 'normal',
      badge: '\\u2705',
      desc: 'Bone density appears within normal range. No significant degenerative changes detected.',
      recIcon: '\\u2695',
      ringColor: '#00e89c',
    },
    Osteopenia: {
      cls: 'osteopenia',
      badge: '\\u26A0',
      desc: 'Mild reduction in bone mineral density detected.',
      recIcon: '\\u26A0',
      ringColor: '#ffc14d',
    },
    Osteoporosis: {
      cls: 'osteoporosis',
      badge: '!!',
      desc: 'Significant bone density loss detected.',
      recIcon: '\\u2695',
      ringColor: '#ff5f6d',
    },
  };

  const fileInput = document.getElementById('file-input');
  const uploadZone = document.getElementById('upload-zone');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg = document.getElementById('preview-img');
  const previewLabel = document.getElementById('preview-label');
  const analyseBtn = document.getElementById('analyse-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnIcon = document.getElementById('btn-icon');
  const errorMsg = document.getElementById('error-msg');
  const formSection = document.getElementById('form-section');
  const resultsPanel = document.getElementById('results-panel');
  const scanOverlay = document.getElementById('scan-overlay');
  const scanMsg = document.getElementById('scan-msg');
  const resetBtn = document.getElementById('reset-btn');

  let uploadedFile = null;

  function getApiBaseUrl() {
    return window.location.protocol === 'file:'
      ? 'http://127.0.0.1:5000'
      : window.location.origin;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[char]));
  }

  function isReadyToAnalyze() {
    return Boolean(
      uploadedFile &&
      document.getElementById('name').value.trim() &&
      document.getElementById('age').value &&
      document.getElementById('sex').value &&
      document.getElementById('weight').value
    );
  }

  function setLoadingState(isLoading) {
    analyseBtn.disabled = isLoading || !isReadyToAnalyze();
    btnSpinner.style.display = isLoading ? 'block' : 'none';
    btnIcon.style.display = isLoading ? 'none' : 'block';
  }

  function buildPayload() {
    const payload = new FormData();
    ['name', 'age', 'sex', 'weight', 'notes'].forEach((id) => {
      payload.append(id, document.getElementById(id).value);
    });
    payload.append('image', uploadedFile, uploadedFile.name);
    return payload;
  }

  function setStepActive(step) {
    for (let i = 1; i <= 4; i++) {
      const circle = document.getElementById(`step-${i}-circle`);
      const label = document.getElementById(`step-${i}-label`);
      if (i < step) {
        circle.className = 'step-circle done';
        label.className = 'step-label done';
        circle.textContent = '\\u2713';
        const line = document.getElementById(`line-${i}`);
        if (line) line.classList.add('done');
      } else if (i === step) {
        circle.className = 'step-circle active';
        label.className = 'step-label active';
        circle.textContent = i;
      } else {
        circle.className = 'step-circle';
        label.className = 'step-label';
        circle.textContent = i;
        const line = document.getElementById(`line-${i}`);
        if (line) line.classList.remove('done');
      }
    }
  }

  function updateBtnState() {
    analyseBtn.disabled = !isReadyToAnalyze();
  }

  function validate() {
    const name = document.getElementById('name').value.trim();
    const age = Number.parseInt(document.getElementById('age').value, 10);
    const sex = document.getElementById('sex').value;
    const weight = Number.parseFloat(document.getElementById('weight').value);

    if (!name) return 'Please enter the patient name.';
    if (!age || age < 1 || age > 120) return 'Please enter a valid age between 1 and 120.';
    if (!sex) return 'Please select biological sex.';
    if (!weight || weight <= 0 || weight > 300) return 'Please enter a valid weight between 1 and 300 kg.';
    if (!uploadedFile) return 'Please upload a knee X-ray image.';
    return null;
  }

  function handleFile(file) {
    uploadedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewWrap.style.display = 'block';
    previewLabel.textContent = file.name.toUpperCase() + ' | READY FOR ANALYSIS';
    updateBtnState();
    setStepActive(2);
  }

  function getProbabilities(result) {
    if (Array.isArray(result.probs)) {
      return result.probs;
    }
    const probabilityMap = (result.prediction && result.prediction.probabilities) || {};
    return CLASS_ORDER.map((label) => Number(probabilityMap[label] || 0));
  }

  function displayResults(result) {
    const label = result.label || (result.prediction && result.prediction.label) || 'Normal';
    const confidence = Number(result.confidence ?? (result.prediction && result.prediction.confidence) ?? 0);
    const probs = getProbabilities(result);
    const info = CLASS_INFO[label] || CLASS_INFO.Normal;
    const patient = result.patient || {};
    const description = (result.prediction && result.prediction.description) || result.info || info.desc;
    const recommendation = (result.prediction && result.prediction.recommendation) || result.recommendation || 'Please consult a qualified clinician.';
    const probIds = ['normal', 'osteopenia', 'osteoporosis'];

    document.getElementById('r-name').textContent = patient.name || document.getElementById('name').value.trim();
    document.getElementById('r-age').textContent = String(patient.age || document.getElementById('age').value) + ' yrs';
    document.getElementById('r-sex').textContent = patient.sex || document.getElementById('sex').value;
    document.getElementById('r-weight').textContent = String(patient.weight || document.getElementById('weight').value) + ' kg';

    const hero = document.getElementById('result-hero');
    const badge = document.getElementById('result-badge');
    hero.className = 'result-hero ' + info.cls;
    badge.className = 'result-badge ' + info.cls;
    badge.textContent = info.badge;

    document.getElementById('result-name').textContent = label;
    document.getElementById('result-name').className = 'result-name ' + info.cls;
    document.getElementById('result-desc').textContent = description;

    const pct = Math.round(confidence * 100);
    const circumference = 175.9;
    const offset = circumference - (pct / 100) * circumference;
    const ring = document.getElementById('ring-fill');
    ring.style.stroke = info.ringColor;
    ring.style.strokeDashoffset = circumference;
    document.getElementById('conf-pct').textContent = pct + '%';
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 100);

    const rec = document.getElementById('recommendation');
    rec.className = 'recommendation ' + info.cls;
    document.getElementById('rec-icon').textContent = info.recIcon;
    document.getElementById('rec-text').innerHTML = '<strong>Clinical Recommendation.</strong> ' + escapeHtml(recommendation);

    probs.forEach((value, index) => {
      const width = (Number(value) * 100).toFixed(1) + '%';
      setTimeout(() => {
        document.getElementById('bar-' + probIds[index]).style.width = width;
        document.getElementById('pct-' + probIds[index]).textContent = width;
      }, 200 + (index * 100));
    });

    document.getElementById('result-img').src = previewImg.src;

    setStepActive(4);
    formSection.style.display = 'none';
    resultsPanel.style.display = 'block';
    resultsPanel.classList.add('fade-in');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) handleFile(file);
  });

  uploadZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadZone.classList.add('drag-over');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
  });

  uploadZone.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  ['name', 'age', 'sex', 'weight'].forEach((id) => {
    document.getElementById(id).addEventListener('input', updateBtnState);
    document.getElementById(id).addEventListener('change', updateBtnState);
  });

  analyseBtn.addEventListener('click', async () => {
    errorMsg.style.display = 'none';
    const errorText = validate();
    if (errorText) {
      errorMsg.textContent = errorText;
      errorMsg.style.display = 'block';
      return;
    }

    setStepActive(3);
    scanOverlay.classList.add('active');
    setLoadingState(true);

    const messages = ['LOADING MODEL', 'PREPROCESSING IMAGE', 'RUNNING DENSENET121', 'CLASSIFYING FEATURES', 'GENERATING REPORT'];
    let messageIndex = 0;
    const msgInterval = setInterval(() => {
      scanMsg.textContent = messages[Math.min(messageIndex++, messages.length - 1)];
    }, 700);

    try {
      const response = await fetch(getApiBaseUrl() + '/api/analyze', {
        method: 'POST',
        body: buildPayload(),
      });
      const result = await response.json();

      if (!response.ok || result.error) {
        throw new Error(result.error || 'Analysis failed. Please try again.');
      }

      displayResults(result);
    } catch (error) {
      setStepActive(2);
      errorMsg.textContent = error.message || 'Unable to reach the backend service.';
      errorMsg.style.display = 'block';
    } finally {
      clearInterval(msgInterval);
      scanOverlay.classList.remove('active');
      setLoadingState(false);
    }
  });

  resetBtn.addEventListener('click', () => {
    uploadedFile = null;
    fileInput.value = '';
    previewWrap.style.display = 'none';
    previewImg.src = '';
    ['name', 'age', 'sex', 'weight', 'notes'].forEach((id) => {
      const element = document.getElementById(id);
      if (element.tagName === 'SELECT') {
        element.selectedIndex = 0;
      } else {
        element.value = '';
      }
    });
    errorMsg.style.display = 'none';
    btnSpinner.style.display = 'none';
    btnIcon.style.display = 'block';
    document.getElementById('ring-fill').style.strokeDashoffset = 175.9;

    ['normal', 'osteopenia', 'osteoporosis'].forEach((id) => {
      document.getElementById('bar-' + id).style.width = '0%';
      document.getElementById('pct-' + id).textContent = '0.0%';
    });

    setStepActive(1);
    updateBtnState();
    resultsPanel.style.display = 'none';
    formSection.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  updateBtnState();
</script>
"""


def render_frontend(frontend_path: Path) -> str:
    html = frontend_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"<script>[\s\S]*?</script>(\s*</body>\s*</html>\s*)$", re.IGNORECASE)
    updated_html, replacements = pattern.subn(
        lambda match: RUNTIME_SCRIPT + match.group(1),
        html,
        count=1,
    )
    if replacements == 0:
        return html
    return updated_html
