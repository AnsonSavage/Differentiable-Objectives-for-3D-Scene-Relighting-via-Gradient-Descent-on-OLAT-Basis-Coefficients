// DOM elements
const grid = document.getElementById('grid');
const statusEl = document.getElementById('status');
const form = document.getElementById('filter-form');
const subdirSelect = document.getElementById('subdir-select');
const sceneSelect = document.getElementById('scene-select');
const promptSelect = document.getElementById('prompt-select');
const lossModelSelect = document.getElementById('loss-model-select');
const refImageSelect = document.getElementById('ref-image-select');
const loadBtn = document.getElementById('load-btn');
const includeFirstCheckbox = document.getElementById('include-first-iter');
const groupByIterationCheckbox = document.getElementById('group-by-iteration');
const template = document.getElementById('image-card-template');
const tooltip = document.getElementById('tooltip');
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modal-image');
const modalCaption = document.getElementById('modal-caption');
const modalClose = document.getElementById('modal-close');
const copyPathBtn = document.getElementById('copy-path-btn');
const modalHeartBtn = document.getElementById('modal-heart-btn');
const modalStarBtn = document.getElementById('modal-star-btn');
const setupOverlay = document.getElementById('runs-dir-setup-overlay');
const setupForm = document.getElementById('runs-dir-form');
const runsDirInput = document.getElementById('runs-dir-input');
const setupError = document.getElementById('runs-dir-error');
const changeRunsDirBtn = document.getElementById('change-runs-dir-btn');
const cancelRunsDirBtn = document.getElementById('runs-dir-cancel-btn');
const resetRunsDirBtn = document.getElementById('runs-dir-reset-btn');
const currentRunsDirLabel = document.getElementById('current-runs-dir-label');

// State
let lastModalData = null;
let hearts = new Set();
let stars = new Set();

function updateRunsDirDisplay() {
  if (currentRunsDirLabel) {
    const isDefault = CONFIG.is_default ? ' (default)' : '';
    currentRunsDirLabel.textContent = `Runs: ${CONFIG.runs_dir_name || 'OPTIMIZATION_RUNS'}${isDefault}`;
    currentRunsDirLabel.title = CONFIG.runs_dir_absolute || '';
  }
}

function showRunsDirSetup(message = '') {
  if(!setupOverlay) return;
  if(setupError) setupError.textContent = message;
  if(runsDirInput) {
    runsDirInput.value = CONFIG.runs_dir_input || CONFIG.runs_dir_absolute || '';
  }
  setupOverlay.hidden = false;
  runsDirInput?.focus();
}

function hideRunsDirSetup() {
  if(!setupOverlay) return;
  setupOverlay.hidden = true;
  if(setupError) setupError.textContent = '';
}

function clearGalleryState() {
  grid.innerHTML = '';
  statusEl.textContent = '';
  sceneSelect.innerHTML = '<option value="" disabled selected>Select a directory first</option>';
  sceneSelect.disabled = true;
  promptSelect.innerHTML = '<option value="" disabled selected>Select a scene first</option>';
  promptSelect.disabled = true;
  lossModelSelect.innerHTML = '<option value="" disabled selected>Select a scene first</option>';
  lossModelSelect.disabled = true;
  refImageSelect.innerHTML = '<option value="" disabled selected>Select a scene first</option>';
  refImageSelect.disabled = true;
  loadBtn.disabled = true;
}

async function initializeGallery() {
  await loadHearts();
  await loadStars();
  await populateSubdirs();
}

function getIterationLabel(image) {
  return image.iteration >= 0 ? `Iter ${image.iteration}` : 'Iter ?';
}

function getOptLabel(optIndex) {
  return optIndex >= 0 ? `Opt ${optIndex}` : 'Opt ?';
}

function getRunLabel(runName) {
  return runName || 'Run';
}

function attachCardHandlers(card, img) {
  card.querySelector('.heart-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleHeart(img.id, card); });
  card.querySelector('.star-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleStar(img.id, card); });
  attachMetadata(card);
}

function renderFlatGallery(images) {
  const frag = document.createDocumentFragment();
  for(const img of images){
    const c = cardHTML(img, template, hearts, stars);
    attachCardHandlers(c, img);
    frag.appendChild(c);
  }
  return frag;
}

function renderGroupedGallery(images) {
  const byRun = new Map();
  for(const img of images) {
    const runName = img.run || 'Run';
    if(!byRun.has(runName)) byRun.set(runName, []);
    byRun.get(runName).push(img);
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'gallery-matrix';

  const sortedRuns = [...byRun.keys()].sort((a, b) => a.localeCompare(b));
  for(const runName of sortedRuns) {
    const runImages = byRun.get(runName);
    const columns = new Set();
    const grouped = new Map();

    for(const img of runImages) {
      const iteration = img.iteration ?? -1;
      const optIndex = img.opt_index ?? -1;
      columns.add(optIndex);
      if(!grouped.has(iteration)) grouped.set(iteration, new Map());
      grouped.get(iteration).set(optIndex, img);
    }

    const sortedIterations = [...grouped.keys()].sort((a, b) => a - b);
    const sortedColumns = [...columns].sort((a, b) => a - b);

    const section = document.createElement('section');
    section.className = 'gallery-run-block';

    const runHeader = document.createElement('div');
    runHeader.className = 'gallery-run-title';
    runHeader.textContent = getRunLabel(runName);
    runHeader.title = runName;
    section.appendChild(runHeader);

    const table = document.createElement('div');
    table.className = 'gallery-run-matrix';
    table.style.setProperty('--gallery-columns', String(sortedColumns.length));

    const header = document.createElement('div');
    header.className = 'gallery-matrix-row gallery-matrix-header';
    const corner = document.createElement('div');
    corner.className = 'gallery-matrix-label gallery-matrix-corner';
    corner.textContent = 'Iteration';
    header.appendChild(corner);
    for(const optIndex of sortedColumns) {
      const label = document.createElement('div');
      label.className = 'gallery-matrix-label';
      label.textContent = getOptLabel(optIndex);
      header.appendChild(label);
    }
    table.appendChild(header);

    for(const iteration of sortedIterations) {
      const row = document.createElement('div');
      row.className = 'gallery-matrix-row';
      const rowLabel = document.createElement('div');
      rowLabel.className = 'gallery-matrix-label gallery-matrix-row-label';
      rowLabel.textContent = getIterationLabel({ iteration });
      row.appendChild(rowLabel);

      const rowImages = grouped.get(iteration);
      for(const optIndex of sortedColumns) {
        const cell = document.createElement('div');
        cell.className = 'gallery-matrix-cell';
        const img = rowImages.get(optIndex);
        if(img) {
          const card = cardHTML(img, template, hearts, stars);
          attachCardHandlers(card, img);
          cell.appendChild(card);
        } else {
          const empty = document.createElement('div');
          empty.className = 'gallery-matrix-empty';
          empty.textContent = '—';
          cell.appendChild(empty);
        }
        row.appendChild(cell);
      }

      table.appendChild(row);
    }

    section.appendChild(table);
    wrapper.appendChild(section);
  }

  return wrapper;
}

async function loadHearts() {
  try { 
    const data = await fetchJSON('/api/hearts'); 
    hearts = new Set(data.images || []); 
  } catch(e) { 
    console.warn('Fav load failed', e); 
  }
}

async function loadStars() {
  try {
    const data = await fetchJSON('/api/stars');
    stars = new Set(data.images || []);
  } catch(e) {
    console.warn('Star load failed', e);
  }
}

async function toggleHeart(id, card) {
  const method = hearts.has(id) ? 'DELETE' : 'POST';
  try {
    const res = await fetch('/api/hearts', { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify({id}) });
    const data = await res.json();
    hearts = new Set(data.images || []);
    // Update card if provided
    if(card) {
      card.classList.toggle('hearted', hearts.has(id));
      const heartEl = card.querySelector('.heart-toggle');
      if(heartEl) heartEl.textContent = hearts.has(id) ? '♥' : '♡';
    }
  } catch(e) { console.error(e); }
}

async function toggleStar(id, card) {
  const method = stars.has(id) ? 'DELETE' : 'POST';
  try {
    const res = await fetch('/api/stars', { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify({id}) });
    const data = await res.json();
    stars = new Set(data.images || []);
    if(card) {
      card.classList.toggle('starred', stars.has(id));
      const starEl = card.querySelector('.star-toggle');
      if(starEl) starEl.textContent = stars.has(id) ? '★' : '☆';
    }
  } catch(e) { console.error(e); }
}

function showTooltip(x,y, meta) {
  tooltip.hidden = false;
  tooltip.style.left = (x+12)+'px';
  tooltip.style.top = (y+12)+'px';
  const escapeHtml = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const refPath = meta.reference_image_path ? escapeHtml(meta.reference_image_path) : '';
  const refThumb = meta.reference_image_path
    ? `<br><img class="ref-thumb" src="/api/reference_image?thumb=1&path=${encodeURIComponent(meta.reference_image_path)}" alt="reference" onerror="this.style.display='none'" />`
    : '';
  tooltip.innerHTML = `<b>${escapeHtml(meta.run)}</b><br>Iter: ${escapeHtml(meta.iteration)} | Opt: ${escapeHtml(meta.opt_index)} | Loss: ${escapeHtml(meta.loss)}<br>${escapeHtml(meta.clip_model)} | ${escapeHtml(meta.criterion)}<br><b>Target:</b> ${escapeHtml(meta.target_prompt)}<br><b>Initial:</b> ${escapeHtml(meta.initial_prompt)}${meta.reference_image_path ? `<br><b>Ref:</b> ${refPath}` : ''}${refThumb}`;
}

function hideTooltip(){ tooltip.hidden = true; }

async function attachMetadata(card) {
  let cached = null;
  card.addEventListener('mousemove', async (e) => {
    if(!cached){
      try { const data = await fetchJSON('/api/metadata?id='+encodeURIComponent(card.dataset.id)); cached = data.tooltip; }
      catch(err){ console.error('Metadata fetch error:', err); return; }
    }
    showTooltip(e.clientX, e.clientY, cached);
  });
  card.addEventListener('mouseleave', hideTooltip);
  card.addEventListener('click', async (e) => {
    console.log('Card clicked! Target:', e.target, 'Card ID:', card.dataset.id);
    // Check if click is on heart/star toggles or their children
    if(e.target.classList.contains('heart-toggle') || e.target.closest('.heart-toggle')) {
      console.log('Heart toggle clicked, skipping modal');
      return;
    }
    if(e.target.classList.contains('star-toggle') || e.target.closest('.star-toggle')) {
      console.log('Fav toggle clicked, skipping modal');
      return; // handled elsewhere
    }
    // Let the JSON link open in a new tab without opening the modal
    if(e.target.classList.contains('meta-json') || e.target.closest('.meta-json')) {
      return;
    }
    if(!cached){
      console.log('Fetching metadata for', card.dataset.id);
      try { const data = await fetchJSON('/api/metadata?id='+encodeURIComponent(card.dataset.id)); cached = data.tooltip; }
      catch(err){ console.error('Metadata fetch error:', err); return; }
    }
    console.log('Opening modal with cached data:', cached);
    console.log('Modal element:', modal, 'Hidden before:', modal.hidden);
    lastModalData = openModal(modal, modalImg, modalCaption, card, cached, formatCaption, hearts, stars, modalHeartBtn, modalStarBtn);
    console.log('Modal opened, hidden after:', modal.hidden, 'lastModalData:', lastModalData);
  });
}

async function loadGallery(scene, target_prompt, loss_model, reference_image, subdir){
  statusEl.textContent = 'Loading...';
  grid.innerHTML='';
  try {
    await loadHearts();
    await loadStars();
    // If "All" is selected, pass empty string to fetch all
    const prompt_param = target_prompt === 'All' ? '' : target_prompt;
    const loss_model_param = loss_model === 'All' ? '' : loss_model;
    const ref_image_param = reference_image === 'All' ? '' : reference_image;
    let url = `/api/gallery?scene=${encodeURIComponent(scene)}&target_prompt=${encodeURIComponent(prompt_param)}`;
    url += `&loss_model=${encodeURIComponent(loss_model_param)}`;
    url += `&reference_image=${encodeURIComponent(ref_image_param)}`;
    if(subdir) url += `&subdir=${encodeURIComponent(subdir)}`;
    // Pass include_first flag (0/1)
    const includeFirst = includeFirstCheckbox && includeFirstCheckbox.checked ? '1' : '0';
    url += `&include_first=${includeFirst}`;
    const data = await fetchJSON(url);
    statusEl.textContent = `${data.count} images.`;
    const grouped = !!groupByIterationCheckbox?.checked;
    if(grouped){
      grid.classList.add('gallery-grid-grouped');
      grid.appendChild(renderGroupedGallery(data.images));
    } else {
      grid.classList.remove('gallery-grid-grouped');
      grid.appendChild(renderFlatGallery(data.images));
    }
  } catch(e) {
    statusEl.textContent = 'Error: '+ e.message;
  }
}

async function populateSubdirs(){
  try {
    const data = await fetchJSON('/api/subdirs');
    subdirSelect.innerHTML = '<option value="" disabled selected>Select a subdirectory</option>';
    for(const s of data.subdirs){
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s; subdirSelect.appendChild(opt);
    }
    subdirSelect.disabled = false;
  } catch(e){
    subdirSelect.innerHTML = '<option value="">Error loading subdirectories</option>';
  }
}

async function populateScenes(subdir){
  try {
    let url = '/api/scenes';
    if(subdir) url += `?subdir=${encodeURIComponent(subdir)}`;
    const data = await fetchJSON(url);
    sceneSelect.innerHTML = '<option value="" disabled selected>Select a scene</option>';
    for(const s of data.scenes){
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s; sceneSelect.appendChild(opt);
    }
    sceneSelect.disabled = false;
  } catch(e){
    sceneSelect.innerHTML = '<option value="">Error loading scenes</option>';
  }
}

async function populatePrompts(scene, subdir){
  promptSelect.disabled = true;
  promptSelect.innerHTML = '<option value="" disabled selected>Loading prompts...</option>';
  try {
    let url = `/api/prompts?scene=${encodeURIComponent(scene)}`;
    if(subdir) url += `&subdir=${encodeURIComponent(subdir)}`;
    const data = await fetchJSON(url);
    promptSelect.innerHTML = '';
    for(const p of data.prompts){
      const opt = document.createElement('option');
      opt.value = p; opt.textContent = p;
      // Set "All" as default selected
      if(p === 'All') opt.selected = true;
      promptSelect.appendChild(opt);
    }
    promptSelect.disabled = false;
  } catch(e){
    promptSelect.innerHTML = '<option value="" disabled>Error loading prompts</option>';
  }
  updateLoadBtnState();
}

async function populateLossModels(scene, subdir){
  lossModelSelect.disabled = true;
  lossModelSelect.innerHTML = '<option value="" disabled selected>Loading loss models...</option>';
  try {
    let url = `/api/loss_models?scene=${encodeURIComponent(scene)}`;
    if(subdir) url += `&subdir=${encodeURIComponent(subdir)}`;
    const data = await fetchJSON(url);
    lossModelSelect.innerHTML = '';
    for(const lm of data.loss_models){
      const opt = document.createElement('option');
      opt.value = lm; opt.textContent = lm;
      if(lm === 'All') opt.selected = true;
      lossModelSelect.appendChild(opt);
    }
    lossModelSelect.disabled = false;
  } catch(e){
    lossModelSelect.innerHTML = '<option value="" disabled>Error loading loss models</option>';
  }
  updateLoadBtnState();
}

async function populateRefImages(scene, subdir){
  refImageSelect.disabled = true;
  refImageSelect.innerHTML = '<option value="" disabled selected>Loading reference images...</option>';
  try {
    let url = `/api/reference_images?scene=${encodeURIComponent(scene)}`;
    if(subdir) url += `&subdir=${encodeURIComponent(subdir)}`;
    const data = await fetchJSON(url);
    refImageSelect.innerHTML = '';
    for(const ri of data.reference_images){
      const opt = document.createElement('option');
      opt.value = ri; opt.textContent = ri;
      if(ri === 'All') opt.selected = true;
      refImageSelect.appendChild(opt);
    }
    refImageSelect.disabled = false;
  } catch(e){
    refImageSelect.innerHTML = '<option value="" disabled>Error loading ref images</option>';
  }
  updateLoadBtnState();
}

function updateLoadBtnState(){
  loadBtn.disabled = !(sceneSelect.value && promptSelect.value && lossModelSelect.value && refImageSelect.value);
}

subdirSelect.addEventListener('change', ()=>{
  // Clear downstream selectors when subdirectory changes
  sceneSelect.value = '';
  sceneSelect.disabled = true;
  promptSelect.value = '';
  promptSelect.disabled = true;
  lossModelSelect.value = '';
  lossModelSelect.disabled = true;
  refImageSelect.value = '';
  refImageSelect.disabled = true;
  grid.innerHTML = '';
  statusEl.textContent = '';
  
  if(subdirSelect.value){
    populateScenes(subdirSelect.value);
  }
});

sceneSelect.addEventListener('change', ()=>{
  // Clear downstream selectors when scene changes
  promptSelect.value = '';
  promptSelect.disabled = true;
  lossModelSelect.value = '';
  lossModelSelect.disabled = true;
  refImageSelect.value = '';
  refImageSelect.disabled = true;
  grid.innerHTML = '';
  statusEl.textContent = '';
  
  if(sceneSelect.value){
    populatePrompts(sceneSelect.value, subdirSelect.value);
    populateLossModels(sceneSelect.value, subdirSelect.value);
    populateRefImages(sceneSelect.value, subdirSelect.value);
  }
  updateLoadBtnState();
});

promptSelect.addEventListener('change', updateLoadBtnState);
lossModelSelect.addEventListener('change', updateLoadBtnState);
refImageSelect.addEventListener('change', updateLoadBtnState);

// Re-load when toggling the checkbox if a scene/prompt is already selected
includeFirstCheckbox?.addEventListener('change', () => {
  if(sceneSelect.value && promptSelect.value && lossModelSelect.value && refImageSelect.value){
    loadGallery(sceneSelect.value, promptSelect.value, lossModelSelect.value, refImageSelect.value, subdirSelect.value);
  }
});

groupByIterationCheckbox?.addEventListener('change', () => {
  if(sceneSelect.value && promptSelect.value && lossModelSelect.value && refImageSelect.value){
    loadGallery(sceneSelect.value, promptSelect.value, lossModelSelect.value, refImageSelect.value, subdirSelect.value);
  }
});

form.addEventListener('submit', (e)=> {
  e.preventDefault();
  if(!(sceneSelect.value && promptSelect.value && lossModelSelect.value && refImageSelect.value)) return;
  loadGallery(sceneSelect.value, promptSelect.value, lossModelSelect.value, refImageSelect.value, subdirSelect.value);
});

// Setup modal handlers (from shared.js)
setupModalHandlers(
  modal, 
  modalClose, 
  copyPathBtn, 
  modalHeartBtn,
  modalStarBtn,
  () => lastModalData,
  (id) => toggleHeart(id, null),
  (id) => toggleStar(id, null),
  () => hearts,
  () => stars
);

// Initialize: load config, then favorites, then populate subdirectories
setupForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const value = runsDirInput?.value.trim() || '';
  if(!value) {
    showRunsDirSetup('Enter a runs directory path or click Reset to Default.');
    return;
  }
  try {
    if(setupError) setupError.textContent = 'Saving...';
    await saveConfig(value);
    hideRunsDirSetup();
    updateRunsDirDisplay();
    clearGalleryState();
    await initializeGallery();
  } catch(err) {
    showRunsDirSetup(err.message);
  }
});

cancelRunsDirBtn?.addEventListener('click', () => {
  hideRunsDirSetup();
});

resetRunsDirBtn?.addEventListener('click', async () => {
  try {
    if(setupError) setupError.textContent = 'Resetting to default...';
    await saveConfig('', true);
    hideRunsDirSetup();
    updateRunsDirDisplay();
    clearGalleryState();
    await initializeGallery();
  } catch(err) {
    showRunsDirSetup(err.message);
  }
});

setupOverlay?.addEventListener('click', (e) => {
  if (e.target === setupOverlay) {
    hideRunsDirSetup();
  }
});

changeRunsDirBtn?.addEventListener('click', () => {
  showRunsDirSetup();
});

(async function init() {
  console.log('Gallery init starting...');
  await loadConfig();
  console.log('Config loaded:', CONFIG);
  updateRunsDirDisplay();
  await initializeGallery();
  console.log('Gallery initialized');
})();
