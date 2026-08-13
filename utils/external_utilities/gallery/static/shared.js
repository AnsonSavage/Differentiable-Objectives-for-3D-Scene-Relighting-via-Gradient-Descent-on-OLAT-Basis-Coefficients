/**
 * Shared utilities and functions for gallery and favorites pages
 */

// Configuration loaded from backend
let CONFIG = { runs_dir: '', runs_dir_name: '', runs_dir_absolute: '', runs_dir_input: '', is_default: true };

async function fetchJSON(url, options = {}) { 
  const r = await fetch(url, options); 
  if(!r.ok) {
    let errorMsg;
    try {
      const errorData = await r.json();
      errorMsg = errorData.error || JSON.stringify(errorData);
    } catch(e) {
      errorMsg = await r.text();
    }
    throw new Error(errorMsg);
  }
  return r.json(); 
}

async function loadConfig() {
  try { 
    CONFIG = await fetchJSON('/api/config'); 
  } catch(e) { 
    console.warn('Config load failed, using defaults', e); 
  }
}

async function saveConfig(runsDir, reset = false) {
  const data = await fetchJSON('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ runs_dir: runsDir, reset: reset }),
  });
  CONFIG = data;
  return CONFIG;
}

function updateToggle(el, isActive, onChar, offChar, activeClass) {
  if(!el) return;
  el.textContent = isActive ? onChar : offChar;
  if(activeClass) el.classList.toggle(activeClass, isActive);
}

function cardHTML(img, template, hearts, stars) {
  const node = template.content.firstElementChild.cloneNode(true);
  const imageEl = node.querySelector('img');
  imageEl.src = '/image/' + img.path; // img.path is relative path from RUNS_DIR
  imageEl.alt = img.filename;
  node.dataset.id = img.id;
  const metaLink = node.querySelector('.meta-json');
  if (metaLink) {
    metaLink.href = '/api/metadata?id=' + encodeURIComponent(img.id);
  }

  const isHearted = !!(hearts && hearts.has(img.id));
  const isStarred = !!(stars && stars.has(img.id));
  node.classList.toggle('hearted', isHearted);
  node.classList.toggle('starred', isStarred);

  updateToggle(node.querySelector('.heart-toggle'), isHearted, '♥', '♡');
  updateToggle(node.querySelector('.star-toggle'), isStarred, '★', '☆');
  return node;
}

function openModal(modal, modalImg, modalCaption, card, meta, formatCaptionFn, hearts, stars, modalHeartBtn, modalStarBtn) {
  console.log('openModal called with:', { modal, modalImg, modalCaption, card, meta });
  const lastModalData = { id: card.dataset.id, meta };
  modalImg.src = card.querySelector('img').src;
  console.log('Modal image src set to:', modalImg.src);
  modalCaption.textContent = formatCaptionFn(card.dataset.id, meta);
  console.log('Modal caption set, now showing modal...');
  
  // Update modal button states if provided
  const id = card.dataset.id;
  if(modalHeartBtn && hearts) {
    const isHearted = hearts.has(id);
    updateToggle(modalHeartBtn, isHearted, '♥', '♡', 'hearted');
  }
  if(modalStarBtn && stars) {
    const isStarred = stars.has(id);
    updateToggle(modalStarBtn, isStarred, '★', '☆', 'starred');
  }
  
  modal.hidden = false;
  console.log('Modal.hidden set to false, current hidden value:', modal.hidden);
  return lastModalData;
}

function formatCaption(id, meta) {
  const absolutePath = CONFIG.runs_dir_absolute || CONFIG.runs_dir || '';
  const batchIndex = meta.opt_index ?? 'N/A';
  return `ID: ${id}\nRun: ${meta.run}\nIteration: ${meta.iteration}\nOpt: ${batchIndex}\nLoss: ${meta.loss}\nModel: ${meta.clip_model}\nCriterion: ${meta.criterion}\nTarget Prompt: ${meta.target_prompt}\nInitial Prompt: ${meta.initial_prompt || ''}\nReference Image: ${meta.reference_image_path || ''}\nImage Path: ${absolutePath}/${id}`;
}

function setupModalHandlers(
  modal,
  modalClose,
  copyPathBtn,
  modalHeartBtn,
  modalStarBtn,
  getLastModalDataFn,
  toggleHeartFn,
  toggleStarFn,
  getHeartsFn,
  getStarsFn
) {
  modalClose?.addEventListener('click', () => { modal.hidden = true; });
  modal.addEventListener('click', (e) => { if(e.target === modal) modal.hidden = true; });
  
  copyPathBtn?.addEventListener('click', async () => {
    const lastModalData = getLastModalDataFn();
    if(!lastModalData) return;
    // Use absolute path for clipboard
    const absolutePath = CONFIG.runs_dir_absolute || CONFIG.runs_dir || '';
    const path = `${absolutePath}/${lastModalData.id}`;
    try { 
      await navigator.clipboard.writeText(path); 
      copyPathBtn.textContent = 'Copied!'; 
      setTimeout(() => copyPathBtn.textContent='Copy Image Path', 1500); 
    } catch(e) { 
      console.warn('Clipboard copy failed', e); 
    }
  });

  // Modal heart button handler
  modalHeartBtn?.addEventListener('click', async () => {
    const lastModalData = getLastModalDataFn();
    if(!lastModalData || !toggleHeartFn) return;

    await toggleHeartFn(lastModalData.id);

    if(getHeartsFn) {
      const hearts = getHeartsFn();
      const isHearted = hearts.has(lastModalData.id);
      updateToggle(modalHeartBtn, isHearted, '♥', '♡', 'hearted');
    }
  });

  // Modal star button handler
  modalStarBtn?.addEventListener('click', async () => {
    const lastModalData = getLastModalDataFn();
    if(!lastModalData || !toggleStarFn) return;

    await toggleStarFn(lastModalData.id);

    if(getStarsFn) {
      const stars = getStarsFn();
      const isStarred = stars.has(lastModalData.id);
      updateToggle(modalStarBtn, isStarred, '★', '☆', 'starred');
    }
  });

  // Ensure modal hidden on initial load
  if(modal && !modal.hasAttribute('hidden')) {
    modal.hidden = true;
  }

  // Escape key handler
  window.addEventListener('keydown', (e) => {
    if(e.key === 'Escape' && !modal.hidden) { modal.hidden = true; }
  });
}
