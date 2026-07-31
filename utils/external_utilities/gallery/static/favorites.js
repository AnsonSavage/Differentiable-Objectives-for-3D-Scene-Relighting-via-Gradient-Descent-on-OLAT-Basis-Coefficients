// DOM elements
const grid = document.getElementById('favorites-grid');
const statusEl = document.getElementById('status');
const template = document.getElementById('image-card-template');
const tooltip = document.getElementById('tooltip');
const modal = document.getElementById('modal');
const modalImg = document.getElementById('modal-image');
const modalCaption = document.getElementById('modal-caption');
const modalClose = document.getElementById('modal-close');
const copyPathBtn = document.getElementById('copy-path-btn');
const modalHeartBtn = document.getElementById('modal-heart-btn');
const modalStarBtn = document.getElementById('modal-star-btn');

// State
let lastModalData = null;
let hearts = new Set();
let stars = new Set();

function cardHTMLHeart(img) {
  const node = cardHTML(img, template, hearts, stars);
  node.classList.add('hearted');
  const heartEl = node.querySelector('.heart-toggle');
  if(heartEl) heartEl.textContent = '♥';
  return node;
}

async function toggleHeart(id, card) {
  try {
    const method = hearts.has(id) ? 'DELETE' : 'POST';
    const res = await fetch('/api/hearts', { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify({id}) });
    const data = await res.json();
    hearts = new Set(data.images || []);
    if(card) {
      // On hearts page, removing the heart removes it from the list
      if(!hearts.has(id)) {
        card.remove();
      } else {
        card.classList.add('hearted');
      }
    }
    statusEl.textContent = `${hearts.size} hearts.`;
  } catch(e) { console.error(e); }
}

async function toggleStar(id, card) {
  try {
    const method = stars.has(id) ? 'DELETE' : 'POST';
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
      try { 
        const data = await fetchJSON('/api/metadata?id='+encodeURIComponent(card.dataset.id)); 
        cached = data.tooltip; 
      } catch(err){ 
        console.error('Metadata fetch error:', err); 
        return; 
      }
    }
    showTooltip(e.clientX, e.clientY, cached);
  });
  card.addEventListener('mouseleave', hideTooltip);
  card.addEventListener('click', async (e) => {
    console.log('Favorite card clicked:', card.dataset.id);
    if(e.target.classList.contains('heart-toggle') || e.target.closest('.heart-toggle')) {
      console.log('Heart toggle clicked, skipping modal');
      return;
    }
    if(e.target.classList.contains('star-toggle') || e.target.closest('.star-toggle')) {
      console.log('Star toggle clicked, skipping modal');
      return;
    }
    // Let the JSON link open in a new tab without opening the modal
    if(e.target.classList.contains('meta-json') || e.target.closest('.meta-json')) {
      return;
    }
    if(!cached){
      console.log('Fetching metadata for', card.dataset.id);
      try { 
        const data = await fetchJSON('/api/metadata?id='+encodeURIComponent(card.dataset.id)); 
        cached = data.tooltip; 
      } catch(err){ 
        console.error('Metadata fetch error:', err); 
        // Show error in modal instead of silently failing
        alert(`Cannot load metadata for this image: ${err.message}\n\nThe run directory may have been moved or deleted.`);
        return; 
      }
    }
    console.log('Opening modal with cached data:', cached);
    lastModalData = openModal(modal, modalImg, modalCaption, card, cached, formatCaption, hearts, stars, modalHeartBtn, modalStarBtn);
    console.log('Modal state - hidden:', modal.hidden);
  });
}

async function loadHearts(){
  try {
    console.log('Fetching hearts from API...');
    const heartsData = await fetchJSON('/api/hearts');
    console.log('Hearts API response:', heartsData);
    hearts = new Set(heartsData.images || []);
    console.log('Hearts set size:', hearts.size);
    if(hearts.size === 0){ statusEl.textContent = 'No hearts yet.'; return; }
    
    // We need to fetch each metadata to build path info; gather by run.
    const images = [];
    const missingRuns = [];
    
    for(const id of hearts){
      // id is the full relative path from RUNS_DIR (e.g., "run_name/file.png" or "subdir/run_name/file.png")
      // Extract just the filename (last part after the last /)
      const parts = id.split('/');
      const filename = parts[parts.length - 1];
      
      // Check if this run exists by trying to fetch its image
      try {
        const testResponse = await fetch('/image/' + id, { method: 'HEAD' });
        if(testResponse.ok) {
          // path is the full id (already relative to RUNS_DIR)
          images.push({ id, filename, path: id, iteration:0, loss:0 });
        } else {
          missingRuns.push(id);
        }
      } catch(e) {
        missingRuns.push(id);
      }
    }
    
    console.log('Images array:', images.length, 'items');
    if(missingRuns.length > 0) {
      console.warn('Skipped', missingRuns.length, 'favorites with missing runs:', missingRuns);
    }
    
    if(images.length === 0) {
      statusEl.textContent = 'No valid favorites found. Some may have been deleted.';
      return;
    }
    
    // We can refine iteration/loss from filename patterns
    for(const img of images){
      const iterMatch = img.filename.match(/iter(\d+)/); if(iterMatch) img.iteration = parseInt(iterMatch[1]);
      const lossMatch = img.filename.match(/loss_(\d+\.\d+)/); if(lossMatch) img.loss = parseFloat(lossMatch[1]);
    }
    const frag = document.createDocumentFragment();
    for(const img of images){
      const c = cardHTMLHeart(img);
      c.querySelector('.heart-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleHeart(img.id, c); });
      c.querySelector('.star-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleStar(img.id, c); });
      attachMetadata(c);
      frag.appendChild(c);
    }
    grid.appendChild(frag);
    console.log('Grid populated with', images.length, 'cards');
    
    let statusMsg = `${images.length} hearts.`;
    if(missingRuns.length > 0) {
      statusMsg += ` (${missingRuns.length} unavailable)`;
    }
    statusEl.textContent = statusMsg;
  } catch(e){ 
    console.error('Load favorites error:', e);
    statusEl.textContent = 'Error: '+e.message; 
  }
}

async function loadStars(){
  try {
    const starsData = await fetchJSON('/api/stars');
    stars = new Set(starsData.images || []);
  } catch(e) {
    console.warn('Star load failed', e);
  }
}

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

// Initialize: load config, then load favorites
(async function init() {
  console.log('Favorites page init starting...');
  await loadConfig();
  console.log('Config loaded:', CONFIG);
  await loadStars();
  await loadHearts();
  console.log('Hearts loaded and rendered');
})();
