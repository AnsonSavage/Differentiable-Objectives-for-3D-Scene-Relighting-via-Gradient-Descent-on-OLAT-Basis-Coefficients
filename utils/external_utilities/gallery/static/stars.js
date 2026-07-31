// DOM elements
const grid = document.getElementById('stars-grid');
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

function cardHTMLStar(img) {
  const node = cardHTML(img, template, hearts, stars);
  node.classList.add('starred');
  const starEl = node.querySelector('.star-toggle');
  if(starEl) starEl.textContent = '★';
  return node;
}

async function toggleHeart(id, card) {
  try {
    const method = hearts.has(id) ? 'DELETE' : 'POST';
    const res = await fetch('/api/hearts', { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify({id}) });
    const data = await res.json();
    hearts = new Set(data.images || []);
    if(card) {
      card.classList.toggle('hearted', hearts.has(id));
      const heartEl = card.querySelector('.heart-toggle');
      if(heartEl) heartEl.textContent = hearts.has(id) ? '♥' : '♡';
    }
  } catch(e) { console.error(e); }
}

async function toggleStar(id, card) {
  try {
    const method = stars.has(id) ? 'DELETE' : 'POST';
    const res = await fetch('/api/stars', { method, headers: {'Content-Type':'application/json'}, body: JSON.stringify({id}) });
    const data = await res.json();
    stars = new Set(data.images || []);
    if(card) {
      // On stars page, un-starring removes it from the list
      if(!stars.has(id)) {
        card.remove();
      } else {
        card.classList.add('starred');
      }
    }
    statusEl.textContent = `${stars.size} stars.`;
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
    if(e.target.classList.contains('heart-toggle') || e.target.closest('.heart-toggle')) return;
    if(e.target.classList.contains('star-toggle') || e.target.closest('.star-toggle')) return;
    if(e.target.classList.contains('meta-json') || e.target.closest('.meta-json')) return;

    if(!cached){
      try {
        const data = await fetchJSON('/api/metadata?id='+encodeURIComponent(card.dataset.id));
        cached = data.tooltip;
      } catch(err){
        alert(`Cannot load metadata for this image: ${err.message}\n\nThe run directory may have been moved or deleted.`);
        return;
      }
    }
    lastModalData = openModal(modal, modalImg, modalCaption, card, cached, formatCaption, hearts, stars, modalHeartBtn, modalStarBtn);
  });
}

async function loadHearts(){
  try {
    const heartsData = await fetchJSON('/api/hearts');
    hearts = new Set(heartsData.images || []);
  } catch(e) {
    console.warn('Hearts load failed', e);
  }
}

async function loadStars(){
  try {
    const starsData = await fetchJSON('/api/stars');
    stars = new Set(starsData.images || []);
  } catch(e) {
    console.warn('Stars load failed', e);
  }
}

async function renderStars(){
  try {
    await loadHearts();
    await loadStars();

    if(stars.size === 0){ statusEl.textContent = 'No stars yet.'; return; }

    const images = [];
    const missingRuns = [];

    for(const id of stars){
      const parts = id.split('/');
      const filename = parts[parts.length - 1];
      try {
        const testResponse = await fetch('/image/' + id, { method: 'HEAD' });
        if(testResponse.ok) {
          images.push({ id, filename, path: id, iteration:0, loss:0 });
        } else {
          missingRuns.push(id);
        }
      } catch(e) {
        missingRuns.push(id);
      }
    }

    if(images.length === 0) {
      statusEl.textContent = 'No valid stars found. Some may have been deleted.';
      return;
    }

    for(const img of images){
      const iterMatch = img.filename.match(/iter(\d+)/); if(iterMatch) img.iteration = parseInt(iterMatch[1]);
      const lossMatch = img.filename.match(/loss_(\d+\.\d+)/); if(lossMatch) img.loss = parseFloat(lossMatch[1]);
    }

    const frag = document.createDocumentFragment();
    for(const img of images){
      const c = cardHTMLStar(img);
      c.querySelector('.heart-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleHeart(img.id, c); });
      c.querySelector('.star-toggle').addEventListener('click', (ev)=>{ ev.stopPropagation(); toggleStar(img.id, c); });
      attachMetadata(c);
      frag.appendChild(c);
    }
    grid.appendChild(frag);

    let statusMsg = `${images.length} stars.`;
    if(missingRuns.length > 0) statusMsg += ` (${missingRuns.length} unavailable)`;
    statusEl.textContent = statusMsg;

  } catch(e){
    statusEl.textContent = 'Error: ' + e.message;
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

(async function init() {
  await loadConfig();
  await renderStars();
})();
