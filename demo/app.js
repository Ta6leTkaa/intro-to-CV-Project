const $ = s => document.querySelector(s);
const state = {file:null, model:'both', ready:false, busy:false, result:null, preview:null, revision:0};
const colors = ['#b4e784','#70d8ec','#ffd783','#a7adff','#ff9db3','#82dcbc'];
function message(text, error=false){$('#message').textContent=text;$('#message').classList.toggle('error',error)}
function updateRun(){ $('#run').disabled=!state.file||!state.ready||state.busy; }
async function jsonResponse(response){ const data=await response.json();if(!response.ok)throw Error(data.error||'Request failed');return data; }
function invalidate(){state.revision++;state.result=null;$('#results').replaceChildren();$('#results').hidden=true;$('#empty').hidden=false;}
function selectFile(file){
 if(state.busy)return;
 if(!file)return;
 if(file.size>15*1024*1024){message('Maximum file size is 15 MB.',true);return;}
 if(!['image/jpeg','image/png','image/webp'].includes(file.type)){message('Choose a JPEG, PNG or WebP image.',true);return;}
 invalidate();state.file=file;$('#filename').textContent=file.name;message(state.ready?'':'Install demo/requirements.txt to enable detection.',!state.ready);
 if(state.preview)URL.revokeObjectURL(state.preview);state.preview=URL.createObjectURL(file);
 const img=new Image();const revision=state.revision;
 img.onload=()=>{if(revision!==state.revision)return;$('#empty').hidden=true;$('#results').hidden=false;const wrap=document.createElement('div');const label=document.createElement('p');label.className='preview-label';label.textContent='Original';wrap.append(label,img);img.className='result-canvas';$('#results').replaceChildren(wrap);$('#image-size').textContent=`${img.naturalWidth} × ${img.naturalHeight} px`;};
 img.onerror=()=>{if(revision!==state.revision)return;state.file=null;message('Could not open this image.',true);updateRun();};img.src=state.preview;updateRun();
}
$('#file').addEventListener('change',e=>{selectFile(e.target.files[0]);$('#examples').value='';});
$('#dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#file').click();}});
['dragenter','dragover'].forEach(type=>$('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.add('drag');}));
['dragleave','drop'].forEach(type=>$('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.remove('drag');}));
$('#dropzone').addEventListener('drop',e=>{selectFile(e.dataTransfer.files[0]);$('#examples').value='';});
let exampleRequest=0;
$('#examples').addEventListener('change',async e=>{const name=e.target.value;if(!name)return;const request=++exampleRequest;try{const response=await fetch('/examples/'+encodeURIComponent(name));if(!response.ok)throw Error('Example not found');const blob=await response.blob();if(request!==exampleRequest)return;selectFile(new File([blob],name,{type:blob.type}));}catch(error){message(error.message,true);}});
for(const key of ['n','s'])$('#conf-'+key).addEventListener('input',e=>{$('#value-'+key).textContent=Number(e.target.value).toFixed(2);if(state.result)message('Run again to apply changes.');});
$('#reset').addEventListener('click',()=>{for(const [key,value]of [['n',.2],['s',.3]]){$('#conf-'+key).value=value;$('#value-'+key).textContent=value.toFixed(2);}if(state.result)message('Run again to apply defaults.');});
document.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>{state.model=button.dataset.model;document.querySelectorAll('[data-model]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-pressed',String(b===button));});if(state.result)message('Run again to apply changes.');}));
document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-tab]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});$('#demo-view').hidden=button.dataset.tab!=='demo';$('#report-view').hidden=button.dataset.tab!=='report';}));
function draw(canvas, img, boxes, activeIndex = null) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  if (!$('#show-boxes').checked) return;
  const scale = Math.max(1, img.naturalWidth / 850);
  ctx.font = `${12 * scale}px sans-serif`;
  const order = boxes.map((box, index) => ({box, index}));
  if (activeIndex !== null) order.sort((a, b) => Number(a.index === activeIndex) - Number(b.index === activeIndex));
  for (const {box, index} of order) {
    const [x1, y1, x2, y2] = box.xyxy;
    const active = index === activeIndex;
    const color = colors[box.class_id % colors.length];
    ctx.globalAlpha = activeIndex === null || active ? 1 : 0.35;
    ctx.lineWidth = (active ? 4 : 2) * scale;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (active) {
      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.restore();
    }
    const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;
    const width = ctx.measureText(label).width + 10 * scale;
    const x = Math.max(0, Math.min(x1, canvas.width - width));
    const y = Math.max(0, y1 - 21 * scale);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width, 21 * scale);
    ctx.fillStyle = '#101610';
    ctx.fillText(label, x + 5 * scale, y + 15 * scale);
  }
  ctx.globalAlpha = 1;
}

function render() {
  if (!state.result) return;
  const data = state.result;
  const root = $('#results');
  root.replaceChildren();
  root.hidden = false;
  root.classList.toggle('compare', data.results.length === 2);
  $('#empty').hidden = true;

  for (const result of data.results) {
    let selected = null;
    let hovered = null;
    const targets = [];
    const rows = [];
    const card = document.createElement('article');
    const head = document.createElement('div');
    head.className = 'result-head';
    const title = document.createElement('strong');
    title.textContent = result.model.toUpperCase();
    const download = document.createElement('button');
    download.className = 'download';
    download.textContent = 'Save PNG';
    download.disabled = true;
    head.append(title, download);
    card.append(head);

    const frame = document.createElement('div');
    frame.className = 'detection-frame';
    const canvas = document.createElement('canvas');
    canvas.className = 'result-canvas';
    canvas.setAttribute('aria-label', `Detections ${result.model}`);
    const layer = document.createElement('div');
    layer.className = 'box-layer';
    layer.setAttribute('role', 'group');
    layer.setAttribute('aria-label', `${result.model} detected objects`);
    layer.hidden = !$('#show-boxes').checked;
    frame.append(canvas, layer);
    card.append(frame);

    const inspection = document.createElement('div');
    inspection.className = 'inspection';
    const description = document.createElement('span');
    description.setAttribute('role', 'status');
    const clear = document.createElement('button');
    clear.className = 'clear-selection';
    clear.textContent = 'Clear';
    clear.hidden = true;
    inspection.append(description, clear);
    card.append(inspection);
    const img = new Image();

    function update() {
      const active = hovered ?? selected;
      if (img.complete && img.naturalWidth) draw(canvas, img, result.boxes, active);
      description.textContent = active === null ? '' : `${result.boxes[active].label} · ${(result.boxes[active].confidence * 100).toFixed(1)}%`;
      clear.hidden = selected === null;
      targets.forEach(({button, index}) => button.setAttribute('aria-pressed', String(index === selected)));
      rows.forEach(({row, button, index}) => {
        row.classList.toggle('active-detection', index === active);
        button.setAttribute('aria-pressed', String(index === selected));
      });
    }
    function select(index) {
      selected = selected === index ? null : index;
      hovered = null;
      update();
    }
    function bind(button, index) {
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') { hovered = index; update(); } });
      button.addEventListener('pointerleave', () => { hovered = null; update(); });
      button.addEventListener('focus', () => { hovered = index; update(); });
      button.addEventListener('blur', () => { hovered = null; update(); });
      button.addEventListener('click', () => select(index));
      button.addEventListener('keydown', event => {
        if (event.key === 'Escape') { selected = null; hovered = null; update(); }
      });
    }
    clear.onclick = () => { selected = null; hovered = null; update(); };
    frame.addEventListener('click', event => {
      if (event.target === layer || event.target === canvas) { selected = null; hovered = null; update(); }
    });
    // Smaller overlapping objects remain reachable above larger person boxes.
    const objects = result.boxes.map((box, index) => ({box, index}));
    objects.sort((a, b) => {
      const area = box => (box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]);
      return area(b.box) - area(a.box);
    });
    for (const {box, index} of objects) {
      const button = document.createElement('button');
      button.className = 'box-target';
      button.setAttribute('aria-label', `${box.label}, ${(box.confidence * 100).toFixed(1)}%, object ${index + 1}`);
      const [x1, y1, x2, y2] = box.xyxy;
      const left = Math.max(0, Math.min(100, x1 / data.width * 100));
      const top = Math.max(0, Math.min(100, y1 / data.height * 100));
      Object.assign(button.style, {
        left: `${left}%`, top: `${top}%`,
        width: `${Math.max(0, Math.min(100, x2 / data.width * 100) - left)}%`,
        height: `${Math.max(0, Math.min(100, y2 / data.height * 100) - top)}%`
      });
      bind(button, index);
      targets.push({button, index});
      layer.append(button);
    }
    img.onload = () => { update(); download.disabled = false; };
    img.src = data.image;
    download.onclick = () => {
      // Export all detections without transient hover/selection highlighting.
      const exported = document.createElement('canvas');
      draw(exported, img, result.boxes);
      exported.toBlob(blob => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.model + '-prediction.png';
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
    };
    const info = document.createElement('div');
    info.className = 'result-info';
    for (const text of [`${result.boxes.length} detections`, `${result.elapsed_ms} ms${result.cold_start ? ' · first run' : ''}`]) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = text;
      info.append(chip);
    }
    card.append(info);
    if (result.boxes.length) {
      const details = document.createElement('details');
      details.className = 'detections';
      const summary = document.createElement('summary');
      summary.textContent = 'Detections';
      details.append(summary);
      const list = document.createElement('div');
      list.className = 'detection-list';
      const table = document.createElement('table');
      table.innerHTML = '<thead><tr><th>Class</th><th>Confidence</th></tr></thead>';
      const tbody = document.createElement('tbody');
      const ranked = result.boxes.map((box, index) => ({box, index})).sort((a, b) => b.box.confidence - a.box.confidence);
      for (const {box, index} of ranked) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        const button = document.createElement('button');
        button.className = 'detection-select';
        button.textContent = box.label;
        button.setAttribute('aria-label', `Select ${box.label}, ${(box.confidence * 100).toFixed(1)}%, object ${index + 1}`);
        button.disabled = !$('#show-boxes').checked;
        bind(button, index);
        cell.append(button);
        const confidence = document.createElement('td');
        confidence.textContent = (box.confidence * 100).toFixed(1) + '%';
        row.append(cell, confidence);
        rows.push({row, button, index});
        tbody.append(row);
      }
      table.append(tbody);
      list.append(table);
      details.append(list);
      card.append(details);
    } else {
      const empty = document.createElement('p');
      empty.className = 'hint';
      empty.textContent = 'No objects detected.';
      card.append(empty);
    }
    root.append(card);
  }
  $('#image-size').textContent = `${data.width} × ${data.height} px`;
}
$('#show-boxes').addEventListener('change',render);
$('#run').addEventListener('click',async()=>{
 if(state.busy||!state.file)return;state.busy=true;updateRun();$('#run').classList.add('busy');$('#run').textContent='Running…';
 const controls=[...document.querySelectorAll('#file,#examples,[data-model],input[type=range],#reset')];controls.forEach(e=>e.disabled=true);message('');
 const params=new URLSearchParams({model:state.model,yolo11n:$('#conf-n').value,yolo11s:$('#conf-s').value});
 try{const data=await jsonResponse(await fetch('/api/predict?'+params,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:state.file}));state.result=data;render();message('');}
 catch(error){message(error.message,true);}finally{state.busy=false;controls.forEach(e=>e.disabled=false);$('#run').classList.remove('busy');$('#run').textContent='Run detection';updateRun();}
});
async function init(){
 try{const status=await jsonResponse(await fetch('/api/status'));state.ready=status.ready;$('#runtime').textContent=status.ready?`● ${status.device.toUpperCase()} · ready`:'Setup required';$('#runtime').classList.toggle('ready',status.ready);const select=$('#examples');select.replaceChildren(new Option(status.examples.length?'Select an example':'No examples available',''));const labels={'test__image182.jpg':'Small helmet','test__image1037.jpeg':'Occluded vest','test__image1125.jpg':'Localization',...status.example_labels};const ordered=status.examples;for(const name of ordered)select.add(new Option(labels[name]||name,name));if(!status.ready)message('Install demo/requirements.txt to enable detection.',true);updateRun();}
 catch(error){message('Could not connect. '+error.message,true);$('#runtime').textContent='Offline';}
 try{const report=await jsonResponse(await fetch('/api/report'));for(const row of report.models){const card=document.createElement('article');card.className='panel metric-card';const title=document.createElement('h2');title.textContent=row.model.toUpperCase();card.append(title);const values=document.createElement('div');values.className='metric-values';for(const [label,key]of [['mAP50','map50_mean'],['mAP50–95','map50_95_mean'],['Macro-F1','macro_f1_mean']]){const item=document.createElement('div');const value=document.createElement('b');value.textContent=Number(row[key]).toFixed(3);const caption=document.createElement('span');caption.textContent=label;item.append(value,caption);values.append(item);}card.append(values);const note=document.createElement('p');note.textContent=`Confidence ${Number(row.confidence).toFixed(2)}`;card.append(note);$('#metrics').append(card);}}
 catch(error){$('#metrics').textContent='Could not load results: '+error.message;}
}
init();
