"use strict";

const $ = (id) => document.getElementById(id);
const state = { catalog: null, manifest: null, base: null, ref: null, data: null,
  feature: 0, target: null, page: 0, images: [], dataset: null };
const referenceCache = new Map();
const matchCache = new Map();
let navigation = 0;
let matchRequest = 0;
let routeNavigation = 0;
const PAGE_SIZE = 12;
const NS = "http://www.w3.org/2000/svg";
const palette = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];

function colour(value) {
  if (!Number.isFinite(value)) return "#d9dcd6";
  const v = Math.max(0, Math.min(1, value)) * 4;
  const i = Math.min(3, Math.floor(v));
  return `rgb(${palette[i].map((c, j) => Math.round(c + (palette[i+1][j]-c)*(v-i))).join(",")})`;
}
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function svg(tag, attrs, text) {
  const el = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  if (text !== undefined) el.textContent = text;
  return el;
}
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
function report(error) { status(error.message, true); }
async function request(url, kind = "json") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load results (${response.status}). Please retry.`);
  return kind === "json" ? response.json() : response.arrayBuffer();
}
async function cached(cache, key, loader, limit) {
  if (!cache.has(key)) {
    const promise = loader().catch((error) => { cache.delete(key); throw error; });
    cache.set(key, promise);
    while (cache.size > limit) cache.delete(cache.keys().next().value);
  }
  return cache.get(key);
}
function findRef(id) { return state.manifest.references.find((r) => r.id === id); }
async function referenceData(ref) {
  const base = new URL(`${ref.id}/`, state.base);
  return cached(referenceCache, base.href, async () => {
    const [buffer, features, stats] = await Promise.all([
      request(new URL("maps.f32", base), "binary"), request(new URL("features.json", base)), request(new URL("stats.json", base))
    ]);
    const maps = new Float32Array(buffer);
    if (maps.length !== ref.shape.reduce((a, b) => a*b, 1) || features.length !== ref.features || stats.length !== ref.features)
      throw new Error("The exported map and feature metadata have different dimensions.");
    return { maps, features, stats };
  }, 3);
}
function mapValue(ref, data, feature, li, ei) {
  return data.maps[(feature*ref.layers.length+li)*ref.epochs.length+ei];
}
function miniature(ref, data, feature) {
  const canvas = document.createElement("canvas");
  canvas.width = ref.epochs.length; canvas.height = ref.layers.length;
  canvas.setAttribute("aria-hidden", "true");
  const context = canvas.getContext("2d");
  for (let l=0; l<ref.layers.length; l++) for (let e=0; e<ref.epochs.length; e++) {
    context.fillStyle = colour(mapValue(ref, data, feature, l, e)); context.fillRect(e,l,1,1);
  }
  return canvas;
}
function fillSelect(id, values, selected) {
  $(id).replaceChildren(...values.map(([value,label]) => {
    const option = node("option", null, label); option.value = value; return option;
  }));
  $(id).value = String(selected); $(id).disabled = false;
}
function updateControls() {
  const epochs = [...new Set(state.manifest.references.map(r => r.epoch))].sort((a,b) => a-b);
  fillSelect("epoch", epochs.map(e => [e, e]), state.ref.epoch);
  const layers = state.manifest.references.filter(r => r.epoch === state.ref.epoch).map(r => r.layer).sort((a,b)=>a-b);
  fillSelect("layer", layers.map(l => [l, l]), state.ref.layer);
  $("feature").max = state.ref.features-1; $("feature").value = state.feature; $("feature").disabled = false;
}
function updateURL(replace = false) {
  const url = new URL(location.href);
  url.searchParams.set("dataset", state.dataset);
  url.searchParams.set("reference", state.ref.id);
  url.searchParams.set("feature", state.feature);
  url.searchParams.set("target", state.target);
  history[replace ? "replaceState" : "pushState"]({}, "", url);
}
function visibleFeatures() {
  return state.data.features.map((f, i) => i).filter(i => !$("active-only").checked || state.data.features[i].images.length);
}
function renderGallery() {
  const visible = visibleFeatures();
  const pages = Math.max(1, Math.ceil(visible.length/PAGE_SIZE));
  state.page = Math.max(0, Math.min(pages-1, state.page));
  $("feature-count").textContent = `${visible.length} / ${state.ref.features}`;
  $("gallery").replaceChildren(...visible.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE).map(feature => {
    const card = node("button", `feature-card${feature===state.feature ? " selected" : ""}`);
    card.type = "button"; card.setAttribute("aria-label", `Open feature ${feature}`);
    card.setAttribute("aria-pressed", String(feature===state.feature));
    card.append(miniature(state.ref, state.data, feature), node("span", null, `Feature ${feature}`));
    card.addEventListener("click", () => selectFeature(feature)); return card;
  }));
  $("page-label").textContent = `${state.page+1} / ${pages}`;
  $("previous").disabled = state.page === 0; $("next").disabled = state.page === pages-1;
}
function renderAccuracy() {
  const accuracy=state.manifest.accuracy, ref=state.ref;
  $("accuracy-context").textContent="";
  $("accuracy-caption").hidden=true;
  if(!accuracy) {
    $("accuracy-chart").replaceChildren(node("p","empty","Accuracy measurements are not available for this experiment."));
    return;
  }
  if(!Array.isArray(accuracy.epochs) || !Array.isArray(accuracy.values) || accuracy.epochs.length!==accuracy.values.length ||
     !accuracy.epochs.length || new Set(accuracy.epochs).size!==accuracy.epochs.length ||
     !accuracy.epochs.every(e=>Number.isInteger(e)&&e>=0) ||
     !accuracy.values.every(v=>Number.isFinite(v)&&v>=0&&v<=1)) {
    $("accuracy-chart").replaceChildren(node("p","empty","Accuracy data is invalid: expected unique epochs and values between 0 and 1."));
    return;
  }
  const values=new Map(accuracy.epochs.map((e,i)=>[e,accuracy.values[i]]));
  const target=state.manifest.points.find(p=>p.id===state.target);
  const describe=epoch=>values.has(epoch)?`${(values.get(epoch)*100).toFixed(1)}%`:"unavailable";
  $("accuracy-context").textContent=`Reference e${ref.epoch}: ${describe(ref.epoch)} · Target e${target.epoch}: ${describe(target.epoch)}`;
  if(!ref.epochs.some(e=>values.has(e))) {
    $("accuracy-chart").replaceChildren(node("p","empty","No accuracy measurements are available for the displayed epochs."));
    return;
  }
  const width=Math.max(650,ref.epochs.length*20+70),cellW=(width-70)/ref.epochs.length;
  const x=i=>45+(i+.5)*cellW,y=value=>135-value*115;
  const drawing=svg("svg",{viewBox:`0 0 ${width} 180`,role:"group","aria-label":"Validation top-1 accuracy by saved epoch"});
  for(const percent of [0,25,50,75,100]) {
    drawing.append(svg("line",{x1:45,x2:width-25,y1:y(percent/100),y2:y(percent/100),stroke:"#dedfd7"}));
    drawing.append(svg("text",{x:38,y:y(percent/100)+4,"text-anchor":"end"},`${percent}%`));
  }
  for(const [epoch,kind,label] of [[ref.epoch,"reference","Reference"],[target.epoch,"target","Selected target"]]) {
    const i=ref.epochs.indexOf(epoch);
    if(i>=0) {
      const marker=svg("line",{x1:x(i),x2:x(i),y1:15,y2:140,class:`${kind}-marker`,"data-epoch":epoch});
      marker.append(svg("title",{},`${label} epoch ${epoch}: ${describe(epoch)}`));drawing.append(marker);
    }
  }
  let path="",connected=false;
  ref.epochs.forEach((epoch,i)=>{
    if(!values.has(epoch)) {connected=false;return;}
    path+=`${connected?"L":"M"}${x(i)},${y(values.get(epoch))} `;connected=true;
  });
  drawing.append(svg("path",{d:path,class:"accuracy-line"}));
  const stride=Math.max(1,Math.ceil(ref.epochs.length/28));
  ref.epochs.forEach((epoch,i)=>{
    if(i%stride===0 || i===ref.epochs.length-1)
      drawing.append(svg("text",{x:x(i),y:155,"text-anchor":"middle"},epoch));
    if(!values.has(epoch))return;
    const label=`Epoch ${epoch}: validation top-1 accuracy ${describe(epoch)}`;
    const point=svg("circle",{cx:x(i),cy:y(values.get(epoch)),r:3.5,class:"accuracy-point",tabindex:0,role:"img","aria-label":label});
    point.append(svg("title",{},label));drawing.append(point);
  });
  drawing.append(svg("text",{x:width/2,y:176,"text-anchor":"middle"},"Saved training epoch"));
  $("accuracy-chart").replaceChildren(drawing);
  $("accuracy-caption").hidden=false;
}
function renderMap() {
  renderAccuracy();
  const ref = state.ref, data = state.data;
  const width = Math.max(650, ref.epochs.length*20+70), cellW=(width-70)/ref.epochs.length, cellH=36;
  const height = ref.layers.length*cellH+75;
  const drawing = svg("svg", {viewBox:`0 0 ${width} ${height}`,role:"group","aria-label":"Interactive epoch by layer evolution map"});
  ref.layers.forEach((layer,li) => {
    drawing.append(svg("text",{x:35,y:24+li*cellH+cellH/2,"text-anchor":"end","dominant-baseline":"middle"},layer));
    ref.epochs.forEach((epoch,ei) => {
      const value = mapValue(ref,data,state.feature,li,ei), target=`e${epoch}-l${layer}`;
      const group = svg("g",{class:"cell",role:"button",tabindex:"0","aria-pressed":String(target===state.target),
        "aria-label":`Epoch ${epoch}, layer ${layer}, similarity ${Number.isFinite(value)?value.toFixed(3):"unavailable"}`});
      group.append(svg("title",{},`Epoch ${epoch} · layer ${layer} · ${Number.isFinite(value)?value.toFixed(4):"missing"}`));
      group.append(svg("rect",{x:45+ei*cellW+1,y:15+li*cellH+1,width:cellW-2,height:cellH-2,rx:1,
        fill:colour(value),stroke:target===state.target?"#e97933":"none","stroke-width":3}));
      if (epoch===ref.epoch && layer===ref.layer)
        group.append(svg("text",{x:45+(ei+.5)*cellW,y:15+(li+.5)*cellH,"text-anchor":"middle","dominant-baseline":"central",style:"fill:#141b1d;font-size:20px;pointer-events:none"},"★"));
      group.addEventListener("click",()=>selectTarget(target));
      group.addEventListener("keydown",event=>{
        if (event.key==="Enter" || event.key===" ") { event.preventDefault(); selectTarget(target, true); }
        const moves={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]};
        if (moves[event.key]) {
          event.preventDefault(); const [dl,de]=moves[event.key];
          const nextL=Math.max(0,Math.min(ref.layers.length-1,li+dl)), nextE=Math.max(0,Math.min(ref.epochs.length-1,ei+de));
          drawing.querySelectorAll(".cell")[nextL*ref.epochs.length+nextE].focus();
        }
      });
      drawing.append(group);
    });
  });
  const stride = Math.max(1,Math.ceil(ref.epochs.length/28));
  ref.epochs.forEach((epoch,ei) => { if(ei%stride===0 || ei===ref.epochs.length-1)
    drawing.append(svg("text",{x:45+(ei+.5)*cellW,y:ref.layers.length*cellH+34,"text-anchor":"middle"},epoch)); });
  drawing.append(svg("text",{x:width/2,y:height-4,"text-anchor":"middle"},"Saved training epoch"));
  drawing.append(svg("text",{transform:`translate(12 ${15+ref.layers.length*cellH/2}) rotate(-90)`,"text-anchor":"middle"},"Layer"));
  $("heatmap").replaceChildren(drawing);
}
function imageElement(index, className) {
  const item=state.images[index], img=node("img",className);
  img.src=new URL(item.url,state.base).href; img.alt=`Evaluation image ${index}, class ${item.label}`; img.loading="lazy";
  return img;
}
function renderImages(container, feature) {
  container.replaceChildren(...feature.images.map((index,i)=>{
    const figure=node("figure");figure.append(imageElement(index),node("figcaption",null,`#${i+1} · ${feature.values[i].toPrecision(3)} · class ${state.images[index].label}`));return figure;
  }));
  if(!feature.images.length) container.append(node("p","empty","This feature has no positive activations on the evaluation images."));
}
function renderDetail() {
  const feature=state.data.features[state.feature], stats=state.data.stats[state.feature];
  $("detail-title").textContent=`Feature ${state.feature}`;
  $("reference-label").textContent=`Epoch ${state.ref.epoch} · Layer ${state.ref.layer}`;
  const items=[["Active images",`${(feature.active_fraction*100).toFixed(1)}%`],["Mean activation",feature.mean_activation.toPrecision(3)],
    ["Label entropy",stats[2]===null?"—":`${stats[2].toFixed(2)} bits`]];
  $("stats").replaceChildren(...items.map(([label,value]) => {const item=node("div","stat",label);item.prepend(node("strong",null,value));return item;}));
  $("image-context").textContent=`Reference SAE · ${state.manifest.sample_count.toLocaleString()} evaluation images`;
  renderImages($("images"),feature);
  renderMap();
}
function selectFeature(feature) {
  if (!Number.isInteger(feature) || feature<0 || feature>=state.ref.features) return;
  state.feature=feature;$("feature").value=feature;
  renderGallery();renderDetail();updateURL();showMatches().catch(report);
}
function selectTarget(target, retainFocus = false) {
  state.target=target;renderMap();updateURL();showMatches().catch(report);
  if(retainFocus) $("heatmap").querySelector('[aria-pressed="true"]').focus({preventScroll:true});
}
async function showMatches() {
  const stamp=++matchRequest, ref=state.ref, feature=state.feature, target=state.target;
  const targetPoint=state.manifest.points.find(p=>p.id===target);
  if(!targetPoint) throw new Error("Selected target is outside the exported grid.");
  $("neighbour-title").textContent=`Matches at epoch ${targetPoint.epoch}, layer ${targetPoint.layer}`;
  const li=ref.layers.indexOf(targetPoint.layer),ei=ref.epochs.indexOf(targetPoint.epoch);
  const combined=mapValue(ref,state.data,feature,li,ei);
  $("target-score").textContent=`Combined similarity ${Number.isFinite(combined)?combined.toFixed(3):"—"}`;
  $("neighbours").replaceChildren(node("p","empty","Loading matching features…"));
  $("best-match-context").textContent=`Target SAE · Epoch ${targetPoint.epoch} · Layer ${targetPoint.layer}`;
  $("best-match-images").replaceChildren(node("p","empty","Loading best match…"));
  const url=new URL(`${ref.id}/matches/${target}.bin`,state.base);
  const targetRef=findRef(target);
  const [buffer,targetData]=await Promise.all([
    cached(matchCache,url.href,()=>request(url,"binary"),16),
    targetRef?referenceData(targetRef):request(new URL(`${target}/features.json`,state.base)).then(features=>({features}))
  ]).catch(error=>{
    if(stamp===matchRequest) {
      $("best-match-images").replaceChildren(node("p","empty","Could not load the best match. Select a cell to retry."));
      $("neighbours").replaceChildren(node("p","empty","Could not load matching features."));
    }
    throw error;
  });
  if(stamp!==matchRequest) return;
  const header=new DataView(buffer),rows=header.getUint32(0,true),k=header.getUint32(4,true);
  if(rows!==ref.features || !k || buffer.byteLength!==8+rows*k*8) throw new Error("Invalid feature-match dimensions.");
  const ids=new Uint32Array(buffer,8,rows*k),scores=new Float32Array(buffer,8+rows*k*4,rows*k);
  const cards=[];
  let bestMatch=null;
  for(let rank=0;rank<k;rank++) {
    const id=ids[feature*k+rank],score=scores[feature*k+rank];
    if(!Number.isFinite(score) || score<=0) continue;
    if(id>=targetData.features.length) throw new Error("A matched feature ID is outside the target SAE.");
    if(bestMatch===null) bestMatch={id,score};
    const card=node("button","neighbour-card");card.type="button";
    const title=node("div","card-title");title.append(node("strong",null,`Feature ${id}`),node("span","score",`ρ ${score.toFixed(3)}`));card.append(title);
    if(targetRef) card.append(miniature(targetRef,targetData,id));
    const images=targetData.features[id].images,preview=node("div","preview");
    if(images.length) preview.append(imageElement(images[0]));
    preview.append(node("span",null,`Target SAE · rank ${rank+1}`));card.append(preview);
    card.append(node("span","open",targetRef?"Open evolution map ↗":"Evolution map not exported"));
    card.disabled=!targetRef;
    if(targetRef) card.addEventListener("click",()=>openReference(targetRef,id,null,true).then(()=>$("detail-title").scrollIntoView({behavior:"smooth",block:"center"})).catch(report));
    cards.push(card);
  }
  $("neighbours").replaceChildren(...(cards.length?cards:[node("p","empty","No positively correlated target features were found for this reference feature.")]));
  if(bestMatch) {
    $("best-match-context").textContent=`Target SAE · Epoch ${targetPoint.epoch} · Layer ${targetPoint.layer} · Feature ${bestMatch.id} · Spearman ρ ${bestMatch.score.toFixed(3)} (independent match)`;
    renderImages($("best-match-images"),targetData.features[bestMatch.id]);
  } else {
    $("best-match-images").replaceChildren(node("p","empty","No positively correlated target features were found for this reference feature."));
  }
}
async function openReference(ref,feature=null,target=null,push=true) {
  const stamp=++navigation;++matchRequest;
  status(`Loading reference epoch ${ref.epoch}, layer ${ref.layer}…`);
  const data=await referenceData(ref);
  if(stamp!==navigation)return;
  state.ref=ref;state.data=data;
  if(feature===null) feature=data.features.reduce((best,f,i)=>f.mean_activation>data.features[best].mean_activation?i:best,0);
  if(!Number.isInteger(feature)||feature<0||feature>=ref.features) throw new Error("Feature ID is outside the selected SAE.");
  state.feature=feature;
  const validTargets=new Set(ref.layers.flatMap(l=>ref.epochs.map(e=>`e${e}-l${l}`)));
  const previous=ref.epochs.filter(e=>e<ref.epoch).at(-1);
  state.target=validTargets.has(target)?target:`e${previous??ref.epochs.find(e=>e!==ref.epoch)??ref.epoch}-l${ref.layer}`;
  const visible=visibleFeatures();state.page=Math.max(0,Math.floor(visible.indexOf(feature)/PAGE_SIZE));
  updateControls();renderGallery();renderDetail();
  $("workspace").hidden=false;$("neighbour-section").hidden=false;
  status(`${state.manifest.title} · ${ref.features.toLocaleString()} features at this reference · ${state.manifest.references.length} reference SAEs`);
  if(push)updateURL();else updateURL(true);
  await showMatches();
}
async function navigateFromURL() {
  const routeStamp=++routeNavigation;++navigation;++matchRequest;
  for(const id of ["epoch","layer","feature"]) $(id).disabled=true;
  const params=new URLSearchParams(location.search),requested=params.get("dataset");
  const dataset=state.catalog.datasets.find(d=>d.id===requested)??state.catalog.datasets[0];
  if(!dataset) throw new Error("No exported experiments are available.");
  if(state.dataset!==dataset.id) {
    ++navigation;++matchRequest;
    $("workspace").hidden=true;$("neighbour-section").hidden=true;
    const manifestUrl=new URL(dataset.manifest,state.catalog.url);
    const manifest=await request(manifestUrl);
    if(manifest.schema_version!==1 || !manifest.references.length) throw new Error("Unsupported or empty experiment export.");
    const images=await request(new URL("images.json",manifestUrl));
    if(routeStamp!==routeNavigation)return;
    state.manifest=manifest;state.images=images;state.base=new URL(".",manifestUrl);state.dataset=dataset.id;
    referenceCache.clear();matchCache.clear();
    $("provenance").textContent=`Experiment: ${manifest.title}. Source run: ${manifest.source_run}. SAE run: ${manifest.run_index}. Saved epoch labels are retained, including epoch 0 where present. ${manifest.sample_count.toLocaleString()} evaluation images. Up to ${manifest.neighbours} independent matches per target cell.`;
  }
  fillSelect("dataset",state.catalog.datasets.map(d=>[d.id,d.title]),dataset.id);
  const requestedRef=params.get("reference"),ref=findRef(requestedRef)??state.manifest.references.at(-1);
  const feature=params.has("feature")?Number(params.get("feature")):null;
  await openReference(ref,feature,params.get("target"),false);
}

$("controls").addEventListener("submit",event=>{event.preventDefault();if(state.ref)selectFeature(Number($("feature").value));});
$("epoch").addEventListener("change",()=>{
  const candidates=state.manifest.references.filter(r=>r.epoch===Number($("epoch").value));
  openReference(candidates.find(r=>r.layer===state.ref.layer)??candidates[0]).catch(report);
});
$("layer").addEventListener("change",()=>openReference(state.manifest.references.find(r=>r.epoch===state.ref.epoch&&r.layer===Number($("layer").value))).catch(report));
$("dataset").addEventListener("change",()=>{
  const url=new URL(location.href);url.search="";url.searchParams.set("dataset",$("dataset").value);history.pushState({},"",url);navigateFromURL().catch(report);
});
$("active-only").addEventListener("change",()=>{state.page=0;renderGallery();});
$("previous").addEventListener("click",()=>{state.page--;renderGallery();});
$("next").addEventListener("click",()=>{state.page++;renderGallery();});
$("share").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(location.href);status("Link copied. It includes the reference feature and selected target cell.");}catch{status("Copy the address from your browser to share this exact view.");}});
window.addEventListener("popstate",()=>navigateFromURL().catch(report));
(async()=>{
  const config=await request("config.json");
  const catalogUrl=new URL(config.catalogUrl,location.href);
  state.catalog=await request(catalogUrl);state.catalog.url=catalogUrl;
  if(state.catalog.schema_version!==1)throw new Error("Unsupported experiment catalog.");
  await navigateFromURL();
})().catch(report);
