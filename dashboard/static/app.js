/* ═══════════════════════════════════════════════════════════════
   Data2Delivery Dashboard — Application Logic
   ═══════════════════════════════════════════════════════════════ */

"use strict";

// ── Global Map References ─────────────────────────────────────
let map;
let infoWindow;
let layers = {
    edges: [],
    optimalPath: [],
    nodes: [],
    activePath: [],
    animation: []
};

let ws = null;
let networkData = null;
let markers = {};
let edgeMarkers = {}; // keyed by "src___tgt"
let cumulativeReward = 0;

// ── High-Contrast Dark Theme Style ──────────────────────────
const darkStyle = [
    { elementType: "geometry", stylers: [{ color: "#0d1322" }] },
    { elementType: "labels.icon", stylers: [{ visibility: "off" }] },
    { elementType: "labels.text.fill", stylers: [{ color: "#4d5d73" }] },
    { elementType: "labels.text.stroke", stylers: [{ color: "#0d1322" }] },
    { featureType: "administrative", elementType: "geometry", stylers: [{ color: "#3c4a46" }] },
    { featureType: "road", elementType: "geometry.fill", stylers: [{ color: "#191f2f" }] },
    { featureType: "water", elementType: "geometry", stylers: [{ color: "#080e1d" }] }
];

// ── Color Palette ───────────────────────────────────────────
const COLORS = {
    network: "#3c4a46",       
    optimal: "#44ddc1",       // Primary Teal
    active: "#68fadd",        
    anomaly: "#ffb4ab",       // Error Coral
    source: "#ffd700",        
    destination: "#44ddc1",   
    activeNode: "#68fadd",    
    metroNode: "#44ddc1",     
    standardNode: "#3c4a46"   
};

/**
 * Global Callback for Google Maps API
 */
window.initMap = function () {
    const mapElement = document.getElementById("map");
    if (!mapElement) return;

    map = new google.maps.Map(mapElement, {
        center: { lat: 22.5, lng: 80 },
        zoom: 5,
        styles: darkStyle,
        disableDefaultUI: true,
        zoomControl: true,
        backgroundColor: "#0d1322"
    });

    infoWindow = new google.maps.InfoWindow();
    document.dispatchEvent(new CustomEvent("mapReady"));
};

// ── UI Initialization ────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // DOM References
    const scenarioSelect   = document.getElementById("scenario-select");
    const agentSelect      = document.getElementById("agent-select");
    const agentStatus      = document.getElementById("agent-status");
    const speedSlider      = document.getElementById("speed-slider");
    const speedLabel       = document.getElementById("speed-label");
    const seedInput        = document.getElementById("seed-input");
    const btnSimulate      = document.getElementById("btn-simulate");
    const btnStop          = document.getElementById("btn-stop");
    const statusBadge      = document.getElementById("status-badge");
    const statusText       = statusBadge.querySelector(".status-text");
    const shipmentCard     = document.getElementById("shipment-card");
    const shipmentInfo     = document.getElementById("shipment-info");
    const metricsCard      = document.getElementById("metrics-card");
    const anomaliesCard    = document.getElementById("anomalies-card");
    const anomaliesList    = document.getElementById("global-anomalies-list");
    const stepLog          = document.getElementById("step-log");
    const deliveryOverlay  = document.getElementById("delivery-overlay");
    const deliveryContent  = document.getElementById("delivery-content");

    const valSteps   = document.getElementById("val-steps");
    const valTime    = document.getElementById("val-time");
    const valCost    = document.getElementById("val-cost");
    const valRisk    = document.getElementById("val-risk");
    const valReward  = document.getElementById("val-reward");
    const valShelf   = document.getElementById("val-shelf");
    const shelfBar   = document.getElementById("shelf-bar");

    // ── Map Loading ─────────────────────────────────────────────
    document.addEventListener("mapReady", () => {
        loadNetwork(scenarioSelect.value);
        checkModelStatus();
    });

    async function loadNetwork(scenario) {
        if (!map) return;
        try {
            const resp = await fetch(`/api/network?scenario=${scenario}`);
            networkData = await resp.json();
            renderNetwork(networkData);
        } catch (e) { console.error("Failed to load network:", e); }
    }

    function renderNetwork(data) {
        Object.keys(layers).forEach(k => clearLayer(k));
        markers = {};
        edgeMarkers = {};

        data.edges.forEach(edge => {
            const polyline = new google.maps.Polyline({
                path: [{ lat: edge.source_lat, lng: edge.source_lng }, { lat: edge.target_lat, lng: edge.target_lng }],
                geodesic: true,
                strokeColor: COLORS.network,
                strokeOpacity: 0.2,
                strokeWeight: 1.5,
                map: map
            });

            layers.edges.push(polyline);
            edgeMarkers[`${edge.source}___${edge.target}`] = { line: polyline, baseColor: COLORS.network };
        });

        data.nodes.forEach(node => {
            const isMetro = node.region_type === "metro";
            const marker = new google.maps.Marker({
                position: { lat: node.lat, lng: node.lng },
                map: map,
                title: node.id,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: isMetro ? 4 : 2,
                    fillColor: isMetro ? COLORS.metroNode : COLORS.standardNode,
                    fillOpacity: 0.6,
                    strokeWeight: 1,
                    strokeColor: "#3c4a46",
                }
            });

            layers.nodes.push(marker);
            markers[node.id] = { marker, lat: node.lat, lng: node.lng, type: node.region_type };
        });
    }

    function clearLayer(name) {
        layers[name].forEach(obj => obj.setMap(null));
        layers[name] = [];
    }

    // ── Simulation Logic ────────────────────────────────────────
    btnSimulate.addEventListener("click", () => {
        if (ws) ws.close();
        resetUI();
        setStatus("running", "In Progress");
        btnSimulate.disabled = true;
        btnStop.disabled = false;

        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${location.host}/ws/simulate`);

        ws.onopen = () => {
            ws.send(JSON.stringify({
                scenario: scenarioSelect.value,
                seed: seedInput.value ? parseInt(seedInput.value) : null,
                speed_ms: parseInt(speedSlider.value),
                agent: agentSelect.value,
            }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "init") handleInit(data);
            else if (data.type === "step") handleStep(data);
            else if (data.type === "done") handleDone(data);
        };

        ws.onclose = () => {
            btnSimulate.disabled = false;
            btnStop.disabled = true;
            if (statusText.textContent === "In Progress") setStatus("ready", "Ready");
        };
    });

    btnStop.addEventListener("click", () => {
        if (ws) ws.close();
        setStatus("ready", "Stopped");
    });

    function handleInit(data) {
        if (data.network) renderNetwork(data.network);
        shipmentCard.style.display = "block";
        metricsCard.style.display = "block";
        anomaliesCard.style.display = "block";

        const s = data.shipment;
        const agentLabel = data.agent_mode === "trained" ? "Neural Navigator v4" : "Heuristic Core (Random)";
        
        shipmentInfo.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <div class="font-headline text-lg font-bold text-primary tracking-tight">${s.product_type}</div>
                <span class="bg-primary/10 text-primary text-[9px] font-black px-2 py-0.5 rounded-full border border-primary/20 uppercase tracking-widest">${s.priority}</span>
            </div>
            <div class="grid grid-cols-2 gap-y-3 gap-x-4">
                <div>
                    <div class="text-[9px] text-outline uppercase font-label tracking-wider mb-0.5">Route Vector</div>
                    <div class="text-[11px] font-bold text-on-surface whitespace-nowrap overflow-hidden text-ellipsis">${data.source} → ${data.destination}</div>
                </div>
                <div>
                    <div class="text-[9px] text-outline uppercase font-label tracking-wider mb-0.5">Payload</div>
                    <div class="text-[11px] font-bold text-on-surface">${s.weight_kg.toLocaleString()} KG</div>
                </div>
                <div>
                    <div class="text-[9px] text-outline uppercase font-label tracking-wider mb-0.5">Integrity Window</div>
                    <div class="text-[11px] font-bold text-on-surface">${s.shelf_life_hours}H MAX</div>
                </div>
                <div>
                    <div class="text-[9px] text-outline uppercase font-label tracking-wider mb-0.5">Control Logic</div>
                    <div class="text-[11px] font-bold text-primary">${agentLabel}</div>
                </div>
            </div>
        `;

        highlightNode(data.source, "source");
        highlightNode(data.destination, "destination");
        if (data.nominal_path) drawOptimalPath(data.nominal_path, true);
    }

    function handleStep(data) {
        cumulativeReward += data.reward;
        
        valSteps.textContent = data.step;
        valTime.textContent = data.total_time.toFixed(1) + "H";
        valCost.textContent = "₹" + Math.round(data.total_cost).toLocaleString("en-IN");
        valRisk.textContent = data.total_risk.toFixed(3);
        valReward.textContent = cumulativeReward.toFixed(1);
        valReward.style.color = cumulativeReward >= 0 ? "#44ddc1" : "#ffb4ab";

        const shelfPct = data.shelf_remaining_pct;
        valShelf.textContent = `${shelfPct}%`;
        shelfBar.style.width = `${shelfPct}%`;
        
        shelfBar.style.backgroundColor = shelfPct < 25 ? "#ffb4ab" : (shelfPct < 50 ? "#ffd700" : "#44ddc1");

        if (data.global_anomalies) {
            updateGlobalAnomaliesList(data.global_anomalies);
            updateMapAnomalies(data.global_anomalies);
        }

        if (data.optimal_path) drawOptimalPath(data.optimal_path, false);
        
        drawPathSegment(data);
        highlightNode(data.to, "active");
        addLogEntry(data);
    }

    function updateGlobalAnomaliesList(anoms) {
        anomaliesList.innerHTML = "";
        let count = 0;

        if (anoms.edges) {
            Object.keys(anoms.edges).forEach(key => {
                const edgeAnoms = anoms.edges[key];
                const [src, tgt] = key.split("___");
                edgeAnoms.forEach(a => {
                    count++;
                    const item = document.createElement("div");
                    item.className = `anomaly-item ${a.type}`;
                    item.innerHTML = `
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-black uppercase text-[9px] tracking-widest">${a.type}</span>
                            <span class="text-primary font-mono text-[9px]">${a.severity}X IMPACT</span>
                        </div>
                        <div class="text-[10px] opacity-60 font-mono">${src} → ${tgt}</div>
                    `;
                    anomaliesList.appendChild(item);
                });
            });
        }

        if (anoms.nodes) {
            Object.keys(anoms.nodes).forEach(nodeId => {
                const nodeAnoms = anoms.nodes[nodeId];
                nodeAnoms.forEach(a => {
                    count++;
                    const item = document.createElement("div");
                    item.className = `anomaly-item ${a.type}`;
                    item.innerHTML = `
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-black uppercase text-[9px] tracking-widest">${a.type}</span>
                            <span class="text-primary font-mono text-[9px]">${a.severity}X IMPACT</span>
                        </div>
                        <div class="text-[10px] opacity-60 font-mono">NODE: ${nodeId}</div>
                    `;
                    anomaliesList.appendChild(item);
                });
            });
        }

        if (count === 0) {
            anomaliesList.innerHTML = '<div class="text-[10px] text-outline text-center py-4 opacity-30 uppercase tracking-widest">No active disruptions</div>';
        }
    }

    function handleDone(data) {
        setStatus("ready", data.delivered ? "Completed" : "System Error");
        
        deliveryOverlay.style.display = "flex";
        setTimeout(() => {
            deliveryOverlay.style.opacity = "1";
            deliveryOverlay.style.pointerEvents = "auto";
            deliveryOverlay.querySelector(".glass-panel").style.transform = "scale(1)";
        }, 10);

        const statusIcon = data.delivered ? "check_circle" : "error";
        const statusColor = data.delivered ? "text-primary" : "text-error";
        
        deliveryContent.innerHTML = `
            <div class="mb-6">
                <span class="material-symbols-outlined text-6xl ${statusColor} mb-4">${statusIcon}</span>
                <h2 class="text-3xl font-black font-headline tracking-tighter mb-2 text-on-surface">
                    ${data.delivered ? "DELIVERY SUCCESS" : "MISSION FAILURE"}
                </h2>
                <p class="text-outline text-xs uppercase tracking-[0.3em] font-label">Simulation Technical Summary</p>
            </div>
            
            <div class="grid grid-cols-2 gap-4 mb-8">
                <div class="p-4 bg-surface-container-lowest/50 rounded-2xl border border-outline-variant/10 text-left">
                    <div class="text-[9px] text-outline uppercase tracking-widest mb-1">Time Elapsed</div>
                    <div class="text-xl font-bold font-label">${data.total_time_hours}H</div>
                </div>
                <div class="p-4 bg-surface-container-lowest/50 rounded-2xl border border-outline-variant/10 text-left">
                    <div class="text-[9px] text-outline uppercase tracking-widest mb-1">Total Resource Cost</div>
                    <div class="text-xl font-bold font-label text-primary">₹${data.total_cost.toLocaleString("en-IN")}</div>
                </div>
                <div class="p-4 bg-surface-container-lowest/50 rounded-2xl border border-outline-variant/10 text-left">
                    <div class="text-[9px] text-outline uppercase tracking-widest mb-1">Hop Count</div>
                    <div class="text-xl font-bold font-label">${data.total_steps} STEPS</div>
                </div>
                <div class="p-4 bg-surface-container-lowest/50 rounded-2xl border border-outline-variant/10 text-left">
                    <div class="text-[9px] text-outline uppercase tracking-widest mb-1">Final Risk Score</div>
                    <div class="text-xl font-bold font-label text-error">${data.total_risk.toFixed(3)}</div>
                </div>
            </div>

            <button onclick="document.getElementById('delivery-overlay').style.display='none'" class="w-full py-4 bg-primary text-on-primary rounded-xl font-black text-xs uppercase tracking-widest active:scale-95 transition-all shadow-xl shadow-primary/20">
                Dismiss Technical Report
            </button>
        `;
    }

    function highlightNode(id, type) {
        const mObj = markers[id];
        if (!mObj) return;
        const icon = mObj.marker.getIcon();
        if (type === "source") { icon.fillColor = COLORS.source; icon.scale = 8; icon.fillOpacity = 1.0; }
        else if (type === "destination") { icon.fillColor = COLORS.destination; icon.scale = 8; icon.fillOpacity = 1.0; }
        else if (type === "active") { icon.fillColor = COLORS.activeNode; icon.scale = 6; icon.fillOpacity = 1.0; }
        mObj.marker.setIcon(icon);
        mObj.marker.setZIndex(1000);
    }

    function drawPathSegment(data) {
        const poly = new google.maps.Polyline({
            path: [{ lat: data.from_lat, lng: data.from_lng }, { lat: data.to_lat, lng: data.to_lng }],
            strokeColor: COLORS.active, strokeOpacity: 1.0, strokeWeight: 4, map: map
        });
        layers.activePath.push(poly);
        animateMarkerAlongPath(data.from_lat, data.from_lng, data.to_lat, data.to_lng, data.vehicle_type);
    }

    function animateMarkerAlongPath(flat, flng, tlat, tlng, vehicleType) {
        const icons = { truck: "🚛", rail: "🚂", air: "✈️", ship: "🚢" };
        const marker = new google.maps.Marker({
            position: { lat: flat, lng: flng },
            map: map,
            icon: { path: google.maps.SymbolPath.CIRCLE, scale: 0 },
            label: { text: icons[vehicleType] || "📦", fontSize: "24px" }
        });
        layers.animation.push(marker);

        let count = 0;
        const numSteps = 40;
        const interval = setInterval(() => {
            count++;
            const fraction = count / numSteps;
            marker.setPosition({ lat: flat + (tlat - flat) * fraction, lng: flng + (tlng - flng) * fraction });
            if (count >= numSteps) { clearInterval(interval); setTimeout(() => marker.setMap(null), 500); }
        }, 20);
    }

    function drawOptimalPath(path, isInitial) {
        if (!path || path.length < 2) return;
        if (!isInitial) layers.optimalPath = layers.optimalPath.filter(l => { if (l.isDyn) { l.setMap(null); return false; } return true; });
        
        const poly = new google.maps.Polyline({
            path: path.map(id => ({ lat: markers[id].lat, lng: markers[id].lng })),
            strokeColor: COLORS.optimal, 
            strokeOpacity: 0.5, 
            strokeWeight: isInitial ? 2 : 4,
            zIndex: 500,
            map: map
        });
        poly.isDyn = !isInitial;
        layers.optimalPath.push(poly);
    }

    function updateMapAnomalies(anoms) {
        Object.keys(edgeMarkers).forEach(k => {
            edgeMarkers[k].line.setOptions({ strokeColor: COLORS.network, strokeOpacity: 0.2, strokeWeight: 1.5 });
        });
        layers.nodes.forEach(m => { 
            const icon = m.getIcon(); 
            const locData = markers[m.title];
            icon.fillColor = locData.type === "metro" ? COLORS.metroNode : COLORS.standardNode; 
            icon.fillOpacity = 0.6;
            icon.scale = locData.type === "metro" ? 4 : 2;
            m.setIcon(icon); 
        });

        if (anoms.edges) {
            Object.keys(anoms.edges).forEach(key => {
                if (edgeMarkers[key]) {
                    edgeMarkers[key].line.setOptions({ strokeColor: COLORS.anomaly, strokeOpacity: 0.8, strokeWeight: 3 });
                }
            });
        }
        if (anoms.nodes) {
            Object.keys(anoms.nodes).forEach(id => {
                if (markers[id]) {
                    const icon = markers[id].marker.getIcon();
                    icon.fillColor = COLORS.anomaly;
                    icon.fillOpacity = 0.9;
                    icon.scale = 5;
                    markers[id].marker.setIcon(icon);
                }
            });
        }
    }

    function addLogEntry(data) {
        const placeholder = stepLog.querySelector(".log-placeholder");
        if (placeholder) placeholder.remove();

        const entry = document.createElement("div");
        entry.className = "log-entry";
        if (data.delivered) entry.classList.add("delivered");

        const anomaliesHtml = data.anomalies.map(a => 
            `<span class="anomaly-tag ${a.type}">${a.type}</span>`
        ).join(" ");

        const icons = { truck: "local_shipping", rail: "train", air: "flight", ship: "directions_boat" };
        const icon = icons[data.vehicle_type] || "package";

        entry.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <span class="text-[9px] font-black text-primary uppercase tracking-[0.2em]">Step ${data.step}</span>
                <span class="material-symbols-outlined text-sm text-outline opacity-40">${icon}</span>
            </div>
            <div class="text-[11px] font-bold mb-2 tracking-tight">${data.from} → ${data.to}</div>
            <div class="grid grid-cols-3 gap-2 opacity-60">
                <div class="text-[9px] font-mono">⏱ ${data.time_hours}h</div>
                <div class="text-[9px] font-mono">💰 ₹${Math.round(data.cost)}</div>
                <div class="text-[9px] font-mono">⚠ ${data.risk.toFixed(2)}</div>
            </div>
            ${anomaliesHtml ? `<div class="mt-2">${anomaliesHtml}</div>` : ""}
        `;
        
        stepLog.prepend(entry);
    }

    function resetUI() {
        deliveryOverlay.style.opacity = "0";
        deliveryOverlay.style.pointerEvents = "none";
        deliveryOverlay.querySelector(".glass-panel").style.transform = "scale(0.95)";
        setTimeout(() => {
            deliveryOverlay.style.display = "none";
        }, 500);
        
        ["activePath", "optimalPath", "animation"].forEach(k => clearLayer(k));
        cumulativeReward = 0;
        stepLog.innerHTML = "";
    }

    function setStatus(type, text) {
        statusText.textContent = text;
        statusBadge.className = `status-badge px-4 py-1.5 bg-surface-container-low rounded-full border border-outline-variant/20 flex items-center ${type === "running" ? "running" : ""}`;
    }

    async function checkModelStatus() {
        try {
            const resp = await fetch("/api/model-status");
            const d = await resp.json();
            agentStatus.innerHTML = d.available ? `<span class="text-[#44ddc1]">●</span> GNN ENGINE: ${d.model}` : "⚠ GNN ENGINE: OFFLINE";
        } catch (e) {}
    }

    scenarioSelect.addEventListener("change", () => loadNetwork(scenarioSelect.value));
    speedSlider.addEventListener("input", () => speedLabel.textContent = (parseInt(speedSlider.value) / 1000).toFixed(1) + "s");
});
