/* ═══════════════════════════════════════════════════════════════
   SupplyChainAI Dashboard — Application Logic (Google Maps Version)
   Handles map rendering, WebSocket simulation, and UI updates
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
    { elementType: "geometry", stylers: [{ color: "#1a1a1a" }] },
    { elementType: "labels.icon", stylers: [{ visibility: "off" }] },
    { elementType: "labels.text.fill", stylers: [{ color: "#616161" }] },
    { elementType: "labels.text.stroke", stylers: [{ color: "#1a1a1a" }] },
    { featureType: "administrative", elementType: "geometry", stylers: [{ color: "#444444" }] },
    { featureType: "road", elementType: "geometry.fill", stylers: [{ color: "#222222" }] },
    { featureType: "water", elementType: "geometry", stylers: [{ color: "#000000" }] }
];

// ── Color Palette ───────────────────────────────────────────
const COLORS = {
    network: "#555555",       // Brighter background network
    optimal: "#00f2ff",       // Cyan for the "Best" path
    active: "#bf00ff",        // Neon Purple for path travelled
    anomaly: "#ff3f34",       // Bright Red for disruptions
    source: "#ffd700",        // Gold
    destination: "#00ff7f",   // Emerald Green
    activeNode: "#ff9f1a",    // Orange for current location
    metroNode: "#3498db",     // Blue for metro hubs
    standardNode: "#444444"   // Gray for normal nodes
};

/**
 * Global Callback for Google Maps API
 */
window.initMap = function () {
    console.log("Initializing Google Maps...");
    const mapElement = document.getElementById("map");
    if (!mapElement) return;

    map = new google.maps.Map(mapElement, {
        center: { lat: 22.5, lng: 80 },
        zoom: 5,
        styles: darkStyle,
        disableDefaultUI: false,
        zoomControl: true,
        mapTypeControl: false,
        streetViewControl: false,
        rotateControl: false,
        fullscreenControl: true,
        backgroundColor: "#1a1a1a"
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
                strokeOpacity: 0.4,
                strokeWeight: 2.0,
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
                    scale: isMetro ? 5 : 3,
                    fillColor: isMetro ? COLORS.metroNode : COLORS.standardNode,
                    fillOpacity: 0.5,
                    strokeWeight: 1,
                    strokeColor: "#333333",
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
        setStatus("running", "Simulating...");
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
            if (statusText.textContent === "Simulating...") setStatus("ready", "Ready");
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
        const agentLabel = data.agent_mode === "trained" ? "🧠 GNN+RL" : "🎲 Random";
        
        // Detailed shipment info grid
        shipmentInfo.innerHTML = `
            <div class="info-item">
                <span class="info-label">Product</span>
                <span class="info-value">${s.product_type}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Priority</span>
                <span class="info-value">${s.priority}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Weight</span>
                <span class="info-value">${s.weight_kg.toLocaleString()} kg</span>
            </div>
            <div class="info-item">
                <span class="info-label">Shelf Life</span>
                <span class="info-value">${s.shelf_life_hours}h</span>
            </div>
            <div class="info-item full-width" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border);">
                <span class="info-label">Origin → Destination</span>
                <span class="info-value">${data.source} → ${data.destination}</span>
            </div>
            <div class="info-item full-width">
                <span class="info-label">Navigation Agent</span>
                <span class="agent-badge ${data.agent_mode}">${agentLabel}</span>
            </div>
        `;

        highlightNode(data.source, "source");
        highlightNode(data.destination, "destination");
        if (data.nominal_path) drawOptimalPath(data.nominal_path, true);
    }

    function handleStep(data) {
        cumulativeReward += data.reward;
        
        // Update Live Metrics
        valSteps.textContent = data.step;
        valTime.textContent = data.total_time.toFixed(1) + "h";
        valCost.textContent = "₹" + Math.round(data.total_cost).toLocaleString("en-IN");
        valRisk.textContent = data.total_risk.toFixed(3);
        valReward.textContent = cumulativeReward.toFixed(1);
        valReward.style.color = cumulativeReward >= 0 ? "var(--green)" : "var(--red)";

        // Update Shelf Life Bar
        const shelfPct = data.shelf_remaining_pct;
        valShelf.textContent = `${shelfPct}% Remaining`;
        shelfBar.style.width = `${shelfPct}%`;
        
        // Dynamic bar color based on remaining life
        shelfBar.classList.remove("warning", "danger");
        if (shelfPct < 25) shelfBar.classList.add("danger");
        else if (shelfPct < 50) shelfBar.classList.add("warning");

        // Update Global Anomalies List
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

        // Process edges
        if (anoms.edges) {
            Object.keys(anoms.edges).forEach(key => {
                const edgeAnoms = anoms.edges[key];
                const [src, tgt] = key.split("___");
                edgeAnoms.forEach(a => {
                    count++;
                    const item = document.createElement("div");
                    item.className = `anomaly-item ${a.type}`;
                    item.innerHTML = `
                        <div class="anomaly-item-header">
                            <span>${a.type.toUpperCase()}</span>
                            <span>${a.severity}x</span>
                        </div>
                        <div class="anomaly-item-target">${src} → ${tgt}</div>
                    `;
                    anomaliesList.appendChild(item);
                });
            });
        }

        // Process nodes
        if (anoms.nodes) {
            Object.keys(anoms.nodes).forEach(nodeId => {
                const nodeAnoms = anoms.nodes[nodeId];
                nodeAnoms.forEach(a => {
                    count++;
                    const item = document.createElement("div");
                    item.className = `anomaly-item ${a.type}`;
                    item.innerHTML = `
                        <div class="anomaly-item-header">
                            <span>${a.type.toUpperCase()}</span>
                            <span>${a.severity}x</span>
                        </div>
                        <div class="anomaly-item-target">City: ${nodeId}</div>
                    `;
                    anomaliesList.appendChild(item);
                });
            });
        }

        if (count === 0) {
            anomaliesList.innerHTML = '<div class="log-placeholder">No active disruptions.</div>';
        }
    }

    function handleDone(data) {
        setStatus("ready", data.delivered ? "Delivered!" : "Failed");
        deliveryOverlay.style.display = "flex";
        deliveryContent.innerHTML = `
            <div class="overlay-icon">${data.delivered ? "✅" : "❌"}</div>
            <h3>${data.delivered ? "Successfully Delivered!" : "Shipment Failed"}</h3>
            <div class="overlay-stats">
                <p>Path: ${data.path.join(" → ")}</p>
                <p>Steps: ${data.total_steps} | Time: ${data.total_time_hours}h</p>
                <p>Cost: ₹${data.total_cost.toLocaleString("en-IN")}</p>
            </div>
        `;
        deliveryContent.parentElement.className = `overlay ${data.delivered ? "success" : "failure"}`;
    }

    // ── Helper Functions ────────────────────────────────────────
    function highlightNode(id, type) {
        const mObj = markers[id];
        if (!mObj) return;
        const icon = mObj.marker.getIcon();
        if (type === "source") { icon.fillColor = COLORS.source; icon.scale = 10; icon.fillOpacity = 1.0; }
        else if (type === "destination") { icon.fillColor = COLORS.destination; icon.scale = 10; icon.fillOpacity = 1.0; }
        else if (type === "active") { icon.fillColor = COLORS.activeNode; icon.scale = 8; icon.fillOpacity = 1.0; }
        mObj.marker.setIcon(icon);
        mObj.marker.setZIndex(1000);
    }

    function drawPathSegment(data) {
        const poly = new google.maps.Polyline({
            path: [{ lat: data.from_lat, lng: data.from_lng }, { lat: data.to_lat, lng: data.to_lng }],
            strokeColor: COLORS.active, strokeOpacity: 1.0, strokeWeight: 6, map: map
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
            label: { text: icons[vehicleType] || "📦", fontSize: "28px" }
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
            strokeOpacity: 0.8, 
            strokeWeight: isInitial ? 3 : 5,
            zIndex: 500,
            icons: [{
                icon: { path: 'M 0,-1 0,1', strokeOpacity: 1, scale: 2 },
                offset: '0',
                repeat: '10px'
            }],
            map: map
        });
        poly.isDyn = !isInitial;
        layers.optimalPath.push(poly);
    }

    function updateMapAnomalies(anoms) {
        // Reset all edges to base
        Object.keys(edgeMarkers).forEach(k => {
            edgeMarkers[k].line.setOptions({ strokeColor: COLORS.network, strokeOpacity: 0.1, strokeWeight: 1.0 });
        });
        // Reset all nodes to base
        layers.nodes.forEach(m => { 
            const icon = m.getIcon(); 
            const locData = markers[m.title];
            icon.fillColor = locData.type === "metro" ? COLORS.metroNode : COLORS.standardNode; 
            icon.fillOpacity = 0.5;
            icon.scale = locData.type === "metro" ? 5 : 3;
            m.setIcon(icon); 
        });

        // Highlight affected edges (Subtle Glow)
        if (anoms.edges) {
            Object.keys(anoms.edges).forEach(key => {
                if (edgeMarkers[key]) {
                    edgeMarkers[key].line.setOptions({ strokeColor: COLORS.anomaly, strokeOpacity: 0.6, strokeWeight: 3 });
                }
            });
        }
        // Highlight affected nodes
        if (anoms.nodes) {
            Object.keys(anoms.nodes).forEach(id => {
                if (markers[id]) {
                    const icon = markers[id].marker.getIcon();
                    icon.fillColor = COLORS.anomaly;
                    icon.fillOpacity = 0.9;
                    icon.scale = 7;
                    markers[id].marker.setIcon(icon);
                }
            });
        }
    }

    function addLogEntry(data) {
        // Remove placeholder if it exists
        const placeholder = stepLog.querySelector(".log-placeholder");
        if (placeholder) placeholder.remove();

        const entry = document.createElement("div");
        entry.className = "log-entry";
        if (data.delivered) entry.classList.add("delivered");

        const anomaliesHtml = data.anomalies.map(a => 
            `<span class="anomaly-tag ${a.type}">${a.type}: ${a.severity}x</span>`
        ).join("");

        entry.innerHTML = `
            <div class="log-entry-header">
                <span class="log-step-num">STEP ${data.step}</span>
                <span class="log-vehicle">${data.vehicle_type}</span>
            </div>
            <div class="log-route">${data.from} → ${data.to}</div>
            <div class="log-details">
                <span>⏱ ${data.time_hours}h</span>
                <span>💰 ₹${Math.round(data.cost).toLocaleString("en-IN")}</span>
                <span>⚠ ${data.risk.toFixed(3)}</span>
            </div>
            ${anomaliesHtml ? `<div class="log-anomalies">${anomaliesHtml}</div>` : ""}
        `;
        
        stepLog.appendChild(entry);
        stepLog.scrollTop = stepLog.scrollHeight;
    }

    function resetUI() {
        deliveryOverlay.style.display = "none";
        ["activePath", "optimalPath", "animation"].forEach(k => clearLayer(k));
        cumulativeReward = 0;
        stepLog.innerHTML = "";
    }

    function setStatus(type, text) {
        statusText.textContent = text;
        statusBadge.className = `status-badge ${type === "running" ? "running" : ""}`;
    }

    async function checkModelStatus() {
        try {
            const resp = await fetch("/api/model-status");
            const d = await resp.json();
            agentStatus.textContent = d.available ? `✅ ${d.model}` : "⚠ No model";
        } catch (e) {}
    }

    scenarioSelect.addEventListener("change", () => loadNetwork(scenarioSelect.value));
    speedSlider.addEventListener("input", () => speedLabel.textContent = (parseInt(speedSlider.value) / 1000).toFixed(1) + "s");
    deliveryOverlay.addEventListener("click", () => deliveryOverlay.style.display = "none");
});
