/* ═══════════════════════════════════════════════════════════════
   SupplyChainAI Dashboard — Application Logic
   Handles map rendering, WebSocket simulation, and UI updates
   ═══════════════════════════════════════════════════════════════ */

(function () {
    "use strict";

    // ── DOM References ──────────────────────────────────────────
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
    const stepLog          = document.getElementById("step-log");
    const deliveryOverlay  = document.getElementById("delivery-overlay");
    const deliveryContent  = document.getElementById("delivery-content");

    // Metric value elements
    const valSteps   = document.getElementById("val-steps");
    const valTime    = document.getElementById("val-time");
    const valCost    = document.getElementById("val-cost");
    const valRisk    = document.getElementById("val-risk");
    const valReward  = document.getElementById("val-reward");
    const valShelf   = document.getElementById("val-shelf");
    const shelfBar   = document.getElementById("shelf-bar");

    // ── Map Setup ───────────────────────────────────────────────
    const map = L.map("map", {
        center: [22.5, 80],
        zoom: 5,
        zoomControl: true,
        attributionControl: true,
    });

    // Dark-themed tile layer
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: "abcd",
        maxZoom: 18,
    }).addTo(map);

    // Map layer groups
    const edgeLayer       = L.layerGroup().addTo(map);
    const nodeLayer       = L.layerGroup().addTo(map);
    const activePathLayer = L.layerGroup().addTo(map);
    const animationLayer  = L.layerGroup().addTo(map);

    // State
    let ws = null;
    let networkData = null;
    let markers = {};
    let cumulativeReward = 0;

    // ── Speed Slider ────────────────────────────────────────────
    speedSlider.addEventListener("input", () => {
        const val = parseInt(speedSlider.value);
        speedLabel.textContent = (val / 1000).toFixed(1) + "s";
    });

    // ── Load Initial Network ────────────────────────────────────
    async function loadNetwork(scenario) {
        const resp = await fetch(`/api/network?scenario=${scenario}`);
        networkData = await resp.json();
        renderNetwork(networkData);
    }

    function renderNetwork(data) {
        edgeLayer.clearLayers();
        nodeLayer.clearLayers();
        activePathLayer.clearLayers();
        animationLayer.clearLayers();
        markers = {};

        // Draw edges first (behind nodes)
        for (const edge of data.edges) {
            const latlngs = [
                [edge.source_lat, edge.source_lng],
                [edge.target_lat, edge.target_lng],
            ];

            const terrainColors = {
                flat: "hsla(200, 60%, 45%, 0.25)",
                hilly: "hsla(30, 60%, 45%, 0.25)",
                mountainous: "hsla(0, 50%, 50%, 0.25)",
                coastal: "hsla(180, 60%, 45%, 0.25)",
            };

            const line = L.polyline(latlngs, {
                color: terrainColors[edge.terrain] || "hsla(200, 60%, 45%, 0.2)",
                weight: 1.5,
                opacity: 0.6,
                dashArray: "4 6",
            });

            line.bindPopup(`
                <div class="popup-title">${edge.source} → ${edge.target}</div>
                <div class="popup-detail">
                    Distance: ${edge.distance_km.toFixed(0)} km<br>
                    Terrain: ${edge.terrain}<br>
                    Road Grade: ${(edge.road_grading * 100).toFixed(0)}%<br>
                    Toll: ₹${edge.toll_cost}
                </div>
            `);

            edgeLayer.addLayer(line);
        }

        // Draw nodes
        const seenNodes = new Set();
        for (const node of data.nodes) {
            if (seenNodes.has(node.id)) continue;
            seenNodes.add(node.id);

            const isMetro = node.region_type === "metro";
            const size = isMetro ? 16 : 12;

            const icon = L.divIcon({
                className: "",
                html: `<div class="city-marker ${isMetro ? "metro" : ""}" id="marker-${node.id}"></div>`,
                iconSize: [size, size],
                iconAnchor: [size / 2, size / 2],
            });

            const marker = L.marker([node.lat, node.lng], { icon })
                .bindPopup(`
                    <div class="popup-title">${node.id}</div>
                    <div class="popup-detail">
                        Type: ${node.region_type}<br>
                        Warehouse: ${node.has_warehouse ? "Yes" : "No"}
                        ${node.has_warehouse ? `<br>Capacity: ${node.warehouse_capacity}<br>Fill: ${(node.fill_ratio * 100).toFixed(0)}%` : ""}
                        ${node.has_cold_storage ? "<br>❄ Cold Storage" : ""}
                    </div>
                `)
                .addTo(nodeLayer);

            // Add label (tooltip)
            marker.bindTooltip(node.id, {
                permanent: true,
                direction: "top",
                offset: [0, -10],
                className: "city-label",
            });

            markers[node.id] = marker;
        }
    }

    // ── Simulation ──────────────────────────────────────────────
    btnSimulate.addEventListener("click", startSimulation);
    btnStop.addEventListener("click", stopSimulation);

    function startSimulation() {
        // Close existing connection
        if (ws) ws.close();

        // Reset UI
        resetUI();
        setStatus("running", "Simulating...");
        btnSimulate.disabled = true;
        btnStop.disabled = false;
        cumulativeReward = 0;

        // Open WebSocket
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
            handleMessage(data);
        };

        ws.onclose = () => {
            btnSimulate.disabled = false;
            btnStop.disabled = true;
            if (statusText.textContent === "Simulating...") {
                setStatus("ready", "Ready");
            }
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            setStatus("error", "Connection Error");
        };
    }

    function stopSimulation() {
        if (ws) {
            ws.close();
            ws = null;
        }
        setStatus("ready", "Stopped");
        btnSimulate.disabled = false;
        btnStop.disabled = true;
    }

    function handleMessage(data) {
        switch (data.type) {
            case "init":
                handleInit(data);
                break;
            case "step":
                handleStep(data);
                break;
            case "done":
                handleDone(data);
                break;
        }
    }

    // ── Init Handler ────────────────────────────────────────────
    function handleInit(data) {
        // Render the network if provided
        if (data.network) {
            renderNetwork(data.network);
        }

        // Show shipment card
        shipmentCard.style.display = "block";
        const s = data.shipment;
        const agentMode = data.agent_mode || "random";
        const agentLabel = agentMode === "trained"
            ? '<span class="agent-badge trained">🧠 GNN+RL</span>'
            : '<span class="agent-badge random">🎲 Random</span>';
        shipmentInfo.innerHTML = `
            <div class="info-item full-width">
                <span class="info-label">Agent</span>
                ${agentLabel}
            </div>
            <div class="info-item">
                <span class="info-label">Product</span>
                <span class="info-value">${s.product_type}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Weight</span>
                <span class="info-value">${s.weight_kg} kg</span>
            </div>
            <div class="info-item">
                <span class="info-label">Priority</span>
                <span class="info-value" style="color: ${priorityColor(s.priority)}">${s.priority.toUpperCase()}</span>
            </div>
            <div class="info-item">
                <span class="info-label">Shelf Life</span>
                <span class="info-value">${s.shelf_life_hours}h</span>
            </div>
            <div class="info-item full-width">
                <span class="info-label">Route</span>
                <span class="info-value">${data.source} → ${data.destination}</span>
            </div>
        `;

        // Show metrics card
        metricsCard.style.display = "block";

        // Highlight source and destination on the map
        highlightNode(data.source, "source");
        highlightNode(data.destination, "destination");
    }

    // ── Step Handler ────────────────────────────────────────────
    function handleStep(data) {
        cumulativeReward += data.reward;

        // Update metrics
        valSteps.textContent = data.step;
        valTime.textContent = data.total_time.toFixed(1) + "h";
        valCost.textContent = "₹" + formatNumber(data.total_cost);
        valRisk.textContent = data.total_risk.toFixed(3);
        valReward.textContent = cumulativeReward.toFixed(1);
        valReward.style.color = cumulativeReward >= 0
            ? "hsl(145, 70%, 50%)"
            : "hsl(0, 75%, 60%)";

        // Shelf life bar
        if (data.shelf_remaining_pct !== undefined) {
            const pct = data.shelf_remaining_pct;
            valShelf.textContent = pct.toFixed(1) + "%";
            shelfBar.style.width = pct + "%";
            shelfBar.className = "progress-bar" +
                (pct < 20 ? " danger" : pct < 50 ? " warning" : "");
        }

        // Draw active path segment
        drawPathSegment(data);

        // Highlight active node
        highlightNode(data.to, "active");

        // Add log entry
        addLogEntry(data);
    }

    // ── Done Handler ────────────────────────────────────────────
    function handleDone(data) {
        setStatus("ready", data.delivered ? "Delivered!" : "Failed");

        deliveryOverlay.style.display = "flex";
        if (data.delivered) {
            deliveryContent.className = "overlay-content success";
            deliveryContent.innerHTML = `
                <span class="overlay-icon">✅</span>
                <h3>Shipment Delivered!</h3>
                <div class="overlay-stats">
                    Path: ${data.path.join(" → ")}<br>
                    Hops: ${data.total_steps}<br>
                    Time: ${data.total_time_hours}h<br>
                    Cost: ₹${formatNumber(data.total_cost)}<br>
                    Risk: ${data.total_risk.toFixed(3)}
                </div>
            `;
        } else {
            deliveryContent.className = "overlay-content failure";
            deliveryContent.innerHTML = `
                <span class="overlay-icon">❌</span>
                <h3>Delivery Failed</h3>
                <div class="overlay-stats">
                    Reached: ${data.path[data.path.length - 1]}<br>
                    Steps: ${data.total_steps}<br>
                    Time: ${data.total_time_hours}h<br>
                    Cost: ₹${formatNumber(data.total_cost)}
                </div>
            `;
        }

        // Auto-hide overlay after 5 seconds
        setTimeout(() => {
            deliveryOverlay.style.display = "none";
        }, 5000);
    }

    // ── Map Helpers ─────────────────────────────────────────────
    function highlightNode(nodeId, type) {
        const marker = markers[nodeId];
        if (!marker) return;

        const el = document.getElementById(`marker-${nodeId}`);
        if (el) {
            // Remove previous active class from all markers
            if (type === "active") {
                document.querySelectorAll(".city-marker.active").forEach(m =>
                    m.classList.remove("active")
                );
            }
            el.classList.add(type);
        }
    }

    function drawPathSegment(data) {
        const latlngs = [
            [data.from_lat, data.from_lng],
            [data.to_lat, data.to_lng],
        ];

        const vehicleColors = {
            truck: "hsl(200, 90%, 60%)",
            rail: "hsl(145, 70%, 55%)",
            air: "hsl(260, 70%, 65%)",
            ship: "hsl(180, 70%, 55%)",
        };

        const color = vehicleColors[data.vehicle_type] || "hsl(200, 90%, 60%)";

        // Glowing animated line
        const bgLine = L.polyline(latlngs, {
            color: color,
            weight: 6,
            opacity: 0.2,
        });
        activePathLayer.addLayer(bgLine);

        const mainLine = L.polyline(latlngs, {
            color: color,
            weight: 3,
            opacity: 0.9,
        });
        activePathLayer.addLayer(mainLine);

        // Animate a dot moving along the path
        animateMarkerAlongPath(latlngs, color, data.vehicle_type);
    }

    function animateMarkerAlongPath(latlngs, color, vehicleType) {
        const vehicleIcons = { truck: "🚛", rail: "🚂", air: "✈️", ship: "🚢" };
        const icon = vehicleIcons[vehicleType] || "📦";

        const start = latlngs[0];
        const end = latlngs[1];
        const steps = 30;
        let step = 0;

        const dot = L.marker(start, {
            icon: L.divIcon({
                className: "",
                html: `<div style="font-size: 20px; filter: drop-shadow(0 0 6px ${color});">${icon}</div>`,
                iconSize: [24, 24],
                iconAnchor: [12, 12],
            }),
        }).addTo(animationLayer);

        const interval = setInterval(() => {
            step++;
            const t = step / steps;
            const lat = start[0] + (end[0] - start[0]) * t;
            const lng = start[1] + (end[1] - start[1]) * t;
            dot.setLatLng([lat, lng]);

            if (step >= steps) {
                clearInterval(interval);
                setTimeout(() => animationLayer.removeLayer(dot), 200);
            }
        }, 25);
    }

    // ── Log Helpers ─────────────────────────────────────────────
    function addLogEntry(data) {
        // Remove placeholder if it exists
        const placeholder = stepLog.querySelector(".log-placeholder");
        if (placeholder) placeholder.remove();

        let anomalyHtml = "";
        if (data.anomalies && data.anomalies.length > 0) {
            anomalyHtml = `
                <div class="log-anomalies">
                    ${data.anomalies.map(a => `
                        <span class="anomaly-tag ${a.type}">${a.type} ${a.severity}×</span>
                    `).join("")}
                </div>
            `;
        }

        const entry = document.createElement("div");
        entry.className = `log-entry${data.delivered ? " delivered" : ""}`;
        entry.innerHTML = `
            <div class="log-entry-header">
                <span class="log-step-num">Step ${data.step}</span>
                <span class="log-vehicle">${vehicleIcon(data.vehicle_type)} ${data.vehicle_type}</span>
            </div>
            <div class="log-route">${data.from} → ${data.to}</div>
            <div class="log-details">
                <span>⏱ ${data.time_hours}h</span>
                <span>₹${formatNumber(data.cost)}</span>
                <span>⚠ ${data.risk.toFixed(3)}</span>
            </div>
            ${anomalyHtml}
        `;

        stepLog.appendChild(entry);
        stepLog.scrollTop = stepLog.scrollHeight;
    }

    // ── UI Helpers ──────────────────────────────────────────────
    function resetUI() {
        deliveryOverlay.style.display = "none";
        activePathLayer.clearLayers();
        animationLayer.clearLayers();
        cumulativeReward = 0;

        // Remove highlighting from markers
        document.querySelectorAll(".city-marker").forEach(m => {
            m.classList.remove("active", "source", "destination");
        });

        // Reset metrics
        valSteps.textContent = "—";
        valTime.textContent = "—";
        valCost.textContent = "—";
        valRisk.textContent = "—";
        valReward.textContent = "—";
        valReward.style.color = "";
        valShelf.textContent = "—";
        shelfBar.style.width = "100%";
        shelfBar.className = "progress-bar";

        // Clear log
        stepLog.innerHTML = "";
    }

    function setStatus(type, text) {
        statusText.textContent = text;
        statusBadge.className = `status-badge ${type === "running" ? "running" : ""}`;
    }

    function formatNumber(num) {
        return Math.round(num).toLocaleString("en-IN");
    }

    function priorityColor(priority) {
        const colors = {
            low: "hsl(215, 15%, 60%)",
            medium: "hsl(50, 90%, 55%)",
            high: "hsl(30, 90%, 55%)",
            critical: "hsl(0, 75%, 60%)",
        };
        return colors[priority] || "hsl(215, 15%, 60%)";
    }

    function vehicleIcon(type) {
        const icons = { truck: "🚛", rail: "🚂", air: "✈️", ship: "🚢" };
        return icons[type] || "📦";
    }

    // ── Init ────────────────────────────────────────────────────
    // Load India network on startup
    loadNetwork("india");

    // Check if a trained model is available
    async function checkModelStatus() {
        try {
            const resp = await fetch("/api/model-status");
            const data = await resp.json();
            if (data.available) {
                agentStatus.textContent = `✅ ${data.model} (ep ${data.episode})`;
                agentStatus.style.color = "hsl(145, 70%, 50%)";
            } else {
                agentStatus.textContent = "⚠ No model found";
                agentStatus.style.color = "hsl(30, 90%, 55%)";
            }
        } catch (e) {
            agentStatus.textContent = "";
        }
    }
    checkModelStatus();

    // Reload network when scenario changes
    scenarioSelect.addEventListener("change", () => {
        loadNetwork(scenarioSelect.value);
    });

    // Click overlay to dismiss
    deliveryOverlay.addEventListener("click", () => {
        deliveryOverlay.style.display = "none";
    });

})();
