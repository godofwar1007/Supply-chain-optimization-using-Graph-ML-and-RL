# Supply Chain Optimization — System Documentation

## Overview

This project implements a production-grade supply chain optimization system using **Heterogeneous Graph Machine Learning** and **Reinforcement Learning**. It replaces traditional fixed-route heuristics with an intelligent agent capable of dynamic routing and vehicle selection in a volatile environment.

### Key Capabilities
- **Dynamic Routing**: Real-time pathfinding on a heterogeneous graph.
- **Anomaly Awareness**: Native handling of stochastic disruptions (weather, traffic, sentiment, geopolitical).
- **Multi-Objective Optimization**: Trade-offs between travel time, total cost, and delivery risk.
- **Real-Time Visualization**: WebSocket-based dashboard for monitoring agent performance and network health.

---

## 1. Graph Architecture

The supply chain is modeled as a **heterogeneous graph** (`torch_geometric.data.HeteroData`) where different entities have specialized feature sets.

### 1.1 Node & Edge Schema

| Entity | Type | Key Features |
|--------|------|--------------|
| **Location** | Node | Lat/Lng, region type, local weather, sentiment risk. |
| **Warehouse** | Node | Current fill ratio, capacity, cold storage availability. |
| **Vehicle** | Node | Type (Truck/Rail/Air/Ship), payload, efficiency, cost-per-km. |
| **Shipment** | Node | Priority, shelf life, weight, destination, product type. |
| **Route** | Edge | Distance, terrain, tolls, dynamic traffic, anomaly factors. |

### 1.2 Connectivity
- `route`: Connects `Location` ↔ `Location`.
- `has_warehouse`: Connects `Location` → `Warehouse`.
- `vehicle_at`: Connects `Vehicle` → `Location`.
- `shipment_at`: Connects `Shipment` → `Location`.
- `shipment_dest`: Connects `Shipment` → `Location` (Destination).

---

## 2. Model Architecture

The system uses an **Actor-Critic** framework powered by a **Heterogeneous Graph Transformer (HGT)**.

### 2.1 GNN Encoder (`gnn_encoder.py`)
- **Architecture**: `HGTConv` (Heterogeneous Graph Transformer).
- **Process**:
    1. Type-specific linear projections map raw features to a 64-dim hidden space.
    2. Multiple layers of attention-based message passing aggregate neighborhood info.
    3. Graph-level pooling combines node embeddings with a **Global Context Vector** (time, seasonality, fuel prices).

### 2.2 PPO Agent (`ppo_agent.py`)
- **Policy Head**: Uses a **Pointer Network** approach.
    - **Neighbor Head**: Scores adjacent nodes for the next hop.
    - **Vehicle Head**: Scores available vehicles at the current location.
- **Value Head**: Predicts expected reward from the current graph state.
- **Action Masking**: Prevents the agent from selecting invalid neighbors or unavailable vehicles.

---

## 3. Simulation Engines

The environment is driven by specialized engines that calculate real-time dynamics:

- **Anomaly Engine**: Simulates stochastic disruptions.
    - *Types*: Weather, Traffic, Sentiment (Social Unrest), Geopolitical.
    - *Impact*: Affects edge `time_hours` and `risk_penalty` via severity multipliers.
- **Cost Calculator**: Computes multi-component costs.
    - *Components*: Fuel, tolls, maintenance, insurance, and mode-switch penalties.
- **Time Engine**: Handles global temporal factors.
    - *Factors*: Day-of-week demand, holiday effects, and seasonality.

---

## 4. Real-Time Dashboard

A modern web dashboard provides deep visibility into the simulation and agent behavior.

### 4.1 Features
- **Map Visualization**: Interactive Leaflet.js map with CartoDB dark tiles.
- **Disruption Layer**: Real-time highlighting of edges/nodes affected by anomalies (weather, traffic, etc.).
- **Optimal Path Baselines**:
    - **Nominal Shortest Path**: Shows the theoretical best route (dashed green).
    - **Dynamic Optimal Path**: Shows the best route from the current node given nominal costs (dashed blue).
- **Performance Metrics**: Live tracking of steps, cost, risk, and shipment shelf life.
- **Step Log**: Detailed breakdown of every routing decision and its associated rewards.

---

## 5. Project Structure

```
.
├── src/
│   ├── environment/    # SupplyChainEnv, AnomalyEngine, CostCalculator
│   ├── features/       # FeatureEngine (Raw state -> HeteroData)
│   ├── models/         # HGT Encoder, PPO Actor-Critic, Training Logic
│   └── utils/          # Graph helpers, visualization
├── dashboard/          # FastAPI backend + Vanilla JS/Leaflet frontend
├── checkpoints/        # Saved model weights (.pt)
├── tests/              # Pytest suite
└── main.py             # CLI entry point
```

---

## 6. Project Status & Roadmap

### ✅ Completed
- Heterogeneous Graph Environment with 12-category feature mapping.
- Stochastic Anomaly Engine (Weather, Traffic, Sentiment, Geopolitical).
- GNN-RL Pipeline using HGT and PPO.
- Multi-head pointer network for next-hop and vehicle selection.
- High-performance Dashboard with real-time WebSocket streaming.
- Optimal path baselines (Static vs. Dynamic).

### 🚀 Future Roadmap
- [ ] **Multi-Shipment Concurrency**: Managing multiple shipments competing for limited vehicle resources.
- [ ] **Warehouse Allocation**: Agent-driven decisions to store cargo temporarily to wait out disruptions.
- [ ] **Real Data Integration**: Feeding OSM road networks and real weather/news APIs.
- [ ] **Explainability Layer**: Visualizing GNN attention weights to explain *why* the agent rerouted.

---

## 7. Setup & Usage

### Training
```bash
python main.py --mode train --scenario india --episodes 1000
```

### Dashboard
```bash
python main.py --mode dashboard
```
Open `http://localhost:8000` to view the interactive visualization.
