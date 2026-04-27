# Supply Chain Optimization with Graph ML & RL

This project is a production-grade supply chain optimization system that uses **Heterogeneous Graph Transformers (HGT)** and **Reinforcement Learning (PPO)** to solve dynamic routing and vehicle selection problems.

## Project Overview

- **Core Technology**: Python 3.13, PyTorch, PyTorch Geometric (GNN), Gymnasium (RL Environment), and `uv` for dependency management.
- **Architecture**:
    - **Environment**: A custom Gymnasium environment (`SupplyChainEnv`) that models a heterogeneous graph of Indian cities (nodes) and transport routes (edges).
    - **Anomaly Engine**: Simulates stochastic disruptions (weather, traffic, sentiment, geopolitics) that affect route costs and risks.
    - **Model**: An Actor-Critic architecture where an HGT encoder processes the graph state and a Pointer Network selects the next hop and vehicle mode.
    - **Dashboard**: A modern, WebSocket-based dashboard (FastAPI + Leaflet.js) for real-time simulation streaming and visualization.

## Building and Running

### Dependency Management
The project uses `uv`. To set up the environment:
```bash
uv sync
```

### Key Commands
All major operations are accessible via `main.py`:

- **Train the Agent**:
  ```bash
  python main.py train
  ```
  Runs PPO training with a 3-phase curriculum (increasing anomaly frequency).

- **Launch Interactive Dashboard**:
  ```bash
  python main.py dashboard
  ```
  Starts the FastAPI server on port 8000. Open `http://localhost:8000` for the map interface.

- **Run Evaluation**:
  ```bash
  python main.py eval    # Runs one episode with the best trained agent
  python main.py random  # Runs one episode with random baseline
  ```

- **Run Tests**:
  ```bash
  pytest                            # Run all tests using pytest (recommended)
  python tests/test_environment.py  # Run environment smoke test manually
  python tests/test_gnn.py          # Run GNN unit test manually
  python tests/test_ppo.py          # Run PPO unit test manually
  ```

## Testing Strategy
The project includes a suite of verification scripts in the `tests/` directory, fully compatible with `pytest`:
- **Environment Smoke Test**: `test_environment.py` runs random episodes to ensure no crashes, valid observation shapes, and sensible reward ranges for both Small and India scenarios.
- **GNN Unit Test**: `test_gnn.py` verifies the forward pass of the HGT encoder, checking output embedding shapes for all node types.
- **PPO Unit Test**: `test_ppo.py` validates both action selection and action evaluation (log-probability calculation) for the Actor-Critic model.

## Development Conventions

- **Graph Structure**: Uses `torch_geometric.data.HeteroData`. The schema includes `Location`, `Warehouse`, `Vehicle`, and `Shipment` nodes with specific connectivity (`route`, `at`, `dest`).
- **Checkpoints**: Models are saved in the `checkpoints/` directory as `best_model.pt`, `final_model.pt`, or `latest_model.pt`.
- **Environment**: The environment state is fully observable as a graph, but engines (`AnomalyEngine`, `TimeEngine`, `CostCalculator`) handle the stochastic and temporal logic.

## Deployment

The project is optimized for **Google Cloud Run**.
- **Dockerfile**: Uses a slim Python 3.13 image and `uv` for minimal footprint.
- **Cloud Run Setup**: Supports WebSockets and handles scaling to zero. Use `--no-cpu-throttling` to keep the simulation logic responsive during active WebSocket sessions.
