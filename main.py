"""
Entry point for the Supply Chain Optimization system.

Usage:
    python main.py train       # Run PPO training with curriculum
    python main.py eval        # Run one episode with trained agent
    python main.py random      # Run one episode with random actions
    python main.py dashboard   # Launch the FastAPI dashboard server
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_training():
    """Launch PPO training."""
    from src.models.train import train_ppo
    train_ppo()


def run_eval(mode: str = "trained"):
    """Run a single episode with the trained or random agent."""
    import torch
    from src.config.scenarios import india_scenario
    from src.environment.supply_chain_env import SupplyChainEnv
    from src.features.feature_engine import FeatureEngine
    from src.models.ppo_agent import ActorCritic

    config = india_scenario()
    env = SupplyChainEnv(config, render_mode="human")
    obs, info = env.reset()

    agent = None
    feature_engine = None

    if mode == "trained":
        ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
        for name in ["best_model.pt", "final_model.pt", "latest_model.pt"]:
            path = os.path.join(ckpt_dir, name)
            if os.path.exists(path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                feature_engine = FeatureEngine()

                metadata = (
                    ['location', 'vehicle', 'shipment'],
                    [('location', 'route', 'location'),
                     ('vehicle', 'vehicle_at', 'location'),
                     ('shipment', 'shipment_at', 'location'),
                     ('shipment', 'shipment_dest', 'location'),
                     ('location', 'rev_vehicle_at', 'vehicle'),
                     ('location', 'rev_shipment_at', 'shipment'),
                     ('location', 'rev_shipment_dest', 'shipment')]
                )

                agent = ActorCritic(
                    metadata=metadata,
                    hidden_channels=64,
                    out_channels=64,
                    max_neighbors=env.max_neighbors,
                    max_vehicles=env.max_vehicles,
                ).to(device)

                checkpoint = torch.load(path, map_location=device, weights_only=False)
                agent.load_state_dict(checkpoint["model_state_dict"])
                agent.eval()
                print(f"Loaded model from {name} (ep {checkpoint.get('episode', '?')})")
                break
        else:
            print("No trained model found, falling back to random actions")
            mode = "random"

    done = False
    truncated = False
    while not (done or truncated):
        if mode == "trained" and agent is not None:
            state_dict = env.get_graph_state()
            hetero_data = feature_engine.build(state_dict).to(device)
            with torch.no_grad():
                action, _, _, _ = agent(hetero_data)
            action = action.cpu().numpy()
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        env.render()


def run_dashboard():
    """Launch the FastAPI dashboard server."""
    import uvicorn
    print("Starting dashboard at http://localhost:8000")
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Defaulting to: train\n")
        run_training()
        return

    command = sys.argv[1].lower()
    if command == "train":
        run_training()
    elif command == "eval":
        run_eval("trained")
    elif command == "random":
        run_eval("random")
    elif command == "dashboard":
        run_dashboard()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
