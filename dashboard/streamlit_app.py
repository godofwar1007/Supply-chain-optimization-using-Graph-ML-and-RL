"""
Streamlit Dashboard — Supply Chain RL Visualization.

Launch:  streamlit run dashboard/streamlit_app.py

Features:
  - Interactive graph with nodes colored by anomaly risk
  - Edges colored by anomaly intensity (green → red)
  - Agent path overlay during live evaluation
  - Real-time metrics: delivery rate, avg reward, travel time
  - Curriculum phase indicator
  - Manual anomaly injection for demo
"""

import os
import sys
import json
import random
import tempfile
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.config.scenarios import india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic


# ═══════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="🚚",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #58a6ff; }
    .stMetric label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 Supply Chain RL — Dashboard")


# ═══════════════════════════════════════════════════════════════════════
# Cached helpers
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_scenario():
    config = india_scenario()
    env = SupplyChainEnv(config, render_mode=None)
    return config, env


@st.cache_resource
def load_agent(_env):
    """Try to load the best trained agent."""
    ckpt_dir = ROOT / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_engine = FeatureEngine()

    for name in ["best_model.pt", "final_model.pt", "latest_model.pt"]:
        path = ckpt_dir / name
        if path.exists():
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
                max_neighbors=_env.max_neighbors,
                max_vehicles=_env.max_vehicles,
            ).to(device)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            agent.load_state_dict(checkpoint["model_state_dict"])
            agent.eval()
            return agent, feature_engine, device, name, checkpoint.get("episode", "?")

    return None, feature_engine, device, None, None


# ═══════════════════════════════════════════════════════════════════════
# Graph visualisation with matplotlib
# ═══════════════════════════════════════════════════════════════════════

def draw_graph(config, env, path_taken=None, anomaly_edges=None):
    """Draw the supply chain graph with anomaly-colored edges."""
    G = env.graph
    pos = {}
    for loc in config.locations:
        pos[loc.name] = (loc.lng, loc.lat)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Edge colors based on anomaly intensity
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        factor = env.anomaly_engine.edge_time_factor(u, v)
        # Green (1.0) → Yellow (1.5) → Red (2.5+)
        intensity = min((factor - 1.0) / 1.5, 1.0)
        r = min(intensity * 2, 1.0)
        g = max(1.0 - intensity * 2, 0.0)
        edge_colors.append((r, g, 0.2, 0.5))
        edge_widths.append(0.5 + intensity * 2)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.05",
    )

    # Path overlay
    if path_taken and len(path_taken) > 1:
        path_edges = list(zip(path_taken[:-1], path_taken[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, ax=ax,
            edge_color="#58a6ff",
            width=3.0,
            arrows=True,
            arrowsize=12,
            connectionstyle="arc3,rad=0.05",
        )

    # Node colors based on risk
    node_colors = []
    for n in G.nodes():
        risk = env.anomaly_engine.node_risk_score(n)
        if path_taken and n in path_taken:
            node_colors.append("#58a6ff")
        elif risk > 0.5:
            node_colors.append("#f85149")
        elif risk > 0.2:
            node_colors.append("#f0883e")
        else:
            node_colors.append("#3fb950")

    node_sizes = []
    for loc in config.locations:
        size = 200
        if loc.region_type == "metro":
            size = 400
        elif loc.region_type in ("port", "hub"):
            size = 300
        node_sizes.append(size)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#30363d",
        linewidths=1.0,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=6,
        font_color="#c9d1d9",
        font_weight="bold",
    )

    ax.set_title(
        "Supply Chain Network — India",
        color="#c9d1d9", fontsize=14, fontweight="bold", pad=10
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Sidebar controls
# ═══════════════════════════════════════════════════════════════════════

st.sidebar.header("⚙️ Controls")
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42)
agent_mode = st.sidebar.selectbox("Agent mode", ["Trained Agent", "Random"])

# Manual anomaly injection
st.sidebar.markdown("---")
st.sidebar.subheader("💥 Manual Anomaly Injection")
inject_node = st.sidebar.selectbox("Target node", ["(none)"] + india_scenario().location_names())
inject_type = st.sidebar.selectbox("Anomaly type", ["weather", "traffic", "sentiment", "geopolitical"])
inject_severity = st.sidebar.slider("Severity", 1.0, 3.0, 2.0, 0.1)

run_button = st.sidebar.button("▶️ Run Episode", type="primary")


# ═══════════════════════════════════════════════════════════════════════
# Training metrics (if available)
# ═══════════════════════════════════════════════════════════════════════

metrics_path = ROOT / "checkpoints" / "training_metrics.csv"

tab1, tab2, tab3 = st.tabs(["🗺️ Live Simulation", "📊 Training Metrics", "ℹ️ System Info"])

with tab2:
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_eps = len(df)
            st.metric("Total Episodes", total_eps)
        with col2:
            if "delivered" in df.columns:
                del_rate = df["delivered"].tail(100).mean() * 100
                st.metric("Delivery Rate (last 100)", f"{del_rate:.0f}%")
        with col3:
            if "reward" in df.columns:
                avg_r = df["reward"].tail(100).mean()
                st.metric("Avg Reward (last 100)", f"{avg_r:.1f}")
        with col4:
            if "phase" in df.columns:
                phase = df["phase"].iloc[-1]
                labels = {1: "Easy", 2: "Medium", 3: "Full"}
                st.metric("Curriculum Phase", f"{phase} ({labels.get(phase, '?')})")

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        if "reward" in df.columns:
            roll = df["reward"].rolling(50).mean()
            axes[0].plot(df["episode"], df["reward"], alpha=0.2, color="#58a6ff")
            axes[0].plot(df["episode"], roll, color="#58a6ff", linewidth=2)
            axes[0].set_title("Reward", color="#c9d1d9")

        if "delivered" in df.columns:
            roll_d = df["delivered"].astype(float).rolling(100).mean() * 100
            axes[1].plot(df["episode"], roll_d, color="#3fb950", linewidth=2)
            axes[1].set_ylim(-5, 105)
            axes[1].set_title("Delivery Rate %", color="#c9d1d9")

        if "total_time_hours" in df.columns:
            roll_t = df["total_time_hours"].rolling(50).mean()
            axes[2].plot(df["episode"], roll_t, color="#79c0ff", linewidth=2)
            axes[2].set_title("Avg Travel Time (h)", color="#c9d1d9")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No training metrics found. Run `python main.py train` first.")

with tab3:
    config, _ = load_scenario()
    st.markdown(f"""
    **Network:** {len(config.locations)} nodes, {len(config.routes)} routes (bidirectional)

    **Anomaly Config (base rates):**
    | Type | Appear | Disappear |
    |---|---|---|
    | Weather | 0.30 | 0.05 |
    | Traffic | 0.40 | 0.10 |
    | Sentiment | 0.15 | 0.02 |
    | Geopolitical | 0.10 | 0.02 |

    **Agent:** HGT encoder (2 layers, 4 heads, 64d) + Pointer-network Actor-Critic

    **Training:** PPO + GAE, 5000 episodes, 3-phase curriculum
    """)


# ═══════════════════════════════════════════════════════════════════════
# Live simulation
# ═══════════════════════════════════════════════════════════════════════

with tab1:
    if run_button:
        config, env = load_scenario()
        agent_obj, feature_engine, device, model_name, model_ep = load_agent(env)

        use_agent = (agent_mode == "Trained Agent" and agent_obj is not None)
        if agent_mode == "Trained Agent" and agent_obj is None:
            st.warning("No trained model found — using random actions.")

        obs, info = env.reset(seed=seed)

        # Manual anomaly injection
        if inject_node != "(none)":
            from src.environment.anomaly_engine import ActiveAnomaly
            anom = ActiveAnomaly(
                anomaly_type=inject_type,
                severity=inject_severity,
                cost_multiplier=1.5 if inject_type in ("sentiment", "geopolitical") else 1.0,
            )
            if inject_node in env.anomaly_engine.node_anomalies:
                env.anomaly_engine.node_anomalies[inject_node].append(anom)
            # Also inject on all edges from that node
            for key in env.anomaly_engine.edge_anomalies:
                if key[0] == inject_node or key[1] == inject_node:
                    env.anomaly_engine.edge_anomalies[key].append(ActiveAnomaly(
                        anomaly_type=inject_type,
                        severity=inject_severity,
                        cost_multiplier=anom.cost_multiplier,
                    ))
            st.success(f"💥 Injected {inject_type} anomaly (severity {inject_severity}x) at **{inject_node}**")

        # Run episode
        done = False
        truncated = False
        total_reward = 0.0
        step_log = []

        while not (done or truncated):
            if use_agent:
                state_dict = env.get_graph_state()
                hetero_data = feature_engine.build(state_dict).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent_obj(hetero_data)
                action = action.cpu().numpy()
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            leg = env.leg_details[-1]
            step_log.append({
                "Step": env.step_count,
                "From": leg["from"],
                "To": leg["to"],
                "Vehicle": leg["vehicle_type"],
                "Time (h)": round(leg["time_hours"], 1),
                "Cost (₹)": round(leg["cost"].total, 0),
                "Risk": round(leg["risk"], 3),
                "Anomalies": len(leg.get("anomalies", [])),
            })

        # Results
        delivered = env.current_node == env.destination
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Status", "✅ Delivered" if delivered else "❌ Failed")
        with col2:
            st.metric("Steps", env.step_count)
        with col3:
            st.metric("Total Time", f"{env.total_time_hours:.1f}h")
        with col4:
            st.metric("Total Cost", f"₹{env.total_cost:,.0f}")
        with col5:
            st.metric("Total Reward", f"{total_reward:.1f}")

        st.markdown(f"**Path:** `{' → '.join(env.path_taken)}`")
        st.markdown(f"**Shipment:** {env.shipment.product_type} "
                    f"({env.shipment.weight_kg:.0f}kg, {env.shipment.priority} priority)")

        # Graph
        fig = draw_graph(config, env, path_taken=env.path_taken)
        st.pyplot(fig)
        plt.close()

        # Step log table
        st.subheader("📋 Step-by-Step Log")
        st.dataframe(pd.DataFrame(step_log), use_container_width=True)

    else:
        # Show static graph on first load
        config, env = load_scenario()
        env.reset(seed=0)
        fig = draw_graph(config, env)
        st.pyplot(fig)
        plt.close()
        st.info("Press **▶️ Run Episode** in the sidebar to start a simulation.")
