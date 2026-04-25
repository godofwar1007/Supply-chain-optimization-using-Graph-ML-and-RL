# Supply Chain Simulation Engine

This document details the mechanics of the custom Gymnasium environment (`SupplyChainEnv`) used for supply chain routing optimization using Reinforcement Learning (RL). The simulation models a realistic logistics network across Indian cities, integrating spatial, temporal, economic, and stochastic elements.

## 1. High-Level Overview

The agent's objective is to successfully route a shipment (cargo) from a randomly selected source city to a destination city across a transportation graph. At each step, the agent must make two simultaneous decisions:
1. **Which neighboring city to travel to next.**
2. **Which vehicle mode to use for that leg.**

The environment is designed as an episodic Markov Decision Process (MDP). If the shipment takes too long and exceeds its "shelf life," or if the agent takes too many steps, the episode fails. A successful episode results in the shipment arriving at its destination while minimizing time, financial cost, and cargo risk.

---

## 2. Spatial Network (The Graph)

The world is modeled as a directed graph where:
- **Nodes** represent cities or key logistics hubs. Nodes contain metadata such as geographical coordinates (latitude/longitude), region type (e.g., metro, port, rural), and warehouse utilization (if present).
- **Edges** represent transportation routes between cities. Edges have physical characteristics: distance (km), terrain type (flat, hilly, mountainous, coastal), road grading, toll costs, and base travel time.

At the beginning of an episode, the environment randomly samples a source and destination. It performs a feasibility check to ensure the destination is reachable within 70% of the shipment's shelf life under base conditions.

---

## 3. Sub-Engines

The environment relies on three underlying engines to calculate the complex realities of the supply chain step-by-step.

### 3.1 Time Engine (`time_engine.py`)
The simulation tracks continuous time (hours, days of the week, months) and uses it to modulate travel efficiency and costs.
- **Traffic Patterns**: Applies time-of-day multipliers. For example, travel during rush hours (8-10 AM, 5-8 PM) significantly increases travel time (up to 1.6x), while night travel is faster.
- **Seasonality & Weather**: Checks for specific months (like the Indian Monsoon from June to September), which increases the probability of weather-related anomalies.
- **Economic Cycles**: Factors in Indian holidays (e.g., Diwali, Republic Day) and festival seasons, which affect demand congestion and fuel prices.

### 3.2 Anomaly Engine (`anomaly_engine.py`)
To prevent the RL agent from simply memorizing static shortest paths, the `AnomalyEngine` injects stochastic disruptions.
- **Disruption Types**: Models weather events, traffic jams, geopolitical issues, and sentiment shifts.
- **Mechanics**: Anomalies spawn and expire probabilistically at each time step on both edges (routes) and nodes (cities). 
- **Impact**: When active, anomalies apply a "severity multiplier" (e.g., 1.5x) to travel time, travel cost, or increase the overall risk score of a route/node.

### 3.3 Cost Calculator (`cost_calculator.py`)
Computes the exact monetary cost (in ₹) for every single leg traveled. It takes into account:
- **Fuel Cost**: Based on distance, vehicle fuel efficiency, base fuel price (which fluctuates seasonally), and cargo weight (heavier loads consume more fuel linearly).
- **Tolls & Mileage**: Fixed costs depending on the specific route taken.
- **Maintenance Cost**: Scaled by the distance traveled and the age of the vehicle.
- **Mode-Switch Cost**: If the agent decides to change vehicle types (e.g., transferring cargo from a Truck to a Train at a hub), a massive transshipment penalty is applied.
- **Insurance Cost**: Calculated dynamically based on the cargo's declared value, the terrain's inherent risk, and any active anomalies on the route.

---

## 4. Agent Interaction

### 4.1 Observation Space
At each step, the agent receives a rich, flat NumPy array containing:
- **Shipment Profile**: Weight, volume, fragility, temperature sensitivity, value, priority, and remaining shelf life.
- **Node Context**: Encoded features of the current city and the target destination.
- **Neighborhood**: Encoded features for all immediate neighbor cities and the connecting edges (distances, terrains, current anomaly modifiers).
- **Fleet Availability**: Specifications of all available vehicle modes (speed, payload capacity, maintenance cost).
- **Global Context**: Cyclical time encodings (sine/cosine of hour, day, month) and flags for holidays/monsoons, allowing the agent to predict future traffic or weather.

### 4.2 Action Space
A `MultiDiscrete` action space: `[max_neighbors, max_vehicles]`. 
The agent picks an index for the next node and an index for the vehicle type. If the agent chooses an invalid neighbor index, a severe penalty is applied, and it is forced to take the first valid route.

---

## 5. Step Logic & Reward Mechanism

When the agent submits an action, the `step()` function executes the following sequence:

1. **Validation & Configuration**: Validates the chosen route and vehicle. Pulls the physical route configuration.
2. **Speed & Time Calculation**: 
   - Computes effective speed: `Base Speed × Terrain Factor × Road Grading`.
   - Computes actual time: Applies time multipliers from traffic, weather anomalies, and port congestion. Adds a small amount of log-normal noise for realism.
3. **Cost & Risk Calculation**: Invokes the `CostCalculator` for the monetary breakdown. Calculates the cargo risk based on terrain, active anomalies, and shipment fragility.
4. **State Update**: Advances the `TimeEngine` by the travel hours. Ticks the `AnomalyEngine` to spawn/expire disruptions. Updates the shipment's current location.
5. **Reward Calculation**:
   - **Step Penalties**: A weighted negative sum of the normalized time taken, monetary cost incurred, and risk experienced.
   - **Spoilage Penalty**: As the shipment's travel time approaches its shelf life limit, an escalating spoilage penalty is applied. If the shelf life is exceeded entirely, the episode terminates with a massive penalty.
   - **Reward Shaping (Potential-based)**: To solve sparse rewards, the environment calculates the shortest-path distance to the destination. Moving closer yields a positive reward (+5.0 * distance reduced), while moving away yields a negative reward.
   - **Loop Penalty**: Visiting the same node multiple times results in an escalating negative penalty to discourage infinite loops.
   - **Arrival Bonus**: Reaching the destination grants a massive one-time bonus (+50.0).

By balancing these diverse constraints, the RL agent learns to navigate dynamic supply chain shocks, avoiding bad weather, routing around traffic, and switching to cheaper or faster vehicle modes only when economically viable.
