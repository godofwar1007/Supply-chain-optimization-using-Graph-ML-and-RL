from src.config.scenarios import india_scenario, small_scenario
from src.environment.supply_chain_env import SupplyChainEnv


def main():
    config = india_scenario()

    env = SupplyChainEnv(config, render_mode="human")

    # Initial observation
    obs, info = env.reset()
    print("Initial observation:", obs)

    # Run one episode
    done = False
    truncated = False
    while not (done or truncated):
        action = env.action_space.sample()  # random actions
        obs, reward, done, truncated, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
