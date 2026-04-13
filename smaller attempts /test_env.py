from environment import supply_chain_env_fixed_route
import gymnasium as gym

env = supply_chain_env_fixed_route()
obs, info = env.reset()
done = False
total_reward = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
print(f"Episode finished. Total reward: {total_reward:.2f} (negative total time: {info['total_time']:.2f} min)")
