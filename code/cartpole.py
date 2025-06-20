import gymnasium as gym


env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for t in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
