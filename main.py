import gymnasium as gym 
import numpy as np
from collections import deque
import torch
from LunarAgent import Agent

# Configurações
num_episodes = 5000
max_time_steps = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
scores_window = deque(maxlen=100)

env = gym.make("LunarLander-v3", render_mode="human")  
agent = Agent(env.observation_space.shape[0], env.action_space.n)

for episode in range(num_episodes):
    state, _ = env.reset()
    score = 0
    for _ in range(max_time_steps):
        action = agent.act(state, epsilon_start)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    
    scores_window.append(score)
    epsilon_start = max(epsilon_end, epsilon_start * epsilon_decay)

    if episode % 2500 == 0:
        torch.save(agent.local_network.state_dict(), f'LunarLander_ep{episode}.pth')

    if episode % 10 == 0:
        print(f'Episode {episode} - Avg Score: {np.mean(scores_window):.2f}')

    if np.mean(scores_window) >= 200:
        print(f"Solved in {episode} episodes!")
        torch.save(agent.local_network.state_dict(), 'LunarLander_solved.pth')
        break