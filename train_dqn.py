import numpy as np
import torch
from GridDeliveryEnv import GridDeliveryEnv
from dqn_agent import DQNAgent

# --- Hyperparameters ---
GRID_SIZE = 10
NUM_AGENTS = 5
DELIVERY_POINTS = [(6, 2), (3, 7)]
EPISODES = 1000
TARGET_UPDATE_EVERY = 10

# --- Initialize environment ---
env = GridDeliveryEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, delivery_points=DELIVERY_POINTS)

# --- Initialize agents ---
state_size_per_agent = 3  # [x, y, has_package]
action_size = env.num_actions
agents = [DQNAgent(state_size_per_agent, action_size) for _ in range(NUM_AGENTS)]

# --- Training Loop ---
for episode in range(EPISODES):
    states = env.reset()  # List of states, one per agent
    total_rewards = [0 for _ in range(NUM_AGENTS)]
    done = False

    while not done:
        actions = [agents[i].act(states[i]) for i in range(NUM_AGENTS)]
        next_states, rewards, done = env.step(actions)

        for i in range(NUM_AGENTS):
            agents[i].step(states[i], actions[i], rewards[i], next_states[i], done)

        states = next_states
        for i in range(NUM_AGENTS):
            total_rewards[i] += rewards[i]

    for agent in agents:
        agent.decay_epsilon()

    if episode % TARGET_UPDATE_EVERY == 0:
        for agent in agents:
            agent.update_target_network()

    print(f"Episode {episode + 1}/{EPISODES} | Rewards: {total_rewards} | Epsilons: {[round(agent.epsilon, 3) for agent in agents]}")

# Save each agent's model
for i, agent in enumerate(agents):
    torch.save(agent.qnetwork.state_dict(), f'dqn_agent_{i}.pth')

print("Training complete, models saved.")
