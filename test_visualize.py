import torch
import matplotlib.pyplot as plt
import numpy as np
from GridDeliveryEnv import GridDeliveryEnv
from dqn_agent import DQNAgent

# --- Config ---
GRID_SIZE = 5
NUM_AGENTS = 1
DELIVERY_POINTS = [(3, 4)]

# --- Load environment and agent ---
env = GridDeliveryEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, delivery_points=DELIVERY_POINTS)
state_size = 3 * NUM_AGENTS
action_size = env.num_actions

agent = DQNAgent(state_size, action_size)
agent.qnetwork.load_state_dict(torch.load("dqn_agent.pth1"))
agent.qnetwork.eval()

# --- Run a single episode for visualization ---
state = env.reset()[0]
path = [env.agent_positions[0]]  # store path for plotting

done = False
while not done:
    action = agent.act(state, use_epsilon=False)  # always use best action
    next_states, rewards, done = env.step([action])
    state = next_states[0]
    path.append(env.agent_positions[0])

# --- Plot path ---
x_coords, y_coords = zip(*path)

fig, ax = plt.subplots()
ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(-0.5, GRID_SIZE - 0.5)
ax.set_xticks(np.arange(0, GRID_SIZE))
ax.set_yticks(np.arange(0, GRID_SIZE))
ax.grid(True)

# Plot start and delivery point
ax.plot(x_coords, y_coords, marker='o', color='blue', label='Agent Path')
ax.plot(x_coords[0], y_coords[0], marker='s', color='green', markersize=10, label='Start')
ax.plot(3, 4, marker='*', color='red', markersize=15, label='Delivery')

ax.set_title("Agent Path from Start to Delivery")
ax.invert_yaxis()  # optional: (0,0) at top-left
ax.legend()
plt.show()
