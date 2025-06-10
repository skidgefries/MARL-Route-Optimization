# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import torch
# from GridDeliveryEnv import GridDeliveryEnv
# from dqn_agent import DQNAgent

# GRID_SIZE = 5
# NUM_AGENTS = 2
# DELIVERY_POINTS = [(4, 4)]

# # Initialize environment and agent
# env = GridDeliveryEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, delivery_points=DELIVERY_POINTS)

# state_size = 3  # [x, y, has_package] per agent
# action_size = env.num_actions

# agent = DQNAgent(state_size, action_size)
# agent.qnetwork.load_state_dict(torch.load('dqn_agent.pth2'))
# agent.qnetwork.eval()

# states = env.reset()

# fig, ax = plt.subplots()

# def draw_grid():
#     ax.clear()
#     # Draw grid lines
#     for x in range(GRID_SIZE + 1):
#         ax.axhline(x, color='black', lw=1)
#         ax.axvline(x, color='black', lw=1)
#     # Draw delivery points
#     for (dx, dy) in DELIVERY_POINTS:
#         rect = patches.Rectangle((dx, dy), 1, 1, linewidth=2, edgecolor='green', facecolor='lightgreen')
#         ax.add_patch(rect)
#     ax.set_xlim(0, GRID_SIZE)
#     ax.set_ylim(0, GRID_SIZE)
#     ax.set_aspect('equal')
#     ax.invert_yaxis()

# # def draw_agents(agent_positions):
# #     colors = ['red', 'blue', 'orange', 'purple', 'cyan']  # extend for more agents
# #     for i, pos in enumerate(agent_positions):
# #         x, y, has_package = pos
# #         circle = patches.Circle((x + 0.5, y + 0.5), 0.3, facecolor=colors[i % len(colors)], alpha=0.8)
# #         ax.add_patch(circle)
# #         edge_color = 'yellow' if has_package else 'black'
# #         circle.set_edgecolor(edge_color)
# #         circle.set_linewidth(2)
# #         ax.text(x + 0.5, y + 0.5, str(i), color='white', ha='center', va='center', fontsize=12)

# def draw_agents(agent_positions):
#     colors = ['red', 'blue', 'orange', 'purple', 'cyan']  # extend colors if needed
#     offset_vals = [(-0.15, -0.15), (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (0, 0)]
#     for i, pos in enumerate(agent_positions):
#         x, y, has_package = pos
#         offset_x, offset_y = offset_vals[i % len(offset_vals)]
#         circle = patches.Circle((x + 0.5 + offset_x, y + 0.5 + offset_y), 0.25, facecolor=colors[i % len(colors)], alpha=0.8)
#         ax.add_patch(circle)
#         edge_color = 'yellow' if has_package else 'black'
#         circle.set_edgecolor(edge_color)
#         circle.set_linewidth(2)
#         ax.text(x + 0.5 + offset_x, y + 0.5 + offset_y, str(i), color='white', ha='center', va='center', fontsize=12)


# done = False
# while not done:
#     draw_grid()
#     draw_agents(states)
#     plt.title('Multi-Agent Delivery Visualization')
#     plt.draw()
#     plt.pause(0.1)

#     actions = [agent.act(state, use_epsilon=False) for state in states]
#     next_states, rewards, done = env.step(actions)
#     states = next_states

# plt.show()


# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib import animation

# from GridDeliveryEnv import GridDeliveryEnv
# from dqn_agent import DQNAgent

# # --- Setup ---
# GRID_SIZE = 5
# NUM_AGENTS = 2
# DELIVERY_POINTS = [(4, 4)]
# MODEL_PATHS = [f'dqn_agent_{i}.pth' for i in range(NUM_AGENTS)]

# env = GridDeliveryEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, delivery_points=DELIVERY_POINTS)

# state_size = 3  # [x, y, has_package]
# action_size = env.num_actions

# # Load agents and their trained networks
# agents = []
# for path in MODEL_PATHS:
#     agent = DQNAgent(state_size, action_size)
#     agent.qnetwork.load_state_dict(torch.load(path))
#     agent.qnetwork.eval()
#     agents.append(agent)

# states = env.reset()

# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_xlim(-0.5, GRID_SIZE - 0.5)
# ax.set_ylim(-0.5, GRID_SIZE - 0.5)
# ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
# ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
# ax.grid(True)

# # Draw delivery points
# for point in DELIVERY_POINTS:
#     rect = patches.Rectangle((point[0]-0.5, point[1]-0.5), 1, 1, linewidth=2, edgecolor='green', facecolor='lightgreen')
#     ax.add_patch(rect)

# agent_circles = []
# agent_labels = []
# colors = ['red', 'blue', 'orange', 'purple', 'cyan']

# for i in range(NUM_AGENTS):
#     circle = plt.Circle((states[i][0], states[i][1]), 0.3, color=colors[i % len(colors)])
#     ax.add_patch(circle)
#     agent_circles.append(circle)
#     label = ax.text(states[i][0], states[i][1], str(i), fontsize=12, ha='center', va='center', color='white', weight='bold')
#     agent_labels.append(label)

# def update(frame):
#     global states
#     if env.done:
#         return agent_circles + agent_labels

#     actions = []
#     for i in range(NUM_AGENTS):
#         state_tensor = torch.FloatTensor(states[i])
#         action = agents[i].act(state_tensor.detach().numpy(), use_epsilon=False)  # greedy
#         actions.append(action)

#     next_states, rewards, done = env.step(actions)
#     states = next_states

#     for i in range(NUM_AGENTS):
#         agent_circles[i].center = (states[i][0], states[i][1])
#         agent_labels[i].set_position((states[i][0], states[i][1]))

#     return agent_circles + agent_labels

# ani = animation.FuncAnimation(fig, update, frames=100, interval=500, blit=True, repeat=False)

# plt.title("Multi-Agent Delivery Route Visualization")
# plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from GridDeliveryEnv import GridDeliveryEnv
from dqn_agent import DQNAgent

GRID_SIZE = 10
NUM_AGENTS = 5
DELIVERY_POINTS = [(6, 2), (3, 7)]  # Now valid within 10x10 grid

# Sanity check
for dx, dy in DELIVERY_POINTS:
    if dx >= GRID_SIZE or dy >= GRID_SIZE:
        raise ValueError(f"Delivery point ({dx},{dy}) is outside grid size {GRID_SIZE}")

# Initialize environment and agents
env = GridDeliveryEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS, delivery_points=DELIVERY_POINTS)

state_size = 3  # [x, y, has_package] per agent
action_size = env.num_actions

agents = []
for i in range(NUM_AGENTS):
    agent = DQNAgent(state_size, action_size)
    agent.qnetwork.load_state_dict(torch.load(f'dqn_agent_{i}.pth'))
    agent.qnetwork.eval()
    agents.append(agent)

states = env.reset()

fig, ax = plt.subplots()

def draw_grid():
    ax.clear()
    for x in range(GRID_SIZE + 1):
        ax.axhline(x, color='black', lw=1)
        ax.axvline(x, color='black', lw=1)
    for (dx, dy) in DELIVERY_POINTS:
        rect = patches.Rectangle((dx, dy), 1, 1, linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(rect)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

def draw_agents(agent_positions):
    colors = ['red', 'blue', 'orange', 'purple', 'cyan']
    offset_vals = [(-0.15, -0.15), (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (0, 0)]
    for i, pos in enumerate(agent_positions):
        x, y, has_package = pos
        offset_x, offset_y = offset_vals[i % len(offset_vals)]
        circle = patches.Circle((x + 0.5 + offset_x, y + 0.5 + offset_y), 0.25, facecolor=colors[i % len(colors)], alpha=0.8)
        ax.add_patch(circle)
        edge_color = 'yellow' if has_package else 'black'
        circle.set_edgecolor(edge_color)
        circle.set_linewidth(2)
        ax.text(x + 0.5 + offset_x, y + 0.5 + offset_y, str(i), color='white', ha='center', va='center', fontsize=12)

done = False
while not done:
    draw_grid()
    draw_agents(states)
    plt.title('Multi-Agent Delivery Visualization')
    plt.draw()
    plt.pause(0.5)

    actions = [agent.act(state, use_epsilon=False) for agent, state in zip(agents, states)]
    next_states, rewards, done = env.step(actions)
    states = next_states

plt.show()
