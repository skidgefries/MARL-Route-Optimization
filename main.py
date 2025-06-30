import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Lightning AI compatibility
import matplotlib.pyplot as plt
from collections import deque
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------  Environment Parameters  ------------------------ #
GRID_SIZE   = 10
NUM_AGENTS  = 5
EPISODES    = 3000
STEPS       = 100

# ------------------------  DQN Hyper-Parameters  -------------------------- #
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.01
EPSILON_DECAY   = 0.9995

LR              = 0.0005
BATCH_SIZE      = 32
MEMORY_CAPACITY = 10000
TARGET_UPDATE   = 100

# Reward parameters
GOAL_REWARD = 50
COLLISION_PENALTY = -5
DISTANCE_REWARD_SCALE = 2
STEP_PENALTY = -0.1

# Actions: right, left, down, up
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# =========================  Environment  ================================== #
class GridEnv:
    def __init__(self):
        # Define actions within the class: right, left, down, up
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.reset()
        self.max_distance = np.sqrt(2) * (GRID_SIZE - 1)

    def reset(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Random starting positions
        self.agent_pos = []
        for _ in range(NUM_AGENTS):
            while True:
                pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
                if pos not in self.agent_pos:
                    self.agent_pos.append(pos)
                    break
        
        # Diverse goals
        goal_positions = [(GRID_SIZE-1, GRID_SIZE-1), (0, GRID_SIZE-1), (GRID_SIZE-1, 0),
                         (GRID_SIZE//2, GRID_SIZE//2), (GRID_SIZE//4, 3*GRID_SIZE//4)]
        self.agent_goals = []
        for i in range(NUM_AGENTS):
            self.agent_goals.append(goal_positions[i % len(goal_positions)])
        
        return self.get_state()

    def get_state(self):
        state = []
        for i in range(NUM_AGENTS):
            x, y = self.agent_pos[i]
            gx, gy = self.agent_goals[i]
            
            # Normalize and add relative information
            rel_x = float(gx - x) / GRID_SIZE
            rel_y = float(gy - y) / GRID_SIZE
            distance = float(np.sqrt(rel_x**2 + rel_y**2))
            
            state.extend([
                float(x) / GRID_SIZE, float(y) / GRID_SIZE,
                float(gx) / GRID_SIZE, float(gy) / GRID_SIZE,
                rel_x, rel_y, distance
            ])
        return np.array(state, dtype=np.float32)

    def step(self, actions_idx):
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)
        next_positions = []
        pos_counts = {}

        # Calculate next positions
        for i, idx in enumerate(actions_idx):
            # Ensure idx is an integer and within bounds
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if len(idx) > 0 else 0
            idx = int(idx) % len(self.actions)  # Ensure valid action index
            
            dx, dy = self.actions[idx]
            x, y = self.agent_pos[i]
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                next_positions.append((nx, ny))
            else:
                next_positions.append((x, y))
                rewards[i] = rewards[i] - 2.0

        # Count collisions
        for pos in next_positions:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        # Update positions and calculate rewards
        for i in range(NUM_AGENTS):
            pos = next_positions[i]
            goal = self.agent_goals[i]
            old_pos = self.agent_pos[i]
            
            # Ensure positions are tuples
            if not isinstance(pos, tuple):
                pos = tuple(pos)
            if not isinstance(goal, tuple):
                goal = tuple(goal)
            if not isinstance(old_pos, tuple):
                old_pos = tuple(old_pos)

            rewards[i] = rewards[i] + STEP_PENALTY

            if pos_counts[pos] > 1:
                rewards[i] = rewards[i] + COLLISION_PENALTY
                continue

            old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal))
            new_dist = np.linalg.norm(np.array(pos) - np.array(goal))
            
            if pos == goal:
                rewards[i] = rewards[i] + GOAL_REWARD
            else:
                dist_improvement = float(old_dist - new_dist)
                rewards[i] = rewards[i] + (DISTANCE_REWARD_SCALE * dist_improvement)

            self.agent_pos[i] = pos

        return self.get_state(), rewards.tolist()

# =========================  Neural Network  =============================== #
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.out(x)

# =========================  DQN Wrapper  ================================== #
class DQN:
    def __init__(self, input_dim, output_dim):
        self.eval_net = Net(input_dim, output_dim).to(device)
        self.target_net = Net(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.steps = 0
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return [random.randint(0, 3) for _ in range(NUM_AGENTS)]
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.eval_net(state_tensor)
            
            # Ensure proper reshaping
            expected_size = NUM_AGENTS * 4
            if q_values.numel() != expected_size:
                print(f"Warning: Q-values size mismatch. Expected {expected_size}, got {q_values.numel()}")
                return [random.randint(0, 3) for _ in range(NUM_AGENTS)]
            
            # Reshape and convert to numpy for safe indexing
            q_values_np = q_values.cpu().numpy().reshape(NUM_AGENTS, 4)
            actions_list = []
            
            for i in range(NUM_AGENTS):
                best_action = int(np.argmax(q_values_np[i]))
                actions_list.append(best_action)
                
            return actions_list

    def store_transition(self, s, a, r, s_, done=False):
        self.memory.append((s, a, r, s_, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, BATCH_SIZE)
        
        # Prepare batch data
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        # Current Q values
        current_q = self.eval_net(states_tensor)
        current_q = current_q.view(BATCH_SIZE, NUM_AGENTS, 4)

        # Next Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_tensor)
            next_q = next_q.view(BATCH_SIZE, NUM_AGENTS, 4)
            next_q_max = torch.max(next_q, dim=2)[0]

        # Compute target using proper tensor operations (Lightning AI friendly)
        # Calculate target values without loops
        target_values = torch.where(
            dones_tensor.unsqueeze(1).expand(-1, NUM_AGENTS),
            rewards_tensor,
            rewards_tensor + GAMMA * next_q_max
        )
        
        # Create target Q values using scatter operation
        target_q = current_q.clone()
        
        # Use advanced indexing to update target_q
        batch_indices = torch.arange(BATCH_SIZE, device=device).unsqueeze(1).expand(-1, NUM_AGENTS)
        agent_indices = torch.arange(NUM_AGENTS, device=device).unsqueeze(0).expand(BATCH_SIZE, -1)
        
        target_q[batch_indices, agent_indices, actions_tensor] = target_values

        # Compute loss
        loss = self.loss_fn(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # Decay epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        return loss.item()

# ========================= IMPROVEMENT 1: Random Baseline ================= #
def random_baseline():
    """Compare against random policy"""
    print("Running random baseline...")
    env = GridEnv()
    random_rewards = []
    random_success = []
    
    for ep in range(100):  # Quick 100 episodes
        state = env.reset()
        ep_reward = 0
        
        for step in range(STEPS):
            actions = [random.randint(0, 3) for _ in range(NUM_AGENTS)]
            next_state, rewards = env.step(actions)
            ep_reward += sum(rewards)
            state = next_state
        
        goals_reached = sum(1 for i in range(NUM_AGENTS) 
                          if env.agent_pos[i] == env.agent_goals[i])
        random_rewards.append(ep_reward)
        random_success.append(goals_reached / NUM_AGENTS)
    
    return np.mean(random_rewards), np.mean(random_success)

# ========================= IMPROVEMENT 2: Convergence Analysis ============ #
def analyze_convergence(success_rates, episodes):
    """Find when training converged"""
    # Find when training "converged" (success rate > 80% for 50 episodes)
    convergence_point = None
    convergence_threshold = 0.8
    window_size = 50
    
    if len(success_rates) < window_size:
        return None
        
    for i in range(window_size, len(success_rates)):
        if np.mean(success_rates[i-window_size:i]) > convergence_threshold:
            convergence_point = i
            break
    
    return convergence_point

# ========================= IMPROVEMENT 3: Agent Behavior Analysis ========= #
def analyze_final_behavior(agent, env):
    """Test trained agents on multiple scenarios"""
    print("Analyzing final agent behavior...")
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Pure exploitation
    
    scenarios = []
    for test in range(10):
        state = env.reset()
        steps_taken = 0
        collisions = 0
        total_reward = 0
        
        for step in range(STEPS):
            actions = agent.choose_action(state)
            next_state, rewards = env.step(actions)
            
            steps_taken += 1
            collisions += sum(1 for r in rewards if r < -3)  # Collision penalty
            total_reward += sum(rewards)
            
            if all(env.agent_pos[i] == env.agent_goals[i] for i in range(NUM_AGENTS)):
                break
                
            state = next_state
        
        final_success = sum(1 for i in range(NUM_AGENTS) 
                          if env.agent_pos[i] == env.agent_goals[i]) / NUM_AGENTS
        scenarios.append({
            "steps": steps_taken, 
            "collisions": collisions,
            "reward": total_reward,
            "success": final_success
        })
    
    agent.epsilon = original_epsilon  # Restore original epsilon
    
    avg_steps = np.mean([s["steps"] for s in scenarios])
    avg_collisions = np.mean([s["collisions"] for s in scenarios])
    avg_reward = np.mean([s["reward"] for s in scenarios])
    avg_success = np.mean([s["success"] for s in scenarios])
    
    return {
        "avg_steps": avg_steps,
        "avg_collisions": avg_collisions,
        "avg_reward": avg_reward,
        "avg_success": avg_success,
        "scenarios": scenarios
    }

# ========================= IMPROVEMENT 4: Real-time Visualization ========= #
def visualize_episode(env, agent, episode_num, show_steps=15):
    """Show agent movements for one episode"""
    if episode_num % 500 != 0:  # Only every 500 episodes
        return
        
    print(f"\n=== Episode {episode_num} Visualization ===")
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Show learned behavior
    
    state = env.reset()
    
    for step in range(show_steps):
        actions = agent.choose_action(state)
        next_state, rewards = env.step(actions)
        
        # Create grid visualization
        grid_vis = np.full((GRID_SIZE, GRID_SIZE), '.', dtype=object)
        
        # Mark goals first
        for i, goal in enumerate(env.agent_goals):
            grid_vis[goal[0], goal[1]] = f'G{i}'
        
        # Mark agents (they override goals if same position)
        for i, pos in enumerate(env.agent_pos):
            if grid_vis[pos[0], pos[1]] == f'G{i}':
                grid_vis[pos[0], pos[1]] = f'✓{i}'  # Agent reached goal
            else:
                grid_vis[pos[0], pos[1]] = f'A{i}'
        
        print(f"\nStep {step + 1}:")
        print("  " + " ".join([str(i) for i in range(GRID_SIZE)]))
        for row_idx, row in enumerate(grid_vis):
            print(f"{row_idx} " + " ".join([str(cell).ljust(2) for cell in row]))
        
        # Show rewards
        print(f"Rewards: {[f'{r:.1f}' for r in rewards]}")
        print(f"Actions: {actions}")
        
        # Check if all reached goals
        if all(env.agent_pos[i] == env.agent_goals[i] for i in range(NUM_AGENTS)):
            print("🎉 All agents reached their goals!")
            break
            
        state = next_state
        time.sleep(0.5)  # Pause for visualization
    
    agent.epsilon = original_epsilon

# =========================  Training Loop  ================================ #
def moving_average(x, w=100):
    if len(x) < w:
        return np.array([])
    return np.convolve(x, np.ones(w)/w, mode='valid')

# Initialize
env = GridEnv()
input_dim = NUM_AGENTS * 7
output_dim = NUM_AGENTS * 4
agent = DQN(input_dim, output_dim)

# Run random baseline first
random_reward, random_success = random_baseline()

# Tracking
total_rewards = []
avg_reward_per_agent = []
success_rates = []
epsilon_history = []
losses = []

print("Starting optimized MARL training...")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Agents: {NUM_AGENTS}, Episodes: {EPISODES}")
print(f"Random baseline - Reward: {random_reward:.2f}, Success: {random_success:.2%}")

for ep in range(EPISODES):
    state = env.reset()
    ep_reward = 0.0
    per_agent_reward = np.zeros(NUM_AGENTS)
    goals_reached = 0
    episode_losses = []

    # Show visualization for some episodes
    if ep % 500 == 0 and ep > 0:
        visualize_episode(env, agent, ep)

    for step in range(STEPS):
        actions = agent.choose_action(state)
        next_state, rewards = env.step(actions)
        
        # Check if all agents reached goals
        done = all(env.agent_pos[i] == env.agent_goals[i] for i in range(NUM_AGENTS))
        
        agent.store_transition(state, actions, rewards, next_state, done)
        loss = agent.learn()
        
        if loss > 0:
            episode_losses.append(loss)

        state = next_state
        ep_reward += sum(rewards)
        per_agent_reward += np.array(rewards)
        
        goals_reached = sum(1 for i in range(NUM_AGENTS) 
                          if env.agent_pos[i] == env.agent_goals[i])
        
        if done:
            break

    # Record metrics
    total_rewards.append(ep_reward)
    avg_reward_per_agent.append(per_agent_reward.mean())
    success_rates.append(goals_reached / NUM_AGENTS)
    epsilon_history.append(agent.epsilon)
    
    if episode_losses:
        losses.append(np.mean(episode_losses))

    # Learning rate decay
    if ep % 500 == 0 and ep > 0:
        agent.scheduler.step()

    # Progress report
    if (ep + 1) % 100 == 0:
        recent_avg = np.mean(total_rewards[-100:])
        recent_success = np.mean(success_rates[-100:])
        print(f"Ep {ep+1:4d}/{EPISODES} | Reward: {ep_reward:6.1f} | "
              f"Avg100: {recent_avg:6.1f} | Success: {recent_success:.2%} | "
              f"ε: {agent.epsilon:.3f}")

print("\nTraining completed!")

# ========================= ENHANCED ANALYSIS ========================= #

# Convergence analysis
convergence_point = analyze_convergence(success_rates, EPISODES)

# Final behavior analysis
behavior_analysis = analyze_final_behavior(agent, env)

# Enhanced final stats
final_100_avg = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
final_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)

print(f"\n=== COMPREHENSIVE RESULTS ===")
print(f"Random Baseline:")
print(f"  - Average reward: {random_reward:.2f}")
print(f"  - Success rate: {random_success:.2%}")

print(f"\nDQN Performance:")
print(f"  - Final avg reward (last 100): {final_100_avg:.2f}")
print(f"  - Final success rate: {final_success:.2%}")
print(f"  - Improvement over random:")
print(f"    * Reward: {final_100_avg/random_reward:.1f}x better")
print(f"    * Success: {final_success/random_success:.1f}x better")

if convergence_point:
    print(f"\nConvergence Analysis:")
    print(f"  - Converged at episode: {convergence_point}")
    print(f"  - Training efficiency: {convergence_point/EPISODES:.1%} of total episodes")
    print(f"  - Episodes to 80% success: {convergence_point}")
else:
    print(f"\nConvergence Analysis:")
    print(f"  - No clear convergence to 80% success rate detected")
    print(f"  - May need more training episodes or hyperparameter tuning")

print(f"\nFinal Agent Behavior Analysis (10 test episodes):")
print(f"  - Average steps to completion: {behavior_analysis['avg_steps']:.1f}")
print(f"  - Average collisions per episode: {behavior_analysis['avg_collisions']:.1f}")
print(f"  - Average test reward: {behavior_analysis['avg_reward']:.1f}")
print(f"  - Average test success rate: {behavior_analysis['avg_success']:.2%}")

print(f"\nTraining Dynamics:")
print(f"  - Final epsilon: {agent.epsilon:.4f}")
print(f"  - Total training steps: {agent.steps}")
print(f"  - Memory utilization: {len(agent.memory)}/{MEMORY_CAPACITY}")

# =========================  Enhanced Plotting  ===================================== #
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Total rewards with baseline comparison
ax1.plot(total_rewards, alpha=0.3, color='blue', label='DQN Raw')
ma_total = moving_average(total_rewards, 100)
if len(ma_total) > 0:
    ax1.plot(range(99, EPISODES), ma_total, color='red', linewidth=2, label='DQN MA(100)')
ax1.axhline(y=random_reward, color='gray', linestyle='--', label=f'Random Baseline ({random_reward:.1f})')
if convergence_point:
    ax1.axvline(x=convergence_point, color='green', linestyle=':', label=f'Convergence ({convergence_point})')
ax1.set_title('Total Reward per Episode')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Success rate with baseline comparison
ax2.plot(success_rates, alpha=0.3, color='green', label='DQN Raw')
ma_success = moving_average(success_rates, 100)
if len(ma_success) > 0:
    ax2.plot(range(99, EPISODES), ma_success, color='darkgreen', linewidth=2, label='DQN MA(100)')
ax2.axhline(y=random_success, color='gray', linestyle='--', label=f'Random Baseline ({random_success:.2%})')
if convergence_point:
    ax2.axvline(x=convergence_point, color='green', linestyle=':', label=f'Convergence ({convergence_point})')
ax2.set_title('Success Rate')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Success Rate')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Epsilon decay
ax3.plot(epsilon_history, color='orange')
ax3.set_title('Epsilon Decay (Exploration Rate)')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Epsilon')
ax3.grid(True, alpha=0.3)

# Average reward per agent
ax4.plot(avg_reward_per_agent, alpha=0.3, color='purple', label='Raw')
ma_avg = moving_average(avg_reward_per_agent, 100)
if len(ma_avg) > 0:
    ax4.plot(range(99, EPISODES), ma_avg, color='darkviolet', linewidth=2, label='MA(100)')
ax4.set_title('Avg Reward per Agent')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Avg Reward')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('marl_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Enhanced analysis complete!")
print("Graph saved: marl_results.png")