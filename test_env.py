import random
from GridDeliveryEnv import GridDeliveryEnv

# Initialize environment
env = GridDeliveryEnv(grid_size=10, num_agents=5, delivery_points=[(6, 2), (3, 7)])

# Reset environment to start
state = env.reset()
print("Initial state:", state)
env.render()

num_episodes = 3

for episode in range(num_episodes):
    print(f"\n=== Episode {episode + 1} ===")
    state = env.reset()
    done = False
    step_num = 0

    while not done:
        # Random actions for both agents
        actions = [random.randint(0, env.num_actions - 1) for _ in range(env.num_agents)]
        next_state, rewards, done = env.step(actions)

        print(f"\nStep {step_num}")
        print(f"Actions taken: {actions} (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:DELIVER)")
        print(f"Rewards: {rewards}")
        print(f"Next state: {next_state}")
        env.render()

        state = next_state
        step_num += 1

        if done:
            print("\nEpisode finished!")

