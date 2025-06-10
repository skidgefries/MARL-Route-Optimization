import numpy as np

class GridDeliveryEnv:
    def __init__(self, grid_size=10, num_agents=5, delivery_points=None):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.delivery_points = delivery_points or [(grid_size-1, grid_size-1)]
        self.num_actions = 5  # Up, Down, Left, Right, Pick/Deliver
        self.reset()

    def reset(self):
        self.agent_positions = [(0, 0) for _ in range(self.num_agents)]
        self.agent_has_package = [1 for _ in range(self.num_agents)]
        self.done = False
        return self._get_states()

    def _get_states(self):
        # State per agent: [x, y, has_package]
        states = []
        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            pkg = self.agent_has_package[i]
            states.append(np.array([x, y, pkg], dtype=np.float32))
        return states

    def _move(self, pos, action):
        x, y = pos
        if action == 0 and y > 0:  # Up
            y -= 1
        elif action == 1 and y < self.grid_size - 1:  # Down
            y += 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.grid_size - 1:  # Right
            x += 1
        return (x, y)

    def step(self, actions):
        rewards = []
        for i in range(self.num_agents):
            if self.agent_has_package[i] == 0:
                # Already delivered
                rewards.append(0)
                continue

            action = actions[i]
            if action in [0,1,2,3]:  # Movement
                self.agent_positions[i] = self._move(self.agent_positions[i], action)
                rewards.append(-0.1)  # Small step penalty
            elif action == 4:  # Pick/Deliver
                if self.agent_positions[i] in self.delivery_points and self.agent_has_package[i] == 1:
                    self.agent_has_package[i] = 0
                    rewards.append(3.0)
                else:
                    rewards.append(-0.5)  # Wrong delivery penalty

        self.done = all(p == 0 for p in self.agent_has_package)
        next_states = self._get_states()

        return next_states, rewards, self.done
