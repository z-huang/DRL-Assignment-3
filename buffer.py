import numpy as np


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple,
        batch_size: int,
        alpha: float,
        beta: float,
        beta_frames: int,
        eps: float,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_frames
        self.max_prio = 1.0
        self.eps = eps

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool8)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self.position = 0
        self._len = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        i = self.position

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.priorities[i] = self.max_prio

        self.position = (self.position + 1) % self.capacity
        self._len = min(self._len + 1, self.capacity)

    def sample(self):
        prios = self.priorities[:len(self)]
        probs = prios / prios.sum()

        indices = np.random.choice(len(self), self.batch_size, p=probs)

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, prio in zip(indices, td_errors):
            self.priorities[idx] = (np.abs(prio) + self.eps) ** self.alpha
            self.max_prio = max(self.max_prio, self.priorities[idx])

    def __len__(self):
        return self._len
