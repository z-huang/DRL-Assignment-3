from collections import deque
import random
import cv2
import gym
import numpy as np
from model import CNNModel
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

# Do not modify the input of the 'act' function and the '__init__' function.


class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.model = CNNModel(
            input_shape=(4, 84, 84),
            num_actions=12
        )
        self.model.load_state_dict(
            torch.load('model.pth', map_location='cpu')
        )
        self.model = self.model.to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

        self.tau = 0.8
        self.epsilon = 0

        self.last_action = None
        self.step = -1
        self.buffer = deque(maxlen=4)
        for _ in range(4):
            self.buffer.append(np.zeros((84, 84), dtype=np.uint8))

    def act(self, observation):
        self.step = (self.step + 1) % 4
        if self.step != 0:
            return self.last_action

        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        self.buffer.append(frame)

        stacked_frames = np.stack(self.buffer)
        if random.random() < self.epsilon:
            action = random.randrange(12)
            # logits = self.model.get_logits(stacked_frames)
            # probs = F.softmax(logits / self.tau)
            # action = torch.multinomial(probs, num_samples=1).item()
        else:
            with torch.no_grad():
                action = self.model.get_action(stacked_frames)
        self.last_action = action
        return action
