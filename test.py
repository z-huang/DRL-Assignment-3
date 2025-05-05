from datetime import datetime
import random
import time
import numpy as np
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from student_agent import Agent
import gym
import warnings

warnings.filterwarnings('ignore')

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = gym.wrappers.RecordVideo(
    env,
    video_folder='mario_videos',
    episode_trigger=lambda episode_id: True,
    name_prefix=datetime.now().strftime('%m%d_%H%M%S')
)

reward_history = []

test_generalization = False

for i in range(10):
    agent = Agent()

    state = env.reset()
    step = 0
    total_reward = 0
    done = False
    start_time = time.time()

    while not done:
        if test_generalization and step < 4:
            action = random.randrange(12)
            step += 1
        else:
            action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        state = next_state
        total_reward += reward

        print(
            f'reward={total_reward}, x={info["x_pos"]}, y={info["y_pos"]}, score={info["score"]}, time={info["time"]}', "\r", end=' ')
        
        if time.time() - start_time >= 120:
            break
    
    reward_history.append(total_reward)
    print()
    print('Total reward:', total_reward)

print(f'Average reward: {int(np.mean(reward_history))}')

try:
    # There is a problem with the destructor of the environment, so an exception is needed to avoid error reporting
    del env
except Exception:
    pass
