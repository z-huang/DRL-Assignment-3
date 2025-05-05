from collections import deque
from datetime import datetime
import gym
from gym import ObservationWrapper, Wrapper
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import cv2


class RandomActionStartWrapper(gym.Wrapper):
    """
    Do random action in the first N frames to improve diversity and generalizability
    """

    def __init__(self, env, num_random_frames=4):
        super().__init__(env)
        self.num_random_frames = num_random_frames

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.num_random_frames):
            action = self.env.action_space.sample()
            obs, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class SkipFrameEnv(Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class FrameDownsample(ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(1, 84, 84),
                                     dtype=np.uint8)
        self.width = width
        self.height = height

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame,
                           (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        frame = frame[None, :, :]
        return frame


class FrameStack(ObservationWrapper):
    def __init__(self, env: gym.Env, num_stack=4):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        obs_space = env.observation_space
        self.frame_shape = obs_space.shape
        self._dtype = obs_space.dtype

        self.observation_space = Box(
            obs_space.low.repeat(num_stack, axis=0),
            obs_space.high.repeat(num_stack, axis=0),
            dtype=self._dtype
        )

    def reset(self):
        for _ in range(self.num_stack):
            self.frames.append(np.zeros(self.frame_shape, dtype=self._dtype))
        return self.observation(self.env.reset())

    def observation(self, obs):
        self.frames.append(obs)
        return np.concatenate(self.frames)


def make_env(
    env_id: str = 'SuperMarioBros-v0',
    record_video: bool = True,
    video_dir_path: str = 'mario_videos',
    video_name_prefix: str = datetime.now().strftime('%m%d_%H%M%S'),
    video_record_freq: int = 100
) -> gym.Env:
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir_path,
            episode_trigger=lambda episode_id: (
                episode_id + 1) % video_record_freq == 0,
            name_prefix=video_name_prefix
        )
    env = SkipFrameEnv(env, skip=4)
    env = FrameDownsample(env)
    env = FrameStack(env, num_stack=4)
    return env
