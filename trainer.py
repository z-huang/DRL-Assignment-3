from pathlib import Path
import random
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import CNNModel
from buffer import PrioritizedReplayBuffer
from logger import Logger


class DQNTrainer:
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        replay_buffer: PrioritizedReplayBuffer,
        output_path: Path,
        lr: float,
        gamma: float,
        target_update_freq: int,
        train_freq: int,
        checkpoint_freq: int,
        log_freq: int,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        device: str,
    ):
        self.output_path = output_path
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = CNNModel(input_shape, num_actions).to(device)
        self.target_net = CNNModel(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.train()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = replay_buffer

        self.n_train_step = 0
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_freq = log_freq

        self.device = device

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self):
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample()

        states = torch.tensor(states, dtype=torch.float32).to(
            self.device) / 255.0
        actions = torch.tensor(
            actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device) / 255.0
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(
            1).to(self.device)
        weights = torch.tensor(
            weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        td_errors = (
            q_values - target_q).abs().detach().cpu().numpy().flatten()

        loss = F.mse_loss(q_values, target_q, reduction='none')
        loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.n_train_step += 1

        if self.n_train_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def evaluate(self, env: gym.Env, num_episodes: int = 1) -> int:
        total_rewards = []

        for _ in (pbar := tqdm(range(num_episodes),
                               desc='Evaluation',
                               position=1,
                               leave=False)):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    action = self.q_net.get_action(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

            pbar.set_postfix({
                'reward': total_reward,
                'x': info['x_pos']
            })

        return int(np.mean(total_rewards))

    def train(self, env: gym.Env, num_episodes: int):
        step = 0
        epsilon = self.epsilon_start

        reward_logger = Logger(log_every=1, moving_avg_window=50)
        best_reward = 5500

        for episode in (pbar := tqdm(range(num_episodes), position=0)):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.q_net.get_action(state)
                next_state, reward, done, info = env.step(action)

                self.replay_buffer.push(
                    state, action, reward, next_state, done)
                total_reward += reward
                step += 1

                if step % self.train_freq == 0 and \
                        len(self.replay_buffer) >= self.replay_buffer.batch_size:
                    self.train_step()
                    epsilon = max(self.epsilon_min,
                                  epsilon * self.epsilon_decay)

                state = next_state

                pbar.set_postfix({
                    'step': step,
                    'ε': epsilon,
                    'reward': total_reward,
                    'x': info['x_pos']
                })

            reward_logger.step(total_reward)

            pbar.set_description(
                f'Average reward={int(np.mean(reward_logger.values[-20:]))}')

            if (episode + 1) % self.checkpoint_freq == 0:
                torch.save(
                    self.q_net.state_dict(),
                    self.output_path / 'ckpt' / f'model_e{episode + 1}.pkl'
                )

            if (episode + 1) % self.log_freq == 0:
                reward_logger.plot(
                    xlabel='Episode',
                    ylabel='Reward',
                    path=self.output_path / 'reward.png'
                )

            if (episode + 1) % 10 == 0 and \
                    np.mean(reward_logger.values[-10]) > best_reward:
                eval_reward = self.evaluate(env)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    torch.save(
                        self.q_net.state_dict(),
                        self.output_path / 'ckpt' / f'reward_{best_reward}.pkl'
                    )
