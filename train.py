import argparse
from datetime import datetime
from pathlib import Path
import torch

from trainer import DQNTrainer
from wrappers import make_env
from buffer import PrioritizedReplayBuffer
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='SuperMarioBros-v0'
    )
    parser.add_argument(
        '-o',
        type=Path,
        default=Path('runs/' + datetime.now().strftime('%m%d_%H%M'))
    )
    return parser.parse_args()


def main(args):
    output_path = args.o
    ckpt_path = output_path / 'ckpt'
    video_path = output_path / 'videos'

    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    video_path.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_id=args.env,
        record_video=False,
        video_dir_path=str(output_path / 'videos'),
        video_name_prefix='mario',
        video_record_freq=50
    )

    trainer = DQNTrainer(
        input_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        replay_buffer=PrioritizedReplayBuffer(
            capacity=100000,
            state_shape=env.observation_space.shape,
            batch_size=32,
            alpha=0.6,
            beta=0.4,
            beta_frames=100000,
            eps=1e-6,
        ),
        output_path=output_path,
        lr=1e-4,
        gamma=0.75,
        target_update_freq=1000,
        train_freq=4,
        checkpoint_freq=50,
        log_freq=10,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.99995,
        device=('cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu')
    )

    trainer.train(env, num_episodes=100000)


if __name__ == '__main__':
    args = parse_args()
    main(args)
