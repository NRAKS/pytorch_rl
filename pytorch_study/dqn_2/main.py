import gym
import numpy as np
import torch as th
import argparse
from train import train
from dqn import DQN
from util import *

parser = argparse.ArgumentParser(description='PyTorch')

if __name__ == "__main__":
    # 環境設定
    parser.add_argument('--env', default='CartPole-v1')
    # パラメータ設定
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--discount_rate', default=0.99, type=float, help='discount_rate')
    parser.add_argument('--bsize', default=32, type=int)
    parser.add_argument('--rmsize', default=50000, type=int)
    parser.add_argument('--decay_epsilon', default=10000, type=int)
    parser.add_argument('--min_epsilon', default=0.02)
    parser.add_argument('--hidden1', default=128, type=int)
    parser.add_argument('--hidden2', default=128, type=int)
    parser.add_argument('--init_w', default=0.003, type=float)
    # トレーニング回数関連
    parser.add_argument('--train_simulation', default=1, type=int)
    parser.add_argument('--train_episode', default=2000, type=int)
    parser.add_argument('--train_step', default=5000)
    parser.add_argument('--pre_step', default=1000)
    parser.add_argument('--update_target_timing', default=1000)

    parser.add_argument('--seed', default=1, type=int)
    # 保存先パス
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--resume', default='default', type=str)

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = gym.make(args.env)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n

    agent = DQN(nb_state, nb_action, args)
    train_cfg = {
        "simulation_times": args.train_simulation,
        "episode_times": args.train_episode,
        "step_times": args.train_step,
        "pre_step_times": args.pre_step
    }
    print("start training...")
    train(agent, env, **train_cfg)
