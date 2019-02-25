import gym
import numpy as np
import torch as th
import argparse

from dqn import DQN

parser = argparse.ArgumentParser(description='PyTorch')

# 環境設定
parser.add_argument('--env', default='CartPole-v1')
# パラメータ設定
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--discount_rate', default=0.99, type=float, help='discount_rate')
parser.add_argument('--bsize', default=64, type=int)
parser.add_argument('--rmsize', default=50000, type=int)
parser.add_argument('--train_iter', default=200000)
parser.add_argument('--decay_epsilon' default=100000)
parser.add_argument('--min_epsilon', default=0.01)


