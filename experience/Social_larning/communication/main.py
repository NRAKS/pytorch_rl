import gym
import numpy as np
import torch as th
import argparse
import time

from train import multi_train_comu as train

from dqn import DQN
from ddqn import DDQN
from make_graph import graph, pie_graph
import enviroment
from util import *

time = time.asctime()
parser = argparse.ArgumentParser(description='PyTorch')

if __name__ == "__main__":
    # 環境設定
    parser.add_argument('--env', default='simple_planning')
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
        args.resume = 'output/{}-{}-{}'.format(args.env, args.train_simulation, time)

    # env = gym.make(args.env)
    env = enviroment.make(args.env)
    # env = enviroment.simple_planning
    print(env)

    if args.seed > 0:
        np.random.seed(args.seed)
        # env.seed(args.seed)

    train_cfg = {
        "simulation_times": args.train_simulation,
        "episode_times": args.train_episode,
        "step_times": args.train_step,
        "pre_step_times": args.pre_step
    }

    # agent_0
    nb_state = env.n_observation
    nb_action = env.n_action

    # agent_1
    # nb_state_1 = env.n_observation
    print("start DQN training...")
    # agent作成
    agent = np.asarray([DQN(nb_state, nb_action, args),
                        DQN(nb_state, nb_action, args)])
    result_dqn = train(agent, env, **train_cfg)
    # print("reward_dqn:{}" .format(reward_dqn))
    # print("start DDQN training...")
    # agent = np.full(2, DDQN(nb_state, nb_action, args))
    # reward_ddqn = train(agent, env, **train_cfg)

    # data = np.array([reward_dqn, reward_ddqn])
    # data = np.array([reward_dqn])
    data_label = np.array([["sum_reward", "dqn 0", "dqn 1"], None, "store"])
    title_label = np.array(["reward", "action_ratio", "store_count"])
    if env.n_action == 3:
        data_label[1] = ["Act", "Do nothing", "Store"]
    elif env.n_action == 2:
        data_label[1] = ["Act", "Do nothing"]
    # data_label = np.array(["dqn", "ddqn"])
    for n in range(len(result_dqn)):
        if n == 1:
            pie_graph(result_dqn[1], data_label[1], path=args.resume, title=title_label[n])
        else:
            graph(result_dqn[n], len(result_dqn[n]), xlabel="episode", ylabel="reward", label=data_label[n], path=args.resume, title=title_label[n])

