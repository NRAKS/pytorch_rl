#!/usr/bin/env python3

import numpy as np
import argparse
from copy import deepcopy
import time

import torch
import gym
from gym.spaces import Box, Discrete

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *

gym.undo_logger_setup()


def train(num_iterations, agent, env,  evaluate, validate_steps, args, output, max_episode_length=None, debug=False):

    for n_agent in range(env.n):
        agent[n_agent].is_training = True
    step = episode = episode_steps = 0
    episode_reward = [0.0]
    agent_reward = [[0.0] for _ in range(env.n)]
    observation = None
    t_start = time.time()

    print("Starting iterations...")
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            for n_agent in range(env.n):
                agent[n_agent].reset(observation)
        # print("observation:{}" .format(observation))
        # agent pick action ...
        if step <= args.warmup:
            action = [agent[n_agent].random_action() for n_agent in range(len(agent))]
        else:
            action = [agent[n_agent].select_action(observation[n_agent]) for n_agent in range(len(agent))]
        
        # env response with next_observation, reward, terminate_info
        # print("action:{}" .format(action))
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        terminal = (episode_steps >= args.max_episode_length)

        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        for n_agent in range(len(agent)):
            agent[n_agent].observe(reward, observation2, done)
        if step > args.warmup:
            for n_agent in range(len(agent)):
                agent[n_agent].update_policy()

        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:

            for n_agent in range(len(agent)):
                policy = np.full(len(agent), lambda x: agent[n_agent].select_action(x, decay_epsilon=False))
                validate_reward = np.full(len(agent), evaluate(env, policy[n_agent], debug=False, visualize=False))
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            for n_agent in range(len(agent)):
                agent[n_agent].save_model(output)

        # update
        step += 1
        episode_steps += 1
        for i, rew in enumerate(reward):
            episode_reward[-1] += rew
            agent_reward[i][-1] += rew
        observation = deepcopy(observation2)

        # for displaying learned policies
        if args.display:
            time.sleep(0.1)
            env.render()
            continue

        if (terminal and (len(episode_reward) % 1000) == 0):
            if debug:
                prGreen('#{}: episode: {} episode_reward: {} steps: {} time: {}'.format(episode, len(episode_reward), episode_reward[episode],step,time.time() - t_start))
                t_start = time.time()
                episode_steps = 0


        if done or terminal: # end of episode
            for n_agent in range(len(agent)):
                agent[n_agent].memory.append(
                    observation,
                    agent[n_agent].select_action(observation),
                    0., False
                )
            # reset
            observation = None
            # episode_steps = 0
            episode_reward.append(0)
            episode += 1


def make_env(scenario_name, arglist=None, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    for n_agent in range(len(agent)):
        agent[n_agent].load_weights(model_path)
        agent[n_agent].is_training = False
        agent[n_agent].eval()
    
    observation = env.reset()
    step = 0
    while True:
        action = [agent[n_agent].select_action(observation[n_agent]) for n_agent in range(len(agent))]
        observation, reward, done_n, info = env.step(action)
        # print("obs:{}" .format(observation[0]))
        time.sleep(0.1)
        env.render()
        done = all(done_n)
        step += 1
        if step % 100 == 0:
            step = 0
            observation = env.reset()
    
    #     policy = np.full(len(agent), lambda x: agent[n_agent].select_action(x, decay_epsilon=False))

    # for i in range(num_episodes):
    #     validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
    #     if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    # parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument("--env", type=str, default="simple", help="name of the scenario script")
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=25, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=60000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')

    parser.add_argument('--display', action="store_true", default=False)
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = make_env(args.env)
    # env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    agent_init_params = []
    
    # ("env.n:{}".format(env.n))
    for acsp, obsp in zip(env.action_space, env.observation_space):
            num_in_pol = obsp.shape[0]  # 状態数
            # print("num_in_pol:{}".format(obsp.shape[0]))
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)  # 行動数
            # print("num_out_pol{}" .format(get_shape(acsp)))
    
            nb_states = num_in_pol
            nb_actions = num_out_pol
            

    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    # print("env_n:{}" .format(env.n))
    # print("obs_shape_n:{}" .format(obs_shape_n))
    # print("env.action_space:{}" .format(env.action_space))
    # print("env.observation_space:{}" .format(env.observation_space))
    # nb_states = env.observation_space[0].shape[0]
    # nb_actions = env.action_space
    # print("nb_actions:{}" .format(nb_actions))

    # if isinstance(nb_actions, Box):
    #     discrete_action = False
    #     get_shape = lambda x: x.shape[0]
    # else:  # Discrete
    #     discrete_action = True
    #     get_shape = lambda x: x.n

    # num_out_pol = get_shape(nb_actions)
    # print("num_out_pol".format(num_out_pol))
    agent = []
    for _ in range(env.n):
        agent.append(DDPG(nb_states, nb_actions, args))
    # print("agent:{}" .format(agent))
    # print("agent_init_params:{}" .format(agent_init_params))


    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
