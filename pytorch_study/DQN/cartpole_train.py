import gym
from itertools import count

import numpy as np
import torch as th
import visdom
import matplotlib as plt

from dqn import DQN

scale_reward = 0.01
use_cuda = th.cuda.is_available()
device = th.device("cuda" if use_cuda else "cpu")
FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor

def main():
    env = gym.make("CartPole-v0")
    agent = DQN(dim_obs=env.observation_space.shape[0],
                dim_act=env.action_space.n,
                batch_size=50,
                capacity=1000)

    reward_record = []

    n_episode = 50
    TARGET_UPDATE = 10

    for i_episode in range(n_episode):
        print("i_episode: {}" .format(i_episode))
        obs = env.reset()
        total_reward = 0.0

        for t in count():
            action = agent.select_action(obs)
            # print(env.step(action[0, 0]))
            obs_, reward, done, _ = env.step(action)

            reward = th.tensor([reward], device=device)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if not done:
                next_obs = obs_
            else:
                next_obs = None
            
            total_reward += reward
            agent.memory.push(obs,
                              action,
                              reward,
                              obs_,
                              done)
            obs = next_obs
            agent.update_policy()

            if done:
                print("break")
                break
        
        if i_episode % TARGET_UPDATE == 0:
            agent.learner_target.load_state_dict(agent.learner.state_dict())
        
        print("Episode: {}, Total_reward: {}" .format(i_episode, total_reward))
main()