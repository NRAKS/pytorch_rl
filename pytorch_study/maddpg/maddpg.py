import numpy as np
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

criterion = nn.MSELoss()

class MADDPG(object):
    def __init__(self, n_states, n_actions, args, n_agent):
        if args.seed > 0:
            self.seed(args.seed)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_agent = n_agent

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.actor = Actor(self.n_states, self.n_actions, **net_cfg)
        self.actor_target = Actor(self.n_states, self.n_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.rate)

        self.critic = Critic(self.n_states, self.n_actions, self.n_agent, **net_cfg)
        self.critic_target = Critic(self.n_states, self.n_actions, self.n_agent, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None
        self.a_t = None
        self.is_trainning = True

        if USE_CUDA: self.cuda()

    def update_policy(self, agents):
        # sample batch
        obs_n=[]
        obs_next_n = []
        act_n = []
        self.replay_sample_idx = self.memory.sample_batch_idxs(self.batch_size)

        for i in range(self.n_agent):
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = agents[i].memory.sample_and_split(self.batch_size, self.replay_sample_idx)
            obs_n.append(state_batch)
            obs_next_n.append(next_state_batch)
            act_n.append(action_batch)

        obs, act, rew, obs_next, done = self.memory.sample_and_split(self.batch_size, self.replay_sample_idx)
        # print("rew:{}" .format(rew))

        # print("self.n:{}" .format(self.n_agent))
        # print("len(obs_next_n[i]):{}" .format(len(obs_next_n)))
        # print("len(obs_next_n[0]):{}" .format(len(obs_next_n[0])))
        # print("len(obs_next_n[0,0]):{}" .format(len(obs_next_n[0][0])))

        # Prepare for the target q batch
        for i in range(1):
            # print("len_obs_next:{}".format(len(obs)))
            # print("len_obs_next_n[0]:{}" .format(len(obs_next_n[0].shape)))
            # print("type_obs_next_n[0]:{}" .format(type(obs_next_n[0])))

            target_act_next_n = []

            for n in range(self.n_agent):
                _target_act_next_n = agents[n].actor_target(to_tensor(obs_next_n[n]))
                target_act_next_n.append(to_numpy(_target_act_next_n))
            target_act_next_n = np.array(target_act_next_n)
            obs_next_n = np.asarray(obs_next_n)

            # print("target_act_next_n.shape:{}" .format(target_act_next_n.shape))
            # print("target_act_next:{}" .format(target_act_next_n))
            # print("target_act_next_n:{}" .format(len(target_act_next_n[0])))
            # print("obs_next_n.shape:{}" .format(obs_next_n.shape))
            # print("obs_next_n+target_act_next_n:{}" .format(np.concatenate((obs_next_n, target_act_next_n), axis=-1)))
            a = np.concatenate((obs_next_n, target_act_next_n), axis=-1)
            # a = np.expand_dims(a, axis=-1)
            if self.n_agent == 3:
                a = np.concatenate((a[0], a[1], a[2]), axis=-1)
            # print("a.shape():{}" .format(a.shape))
            # a = a.transpose(2, 1, 3, 0)
            # print("a.shape():{}" .format(a.shape))
            # a = a.reshape(1024, -1)
            # print("a.shape():{}" .format(a.shape))
            # print("a.type():{}" .format(type(a)))
             
            # print("len_to_tensor(np.asarray([obs_next_n]+[target_act_next_n])):{}" .format(to_tensor(a))))

            next_q_values = self.critic_target(to_tensor(a))
            # print("next_q_values:{}" .format(next_q_values))
            # print("rew:{}" .format(to_tensor(rew)))

            # target_q_batch = to_tensor(rew) + self.discount * to_tensor(done.astype(np.float)) * next_q_values
            # print("done:{}" .format(done))
            target_q_batch = to_tensor(rew) + self.discount * to_tensor(done.astype(np.float)) * next_q_values
            # print("target_q_batch:{}" .format(target_q_batch))
            # print("done.astype(np.float):{}" .format(done.astype(np.float)))
            # print("done:{}" .format(done))
        # print("target_q_batch:{}" .format(target_q_batch))
        # Critic update
        self.critic.zero_grad()

        b = np.concatenate((obs_n, act_n), axis=-1)
        # b = np.expand_dims(b, axis=-1)
        if self.n_agent == 3:
            b = np.concatenate((b[0], b[1], b[2]), axis=-1)
        # print("b.shape:{}" .format(b.shape))
        # b = b.transpose(2, 1, 3, 0)
        # print("b.shape:{}" .format(b.shape))
        # b = b.reshape(1024, -1)
        # print("b.shape:{}" .format(b.shape))

        q_batch = self.critic(to_tensor(b))

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        # c = np.expand_dims(obs_n, axis=-1)
        # c = c.transpose(3, 1, 2, 0)
        # c = c.reshape(1024, -1)
        # print("type(self.actor(to_tensor(c))".format(type(self.actor(to_tensor(obs_n)))))
        # print("c.shape:{}" .format(c.shape))

        policy_loss = -self.critic(to_tensor(b)).mean()
        p_out = self.actor(to_tensor(obs))
        # print("p_out:{}" .format(p_out))
        # print("p_out.shape:{}" .format(to_numpy(p_out).shape))
        policy_loss += (p_out**2).mean() * 1e-3
        # print("policy_loss:{}" .format(policy_loss))
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)
        self.actor_optim.step()

    def target_update(self):
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        # hard_update(self.actor_target, self.actor)
        # hard_update(self.critic_target, self.critic)


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def observe(self, r_t, s_t1, done):
        if self.is_trainning:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(0., 1., self.n_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        if random.random() <= max(self.epsilon, 1e-1):
            action = self.random_action()
        else:
            action = to_numpy(
                self.actor(to_tensor(np.array([s_t])))
            ).squeeze(0)
            # action += self.is_trainning * max(self.epsilon, 1e-1) * self.random_process.sample()
            action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
