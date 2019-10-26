from model import Learner
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
# from params import scale_reward

scale_reward = 0.01
use_cuda = th.cuda.is_available()
device = th.device("cuda" if th.cuda.is_available() else "cpu")
FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class DQN():
    def __init__(self, dim_obs, dim_act, batch_size, capacity):
        self.learner = Learner(dim_obs, dim_act).to(device)
        self.learner_target = deepcopy(self.learner)

        self.n_agents = 1
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size

        self.GAMMA = 0.95
        self.tau = 0.01

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200

        self.learner_optimizer = Adam(self.learner.parameters(), lr=0.001)

        self.step_done = 0
        self.episode_done = 0

    def update_policy(self):
        # replayメモリが十分にたまってない場合
        if len(self.memory) < self.batch_size:
            return

        print("called update_policy")
        minibatch = self.memory.sample(self.batch_size)
        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        new_state = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        y_label = []
        q_prime = self.learner(th.from_numpy(np.array(new_state)).float()).data.numpy()
        #get the y_label e.t. the r+Q(s',a',w-)
        for i in xrange(self.batch_size):
            if done[i]:
                y_label.append(reward[i])
            else:
                y_label.append(reward[i] + np.max(q_prime[i]))

        # the input for the minibatch
        # Q(s,a,w)
        state_input = th.from_numpy(np.array(state)).float()
        action_input = th.from_numpy(np.array(action))
        out = self.learner(Variable(state_input))
        y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

        self.learner_optimizer.zero_grad()
        loss = self.mse(y_out, Variable(th.from_numpy(np.array(y_label)).float()))
        loss.backward()
        self.learner_optimizer.step()
        # transitions = self.memory.sample(self.batch_size)
        # batch = Experience(*zip(*transitions))
        # non_final_mask = th.tensor(tuple(map(lambda s: s is not None, batch.next_states)), device=device, dtype=th.uint8)
        # non_final_next_states = th.cat([s for s in batch.next_states if s is not None])

        # # state_batch: batch_size * dim_obs
        # state_batch = Variable(th.cat(batch.states))
        # action_batch = Variable(th.cat(batch.actions))
        # reward_batch = Variable(th.cat(batch.rewards))

        # state_action_values = self.learner(state_batch).gather(1, action_batch)

        # next_state_values = th.zeros(self.batch_size, device=device)
        # next_state_values[non_final_mask] = self.learner_target(non_final_next_states).max(1)[0].detach()
        
        # expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # self.learner_optimizer.zero_grad()
        # loss.backward()
        # for param in self.learner.parameters():
        #     param.grad.data.clamp_(-1, 1)

        # self.learner_optimizer.step()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.step_done / self.EPS_DECAY)

        self.step_done += 1
        if sample > eps_threshold:
            with th.no_grad():
                return np.argmax(self.learner(state).data.numpy())

        else:
            return np.random.randint(self.n_actions)
