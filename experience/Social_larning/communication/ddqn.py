import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from model import Learner
from memory import ReplayMemory
from util import *

criterion = nn.MSELoss()


class DDQN(object):
    def __init__(self, n_states, n_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.n_states = n_states
        self.n_actions = n_actions

        # create agent network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            'init_w': args.init_w
        }
        self.agent = Learner(self.n_states, self.n_actions, **net_cfg)
        self.target = Learner(self.n_states, self.n_actions, **net_cfg)
        self.agent_optim = Adam(self.agent.parameters(), lr=args.lr)

        self.update_target_steps = args.update_target_timing

        hard_update(self.target, self.agent)

        # create replay memory
        self.memory = ReplayMemory(capacity=args.rmsize)

        # hyper parameters
        self.batch_size = args.bsize
        self.discount_rate = args.discount_rate
        self.decay_epsilon = 1 / args.decay_epsilon
        self.min_epsilon = args.min_epsilon
        
        self.epsilon = 1.0
        
        if USE_CUDA: self.cuda()

    def update(self, step):
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        q_predict = self.agent(to_tensor(state_batch))
        n_q_predict = self.agent(to_tensor(next_state_batch))
        q_batch = torch.zeros(self.batch_size, 1)
        n_act_batch = np.zeros(self.batch_size)
        next_q_value = torch.zeros(self.batch_size, 1)

        for n in range(self.batch_size):
            q_batch[n] = q_predict[n][action_batch[n]]
            n_act_batch = torch.argmax(n_q_predict[n])
            # print(n_act_batch)
            # print(self.target(to_tensor(next_state_batch[n])))
            next_q_value[n] = self.target(to_tensor(next_state_batch[n]))[n_act_batch]

        # next_q_value = torch.max(self.target(to_tensor(next_state_batch)), 1)[0].reshape(self.batch_size, 1)
        
        # next_q_value = self.target(to_tensor(next_state_batch))[n_act_batch]

        target_q_batch = to_tensor(reward_batch).reshape(self.batch_size, 1) + self.discount_rate * next_q_value * to_tensor(1-terminal_batch.astype(np.float).reshape(self.batch_size, 1))

        # q_predict = self.agent(to_tensor(state_batch))
        # print("q_predict:{}" .format(q_predict))
        # q_batch = torch.zeros(self.batch_size, 1)
        # print("q_batch:{}" .format(q_batch.shape))
        # print("q_batch:{}" .format(q_batch))
        value_loss = criterion(q_batch, target_q_batch)
        # print("loss:{}" .format(value_loss))
        self.agent.zero_grad()
        value_loss.backward()
        self.agent_optim.step()

        if step % self.update_target_steps == 0:
            # print("update target")
            self.update_target()

    def update_target(self):
        hard_update(self.target, self.agent)

    def random_action(self):
        action = np.random.uniform(-1., 1., self.n_actions)
        # self.a_t = action
        
        action = np.argmax(action)

        # idx = np.where(action == max(action))

        # action = np.random.choice(idx[0])
        # print(action)
        return action

    def select_action(self, s_t, decay_epsilon=True):
        if np.random.random () < self.epsilon:
            action = self.random_action()
        else:
            action = to_numpy(
                self.agent(to_tensor(np.array([s_t])))
            ).squeeze(0)
            # print("action:{}".format(action))
            action = np.argmax(action)
            # idx = np.where(action == max(action))
            # action = np.random.choice(idx[0])
            
            # print("action:{}" .format(action))
            # action = np.clip(action, -1, 1)

        if self.epsilon > self.min_epsilon and decay_epsilon:
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_epsilon)    

        return action

    def observe(self, obs, act, new_obs, rew, done):
        items = np.asarray([obs, act, new_obs, rew, done])
        self.memory.push(items)

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
