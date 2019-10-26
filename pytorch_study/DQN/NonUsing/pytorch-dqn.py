from __future__ import division
from collections import deque
import gym
from gym import wrappers
import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Variable


#np.random.seed(3)

env = gym.make('CartPole-v0')


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.f1 = nn.Linear(input_size, 200)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(200, output_size)

        self.experience_replay = deque()
        self.epsilon = 1
        self.action_num = output_size

        self.batch_size = 16
        self.memory_size = 10000
        self.gamma = 0.9
        self.mse = criterion = nn.MSELoss()

    def forward(self, x):
        out = self.f1(x)
        out = self.relu(out)
        out = self.f2(out)
        return out

    def sample_action(self, epoch, state):
        '''
        use e-greedy
        '''
        if epoch == 0:
            return np.argmax(self.forward(state).data.numpy())

        self.epsilon /= epoch

        greedy = np.random.rand()
        if greedy < self.epsilon:
            action = np.random.randint(self.action_num)
        else:
            action = np.argmax(self.forward(state).data.numpy())
        return action
    
    def compute(self, state, action, reward, new_state, done, optimizer):
        self.experience_replay.append((state, action, reward, new_state, done))
        if len(self.experience_replay) > self.memory_size:
            self.experience_replay.popleft()
        if len(self.experience_replay) > self.batch_size:
            self.train(optimizer)

    def train(self, optimizer):
        minibatch = random.sample(self.experience_replay, self.batch_size)
        
        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        new_state = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        y_label = []
        q_prime = self.forward(Variable(torch.from_numpy(np.array(new_state)).float())).data.numpy()
        #get the y_label e.t. the r+Q(s',a',w-)
        for i in range(self.batch_size):
            if done[i]:
                y_label.append(reward[i])
            else:
                y_label.append(reward[i] + np.max(q_prime[i]))

        # the input for the minibatch
        # Q(s,a,w)
        state_input = torch.from_numpy(np.array(state)).float()
        action_input = torch.from_numpy(np.array(action))
        out = self.forward(Variable(state_input))
        y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

        optimizer.zero_grad()
        loss = self.mse(y_out, Variable(torch.from_numpy(np.array(y_label)).float()))
        loss.backward()
        optimizer.step()


state_dim = env.observation_space.high.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, action_dim)

optimizer = torch.optim.Adam(agent.parameters(),lr=1e-3)  

for i in range(1000):
    state = env.reset()
    while True:
        tensor_state = torch.from_numpy(np.expand_dims(state, axis = 0)).float()
        action = agent.sample_action(i + 1, Variable(tensor_state))
        state_new, reward, done, info = env.step(action)
        agent.compute(state, action, reward, state_new, done, optimizer)
        state = state_new
        if done:
            break
    
    if i % 100 == 99:
        total_reward = 0
        for step in range(100):
            state = env.reset()
            while True:
                env.render()
                tensor_state = torch.from_numpy(np.expand_dims(state, axis = 0)).float()
                action = agent.sample_action(0, Variable(tensor_state))
                state,reward,done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / 100
        # print 'episode: ',100,'Evaluation Average Reward:',ave_reward
        if ave_reward >= 200:
            break

env = gym.wrappers.Monitor(env, 'monitor', force = True)
for step in range(100):
            state = env.reset()
            while True:
                env.render()
                tensor_state = torch.from_numpy(np.expand_dims(state, axis = 0)).float()
                action = agent.sample_action(0, Variable(tensor_state))
                state,reward,done,_ = env.step(action)
                total_reward += reward
                if done:
                    break
env.close()
gym.upload('monitor', api_key='sk_KqiSu4iIThqHmgjcONDChg')