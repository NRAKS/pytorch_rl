import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

from memory import ReplayMemory
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        # self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.l3(x)
        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, args):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount_rate = args.discount_rate
        self.tau = args.tau
        self.batch_size = args.bsize
        self.memory = ReplayMemory(capacity=args.rmsize)


    def select_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        print(state)
        print(self.actor(state))
        return self.actor(state).cpu().data.numpy().flatten()

    def random_action(self):
        return 

    def observe(self, obs, act, new_obs, rew, done):
        items = np.asarray([obs, act, new_obs, rew, done])
        self.memory.push(items)

    def update(self, step):
        st_batch, act_batch, nst_batch, rwd_batch, terminal_batch = self.ReplayMemory(self.batch_size)

        state = torch.FloatTensor(st_batch).to(device)
        action = torch.FloatTensor(act_batch).to(device)
        next_state = torch.FloatTensor(nst_batch).to(device)
        reward = torch.FloatTensor(rwd_batch).to(device)
        done = torch.FloatTensor(terminal_batch).to(device)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))

        target_Q = reward + (self.discount_rate * target_Q) * (1-done).detach()

        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    

        # def save(self, filename, directory):
        #     torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        #     torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


        # def load(self, filename, directory):
        #     self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        #     self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

        