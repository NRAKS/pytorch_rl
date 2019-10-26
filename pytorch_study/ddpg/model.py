import torch
import torch.nn as nn
import torch.nn.functional as F

# 学習モデルの作成
class Learner(nn.Module):
    def __init__(self, dim_obs, dim_action, hidden1=400, hidden2=300, init_w=3e-3):
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(dim_obs, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, dim_action)
        # self.relu = F.relu
        # self.tanh = F.tanh
        self.softmax = nn.Softmax()
        # self.init_weights(init_w)

    # def init_weights(self, init_w):
    #     self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
    #     self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
    #     self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        # out = F.relu(out)
        # out = self.softmax(out)

        return out
