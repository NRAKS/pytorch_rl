import torch as th
import torch.nn as nn
import torch.nn.functional as F


# 学習モデルの作成
class Learner(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Learner, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = self.FC3(result)

        return result
