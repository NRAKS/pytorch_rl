"環境まとめファイル"

import numpy as np
from copy import deepcopy


# 基本構成
class simple_planning(object):
    def __init__(self):
        # todo
        # 基本情報
        self.n_agent = 2
        # 状態定義
        self.n_observation = 2  # time, opponent_action
        self.init_observation = np.zeros((self.n_agent, self.n_observation))
        self.observation = np.zeros((self.n_agent, self.n_observation))
        self.obs_time = 0
        self.obs_opponent_action = 1
        # 行動定義
        # 0:何もしない
        # 1:actionを起こす
        self.n_action = 2
        # 報酬定義
        self.reward = np.array([0., 1., -1.])
        # 終端状態定義
        self.done_terms = 500

    def evaluate_next_state(self, action_list):
        # todo
        # 次状態遷移定義
        n_obs = deepcopy(self.observation)
        for n in range(self.n_agent):
            n_obs[n, self.obs_time] += 1
            n_obs[n, self.obs_opponent_action] = action_list[n-1]
        self.observation = deepcopy(n_obs)
        return n_obs

    def evaluate_reward(self, state):
        # todo
        # 報酬判定
        reward = np.zeros(self.n_agent)
        if state[0, 1] != state[1, 1]:
            if state[0, 1] == 0:
                reward[0] = self.reward[1]
            else:
                reward[1] = self.reward[1]
        else:
            pass
        return reward

    def evaluate_done(self, n_obs):
        # todo
        # 終端状態判定
        # print("n_obs:{}" .format(n_obs[0, 0]))
        if n_obs[0, 0] >= self.done_terms:
            return True
        else:
            return False

    def evaluate_info(self):
        # todo
        # 補足情報定義

        return False

    def step(self, action):
        # todo
        n_obs = self.evaluate_next_state(action)
        reward = self.evaluate_reward(n_obs)
        done = self.evaluate_done(n_obs)
        info = self.evaluate_info()

        return n_obs, reward, done, info

    def reset(self):
        self.observation = deepcopy(self.init_observation)
        return self.observation


class simple_planning_ex(simple_planning):

    def __init__(self):
        super().__init__()
        self.n_action = 3
        self.obs_store = 2
        self.n_observation = 3  # time, opponent_action
        self.init_observation = np.zeros((self.n_agent, self.n_observation))
        self.observation = np.zeros((self.n_agent, self.n_observation))
        self.obs_time = 0
        self.obs_opponent_action = 1
        self.obs_store = 2

    def evaluate_next_state(self, action_list):
        # todo
        # 次状態遷移定義
        n_obs = deepcopy(self.observation)
        for n in range(self.n_agent):
            n_obs[n, self.obs_time] += 1
            n_obs[n, self.obs_opponent_action] = action_list[n-1]
            if action_list[n] == 2:
                n_obs[n, self.obs_store] += 1
        self.observation = deepcopy(n_obs)
        return n_obs

    def evaluate_reward(self, state):
        # todo
        # 報酬判定
        reward = np.zeros(self.n_agent)

        if state[0, 1] == 0 and state[1, 1] == 0:
            if state[0, 2] > state[1, 2]:
                reward[0] = self.reward[1]

            elif state[0, 2] < state[1, 2]:
                reward[1] = self.reward[1]

        elif state[0, 1] == 1 and state[1, 1] == 0:
            reward[1] = self.reward[1]

        elif state[0, 1] == 0 and state[1, 1] == 1:
            reward[0] = self.reward[1]
        
        else:
            pass

        # if state[0, 1] == 0 and state[0, 2] > state[1, 2]:
        #     reward[0] = self.reward[1]
        # elif state[1, 1] == 0 and state[1, 2] > state[0, 2]:
        #     reward[1] = self.reward[1]
        # else:
        #     pass
        return reward

def make(name):
    if name == 'simple_planning':
        return simple_planning()
    elif name == 'simple_planning_ex':
        return simple_planning_ex()