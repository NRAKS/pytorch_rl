"""
python3
実験用タスクまとめクラス
基本的にやることは、
    環境設定に必要な情報を受け取り、初期設定
    行動と現状態を受け取り、次の状態を決定
    報酬を決定
    次状態などを返す
"""

import numpy as np


class GlidWorld(object):
    # 簡単なステージを作る
    def __init__(self, row, col, start, goal=None):
        self.row = row
        self.col = col
        self.start_state = start
        self.goal_state = goal
        self.num_state = row * col
        self.num_action = 5
        self.reward = 0
        self.done = False

    # 座標に変換
    def coord_to_state(self, row, col):
        return ((row * self.col) + col)

    # 座標からx軸を算出
    def state_to_row(self, state):
        return ((int)(state / self.col))

    # 座標からy軸を算出
    def state_to_col(self, state):
        return (state % self.col)

    # 次の座標を算出
    def evaluate_next_state(self, state, action):
        UPPER = 0
        LOWER = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

        row = self.state_to_row(state)
        col = self.state_to_col(state)

        if action == UPPER:
            if (row) > 0:
                row -= 1
        elif action == LOWER:
            if (row) < (self.row-1):
                row += 1
        elif action == RIGHT:
            if (col) < (self.col-1):
                col += 1
        elif action == LEFT:
            if (col) > 0:
                col -= 1
        elif action == STOP:
            pass

        self.next_state = self.coord_to_state(row, col)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal_state:
            self.reward = 1
        else:
            self.reward = 0

    def evaluate_done(self):
        if self.reward > 0:
            self.done = True
        else:
            self.done = False

    # 行動
    def step(self, state, action):
        self.evaluate_next_state(state, action)
        self.evaluate_reward(self.next_state)
        self.evaluate_done()

        return self.next_state, self.reward, self.done


# 崖歩きタスク
class CriffWorld(GlidWorld):
    def __init__(self, Row, Col, start, goal):
        super().__init__()
        self.row = Row
        self.col = Col
        self.start = start
        self.goal = goal

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal:
            self.reward = 1

        elif (self.row * (self.col - 1) + 1 <= state
              and state <= self.row * self.col - 2):
            self.reward = -10

        else:
            self.reward = 0


class GlidWorldSatori(GlidWorld):
    def __init__(self, row=7, col=7, start=None, goal=[9, 11, 15, 19, 29, 33, 37, 39]):
        super().__init__(row, col, start, goal)
        if start == None:
            self.start = int(row * col / 2)
        self.num_action = 4

    def evaluate_reward(self, state):
        for n in range(len(self.goal_state)):
            if state == self.goal_state[n]:
                self.reward = n + 1
        self.reward = 0


# 真の佐鳥ワールド
class GlidWorldSatori_True(GlidWorld):
    def __init__(self, number):
        self.side_space = number
        row = col = 4 * number + 1
        super().__init__(row, col, start=None)
        self.start = int(self.row * self.col / 2)
        self.num_state = self.row * self.col
        self.min_step = 3 * number
        self.middle = 2 * number - 1

        self.goal_state = np.zeros(self.num_state)

        # ゴール作成
        self.goal_state[super().coord_to_state(0, self.side_space)] = 8
        self.goal_state[super().coord_to_state(0, self.side_space + self.middle + 1)] = 1
        
        self.goal_state[super().coord_to_state(self.side_space, 0)] = 3
        self.goal_state[super().coord_to_state(self.side_space, self.col-1)] = 5

        self.goal_state[super().coord_to_state(self.side_space + self.middle + 1, 0)] = 6
        self.goal_state[super().coord_to_state(self.side_space + self.middle + 1, self.col - 1)] = 4

        self.goal_state[super().coord_to_state(self.row - 1, self.side_space)] = 2
        self.goal_state[super().coord_to_state(self.row - 1, self.side_space + self.middle + 1)] = 7

    def evaluate_reward(self, state):
        self.reward = self.goal_state[state]


class GlidWorld3Way(GlidWorld):
    def __init__(self, row, col, start, goal, criff_state=[]):
        super().__init__(row, col, start, goal)
        self.criff_state = criff_state
    
    def evaluate_reward(self, state):
        for n in range(len(self.criff_state)):
            if state == self.criff_state[n]:
                return -10
        else:
            return 0
