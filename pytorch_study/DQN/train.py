import numpy as np
import sys
from copy import deepcopy

def train(agent, env, simulation_times, episode_times, step_times, pre_step_times):
    # todo_記録用のnumpy配列作成
    reward_graph = np.zeros(episode_times)

    for n_simulation in range(simulation_times):
        # todo_初期化処理
        simulation_step = 0
        reward_epi = np.zeros(episode_times)

        for n_episode in range(episode_times):
            # todo_初期化処理
            # 記録用
            # 初期化処理
            obs = env.reset()
            step = 0
            sys.stdout.write("\rsimu:%s/%s, epi:%s/%s" % (str(n_simulation), str(simulation_times-1), str(n_episode), str(episode_times-1)))
            while True:
                if simulation_step < pre_step_times:
                    action = agent.random_action()
                else:
                    action = agent.select_action(obs)

                new_obs, rew, done, info = env.step(action)
                reward_epi[n_episode] += rew
                agent.observe(obs, action, new_obs, rew, done)

                if step >= step_times:
                    done = True

                if simulation_step > pre_step_times:
                    agent.update(simulation_step)

                step += 1
                simulation_step += 1
                obs = deepcopy(new_obs)

                if done:
                    # print("episode:{} simulation_step:{}, reward:{}, eps:{}" .format(n_episode, simulation_step, reward_epi[n_episode], agent.epsilon))
                    reward_graph[n_episode] += reward_epi[n_episode]
                    break

    return reward_graph / simulation_times
