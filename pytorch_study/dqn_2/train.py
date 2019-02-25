import numpy as np
from copy import deepcopy

def train(agent, env, simulation_times, episode_times, step_times, pre_step_times):
    # todo_記録用のnumpy配列作成

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
                    print("episode:{} simulation_step:{}, reward:{}, eps:{}" .format(n_episode, simulation_step, reward_epi[n_episode], agent.epsilon))
                    break

                