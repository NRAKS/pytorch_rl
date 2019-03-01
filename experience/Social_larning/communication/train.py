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


# マルチエージェントトレーニング
def multi_train(agent, env, simulation_times, episode_times, step_times, pre_step_times):
    # todo_記録用のnumpy配列作成
    reward_graph = np.zeros((len(agent)+1, episode_times))
    action_graph = np.zeros((len(agent), env.n_action))
    store_graph_epi = np.zeros((len(agent), episode_times))
    for n_simulation in range(simulation_times):
        # todo_初期化処理
        simulation_step = 0
        reward_epi = np.zeros((len(agent)+1, episode_times))

        for n_episode in range(episode_times):
            # todo_初期化処理
            # 記録用
            # 初期化処理
            obs = env.reset()
            step = 0
            sys.stdout.write("\rsimu:%s/%s, epi:%s/%s" % (str(n_simulation), str(simulation_times-1), str(n_episode), str(episode_times-1)))
            while True:
                # print("step:{}" .format(step))
                action = np.full(len(agent), None)
                for n_agent in range(len(agent)):
                    if simulation_step < pre_step_times:
                        action[n_agent] = agent[n_agent].random_action()
                    else:
                        action[n_agent] = agent[n_agent].select_action(obs[n_agent])
                    action_graph[n_agent, action[n_agent]] += 1
                    if action[n_agent] == 2:
                        store_graph_epi[n_agent, n_episode] += 1

                # print(action.shape)
                new_obs, rew, done, info = env.step(action)
                # print("new_obs:{}. rew:{}, done:{}, info:{}" .format(new_obs, rew, done, info))
                for n_agent in range(len(agent)):
                    reward_epi[n_agent, n_episode] += rew[n_agent]
                    agent[n_agent].observe(obs[n_agent], action[n_agent], new_obs[n_agent], rew[n_agent], done)

                if step >= step_times:
                    done = True

                if simulation_step > pre_step_times:
                    for n_agent in range(len(agent)):
                        agent[n_agent].update(simulation_step)

                step += 1
                simulation_step += 1
                obs = deepcopy(new_obs)

                # print("done:{}" .format(done))
                if done:
                    print("episode:{} simulation_step:{}, reward_1:{}, reward_2:{}" .format(n_episode, simulation_step, reward_epi[0, n_episode], reward_epi[1, n_episode]))
                    break
            for n in range(len(agent)):     
                reward_graph[n, n_episode] += reward_epi[n, n_episode]

    reward_graph[-1] = reward_graph[0] + reward_graph[1]
    a = deepcopy(reward_graph[2])
    reward_graph[2] = reward_graph[1]
    reward_graph[1] = reward_graph[0]
    reward_graph[0] = a
    reward_graph /= simulation_times
    action_graph /= simulation_times
    store_graph_epi /= simulation_times

    return np.asarray([reward_graph, action_graph, store_graph_epi])


# コミュニケーションマルチエージェントトレーニング
def multi_train_comu(agent, env, simulation_times, episode_times, step_times, pre_step_times):
    # todo_記録用のnumpy配列作成
    reward_graph = np.zeros((len(agent)+1, episode_times))
    action_graph = np.zeros((len(agent), env.n_action))
    store_graph_epi = np.zeros((len(agent), episode_times))
    for n_simulation in range(simulation_times):
        # todo_初期化処理
        simulation_step = 0
        reward_epi = np.zeros((len(agent)+1, episode_times))

        for n_episode in range(episode_times):
            # todo_初期化処理
            # 記録用
            # 初期化処理
            obs = env.reset()
            # obs_agent1 = deepcopy(obs)
            step = 0
            sys.stdout.write("\rsimu:%s/%s, epi:%s/%s" % (str(n_simulation), str(simulation_times-1), str(n_episode), str(episode_times-1)))
            while True:
                # print("step:{}" .format(step))
                action = np.full(len(agent), None)
                for n_agent in range(len(agent)):
                    if n_agent == 0:
                        if simulation_step < pre_step_times:
                            action[n_agent] = agent[n_agent].random_action()
                        else:
                            action[n_agent] = agent[n_agent].select_action(obs[n_agent])

                    elif n_agent == 1:
                        # obs_agent1[n_agent, -1] = action[0]
                        obs[n_agent, -1] = action[0]    # 相手の行動についての情報を追加

                        # print("obs:{}" .format(obs))
                        # print("n_obs:{}" .format(new_obs))
                        if step > 0:
                            # 行動決定の直前にエージェントの観測情報に加える
                            # print("action[n_agent]:{}" .format(action[0]))
                            agent[n_agent].observe(obs_agent1[n_agent], pre_action[n_agent], obs[n_agent], rew[n_agent], done)

                        if simulation_step < pre_step_times:
                            action[n_agent] = agent[n_agent].random_action()
                        else:
                            action[n_agent] = agent[n_agent].select_action(obs[n_agent])

                    action_graph[n_agent, action[n_agent]] += 1
                    if action[n_agent] == 2:
                        store_graph_epi[n_agent, n_episode] += 1

                # print(action.shape)
                new_obs, rew, done, info = env.step(action)
                # print("new_obs:{}. rew:{}, done:{}, info:{}" .format(new_obs, rew, done, info))

                if step >= step_times:
                    done = True

                for n_agent in range(len(agent)):
                    reward_epi[n_agent, n_episode] += rew[n_agent]
                    if n_agent == 0 or done:    # 終端状態であれば1番目のエージェントの更新もする
                        agent[n_agent].observe(obs[n_agent], action[n_agent], new_obs[n_agent], rew[n_agent], done)

                if simulation_step > pre_step_times:
                    for n_agent in range(len(agent)):
                        # print("n_agent:{}" .format(n_agent))
                        agent[n_agent].update(simulation_step)

                step += 1
                simulation_step += 1
                obs_agent1 = deepcopy(obs)
                pre_action = deepcopy(action)
                obs = deepcopy(new_obs)

                # print("done:{}" .format(done))
                if done:
                    print("episode:{} simulation_step:{}, reward_1:{}, reward_2:{}" .format(n_episode, simulation_step, reward_epi[0, n_episode], reward_epi[1, n_episode]))
                    break
            for n in range(len(agent)):     
                reward_graph[n, n_episode] += reward_epi[n, n_episode]

    reward_graph[-1] = reward_graph[0] + reward_graph[1]
    a = deepcopy(reward_graph[2])
    reward_graph[2] = reward_graph[1]
    reward_graph[1] = reward_graph[0]
    reward_graph[0] = a
    reward_graph /= simulation_times
    action_graph /= simulation_times
    store_graph_epi /= simulation_times

    return np.asarray([reward_graph, action_graph, store_graph_epi])