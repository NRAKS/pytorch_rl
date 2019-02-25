import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import time
import seaborn as sns
import pickle
import copy
import sys

params = {'mathtext.default': 'regular'}   
plt.rcParams.update(params)
fontsize=18
alpha = 1.0
alpha_dec = 0.25
time = time.asctime()


def save_pickle(data, title):
    os.makedirs("output/{}/config" .format(time), exist_ok=True)
    with open("output/{}/config/{}.pickle" .format(time, title), mode='wb') as f:
        pickle.dump(data, f)


def graph(graph, NUM_AGENT, xlabel=None, ylabel=None, policy_name=None, label=None, color=None, y_lim=None, path=None, title=None):

    os.makedirs(path, exist_ok=True)
    # print("{}, {}グラフ作成" .format(policy_name, ylabel))

    if y_lim is not None:
        plt.ylim(y_lim)

    # print(len(graph))

    for n_agent in range(NUM_AGENT):
        # print("a")
        if label is None:
            plt.plot(graph[n_agent], alpha=alpha-(n_agent*alpha_dec), label="agent {}" .format(n_agent))
        else:
            plt.plot(graph[n_agent], alpha=alpha-(n_agent*alpha_dec), label="{}" .format(label[n_agent]))
    plt.legend()
    plt.title("{} time development {} agent" .format(ylabel, NUM_AGENT))
    plt.xlabel("{}" .format(xlabel))
    plt.ylabel("{}" .format(ylabel))
    plt.legend(fontsize=fontsize)
    plt.savefig(path+"/{}" .format(title))
    # plt.show()
    plt.figure()


def graph_regret(graph, ideal, NUM_AGENT, xlabel=None, ylabel="regret", policy_name=None, label=None, color=None, ylim=None):

    os.makedirs("output/{}/{}_agents/{}" .format(time, NUM_AGENT, policy_name), exist_ok=True)

    # print("graph:{}" .format(graph))

    ideal_list = np.zeros(len(graph[0]))
    for n in range(len(graph[0])):
        ideal_list[n] = ideal_list[n-1]+ideal

    print("{}, {}グラフ作成" .format(policy_name, ylabel))

    for n_agent in range(len(graph)):
        if label is None:
            plt.plot(ideal_list - graph[n_agent], alpha=alpha-(n_agent*alpha_dec), label="agent {}" .format(n_agent))
        else:
            
            plt.plot(ideal_list - graph[n_agent], alpha=alpha-(n_agent*alpha_dec), label="{}" .format(label[n_agent]), color=color[n_agent])
    plt.legend()
    plt.title("{} time development {} agent" .format(ylabel, NUM_AGENT))
    plt.xlabel("{}" .format(xlabel))
    plt.ylabel("{}" .format(ylabel))
    plt.legend(fontsize=fontsize)
    plt.savefig("output/{}/{}_agents/{}/{}_{}" .format(time, NUM_AGENT, policy_name, ylabel, policy_name))
    # plt.show()
    plt.figure()


def graph_comparison(list, ylabel):
    os.makedirs("output/{}/comparison" .format(time), exist_ok=True)
    for n in range(len(list[0])):
        for n_agent in range(len(list)):
            if list[n_agent][n] is None:
                pass
            else:
                if n_agent+1 == 1:
                    plt.plot(list[n_agent][n], alpha=alpha-(n_agent*alpha_dec), label="{} agent" .format(n_agent+1))
                else:
                    plt.plot(list[n_agent][n], alpha=alpha-(n_agent*alpha_dec), label="{} agents" .format(n_agent+1))
        plt.legend()
        plt.title("{} time development" .format(ylabel))
        plt.xlabel("episode")
        plt.ylabel("{}" .format(ylabel))
        plt.legend(fontsize=fontsize)

        if n == 0:
            name = "GRC"
        elif n == 1:
            name = "RS"
        elif n == 2:
            name = "eps-decrease"
            
        plt.savefig("output/{}/comparison/{}_{}" .format(time, ylabel, name))
        # plt.show()
        plt.figure()


def graph_comparison_regret(list, ideal, ylabel="regret"):
    os.makedirs("output/{}/comparison" .format(time), exist_ok=True)
    ideal_list = np.zeros((len(list[0][0])))
    for n in range(len(list[0][0])):
        ideal_list[n] = ideal_list[n-1] + ideal

    for n in range(len(list[0])):
        for n_agent in range(len(list)):
            if list[n_agent][n] is None:
                pass
            else:
                if n_agent+1 == 1:
                    plt.plot(list[n_agent][n], alpha=alpha-(n_agent*alpha_dec), label="{} agent" .format(n_agent+1))
                else:
                    plt.plot(ideal_list - list[n_agent][n], alpha=alpha-(n_agent*alpha_dec), label="{} agents" .format(n_agent+1))
        plt.legend()
        plt.title("{} time development" .format(ylabel))
        plt.xlabel("episode")
        plt.ylabel("{}" .format(ylabel))
        plt.legend(fontsize=fontsize)
        if n == 0:
            name = "GRC"
        elif n == 1:
            name = "RS"
        elif n == 2:
            name = "eps-decrease"
            
        plt.savefig("output/{}/comparison/{}_{}" .format(time, ylabel, name))
        # plt.show()
        plt.figure()


def draw_heatmap(data, NUM_AGENT, env, element_name, policy_name, episode_time=None):
    if episode_time is None:
        os.makedirs("output/{}/{}_agents/{}" .format(time, NUM_AGENT, policy_name), exist_ok=True)
    else:
        os.makedirs("output/{}/{}_agents/{}/state_count_per_epi" .format(time, NUM_AGENT, policy_name), exist_ok=True)
    # 描画する
    if NUM_AGENT == 1:
        size = 1
    elif 1 < NUM_AGENT <= 4:
        size = 2
    
    fig = plt.figure()

    for n in range(NUM_AGENT):
        _data = data[n].reshape(env.row, env.col)
        # print(_data.shape)

        ax = fig.add_subplot(size, size, int(n+1))
        sns.heatmap(_data, ax=ax)
        # ims = sns.heatmap(_data, ax=ax)

    # fig.set_xticks(np.arange(data[0].shape[0]) + 0.5, minor=False)
    # fig.set_yticks(np.arange(data[0].shape[1]) + 0.5, minor=False)

    # # ax.invert_yaxis()
    # # ax.xaxis.tick_top()

    # # ax.set_xticklabels(row_labels, minor=False)
    # # ax.set_yticklabels(column_labels, minor=False)
    # fig.show()
    if episode_time is None:
        plt.savefig("output/{}/{}_agents/{}/{}_{}_heatmap" .format(time, NUM_AGENT, policy_name, element_name, policy_name))
    else:
        plt.savefig("output/{}/{}_agents/{}/state_count_per_epi/{}_{}_{}_heatmap" .format(time, NUM_AGENT, policy_name, episode_time, element_name, policy_name))

    plt.figure()

def make_heat_animation(data, label, NUM_AGENT, env, policy_name):
    fig = plt.figure()

    def update(i, title, NUM_AGENT, env, length):

        if i != 0:
            plt.clf()
            # plt.cla()

        sys.stdout.write("描画中\r%s/%s" % (str(i), str(length)))
        # 描画する
        if NUM_AGENT == 1:
            size = 1
        elif 1 < NUM_AGENT <= 4:
            size = 2
        for n in range(NUM_AGENT):
            _data = data[i, n].reshape(env.row, env.col)
        # print(_data.shape)

            ax = fig.add_subplot(size, size, int(n+1))
            sns.heatmap(np.log(_data+1), ax=ax)
            plt.title('epi:' + str(i))

    ani = animation.FuncAnimation(fig, update, fargs=('State Visits', NUM_AGENT, env, len(data)), interval=100, frames=len(data))

    ani.save("output/{}/{}_agents/{}/{}.gif".format(time, NUM_AGENT, policy_name,label), writer='imagemagick')


def pie_graph(data, label, path, title):
    # 描画する
    if len(data) == 1:
        size = 1
    elif 1 < len(data) <= 4:
        size = 2
    
    fig = plt.figure()

    for n in range(len(data)):
        # print(_data.shape)

        ax = fig.add_subplot(size, size, int(n+1))
        plt.pie(data[n], labels=label, counterclock=False, startangle=90, autopct="%1.1f%%")
        plt.axis('equal')
    plt.savefig(path+"/{}" .format(title))
    plt.figure()