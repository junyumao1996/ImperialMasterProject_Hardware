import numpy as np
import matplotlib.pyplot as plt


def plot_single_reward_curve(rewards):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(len(rewards)), rewards)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward per episode')
    plt.grid()
    plt.show()
    plt.close(fig)
    return fig

def save_single_reward_curve1(rewards, parent_dir_path):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(len(rewards)), rewards)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward per episode')
    plt.grid()
    fig.savefig(parent_dir_path + '/reward_plot')
    plt.close(fig)
    return None

def save_single_reward_curve2(parent_dir_path, filename):
    data = np.loadtxt(parent_dir_path + '/' + filename)
    rewards = data[:, 1]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(len(rewards)), rewards)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Reward per episode')
    plt.grid()
    fig.savefig(parent_dir_path + '/reward_plot')
    plt.close(fig)
    return None

def plot_multiple_reward_curve(rewards_list, legends=None):
    fig = plt.figure()
    for i, item in enumerate(rewards_list):
        plt.plot(range(len(item)), item)
    plt.xlabel('Iterations')
    plt.ylabel('Reward per episode')
    if legends != None:
        plt.legend(legends)
    else:
        pass
    plt.grid()
    plt.show()
    plt.close(fig)
    return fig






