import time
import numpy as np
from hiive.mdptoolbox import mdp, example
from matplotlib import pyplot as plt
from MyOpenAI_MDPToolbox import MyOpenAI_MDPToolbox


MAP8 = ['SHFFFFFF', 'FFFFFHFF', 'HFHFFFFF', 'FHFFHFFF', 'FHFFFFFF', 'FFFFFFFF', 'FHHFFFHF', 'FFFFHHHG']
MAP20 = ['SFFHFFHFFHFFFFFFFFHH', 'FFFHFFFFHFHFHHFFFFFF', 'FHFFHHFFHFFHFFFFFFHF', 'FFHFFFFFFFHFFFHFFFFF',
         'HFFFFFFFFHFFFHHFHFFF', 'FFHFFFFFHHFFFFFFFFFF', 'FFFFFFFFFHFFFHFFHFFF', 'FHHFFFFFFFHHFHFFHFFF',
         'FFFHFFFFFFFFFFFHFFFF', 'FFFFFFFFFHFFHFFHFFFF', 'FFFFFFHFFFFFFFFFFFFF', 'FHFHFFFFFFFFFHFFFHFF',
         'HFFFFFFFFFFHFFHFHFFF', 'FFFFFFFFFFFHFHFFHFFF', 'FFFFFFFFFFFHFHFHFFFF', 'FHHHFFFFFFFFFFFHFFHF',
         'FFFHFFFHFFFFHFHFFFFF', 'FFFHHFFFFHFFFFHHFFFF', 'FFFFFFHHFFFFFFFFFFFF', 'FHFFFFFFHFHHFHFFFHFG']
colors = {'S': 'green', 'F': 'skyblue', 'H': 'black', 'G': 'gold'}
directions = {3: '⬆', 2: '➡', 1: '⬇', 0: '⬅'}


def plot_algorithm_results(algorithm, P, R, gamma=.9, max_iterations=1000, axes=None, verbose=False):
    if algorithm == "VI":
        learner = mdp.ValueIteration(P, R, gamma, max_iter=max_iterations, run_stat_frequency=1)
    elif algorithm == "PI":
        learner = mdp.PolicyIteration(P, R, gamma, max_iter=max_iterations, run_stat_frequency=1)
    else:
        learner = mdp.QLearning(P, R, gamma, run_stat_frequency=1)
    if verbose:
        learner.setVerbose()
    now = time.perf_counter()
    res = learner.run()
    wall_clock = time.perf_counter() - now
    print(f"Wall clock time: {wall_clock * 1e3:.4f} ms")
    rewards = []
    variation = []
    iterations = []
    for r in res:
        rewards.append(r["Reward"])
        variation.append(r["Error"])
        iterations.append(r["Iteration"])

    plot_results(rewards, variation, iterations, axes)
    return learner.policy


def plot_results(rewards, variation, iterations, axes):
    if axes is None:
        _, axes = plt.subplots(1, 2)
    # Plot wall clock time
    axes[0].set_title("Reward per iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Reward")
    axes[0].grid()
    axes[0].plot(iterations, rewards)
    # Plot variance ratio
    axes[1].set_title("Variation per Iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Variation")
    axes[1].grid()
    axes[1].plot(iterations, variation)
    plt.show()


def show_policy_map(title, policy, map_desc):
    fig = plt.figure()
    size = len(map_desc)
    ax = fig.add_subplot(111, xlim=(0, size), ylim=(0, size))
    font_size = 'x-large'
    plt.title(title)
    if type(policy) == tuple:
        policy = np.array(policy).reshape(size, size)
    for i in range(size):
        for j in range(size):
            y = size - i - 1
            x = j
            p = plt.Rectangle([x, y], 1, 1)
            p.set_facecolor(colors[str(map_desc[i][j])])
            ax.add_patch(p)

            text = ax.text(x+0.5, y+0.5, directions[policy[i, j]], weight='bold', size=font_size,
                           horizontalalignment='center', verticalalignment='center', color='w')
    plt.show()


def show_forest_policy(policy):
    policy = list(policy)
    last_seen = policy[0]
    last_index = 0
    forest_actions = {0: "Wait", 1: "Cut"}
    for state in range(1, len(policy)):
        if last_seen != policy[state]:
            print(f"For ages ranging from {last_index} to {state-1} is better to {forest_actions[policy[last_seen]]}")
            last_index = state
            last_seen = policy[state]
    print(f"For ages ranging from {last_index} to {len(policy)} is better to {forest_actions[policy[last_seen]]}")


def policy_to_str(policy):
    to_plot = []
    forest_actions = {0: "Wait", 1: "Cut"}
    for action in policy:
        to_plot.append(forest_actions[action])
    return to_plot


def show_forest_policy_ql(policy, axes=None):
    policy = list(policy)
    range_ = [i for i in range(len(policy))]
    to_plot = policy_to_str(policy)
    if axes is None:
        _, axes = plt.subplots(1, 1)
    axes.set_title(f"Best Policy for {len(policy)} states")
    axes.set_xlabel("Year")
    axes.set_ylabel("Action")
    axes.grid()
    axes.plot(range_, to_plot)
    plt.show()


def generate_comparative_plots(nb_states, gamma=.9, max_iterations=1000):
    P, R = example.forest(S=nb_states, r1=4, r2=2, p=0.1)
    vi = mdp.ValueIteration(P, R, gamma, max_iter=max_iterations, run_stat_frequency=1)
    pi = mdp.PolicyIteration(P, R, gamma, max_iter=max_iterations, run_stat_frequency=1)
    ql = mdp.QLearning(P, R, gamma, run_stat_frequency=1)
    vi.run()
    pi.run()
    ql.run()
    vi_pol = policy_to_str(vi.policy)
    pi_pol = policy_to_str(pi.policy)
    ql_pol = policy_to_str(ql.policy)
    range_ = [i for i in range(len(vi_pol))]
    _, axes = plt.subplots(1, 1)
    axes.set_title(f"Best Policy for {len(vi_pol)} states")
    axes.set_xlabel("Year")
    axes.set_ylabel("Action")
    axes.grid()
    axes.plot(range_, vi_pol, color="r", label="VI")
    axes.plot(range_, pi_pol, color="g", label="PI")
    axes.plot(range_, ql_pol, color="b", label="QL")
    axes.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    generate_comparative_plots(10)
    # P, R = example.forest(S=1000, r1=4, r2=2, p=0.1)
    # _, axes = plt.subplots(1, 2, figsize=(20, 5))
    # best_policy = plot_algorithm_results("QL", P, R, axes=axes, verbose=True)
    # show_forest_policy_ql(best_policy)
