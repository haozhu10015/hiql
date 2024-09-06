import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style='ticks', font_scale=1.5)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']

from gridworld import GridWorld
from algorithms import value_iteration, policy_eval


if __name__ == '__main__':
    envr = GridWorld()
    if not os.path.exists('../figures'):
        os.makedirs('../figures')

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 4.5))
    cmap = 'viridis'
    # ground truth
    r_goal = np.zeros(envr.num_states)
    r_goal[envr.state_to_int(envr.goal_state)] = 1
    v_goal_gt = value_iteration(reward=r_goal, P=envr.P, num_actions=envr.num_actions,
                                num_states=envr.num_states, discount=envr.gamma)
    r_return = np.zeros(envr.num_states)
    r_return[envr.state_to_int(envr.initial_state)] = 1
    v_return_gt = value_iteration(reward=r_return, P=envr.P, num_actions=envr.num_actions,
                                  num_states=envr.num_states, discount=envr.gamma)
    im = axs[0, 0].imshow(v_goal_gt.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[0, 0], location='bottom', pad=0.05, ticks=[np.min(v_goal_gt), np.max(v_goal_gt)],
                 format='${x:.1f}$')
    im = axs[1, 0].imshow(v_return_gt.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[1, 0], location='bottom', pad=0.05, ticks=[np.min(v_return_gt), np.max(v_return_gt)],
                 format='${x:.1f}$')

    # iavi
    pi = np.load('../outputs/train/iavi/1024/fold_2/q.npy')
    pi = np.exp(pi) / np.sum(np.exp(pi), axis=-1, keepdims=True)
    v_goal = policy_eval(pi, r_goal, envr.P, envr.num_states, envr.gamma)
    v_return = policy_eval(pi, r_return, envr.P, envr.num_states, envr.gamma)
    im = axs[0, 1].imshow(v_goal.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[0, 1], location='bottom', pad=0.05, ticks=[np.min(v_goal), np.max(v_goal)],
                 format='${x:.1f}$')
    im = axs[1, 1].imshow(v_return.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[1, 1], location='bottom', pad=0.05, ticks=[np.min(v_return), np.max(v_return)],
                 format='${x:.1f}$')

    # hiavi
    pis = []
    for l_idx in range(2):
        pi = np.load(f'../outputs/train/hiavi/1024/fold_0/q_{l_idx}.npy')
        pi = np.exp(pi) / np.sum(np.exp(pi), axis=-1, keepdims=True)
        pis.append(pi)
    v_goal = policy_eval(pis[1], r_goal, envr.P, envr.num_states, envr.gamma)
    v_return = policy_eval(pis[0], r_return, envr.P, envr.num_states, envr.gamma)
    im = axs[0, 2].imshow(v_goal.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[0, 2], location='bottom', pad=0.05, ticks=[np.min(v_goal), np.max(v_goal)],
                 format='${x:.1f}$')
    im = axs[1, 2].imshow(v_return.reshape(5, 5), cmap=cmap)
    plt.colorbar(im, ax=axs[1, 2], location='bottom', pad=0.05, ticks=[np.min(v_return), np.max(v_return)],
                 format='${x:.1f}$')

    for r_idx in range(axs.shape[0]):
        for c_idx in range(axs.shape[1]):
            axs[r_idx, c_idx].set_xticks([])
            axs[r_idx, c_idx].set_yticks([])

    fig.add_artist(plt.Line2D((.36, .36), (0.05, 0.99), color="k", linestyle='dashed', linewidth=2))

    fig.text(0.082, 0.95, 'ground truth', fontsize=20)
    fig.text(0.61, 0.95, 'HIAVI', fontsize=20)
    axs[0, 0].set_ylabel("`goal'", fontsize=20)
    axs[1, 0].set_ylabel("`abandon'", fontsize=20)
    axs[0, 1].set_title('(1 intention)', fontsize=15)
    axs[0, 2].set_title('(2 intentions)', fontsize=15)

    plt.tight_layout(h_pad=0.1)
    fig.savefig('../figures/vs.pdf', bbox_inches='tight')
