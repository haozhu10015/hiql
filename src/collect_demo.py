import os
import json

import numpy as np

from gridworld import GridWorld
from algorithms import value_iteration, vi_policy

if __name__ == '__main__':
    np.random.seed(10015)
    num_trajs = 1024
    p_barrier = 0.3
    p_t = 0.5

    envr = GridWorld()

    r_goal = np.zeros(envr.num_states)
    r_goal[envr.state_to_int(envr.goal_state)] = 1
    v_goal = value_iteration(reward=r_goal, P=envr.P, num_actions=envr.num_actions,
                             num_states=envr.num_states, discount=envr.gamma)
    pi_goal = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                        P=envr.P, reward=r_goal, discount=envr.gamma, stochastic=False)

    r_return = np.zeros(envr.num_states)
    r_return[envr.state_to_int(envr.initial_state)] = 1
    v_return = value_iteration(reward=r_return, P=envr.P, num_actions=envr.num_actions,
                               num_states=envr.num_states, discount=envr.gamma)
    pi_return = vi_policy(num_states=envr.num_states, num_actions=envr.num_actions,
                          P=envr.P, reward=r_return, discount=envr.gamma, stochastic=False)

    pis = [pi_goal, pi_return]

    trajs = []
    latents = []
    for repeat in range(num_trajs):
        traj = []
        latent = []
        s = envr.state_to_int(envr.initial_state)
        pi_idx = 0
        t = 0
        while True:
            if envr.int_to_state(s) in envr.barriers:
                if np.random.uniform() < p_barrier:
                    pi_idx = 1 - pi_idx
            elif t == 8:
                if np.random.uniform() < p_t:
                    pi_idx = 1

            pi = pis[pi_idx]
            a = np.random.choice(envr.num_actions, p=pi[s])
            ns, done = envr.step(s, a)

            traj.append([s, a, ns])
            latent.append(pi_idx)
            s = ns
            t += 1
            if done:
                break
        trajs.append(traj)
        latents.append(latent)
    if not os.path.exists('../data'):
        os.makedirs('../data')
    with open('../data/trajs.js', 'w') as f:
        json.dump(trajs, f)
    with open('../data/latents.js', 'w') as f:
        json.dump(latents, f)
