import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from gridworld import GridWorld
from algorithms import HIAVI


if __name__ == '__main__':
    num_folds = 5
    num_repeats = 10
    np.random.seed(42)

    output_dir = f'../outputs/train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    envr = GridWorld()
    with open('../data/trajs.js') as f:
        trajs = json.load(f)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10015)
    for num_trajs in np.arange(24, 1025, 100):
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.infty
            best_ll = None
            for repeats in range(num_repeats):
                model = HIAVI(num_latents=2, num_states=envr.num_states, num_actions=envr.num_actions,
                                train_trajs=train_trajs, test_trajs=test_trajs, P=envr.P, discount=envr.gamma)
                ll, logp_init, logp_tr, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == 1024:
                        param_dir = os.path.join(output_dir, f'hiavi/{num_trajs}/fold_{kf_idx}')
                        if not os.path.exists(param_dir):
                            os.makedirs(param_dir)
                        np.save(os.path.join(param_dir, 'logp_init.npy'), logp_init)
                        np.save(os.path.join(param_dir, 'logp_tr.npy'), logp_tr)
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(output_dir, 'll_hiavi.csv'), index=False)
