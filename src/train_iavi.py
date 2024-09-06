import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from gridworld import GridWorld
from algorithms import IAVI


if __name__ == '__main__':
    num_folds = 5
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
            pi = np.zeros((envr.num_states, envr.num_actions))
            for train_idx in train_idxes:
                for s, a, ns in trajs[train_idx]:
                    pi[s, a] += 1
            pi[pi.sum(axis=1) == 0] = 1e-6
            pi /= pi.sum(axis=1).reshape(-1, 1)

            agent = IAVI(num_states=envr.num_states, num_actions=envr.num_actions,
                         P=envr.P, expert_policy=pi, discount=envr.gamma)
            agent.train()

            pi_hat = np.exp(agent.q) / np.sum(np.exp(agent.q), axis=-1, keepdims=True)
            ll = {'train': [], 'test': []}
            for ds in ['train', 'test']:
                input_idxes = eval(f'{ds}_idxes')
                for idx in input_idxes:
                    like = []
                    for s, a, ns in trajs[idx]:
                        like.append(pi_hat[s, a])
                    like = np.log(like)
                    ll[ds].append(np.mean(like))
            if num_trajs == 1024:
                param_dir = os.path.join(output_dir, f'iavi/{num_trajs}/fold_{kf_idx}')
                if not os.path.exists(param_dir):
                    os.makedirs(param_dir)
                np.save(os.path.join(param_dir, f'r.npy'), agent.r)
                np.save(os.path.join(param_dir, f'q.npy'), agent.q)

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, np.mean(ll['train']), np.mean(ll['test'])]
            output_df.to_csv(os.path.join(output_dir, 'll_iavi.csv'), index=False)
