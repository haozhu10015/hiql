import numpy as np
from scipy.special import logsumexp


def value_iteration(reward, P, num_states, num_actions, discount, threshold=1e-2):
    """
    Calculate the optimal state value function of given enviroment.

    :param reward: reward vector. nparray. (states, )
    :param P: transition probability p(st | s, a). nparray. (states, states, actions).
    :param discount: discount rate gamma. float. Default: 0.99
    :param num_states: number of states. int.
    :param num_actions: number of actions. int.
    :param threshold: stop when difference smaller than threshold. float.
    :return: optimal state value function. nparray. (states)
    """

    v = np.zeros(num_states)

    while True:
        delta = 0

        for s in range(num_states):
            max_v = float("-inf")
            for a in range(num_actions):
                tp = P[s, :, a]
                max_v = max(max_v, np.dot(tp, (reward + discount * v)))

            diff = abs(v[s] - max_v)
            delta = max(delta, diff)

            v[s] = max_v

        if delta < threshold:
            break

    return v


def vi_policy(num_states, num_actions, P, reward, discount, stochastic=True, threshold=1e-2):
    """
    Find the optimal policy.

    num_states: Number of states. int.
    num_actions: Number of actions. int.
    P: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    v = value_iteration(reward, P, num_states, num_actions, discount, threshold)

    policy = np.zeros((num_states, num_actions))
    if stochastic:
        for s in range(num_states):
            for a in range(num_actions):
                p = P[s, :, a]
                policy[s, a] = p.dot(reward + discount*v)
        policy -= policy.max(axis=1).reshape((num_states, 1))  # For numerical stability.
        policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((num_states, 1))

    else:
        def _policy(s):
            return max(range(num_actions),
                       key=lambda a: sum(P[s, k, a] *
                                         (reward[k] + discount * v[k])
                                         for k in range(num_states)))
        for s in range(num_states):
            policy[s, _policy(s)] = 1
    return policy


def policy_eval(policy, reward, P, num_states, discount, threshold=1e-2):
    """
    Policy evaluation.

    :param policy: policy to evaluation. nparray. (states, actions).
    :param reward: ground truth reward of the enviroment. nparray. (states, ).
    :param P: transition probability p(st | s, a). nparray. (states, states, actions).
    :param num_states: number of states in the enviroment. int.
    :param discount: discount rate gamma. float.
    :param threshold: stop when difference smaller than threshold. float.
    :return: state value estimation for given policy. nparray. (states, ).
    """
    v = np.zeros(num_states)
    while True:
        delta = 0
        for s in range(num_states):
            pi = policy[s]
            tp = P[s, :, :]
            target = np.dot(pi, np.matmul(tp.T, (reward + discount * v).reshape(-1, 1)))

            delta = max(delta, np.abs(target - v[s]))

            v[s] = target

        if delta < threshold:
            break

    return v


class IAVI:
    def __init__(self, num_states, num_actions, P, expert_policy, discount, threshold=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P
        self.expert_policy = expert_policy
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6

        self.r = np.random.randn(self.num_states, self.num_actions)
        self.q = np.random.randn(self.num_states, self.num_actions)

    def train(self):
        X = np.ones((self.num_actions, self.num_actions))
        X *= -1 / (self.num_actions - 1)
        for i in range(self.num_actions):
            X[i, i] = 1

        e = 0
        while True:
            e += 1
            delta = 0
            for s in range(self.num_states):
                tp = self.P[s, :, :]
                eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                    tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)

                Y = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    eta_a = eta[a]
                    action_b = [b for b in range(self.num_actions) if b != a]
                    eta_b = eta[action_b]
                    Y[a] = eta_a - 1 / (self.num_actions - 1) * np.sum(eta_b)

                r = np.linalg.lstsq(X, Y, rcond=None)[0]

                delta = max(delta, np.max(np.abs(self.r[s, :] - r)))

                self.r[s, :] = r
                self.q[s, :] = r + self.discount * np.matmul(tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)

            if delta < self.threshold:
                break


class HIAVI:
    def __init__(self, num_latents, num_states, num_actions, P, train_trajs, test_trajs, discount):
        self.num_latents = num_latents
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P
        self.discount = discount
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs

    def _get_mc_probs(self, pis, trajs, logp_init, logp_tr):
        num_latents = logp_init.shape[0]
        logp_gammas = []
        logp_xis = []
        lls = []
        for traj in trajs:
            logp_obs = []
            for s, a, ns in traj:
                logp_obs.append([np.log(pis[l_idx][s, a]) for l_idx in range(num_latents)])
            logp_obs = np.array(logp_obs)

            logp_alpha_prev = logp_init + logp_obs[0]
            logp_alpha = [logp_alpha_prev]
            for lpo in logp_obs[1:]:
                logp_alpha_prev = logsumexp(logp_alpha_prev + logp_tr.T, axis=-1)
                logp_alpha_prev += lpo
                logp_alpha.append(logp_alpha_prev)

            logp_beta_next = np.log(np.ones((num_latents,)))
            logp_beta = [logp_beta_next]
            for lpo_idx, lpo in enumerate(reversed(logp_obs[1:])):
                logp_beta_next += lpo
                logp_beta_next = logsumexp(logp_beta_next + logp_tr, axis=-1)
                logp_beta.append(logp_beta_next)

            logp_alpha = np.array(logp_alpha)
            logp_beta = np.array(logp_beta[::-1])

            logp_gamma = logp_alpha + logp_beta
            logp_gamma -= logsumexp(logp_gamma, axis=-1, keepdims=True)

            logp_xi = []
            for lpa_idx, lpa in enumerate(logp_alpha[:-1]):
                lpx = lpa[:, np.newaxis] + logp_tr + logp_beta[lpa_idx + 1] + logp_obs[lpa_idx + 1]
                lpx -= logsumexp(lpx)
                logp_xi.append(lpx)
            logp_xi = np.array(logp_xi)

            ll = logsumexp(logp_gamma + logp_obs, axis=-1)

            lls.append(ll)
            logp_gammas.append(logp_gamma)
            logp_xis.append(logp_xi)

        return logp_gammas, logp_xis, lls

    def fit(self):
        p_init = np.abs(np.random.randn(2))
        p_init /= np.sum(p_init)
        p_tr = 0.95 * np.identity(2)
        p_tr += np.abs(np.random.normal(0, 0.05, (2, 2)))
        p_tr /= np.sum(p_tr, axis=-1, keepdims=True)
        logp_init = np.log(p_init)
        logp_tr = np.log(p_tr)

        pis = []
        for l_idx in range(self.num_latents):
            pi = np.abs(np.random.randn(self.num_states, self.num_actions))
            pi /= np.sum(pi, axis=-1, keepdims=True)
            pis.append(pi)
        logp_gammas, *_ = self._get_mc_probs(pis, self.train_trajs, logp_init, logp_tr)

        while True:
            z_hat = np.argmax(np.vstack(logp_gammas), axis=-1)
            pis = []
            agents = []
            for latent_idx in range(self.num_latents):
                inputs = []
                for session_idx, session_trajs in enumerate(self.train_trajs):
                    logp_gamma = logp_gammas[session_idx]
                    for traj_idx, traj in enumerate(session_trajs):
                        if np.random.uniform() > np.exp(logp_gamma[traj_idx, latent_idx]):
                            continue
                        inputs.append(traj)

                expert_pi = np.zeros((self.num_states, self.num_actions))
                for s, a, ns in inputs:
                    expert_pi[s, a] += 1
                expert_pi[expert_pi.sum(axis=1) == 0] = 1e-6
                expert_pi /= expert_pi.sum(axis=1).reshape(-1, 1)
                agent = IAVI(num_states=self.num_states, num_actions=self.num_actions,
                             P=self.P, expert_policy=expert_pi, discount=self.discount)
                agent.train()

                agents.append(agent)
                q = agent.q
                pis.append(np.exp(q) / np.sum(np.exp(q), axis=-1, keepdims=True))

            logp_gammas, logp_xis, _ = self._get_mc_probs(pis, self.train_trajs, logp_init, logp_tr)

            logp_init = logsumexp([logp_gamma[0] for logp_gamma in logp_gammas], b=1 / len(logp_gammas), axis=0)
            logp_tr = logsumexp(np.concatenate(logp_xis), axis=0)
            logp_tr -= logsumexp(np.concatenate([logp_gamma[:-1] for logp_gamma in logp_gammas]), axis=0,
                                 keepdims=True).T
            logp_tr -= logsumexp(logp_tr, axis=-1, keepdims=True)

            if (z_hat == np.argmax(np.vstack(logp_gammas), axis=-1)).all():
                break

        # Evaluation
        ll = {}
        for ds in ['train', 'test']:
            inputs = eval(f'self.{ds}_trajs')
            *_, lls = self._get_mc_probs(pis, inputs, logp_init, logp_tr)
            lls = np.mean(np.hstack(lls))
            ll[ds] = np.mean(lls)

        return ll, logp_init, logp_tr, agents

    def predict(self, pis, trajs, logp_init, logp_tr):
        logp_gammas, *_ = self._get_mc_probs(pis, trajs, logp_init, logp_tr)
        return logp_gammas
