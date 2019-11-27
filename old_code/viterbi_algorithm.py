from collections import deque

import numpy as np
import operator


def viterbi_algorithm(observations, states, start_dist, trans_prob, emission_prob):
    #initializations
    n_s, n_o = len(states), len(observations)
    T1 = np.zeros((n_s, n_o))
    T2 = np.zeros((n_s, n_o), dtype=int)

    #forwared computation
    for i, state in enumerate(states):
        T1[i, 0] = start_dist(state) * emission_prob(observations[0], (state[0], state[0]))
        T2[i, 0] = 0
    for i, observation in enumerate(observations[1:]):
        for j, state in enumerate(states):
            p_max, k_max = max(((T1[k, i] * trans_prob(pre_state, state), k) for k, pre_state in enumerate(states)), key=operator.itemgetter(0))
            T1[j, i + 1] = emission_prob(observation, state) * p_max
            T2[j, i + 1] = k_max

    #backwared computation
    X = np.empty(n_o, dtype=object)
    Z = np.empty(n_o, dtype=int)
    p_max, state, k = max(((T1[k, -1], state, k) for k, state in enumerate(states)), key=operator.itemgetter(0))
    Z[-1], X[-1] = k, state
    for i in reversed(range(1, n_o)):
        Z[i - 1] = T2[Z[i], i]
        X[i - 1] = states[Z[i - 1]]
    return list(X)

