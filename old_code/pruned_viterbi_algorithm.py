from collections import deque

import numpy as np
import operator


def viterbi_algorithm(observations, states_range, state_dimension, possible_starts, pre_states, observation_states, trans_prob, emission_prob):
    #initializations
    n_o = len(observations)
    shape = (n_o,) + ((states_range,) * state_dimension)
    T1 = np.zeros(shape, dtype=float)
    T2 = np.zeros(shape, dtype=object)

    #forwared computation
    for state, p_state in possible_starts: #upgrade: possible starts matrix
        for i in range(states_range):
            T1[(0, i, state)] = p_state * emission_prob(observations[0], (state,))

    for i, observation in enumerate(observations[1:]):
        states = observation_states(observation)
        for state in states:
            p_max, pre_max = max(((T1[(i,) + pre_state] * trans_prob(pre_state, state), pre_state) for pre_state in pre_states(state)), key=operator.itemgetter(0))
            s_index = (i + 1,) + state
            T1[s_index] = emission_prob(observation, state) * p_max
            T2[s_index] = pre_max

    #backwared computation
    X = np.empty(n_o, dtype=object)
    X[-1] = np.unravel_index(T1[-1].argmax(), T1[-1].shape)
    for i in reversed(range(1, n_o)):
        X[i - 1] = T2[(i,) + X[i]]
    return X

