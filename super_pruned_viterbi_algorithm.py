from collections import deque

import numpy as np
import operator

#this super pruned version of viterbo algorithm can classify over 40,000 words in 1500 sentences in less then 10 seconds
def viterbi_algorithm(observations, states_range, state_dimension, dim2_possible_starts, pre_states, observation_states, trans_prob, emission_prob):
    #initializations
    n_o = len(observations)
    shape = (n_o,) + ((states_range,) * state_dimension)
    T1 = np.zeros(shape, dtype=float)
    T2 = np.zeros(shape, dtype=object)

    #forwared computation
    assigned_states = set()
    if n_o > 1:  #fix: the case of only one word
        for state, p_state in dim2_possible_starts:
            res = emission_prob(observations[0], (state[0],)) * emission_prob(observations[1], state) * p_state
            if res > 0:
                T1[(1,) + state] = res
                T2[(1,) + state] = (state[0], state[0])
                assigned_states.add(state)


    for i, observation in enumerate(observations[2:]):
        states = observation_states(observation)
        prevoius_states = assigned_states
        assigned_states = set()
        for state in states:
            try:
                p_max, pre_max = max(((T1[(i+1,) + pre_state] * trans_prob(pre_state, state), pre_state) for pre_state in prevoius_states.intersection(set(pre_states(state)))), key=operator.itemgetter(0))
            except:
                continue
            res = emission_prob(observation, state) * p_max
            if res > 0:
                s_index = (i + 2,) + state
                T1[s_index] = res
                T2[s_index] = pre_max
                assigned_states.add(state)

    #backwared computation
    X = np.empty(n_o, dtype=object)
    try:
        X[-1] = np.unravel_index(T1[-1].argmax(), T1[-1].shape)
        for i in reversed(range(1, n_o)):
            X[i - 1] = T2[(i,) + X[i]]
    except:
        return None #in case of failour
    return [x[1] for x in X]

