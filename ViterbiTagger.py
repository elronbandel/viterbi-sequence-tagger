from collections import Counter, deque, namedtuple

from MLE import MLE, read_counter_from_file
from old_code import data_tools
from HMMGraph import *
import numpy as np
from pruned_viterbi_algorithm import viterbi_algorithm as viterbi

Record = namedtuple('Record', ['p','node', 'source'])

class ViterbiTagger:
    def __init__(self, mle, n_gram = 3):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1


    def trans_prob(self, state0, state1):
        return self.mle.getQi(state0[0], state1[0], state1[1])

    def emission_prob(self, word, state):
        return self.mle.getEi(word, state[-1])

    def observation_states(self, observation):
        states = self.mle.wordidx2states[observation]
        return states

    def pre_states(self, state):
        return self.mle.pre_states_dict[state]

    def tag(self, line):
        observations = [self.mle.word2idx[self.mle.symbolizer(word)] for word in line]
        states_range = self.mle.n_tags
        state_dim = 2
        possible_starts = self.mle.possible_starts.items()


        result = viterbi(observations, states_range, state_dim, possible_starts, self.pre_states, self.observation_states , self.trans_prob, self.emission_prob)
        tags = [self.mle.tags[state[1]] for state in result]
        return tags

def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    from timeit import default_timer as timer
    start = timer()
    line = "Their mission is to keep clients from fleeing the market , as individual investors did in droves after the crash in October".split()
    tagger = ViterbiTagger(mle)
    print((timer() - start) * 1000000000)
    print(tagger.tag(line))




if __name__ == '__main__':
    test()