from MLE import MLE, read_counter_from_file
from old_code.HMMGraph import *
from super_pruned_viterbi_algorithm import viterbi_algorithm as viterbi
from GreedyTagger import *

Record = namedtuple('Record', ['p','node', 'source'])

class ViterbiTagger:
    def __init__(self, mle, n_gram = 3):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1
        self.emergency_tagger = GreedyTagger(mle, n_gram)


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
        dim2_possible_starts = self.mle.dim2_possible_starts


        result = viterbi(observations, states_range, state_dim, dim2_possible_starts, self.pre_states, self.observation_states , self.trans_prob, self.emission_prob)
        if result:
            return [self.mle.tags[tag] for tag in result]
        else:
            return self.emergency_tagger.tag(line)


def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    from timeit import default_timer as timer
    start = timer()
    line = "It would like to peg the ceiling on Federal Housing Administration mortgage guarantees to 95 % of the median price in a particular market , instead of limiting it to $ 101,250 ; reduce ( or even eliminate ) FHA down-payment requirements and increase the availability of variable-rate mortgages ; expand the Veterans Affairs Department loan guarantee program ; provide `` adequate '' funding for the Farmers Home Administration ( FmHA ) ; increase federal funding and tax incentives for the construction of low-income and rental housing , including $ 4 billion in block grants to states and localities ; and `` fully fund '' the McKinney Act , a $ 656 million potpourri for the homeless .".split()
    tagger = ViterbiTagger(mle)
    print((timer() - start) * 1000000000)
    print(tagger.tag(line))




if __name__ == '__main__':
    test()