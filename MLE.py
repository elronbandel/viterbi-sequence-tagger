import numpy as np

from DataTools import DataProcessor, DataLoader
import DataTools
from collections import Counter

def read_counter_from_file(file):
    counter = Counter()
    with open(file) as f:
        for line in f:
            key, count = line.split('\t')
            counter[tuple(key.split())] = int(count)
    return counter


class MLE:
    def __init__(self, q_counter, e_counter):
        self.q_counter = q_counter
        self.e_counter = e_counter
        #collections to retrieve tags general information about tags:
        self.t_counter = Counter()
        self.tags = list()
        self.tag2idx = dict()
        self.init_tags_collections()
        #words information
        self.words = list()
        self.word2idx = dict()
        self.init_words_collections()
        self.e_matrix = np.zeros((len(self.words), self.n_tags))
        self.init_e_matrix()
        #generate symbolizer to symblized unkwon words
        self.symbolizer = self.generate__symbolizer()
        #possible states
        self.possible_states = set()
        self.init_possible_states()
        #q matrix
        self.q_matrix = np.zeros((self.n_tags, self.n_tags, self.n_tags))
        self.init_q_matrix()
        #s_matrix - the probability of tag to be in the begginning of sentence
        self.possible_starts = dict()
        self.dim2_possible_starts = list()
        self.init_possible_starts()
        #pre states = given state what are the states could be before it
        self.pre_states_dict = dict()
        self.init_pre_states_dict()
        #create array for the possible states of every word
        self.wordidx2states = np.empty((len(self.words),), dtype=list)
        self.init_wordidx2states()


    def init_words_collections(self):
        self.words = list(set((x[0] for x in self.e_counter.keys())))
        for i, word in enumerate(self.words):
            self.word2idx[word] = i

    def init_e_matrix(self):
        for word_tag, count in self.e_counter.items():
            word, tag = word_tag
            if word == DataTools.START_SYM:
                continue
            self.e_matrix[self.word2idx[word], self.tag2idx[tag]] = count / self.t_counter[tag]

    def init_tags_collections(self):
        self.tags = set()
        for word_tag, count in self.e_counter.items():
            self.t_counter[word_tag[1]] += count
            self.tags.add(word_tag[1])
        self.tags.remove(DataTools.START_SYM)
        self.tags = list(self.tags)
        self.n_tags = len(self.tags)
        for i, tag in enumerate(self.tags):
            self.tag2idx[tag] = i


    def generate__symbolizer(self):
        vocab = set(self.words)
        return DataTools.get_symbolizer(vocab)



    def init_possible_states(self):
        for i, tag_i in enumerate(self.tags):
            for j, tag_j in enumerate(self.tags):
                if (tag_i, tag_j) in self.q_counter:
                    self.possible_states.add((i, j))


    def init_q_matrix(self):
        n = len(self.tags)
        self.q_matrix = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                if (i, j) not in self.possible_states:
                    continue
                for k in range(n):
                    if (j,k) not in self.possible_states:
                        continue
                    t1, t2, t3 = self.tags[i], self.tags[j], self.tags[k]
                    a = self.q_counter[(t1, t2, t3)] / self.q_counter[(t1, t2)]
                    b = self.q_counter[(t2, t3)] / self.t_counter[t2]

                    self.q_matrix[i, j, k] = 0.7 * a + 0.2 * b + 0.1


    def init_possible_starts(self):
        start = DataTools.START_SYM
        s2p = self.q_counter[(start, start)]
        for i, tag in enumerate(self.tags):
            self.possible_starts[i] = self.q_counter[(start, start, tag)] / s2p
        self.dim2_possible_starts = dict()
        for i, p in self.possible_starts.items():
            for j in range(self.n_tags):
                try:
                    res = (self.q_counter[(start, self.tags[i], self.tags[j])] / self.q_counter[(start, self.tags[i])]) * p
                    if res > 0:
                        self.dim2_possible_starts[(i, j)] = res
                except:
                    break
        self.dim2_possible_starts = list(self.dim2_possible_starts.items())



    def init_pre_states_dict(self):
        for state in self.possible_states:
            self.pre_states_dict[state] = set()
            for pre_state in  self.possible_states:
                if pre_state[-1] == state[-2]:
                    self.pre_states_dict[state].add(pre_state)

    def init_wordidx2states(self):
        for word, tag in self.e_counter.keys():
            if tag == DataTools.START_SYM:
                continue
            tag_idx = self.tag2idx[tag]
            word_idx = self.word2idx[word]
            if self.wordidx2states[word_idx] is None:
                self.wordidx2states[word_idx] = list()
            for i in range(self.n_tags):
                state = (i, tag_idx)
                if state in self.possible_states:
                    self.wordidx2states[word_idx].append(state)

    def getTags(self):
        return self.tags

    def getQi(self, i, j, k):
        return self.q_matrix[i, j, k]

    def getQ(self, t1, t2, t3):
        self.getQi(self.tag2idx[t1], self.tag2idx[t2], self.tag2idx[t3])

    def getEi(self, word_idx, tag_idx):
        return self.e_matrix[word_idx, tag_idx]

    def getE(self, word, tag):
        word = self.symbolizer(word)
        return self.e_matrix[self.word2idx[word]][self.tag2idx[tag]]