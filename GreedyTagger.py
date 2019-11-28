from MLE import MLE, read_counter_from_file
import operator


#gosh

class GreedyTagger:
    def __init__(self, mle, n_gram = 3):
        self.mle = mle
        self.n_gram = n_gram
        self.n_gram_offset = n_gram - 1

    def tag(self, line):
        observations = [self.mle.word2idx[self.mle.symbolizer(word)] for word in line]
        return self.greedy(observations)

    def greedy(self, observations):
        tags = list()
        p_first, first_tag = max(
            ((p * self.mle.getEi(observations[0], tag), tag) for tag, p in self.mle.possible_starts.items()),
            key=operator.itemgetter(0))
        tags.append(first_tag)
        possible_nexts = list(filter(lambda state: state[0][0] == first_tag, self.mle.dim2_possible_starts))
        try:
            p, next_state = max((((p * self.mle.getEi(observations[1], state[0])), state) for state, p in possible_nexts),
                            key=operator.itemgetter(0))
        except:
            return [self.mle.tags[tag] for tag in tags]
        # p, next_state = max(((p_state * self.mle.getEi(observations[0], state[0]) * self.mle.getEi(observations[1], state[1]) ,state) for state, p_state in self.mle.dim2_possible_starts), key=operator.itemgetter(0))

        tags.append(next_state[1])
        for observation in observations[2:]:
            p, tag = max(
                (((self.mle.getEi(observation, tag) * self.mle.getQi(next_state[0], next_state[1], tag)), tag) for tag
                 in range(self.mle.n_tags)), key=operator.itemgetter(0))
            next_state = (next_state[1], tag)
            tags.append(next_state[1])

        return [self.mle.tags[tag] for tag in tags]




def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    line = "he walked home quickly".split()
    tagger = GreedyTagger(mle)
    print(tagger.tag(line))




if __name__ == '__main__':
    test()