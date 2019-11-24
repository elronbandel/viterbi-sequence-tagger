from MLETrain import MLE, read_counter_from_file
from old_code import data_tools


def generate_hmm_table(mle, sentence, n_gram):

    tags = mle.getTags()
    table = [None] * len(sentence)
    for i in range(n_gram - 1, len(sentence) - (n_gram - 1)):
        table[i] = dict()
    #genrate probabilities of transitions from start symbols
    START_SYM = sentence[0]
    i = n_gram - 1
    for tag in tags:
        tup = (START_SYM, ) * (n_gram - 2) + (tag, )
        table[i][tup] = mle.getE(word=sentence[i], tag=tag) * mle.getQ(tup)
    for i in range(n_gram, len(sentence) - (n_gram - 1)):
        for tup_before in table[i-1].keys():
            for tag in tags:
                transition = tup_before + (tag,)
                tup = tup_before[1:] + (tag,)
                table[i][tup] = mle.getE(word=sentence[i], tag=tag) * mle.getQ(transition)
    return table[n_gram - 1: -(n_gram - 1)]


def test():
    q_counter = read_counter_from_file("q.mle")
    e_counter = read_counter_from_file("e.mle")
    mle = MLE(q_counter, e_counter)
    line = "he walked home quickly".split()
    hmm = generate_hmm_table(mle, data_tools.add_start_symbol_to_list_of_words(line, n_gram=3), n_gram=3)
    import operator
    tags = []
    value = operator.itemgetter(1)
    prediction = max(hmm[0].items(), key=value)[0][-1]
    tags.append(prediction)
    for col in hmm[1:]:
        next_options = list(filter(lambda t: t[0][0] == prediction, col.items()))
        prediction = max(next_options, key=operator.itemgetter(1))[0][-1]

        tags.append(prediction)
    print(tags)
    print(line)





if __name__ == '__main__':
    test()