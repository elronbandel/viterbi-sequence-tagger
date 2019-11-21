from collections import namedtuple, Counter

UNK_SYM = "*UNK*"
NUM_SYM = "*NUM*"
WordTag = namedtuple('WordTag', ['word', 'tag'])

"""
" loading the tagged sentences into list of WordTags lists
"""
def load(file):
    word_tag = lambda token : WordTag(token[0], token[1])
    with open(file, 'r') as f:
        return [[word_tag(token.split('/')) for token in line.split()] for line in f]



def words(sentence):
    return [token.word for token in sentence]


def tags(sentence):
    return [token.tag for token in sentence]


def count_words(data):
    counter = Counter()
    for sentence in data:
        for word in words(sentence):
            counter[word] += 1
    return counter


#####################################################################################
## pre-processing
#####################################################################################
def preprocess(data):
    data = replace_numbers_with_sym(data)
    vocab = build_vocab(data)
    return maniplulate_data_to_fit_vocab(data, vocab)

def replace_numbers_with_sym(data):
    def is_num_str(string):
        try:
            float(string)
        except:
            return False
        return True

    def filter(word_tag):
        if is_num_str(word_tag.word):
            return WordTag(NUM_SYM, word_tag.tag)
        return word_tag
    return [[filter(word_tag) for word_tag in line] for line in data]

def build_vocab(data, p=0.8):
    count = count_words(data)
    threshold = int(p * len(count))
    most_common = count.most_common()[:threshold]
    vocab = set(map(lambda pair: pair[0], most_common))
    return vocab

def maniplulate_data_to_fit_vocab(data, vocab):
    def filter(word_tag):
        if word_tag.word in vocab:
            return word_tag
        return WordTag(UNK_SYM, word_tag.tag)
    return [[filter(word_tag) for word_tag in line] for line in data]

def test():
    data = load("data/ass1-tagger-train")
    data = preprocess(data)
    for sentence in data:
        print(tags(sentence))
        print(words(sentence))






if __name__ == '__main__':
    test()