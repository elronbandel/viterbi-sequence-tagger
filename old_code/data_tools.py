import re
from collections import namedtuple, Counter

UNK_SYM = "*UNK*"
NUM_SYM = "*NUM*"
START_SYM = "*START*"
WordTag = namedtuple('WordTag', ['word', 'tag'])

"""
" loading the tagged sentences into list of WordTags lists
"""
def load_tagged_data(file):
    #in case there is more than 2 cells it means there was more than one '/' in the original token
    word_tag = lambda token : WordTag('/'.join(token[:-1]), token[-1])
    with open(file, 'r') as f:
        return [[word_tag(token.split('/')) for token in line.split()] for line in f]


def save_tagged_data(data, file):
    with open(file, 'w+') as f:
        for line in data:
            f.write(' '.join(['/'.join([word, tag]) for word, tag in line]) + '\n')

def load_untagged_data(file, n_gram=3):
    with open(file, 'r') as f:
        return [add_start_symbol_to_list_of_words(line.split(), n_gram) for line in f]

def str_to_wordtag(str):
    word, tag = str.split()
    return WordTag(word, tag)

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

def count_tags(data):
    counter = Counter()
    for sentence in data:
        for tag in tags(sentence):
            counter[tag] += 1
    return counter


#####################################################################################
## pre-processing
#####################################################################################
def preprocess(data):
    data = apply_symbolizer(data, get_symbolizer())
    data = add_start_symbol_to_data(data)
    vocab = build_vocab(data)
    return maniplulate_data_to_fit_vocab(data, vocab)

def is_num(string):
    regex = re.compile('^(([1-9]+[0-9]*)|([0])|([1-9]([0-9]?)([0-9]?)(([,][0-9]{3})*)))([.][0-9]*)?$')
    return bool(regex.match(string))


def get_symbolizer(vocab=None):
    def symbolizer(word):
        if is_num(word):
            return NUM_SYM
        if vocab:
            if word not in vocab:
                return UNK_SYM
        return word
    return symbolizer


def apply_symbolizer(symbolizer, data):
    def generate(word_tag):
        return WordTag(symbolizer(word_tag.word), word_tag.tag)
    return [[generate(word_tag) for word_tag in line] for line in data]

def add_start_symbol_to_line_of_word_tags(line, n_gram=3):
    t = n_gram - 1
    return ([WordTag(START_SYM, START_SYM)] * t) + line

def add_start_symbol_to_list_of_words(line, n_gram):
    t = n_gram - 1
    return ([START_SYM] * t) + line

def add_start_symbol_to_data(data, n_gram=3):
    t = n_gram - 1
    return [add_start_symbol_to_line_of_word_tags(line, n_gram) for line in data]


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
    data = load_tagged_data("data/ass1-tagger-train")
    data = preprocess(data)
    for sentence in data:
        print(tags(sentence))
        print(words(sentence))






if __name__ == '__main__':
    test()