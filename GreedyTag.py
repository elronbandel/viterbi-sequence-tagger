from DataTools import *
from TaggingMachine import TaggingMachine
from GreedyTagger import *

def main(argv):
    n_gram = 3
    #args for test: data/ass1-tagger-dev-input q.mle e.mle ass1-tagger-dev-output data/ass1-tagger-dev
    prog , input_file, qmle_file, emle_file, output_file, extra_file = argv
    q_counter = read_counter_from_file(qmle_file)
    e_counter = read_counter_from_file(emle_file)
    mle = MLE(q_counter, e_counter)
    dp = DataProcessor(n_gram=n_gram, add_start_symbols=False)
    gt = GreedyTagger(mle)
    tagger = TaggingMachine(gt)
    input = DataLoader.load_untagged_data(input_file)
    data = dp.preprocess_untagged(input)
    tagged = tagger.tag(data)
    output = dp.deprocess(tagged)
    DataLoader.save_tagged_data(output, output_file)




if __name__ == '__main__':
    from sys import argv
    main(argv)