from DataTools import *
from TaggingMachine import TaggingMachine
from ViterbiTagger import *

def main(argv):
    n_gram = 3
    #args for test: data/ass1-tagger-dev-input q.mle e.mle ass1-viterbi-dev-output data/ass1-tagger-dev
    prog , input_file, qmle_file, emle_file, output_file, extra_file = argv
    q_counter = read_counter_from_file(qmle_file)
    e_counter = read_counter_from_file(emle_file)
    mle = MLE(q_counter, e_counter)
    dp = DataProcessor(n_gram=n_gram, add_start_symbols=False)
    gt = ViterbiTagger(mle)
    tagger = TaggingMachine(gt)
    input = DataLoader.load_untagged_data(input_file)
    data = dp.preprocess_untagged(input)
    from timeit import default_timer as timer
    start = timer()
    tagged = tagger.tag(data)
    print(timer() - start)
    output = dp.deprocess(tagged)
    DataLoader.save_tagged_data(output, output_file)




if __name__ == '__main__':
    from sys import argv
    main(argv)