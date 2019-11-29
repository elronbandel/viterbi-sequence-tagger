from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import dump_svmlight_file


def convert(input_file, output_file, feature_map_file):
    tags_dicts, feat_dicts = [], []
    with open(input_file, 'r') as input:
        for line in input:
            tokens = line.split()
            tags_dicts.append({"tag" : tokens[0]})
            feat_dict = dict()
            for token in tokens[1:]:
                key, val = token.split(':', 1)
                feat_dict[key] = val
            feat_dicts.append(feat_dict)
    tvec, fvec = DictVectorizer(), DictVectorizer()
    tags_matrix = tvec.fit_transform(tags_dicts)
    feat_matrix = fvec.fit_transform(feat_dicts)
    dump_svmlight_file(feat_matrix, tags_matrix, output_file, multilabel=True)
    with open(feature_map_file, 'w+') as output:
        for i, feature in enumerate(fvec.get_feature_names()):
            output.write("{}\t{}\n".format(feature,i))
        for i, tag in enumerate(tvec.get_feature_names()):
            output.write("{}\t{}\n".format(tag, i))






