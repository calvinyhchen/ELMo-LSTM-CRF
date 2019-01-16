from __future__ import print_function

import model.utils as utils

import argparse
import codecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--train_file', default='./data/CASE-REPORTS-IOBES/train.iobes', help='path to training file')
    parser.add_argument('--dev_file', default='./data/CASE-REPORTS-IOBES/devel.iobes', help='path to development file')
    parser.add_argument('--test_file', default='./data/CASE-REPORTS-IOBES/test.iobes', help='path to test file')
    parser.add_argument('--pre_path', default='./pretrain_elmo/case_report/', help='path to preprocessed file')
    args = parser.parse_args()

    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    train_features, train_labels, _, _, _ = utils.generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=5, if_shrink_w_feature=False)
    dev_features, dev_labels = utils.read_corpus(dev_lines)
    test_features, test_labels = utils.read_corpus(test_lines)

    with open(args.pre_path + "train.txt", "w") as train_file:
        print(len(train_features))
        for train_feature in train_features:
            sentence = ""
            for w in train_feature:
                sentence += w + " "
            train_file.write(sentence.rstrip() + "\n")

    with open(args.pre_path + "dev.txt", "w") as dev_file:
        print(len(dev_features))
        for dev_feature in dev_features:
            sentence = ""
            for w in dev_feature:
                sentence += w + " "
            dev_file.write(sentence.rstrip() + "\n")

    with open(args.pre_path + "test.txt", "w") as test_file:
        print(len(test_features))
        for test_feature in test_features:
            sentence = ""
            for w in test_feature:
                sentence += w + " "
            test_file.write(sentence.rstrip() + "\n")