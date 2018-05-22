"""
COM4513 CWI Shared Task Assignment.

Rob Ede
aca14re
March 2018

Based on my Lab 4
"""

# python imports
import csv
import re
from argparse import ArgumentParser
from os import listdir, environ
from os.path import join
from random import seed, random

# anaconda imports
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# local imports
from utils.helpers import progress, debug
from utils.scorer import report_score
from utils.perceptron import Perceptron
from utils.scrabble_score import scrabble_score
from utils.common_words import english as common_english

# set seeds for reproducability
seed(4)
np.random.seed(4)


# === configure command line arguments === #

parser = ArgumentParser(
    description='Rob Ede - Lab 4: Perceptron Based Sentiment Analysis')

# parser.add_argument('path', help='Directory of segmented review documents')
parser.add_argument('-x', '--train', type=int,
                    dest='num_train', default=800,
                    help='Use NUM training samples', metavar='NUM')
parser.add_argument('-y', '--test', type=int,
                    dest='num_test', default=200,
                    help='Use NUM testing samples', metavar='NUM')
parser.add_argument('-e', '--epochs', type=int,
                    dest='epochs', default=1,
                    help='Maximum epochs for training', metavar='NUM')
parser.add_argument('-s', '--shuffle',
                    action='store_true', dest='shuffle', default=False,
                    help='Turn on array shuffling')
parser.add_argument('-a', '--average',
                    action='store_true', dest='avg', default=False,
                    help='Test with average vector instead of last vector')
parser.add_argument('-p', '--top',
                    action='store_true', dest='top', default=False,
                    help='Print the top ten features for each class')
parser.add_argument('-g', '--graph',
                    action='store_true', dest='graph', default=False,
                    help='Graph the learning progress')
parser.add_argument('-v', '--verbose',
                    action='store_true', dest='verbose', default=False,
                    help='Print progress information')
parser.add_argument('-vv', '--very-verbose',
                    action='store_true', dest='very_verbose', default=False,
                    help='Print debug information')

opts = parser.parse_args()

if opts.verbose:
    environ['DEBUG_LEVEL'] = "1"

if opts.very_verbose:
    environ['DEBUG_LEVEL'] = "2"

# === regular expressions === #

# matches non-word characters
re_wsep = re.compile(r'[^\w\']+')

# matches line break
re_lsep = re.compile(r'\t+')

# matches tab separators
re_fsep = re.compile(r'\t+')


# === data structures === #

tagdict = nltk.load('help/tagsets/upenn_tagset.pickle')
tags = list(tagdict.keys())


def read_data(filename):
    f = open(filename, encoding='utf-8', newline='')

    fns = ('id', 'snt', 'soff', 'eoff', 'word', 'nsee', 'fsee', 'nmark', 'fmark', 'gsbin', 'gsprob')
    rows = csv.DictReader(f, dialect='excel-tab', fieldnames=fns)

    data = []

    for row in rows:
        row['soff'] = int(row['soff'])
        row['eoff'] = int(row['eoff'])
        row['nsee'] = int(row['nsee'])
        row['fsee'] = int(row['fsee'])
        row['nmark'] = int(row['nmark'])
        row['fmark'] = int(row['fmark'])
        row['gsbin'] = int(row['gsbin'])
        row['gsprob'] = float(row['gsprob'])
        data.append(dict(row))

    return data


def build_dataset(data):
    all = []

    for sample in data:
        is_complex = 1 if int(sample['nmark']) + int(sample['fmark']) > 0 else -1

        features = {
            'pos': nltk.pos_tag(nltk.word_tokenize(sample['word']))[0][1],
            'len': len(sample['word']),
            'scrabble': scrabble_score(sample['word']),
            'common': int(sample['word'] in common_english)
        }

        pos_1_hot = []
        for i in range(len(tags)):
            if features['pos'] == tags[i]:
                pos_1_hot.append(1)
            else:
                pos_1_hot.append(0)

        feature_vec = np.array(
            pos_1_hot +
            [features['len']] +
            [features['common']] +
            [features['scrabble']]
        )

        all.append((feature_vec, is_complex, sample))

    return all


# === training === #

progress('reading training data')

data = read_data('./datasets/english/English_Train.tsv')
all = build_dataset(data)


progress('training perceptron')

per = Perceptron(eta=0.01, epochs=opts.epochs,
                 avg=opts.avg, shuffle=opts.shuffle)
per.train(all)

progress('perceptron trained in', per.convergance_epochs(),
         '/', opts.epochs, '(max) epochs')

if opts.graph:
    per.plot_training_error()


progress('reading test data')

data = read_data('./datasets/english/News_Test.tsv')
all = build_dataset(data)


progress('testing')

gold = [x[2]['gsbin'] for x in all]
pred = [max(0, per.predict(x[0])) for x in all]


progress('evaluating')

report_score(gold, pred, detailed=True)
