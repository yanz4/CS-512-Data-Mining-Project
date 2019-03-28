# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:29:35 2019

@author: msq96
"""


import os
import warnings

import gensim
from gensim.models import Word2Vec


def trim_rule(word, count, min_count):
    if '||' in word:
        return gensim.utils.RULE_KEEP
    else:
        return gensim.utils.RULE_DEFAULT

def word2vec(dataname, data_dir='../data/', num_thread=4):
    print('%s starts!'%dataname)
    corpus_file = data_dir + dataname + '/corpus.txt'
    eid2embed_file = data_dir + dataname + '/eid2embed.txt'
    vocab_file = data_dir + dataname + '/vocab.txt'
    model_file = data_dir + dataname + '/word2vec.pth'

    if os.path.isfile(model_file):
        print('Word2Vec model of %s corpus already exists!'%dataname)
        return None

    model = Word2Vec(corpus_file=corpus_file, size=100, sg=1, hs=0, negative=12, window=5, min_count=5, workers=num_thread,
                     alpha=0.025, min_alpha=0.025 * 0.0001, sample=1e-3, iter=5, trim_rule=trim_rule)
    model.save(model_file)

    word_vectors = model.wv

    f = open(eid2embed_file, 'w')
    with open(vocab_file, 'r') as fin:
        for line in fin:
            try:
                seg = line.strip('\r\n').split('\t')
                one_eid2embed = seg[1] + ' ' + ' '.join(word_vectors[seg[0]+'||'].astype(str)) + '\n'
                f.write(one_eid2embed)
            except KeyError:
                warnings.warn('Cannot find the embedding of "%s"!'%(seg[0]+'||'))
    f.close()

if __name__ == '__main__':
    corpus = ['NYT', 'Wiki']
    list(map(word2vec, corpus))
