# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:58:12 2019

@author: msq96
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def evaluate_per_query(result, groundtruth, verbose=False):
    if len(result) == 0:
        return {'AveP': 0.0, 'F1': 0.0, 'Jaccard': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'PR_P': [0.0] * 20,
                'PR_R': [0.0] * 20}
    hit = miss = 0
    avep = 0.0
    ps = []
    rs = []
    for i in result:
        if i in groundtruth:
            hit += 1
        else:
            miss += 1
        p = hit / (hit + miss)
        r = hit / len(groundtruth)
        avep += p
        ps.append(p)
        rs.append(r)
    avep /= len(result)
    precision = hit / (hit + miss) if hit + miss > 0 else 0.0
    recall = hit / len(groundtruth)
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    jaccard = len(set(result) & set(groundtruth)) / (
            len(result) + len(groundtruth) - len(set(result) & set(groundtruth)))

    if verbose:
        print('Recall list =', rs)
        print('Precision list =', ps)
        print('Precision={0:.6f}  Recall={1:.6f}'.format(precision, recall))
        print('AveP={0:.6f}  F1={1:.6f}  Jaccard={2:.6f}'.format(avep, F1, jaccard))
    return {'AveP': avep, 'F1': F1, 'Jaccard': jaccard, 'Precision': precision, 'Recall': recall, 'PR_P': ps,
            'PR_R': rs}


def evaluate_corpus(dataname='NYT', json_dir='./results_baseline/', algo='word2vec'):
    mAPs = {'mAP10': [], 'mAP20': [], 'mAP50': [], 'mAP100': []}
    class_order = []
    for each_class in os.listdir(json_dir):
        if dataname in each_class:
            class_order.append(each_class)
            records = json.load(open(json_dir + each_class))['epochs']
            APs = {'AP10': [], 'AP20': [], 'AP50': [], 'AP100': []}
            for each_query in records:
                ps = each_query['eval_scores']['PR_P']
                APs['AP10'].append(np.mean(ps[:10]))
                APs['AP20'].append(np.mean(ps[:20]))
                APs['AP50'].append(np.mean(ps[:50]))
                APs['AP100'].append(np.mean(ps[:100]))
            for k, v in APs.items():
                mAPs['m' + k].append(np.mean(v))

    m_mAP = {}
    for k, v in mAPs.items():
        m_mAP['m_' + k] = np.mean(v)

    with open(algo + '_' + dataname + '-stats.txt', 'w') as f:
        f.write(str(m_mAP).replace(',', '\n'))
        f.write('\n-----------------------------------------------')
        f.write('-----------------------------------------------\n')
        f.write(str(mAPs).replace(',', '\n'))
        f.write('\n-----------------------------------------------')
        f.write('-----------------------------------------------\n')
        f.write(str(class_order).replace(',', '\n'))


def get_ranked_list(query, all_terms, all_embs, topn=100):
    query_emb = np.zeros((len(query), all_embs.shape[1]))
    fill_mean = np.mean(all_embs, axis=0)
    for idx, each_term in enumerate(query):
        # idx_of_term_embs could be empty if there exists UNK, so fill UNK using the mean of all extracted embeddings.
        idx_of_term_embs = np.argwhere(each_term + '||' == all_terms).reshape(-1)
        if idx_of_term_embs.shape[0] != 0:
            query_emb[idx] = np.mean(all_embs[idx_of_term_embs], axis=0)
        else:
            query_emb[idx] = fill_mean.copy()

    query_emb = np.mean(query_emb, axis=0, keepdims=True)

    scores = cosine_similarity(query_emb, all_embs).reshape(-1)
#    scores = np.stack(
#        [cosine_similarity(query_emb, each_emb.reshape(1, -1)).reshape(-1) for each_emb in all_embs]).reshape(-1)

    ranked_index = np.argsort(-scores)

    ranked_list = all_terms[ranked_index]
    ranked_scores = scores[ranked_index]

    idx = 0
    res = []
    unique_topn = []
    while len(unique_topn) < topn and idx < len(ranked_list):
        term = ranked_list[idx]
        if term not in unique_topn:
            unique_topn.append(term)
            res.append({'term': term, 'score': ranked_scores[idx]})
        idx += 1
    return res


def generate_term_embeddings(dataname, data_dir='../data/'):
    model = Word2Vec.load(data_dir + dataname + '/' + "word2vec.pth", mmap='r')
    vocab = open(data_dir + dataname + './vocab.txt', 'r', encoding='utf8').read().strip().split('\n')

    fill_mean = np.mean(model.wv[list(model.wv.vocab.keys())], axis=0)

    all_terms = np.array([each_term.split('\t')[0]+'||' for each_term in vocab])

    all_embs = []
    for each_term in all_terms:
        try:
            all_embs.append(model.wv[each_term])
        except KeyError:
            all_embs.append(fill_mean)
    all_embs = np.stack(all_embs)

#    word_vectors = model.wv
#    all_terms = list(word_vectors.vocab.keys())
#    all_embs = word_vectors[all_terms]
#    all_embs = np.array(all_embs)
#    all_terms = np.array(all_terms)
    return all_terms, all_embs


if __name__ in '__main__':
    data_dir = '../data/'
    dataname = 'Wiki'
    classes = data_dir + dataname + '/classes/'
    queries = data_dir + dataname + '/queries/'
    if not os.path.isdir('results_baseline'):
        os.mkdir('results_baseline')

    all_terms, all_embs = generate_term_embeddings(dataname=dataname, data_dir=data_dir)

    all_classes = [each.split('class')[1] for each in os.listdir(classes)]

    for each_class in all_classes:
        print('-----------------------------------------------------------')
        print('%s in %s evaluation starts!' % (each_class[1:], dataname))
        evaluation_class = classes + 'class' + each_class
        evaluation_query = queries + 'query' + each_class
        filename = 'word2vec%s_%s.json' % (each_class[:-4], dataname)

        '''if filename in os.listdir('results_baseline'):
            print('%s exists! Skip this class!')
            continue '''

        with open('results_baseline/' + filename, 'w') as fout:
            queries_for_class = open(evaluation_query, 'r', encoding='utf8').read().strip().split('\n')
            queries_for_class = [[each_term for each_syn_group in eval(each_query) for each_term in each_syn_group] for
                                 each_query in queries_for_class]

            gt_for_class = open(evaluation_class, 'r', encoding='utf8').read().strip().split('\n')
            gt_for_class = [each_term for each_syn_group in gt_for_class for each_term in eval(each_syn_group)]

            record = []
            for query in tqdm(queries_for_class):
                result = get_ranked_list(query, all_terms, all_embs, topn=100)
                # Note in class.txt, terms are like abc||id rather than abc||id||
                eval_socres = evaluate_per_query([each['term'][:-2] for each in result], gt_for_class)

                record.append(
                    {'groundtruth': gt_for_class, 'seed': query, 'result': result, 'eval_scores': eval_socres})
            json.dump({'epochs': record}, fout)

    evaluate_corpus(dataname=dataname)
