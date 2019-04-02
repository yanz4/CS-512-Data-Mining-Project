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
    jaccard = len(set(result) & set(groundtruth)) / (len(result) + len(groundtruth) - len(set(result) & set(groundtruth)))

    if verbose:
        print('Recall list =', rs)
        print('Precision list =', ps)
        print('Precision={0:.6f}  Recall={1:.6f}'.format(precision, recall))
        print('AveP={0:.6f}  F1={1:.6f}  Jaccard={2:.6f}'.format(avep, F1, jaccard))
    return {'AveP': avep, 'F1': F1, 'Jaccard': jaccard, 'Precision': precision, 'Recall': recall, 'PR_P': ps, 'PR_R': rs}


def evaluate_corpus(dataname='NYT', json_dir='./results_baseline/', algo='bert'):

    mAPs = {'mAP10':[], 'mAP20':[], 'mAP50':[], 'mAP100':[]}
    class_order = []
    for each_class in os.listdir(json_dir):
        if dataname in each_class:
            class_order.append(each_class)
            records = json.load(open(json_dir+each_class))['epochs']
            APs = {'AP10':[], 'AP20':[], 'AP50':[], 'AP100':[]}
            for each_query in records:
                ps = each_query['eval_scores']['PR_P']
                APs['AP10'].append(np.mean(ps[:10]))
                APs['AP20'].append(np.mean(ps[:20]))
                APs['AP50'].append(np.mean(ps[:50]))
                APs['AP100'].append(np.mean(ps[:100]))
            for k, v in APs.items():
                mAPs['m'+k].append(np.mean(v))

    m_mAP = {}
    for k, v in mAPs.items():
        m_mAP['m_'+k] = np.mean(v)

    with open(algo + '_' + dataname + '-stats.txt', 'w') as f:
        f.write(str(m_mAP).replace(',', '\n'))
        f.write('\n-----------------------------------------------')
        f.write('-----------------------------------------------\n')
        f.write(str(mAPs).replace(',', '\n'))
        f.write('\n-----------------------------------------------')
        f.write('-----------------------------------------------\n')
        f.write(str(class_order).replace(',', '\n'))


def get_pickle_info(pickle_filename, pool):
    num_batches = 0
    num_terms = 0
    with open(pickle_filename, 'rb') as f:
        while 1:
            try:
                bert_embs_batch = pickle.load(f)
                num_batches += 1
                for sent in bert_embs_batch.values():
                    num_terms += len(sent['sent_info'])
            except EOFError:
                break

    emb_dim = pool(sent['sent_info'][0]['embedding'], axis=1).reshape(-1).shape[0]

    return num_batches, num_terms, emb_dim


def generate_term_embeddings(dataname, data_dir='../data/', pooling_strategy='mean'):
    if pooling_strategy == 'mean':
        pool = np.mean
    elif pooling_strategy == 'max':
        pool = np.max


    pickle_filename = data_dir + dataname + '/bert_embeddings.pickle'

    print('Calculating the time needed...')
    num_batches, num_terms, emb_dim = get_pickle_info(pickle_filename, pool)

    fin = open(pickle_filename, 'rb')

    all_terms = []
    all_embs = np.zeros((num_terms, emb_dim), dtype=np.float16)
    idx = 0
    print('Generating term embeddings...')
    for i in tqdm(range(num_batches)):
        try:
            bert_embs_batch = pickle.load(fin)
            for sent in bert_embs_batch.values():
                sent_info = sent['sent_info']
                for term in sent_info:
                    emb = pool(term['embedding'], axis=1).reshape(-1)

                    all_terms.append(term['term'])
                    all_embs[idx] = emb
                    idx += 1
        except EOFError:
            warnings.warn('EOFError encoutered! Break loop now!')
            break

    fin.close()
    assert len(all_terms) == all_embs.shape[0]

    # Slow to save. Memory overflow.
#    with open(data_dir + dataname + '/bert_term_embeddings.pickle', 'wb') as fout:
#        pickle.dump([all_terms, all_embs], fout, protocol=4)

    return np.stack(all_terms), all_embs


def get_ranked_list(query, all_terms, all_embs, topn=100):
    avg_emb = np.zeros((len(query), all_embs.shape[1]))
    for idx, each_term in enumerate(query):
        # TODO Known bug: avg_emb[idx] could be empty if there exist <UNK>.
        avg_emb[idx] = np.mean(all_embs[np.argwhere(each_term + '||' == all_terms).reshape(-1)], axis=0)
    avg_emb = np.mean(avg_emb, axis=0, keepdims=True)

    # Memory overflow.
#    scores = cosine_similarity(avg_emb, all_embs).reshape(-1)
    # Naive implementation, slow but doesn't require much memory.
    scores = np.stack([cosine_similarity(avg_emb, each_emb.reshape(1, -1)).reshape(-1) for each_emb in all_embs]).reshape(-1)

    ranked_index = np.argsort(-scores)
    ranked_list = all_terms[ranked_index]
    ranked_scores = scores[ranked_index]

    idx = 0
    res = []
    unique_topn = []
    while len(unique_topn) < topn and idx < ranked_list.shape[0]:
        term = ranked_list[idx]
        if term not in unique_topn:
            unique_topn.append(term)
            res.append({'term': term, 'score': ranked_scores[idx]})
        idx += 1

    return res


if __name__ in '__main__':
    data_dir='../data/'
    dataname='Wiki'
    pooling_strategy = 'mean'
    classes = data_dir + dataname + '/classes/'
    queries = data_dir + dataname + '/queries/'
    if not os.path.isdir('results_baseline'):
        os.mkdir('results_baseline')

    all_terms, all_embs = generate_term_embeddings(dataname=dataname, data_dir=data_dir, pooling_strategy=pooling_strategy)

    all_classes = [each.split('class')[1] for each in os.listdir(classes)]

    for each_class in all_classes:
        print('-----------------------------------------------------------')
        print('%s in %s evaluation starts!'%(each_class[1:], dataname))
        evaluation_class = classes + 'class' + each_class
        evaluation_query = queries + 'query' + each_class
        filename = 'bert%s_%s.json'%(each_class[:-4], dataname)

        with open('results_baseline/' + filename, 'w') as fout:
            queries_for_class = open(evaluation_query, 'r', encoding='utf8').read().strip().split('\n')
            queries_for_class = [[each_term for each_syn_group in eval(each_query) for each_term in each_syn_group] for each_query in queries_for_class]

            gt_for_class = open(evaluation_class, 'r', encoding='utf8').read().strip().split('\n')
            gt_for_class = [each_term for each_syn_group in gt_for_class for each_term in eval(each_syn_group)]

            record = []
            for query in tqdm(queries_for_class):
#                print(query)
                result = get_ranked_list(query, all_terms, all_embs, topn=100)
                # Note in class.txt, terms are like abc||id rather than abc||id||
                eval_socres = evaluate_per_query([each['term'][:-2] for each in result], gt_for_class)

                record.append({'groundtruth' : gt_for_class, 'seed' : query, 'result' : result, 'eval_scores':eval_socres})
            json.dump({'epochs' : record}, fout)

    evaluate_corpus(dataname=dataname)
