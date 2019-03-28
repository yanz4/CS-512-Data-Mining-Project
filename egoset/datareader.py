import sys
import time
import mmap
import warnings
from tqdm import tqdm
from collections import defaultdict

import numpy as np


def evaluate(result, groundtruth):
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
    print('Recall list =', rs)
    print('Precision list =', ps)
    print('Precision={0:.6f}  Recall={1:.6f}'.format(precision, recall))
    print('AveP={0:.6f}  F1={1:.6f}  Jaccard={2:.6f}'.format(avep, F1, jaccard))
    return {'AveP': avep, 'F1': F1, 'Jaccard': jaccard, 'Precision': precision, 'Recall': recall, 'PR_P': ps, 'PR_R': rs}


class Logger(object):
    def __init__(self, filename="aaa.txt", silent=False):
        self.silent = silent
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        if not self.silent:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        if not self.silent:
            self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


class read_data:
    def __init__(self, data, evaluation_class, evaluation_query):
        self.file = data
        self.evaluation_class = evaluation_class
        self.evaluation_query = evaluation_query
        self.eidToEntityMap = {}
        self.EntityToeidMap = {}
        self.skipgramsByEidMap = {}
        self.eidsBySkipgramMap = {}
        self.weightByEidAndSkipgramMap = {}
        self.eidToEmbeddingMap = {}
        self.wordToEmbeddingMap = {}
        self.queries = []
        self.gt_results = []

    def loadEidToEntityMap(self, filename):
        start = time.time()
        with open(filename, 'r') as fin:
            for line in fin:
                seg = line.strip('\r\n').split('\t')
                self.eidToEntityMap[int(seg[1])] = seg[0]
                self.EntityToeidMap[seg[0]] = int(seg[1])
        end = time.time()
        print('[readfile.py] Done loading \'%s\' using %.1f seconds' % (filename.split('/')[-1], end - start))

    def loadEidAndFeatureMap(self, folder):
        eidAndPattern2count = self.loadWeightByEidAndFeatureMap(folder + 'eidFSGCounts.txt')
        eid2patterns = defaultdict(set)
        pattern2eids = defaultdict(set)
        eid2count = defaultdict(int)
        pattern2count = defaultdict(int)
        for (eid, pattern), count in eidAndPattern2count.items():
            eid2patterns[eid].add(pattern)
            pattern2eids[pattern].add(eid)
            eid2count[eid] += count
            pattern2count[pattern] += count

        eidAndPattern2strength = dict()
        E = len(eid2patterns)
        for (eid, pattern), count in eidAndPattern2count.items():
            score_old = np.log(1 + count) * (np.log(E) - np.log(pattern2count[pattern]))
            # score_new = (1 + np.log(count)) * (np.log(1 + E) - np.log(1 + len(pattern2eids[pattern])) + 1)
            eidAndPattern2strength[(eid, pattern)] = score_old

        self.weightByEidAndSkipgramMap = eidAndPattern2strength
        self.skipgramsByEidMap = eid2patterns
        self.eidsBySkipgramMap = pattern2eids

    def loadEidToEmbeddingMap(self, filename):
        start = time.time()
        with open(filename, 'r') as fin:
            for line in fin:
                seg = line.strip('\r\n').split(' ')
                self.eidToEmbeddingMap[int(seg[0])] = np.array([float(i) for i in seg[1:]])
        end = time.time()
        print('[readfile.py] Done loading \'%s\' using %.1f seconds' % (filename.split('/')[-1], end - start))

    def loadQueriesandGroundTruthResults(self, evaluation_class, evaluation_query):
        group = []
        with open(evaluation_class, 'r') as file:
            for line in file:
                line = line.replace('"', "'")
                synset = line.strip("\r\n[]'").split("', '")
                group.extend(synset)
        with open(evaluation_query, 'r') as file:
            for line in file:
                if len(line) <= 3:
                    continue
                line = line.replace('"', "'")
                fin = line.strip('\r\n[]').split('], [')
                query = [j for i in fin for j in i.strip("'").split("', '")]
                self.queries.append(query)
                self.gt_results.append(group)

    def print_seeds(self, seedEids, query_id, print_info):
        print_info[query_id] += ('====================================== Query %d ======================================\n' % query_id)
        flag = 0
        if isinstance(seedEids[0], str):
            seedEids_temp = list()
            for i in seedEids:
                if i in self.EntityToeidMap.keys():  # Only in the vocalbulary list
                    seedEids_temp.append(self.EntityToeidMap[i])
                else:
                    flag = 1
                    print('Seed ' + i + ' is not in the \'entity2id.txt\'.')
                    print_info[query_id] += ('Seed ' + i + ' is not in the \'entity2id.txt\'.\n')
                    warnings.warn('Seed ' + i + ' is not in the \'entity2id.txt\'.')
            seedEids = seedEids_temp
            del seedEids_temp
        print('----------------------------------------------')
        print('----------------------------------------------')
        print('seeds:')
        for eid in seedEids:
            print(eid, self.eidToEntityMap[eid])
        print('----------------------------------------------')

        print_info[query_id] += '----------------------------------------------\n'
        print_info[query_id] += '----------------------------------------------\n'
        print_info[query_id] += 'seeds:\n'
        for eid in seedEids:
            print_info[query_id] += '((' + str(eid) + ', ' + self.eidToEntityMap[eid] + '))\n'
        print_info[query_id] += '----------------------------------------------\n'
        return seedEids, flag

    '''The rest are for SetExpan'''
    def get_num_lines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def loadWeightByEidAndFeatureMap(self, filename, idx=-1):
        """Load the (eid, feature) -> strength

        :param filename:
        :param idx: The index column of weight, default is the last column
        :return:
        """
        weightByEidAndFeatureMap = {}
        with open(filename, 'r', encoding='utf8') as fin:
            for line in tqdm(fin, total=self.get_num_lines(filename), desc="Loading: {}".format(filename)):
                seg = line.strip('\r\n').split('\t')
                eid = int(seg[0])
                feature = seg[1]
                weight = float(seg[idx])
                weightByEidAndFeatureMap[(eid, feature)] = weight
        return weightByEidAndFeatureMap

    def loadall(self):
        print(self.file)
        folder = '../data/' + self.file + '/'
        self.loadEidToEntityMap(folder + 'vocab.txt')
        self.loadEidToEmbeddingMap(folder + 'eid2embed.txt')
        self.loadEidAndFeatureMap(folder)
        self.loadQueriesandGroundTruthResults(self.evaluation_class, self.evaluation_query)

        return self
