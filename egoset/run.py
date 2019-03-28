# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:09:19 2019

@author: msq96
"""

import os
import json
import subprocess
import numpy as np


def evaluate(dataname='NYT', json_dir='./results_baseline/'):

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

    with open(dataname+'-stats.txt', 'w') as f:
        f.write(str(m_mAP).replace(',', '\n'))
        f.write('\n-----------------------------------------------\n')
        f.write('\n-----------------------------------------------\n')
        f.write(str(mAPs).replace(',', '\n'))
        f.write('\n-----------------------------------------------\n')
        f.write('\n-----------------------------------------------\n')
        f.write(str(class_order).replace(',', '\n'))


def main(dataname, data_dir='../data/', threads='1'):

    classes = data_dir + dataname + '/classes/'
    queries = data_dir + dataname + '/queries/'

    all_classes = [each.split('class')[1] for each in os.listdir(classes)]

    for each_class in all_classes:
        print('%s in %s evaluation starts!'%(each_class, dataname))
        evaluation_class = classes + 'class' + each_class
        evaluation_query = queries + 'query' + each_class
        filename = 'egoset%s_%s.json'%(each_class[:-4], dataname)

        subprocess.check_call(['python', 'egoset.py', '-evaluation_class', evaluation_class, '-evaluation_query', evaluation_query,
                               '-data', dataname, '-threads', threads, '-filename', filename])
    evaluate(dataname=dataname)

if __name__ == '__main__':
    corpus = ['NYT', 'Wiki']
    list(map(main, corpus))
