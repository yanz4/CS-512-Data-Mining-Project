import os
import sys
import time
import json
import argparse
import warnings
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

import networkx as nx
from community import community_louvain
from datasketch import WeightedMinHashGenerator

import torch
import numpy as np

from datareader import read_data, Logger, evaluate


def merge_ranklists(ranklists, seeds):
    if type(ranklists[0]) in [int, np.int64]:
        result = []
        for i, eid in enumerate(ranklists):
            if eid not in seeds:
                result.append((eid, 1.0 / (i + 1)))
        result = result[:100]
        # assert len(result) >= 50
        return result
    score = defaultdict(float)
    seeds = set(seeds)
    for ranklist in ranklists:
        for i, eid in enumerate(ranklist):
            if eid not in seeds:
                score[eid] += 1.0 / (i + 1)
    result = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:100]
    # print(len(result), ranklists)
    # assert len(result) >= 50
    return result

def get_good_sgs(seedEids, data, query_id, print_info, size1=5, size2=50):
    start = time.time()
    sgs = set()  # sgs contains good skipgrams (with 5-50 neighbors)
    if data.file == 'sample_dataset':
        size1 = 1

    # sum of (good skipgrams for each seed).
    for eid in seedEids:  # good skipgrams for all seeds. (Assume all seeds have identical and non-conflict facets)
        for sg in data.skipgramsByEidMap[eid]:
            size = len(data.eidsBySkipgramMap[sg])
            if (size1 < size < size2) and (sg not in sgs):
                sgs.add(sg)
                # print sg, size

    # # good skipgrams that satisfied for all seeds. (This does NOT work well, we will get very few sgs then.)
    # sgs_candidate = skipgramsByEidMap[seedEids[0]]
    # for i in range(1, len(seedEids)):
    #     sgs_candidate = sgs_candidate.intersection(skipgramsByEidMap[seedEids[i]])
    # for sg in sgs_candidate:
    #     size = len(eidsBySkipgramMap[sg])
    #     if size1 < size < size2:
    #         sgs.add(sg)
    #         print sg, size

    # print list(sgs)
    end = time.time()
    print('[utils.py] Done getting good skipgrams using %.1f seconds' % (end - start))
    print_info[query_id] += ('[utils.py] Done getting good skipgrams using %.1f seconds\n' % (end - start))
    return list(sgs)

def get_candidate_eids(sgs, data, query_id, print_info):
    # candidate_eids have at least one common sgs (good skipgrams) with seeds.
    start = time.time()
    candidate_eids = set()
    for sg in sgs:
        for eid in data.eidsBySkipgramMap[sg]:
            # strongestType = get_strongestType(eid, typesByEidMap, weightByEidAndTypeMap)
            # if strongestType == coreType:  # ??????????????????????????????
            candidate_eids.add(eid)
    # candidate_eids |= set(seedEids)  # In case seedEids are not inside the candidate_eids.
    end = time.time()
    print('[utils.py] Done finding candidate eids using %.1f seconds' % (end - start))
    print_info[query_id] += ('[utils.py] Done finding candidate eids using %.1f seconds\n' % (end - start))
    return candidate_eids

def expand_single_query_egoset(seedEids_and_gt_result):
    def check_seedEids_in_candidate_eids(seedEids, candidate_eids, query_id, print_info):
        diff = set(seedEids) - candidate_eids
        if diff:
            for i in diff: seedEids.remove(i)
            print('Seed ' + str(diff) + ' is not in candidate_eids and is deleted from the seed list.')
            print_info[query_id] += (
                    'Seed ' + str(diff) + ' is not in candidate_eids and is deleted from the seed list.\n')
            warnings.warn('Seed ' + str(diff) + ' is not in candidate_eids and is deleted from the seed list.')
        return seedEids

    def create_idxByXXXMap(XXX):
        idxByXXXMap = {}
        ct = 0
        for X in list(XXX):
            idxByXXXMap[X] = ct
            ct += 1
        return idxByXXXMap

    def generate_word_vectors(candidate_eids, sgs, idxByCandEidMap, idxBySgMap, data, query_id, print_info):
        # word_vectors has the weight for each skipgram and each candidate word using Equation (2). Size(word_vectors) = No. of candidate words X No. of sgs.
        start = time.time()
        word_vectors = np.zeros((len(candidate_eids), len(sgs)))
        for eid in candidate_eids:
            for sg in sgs:
                if sg in data.skipgramsByEidMap[eid]:
                    word_vectors[idxByCandEidMap[eid]][idxBySgMap[sg]] = data.weightByEidAndSkipgramMap[(eid, sg)]
        end = time.time()
        print('[utils.py] Done generating word_vectors using %.1f seconds' % (end - start))
        print_info[query_id] += ('[utils.py] Done generating word_vectors using %.1f seconds\n' % (end - start))
        return word_vectors

    def fast_cosine_similarity(X, Y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Use', device)
        target_embed = torch.FloatTensor(X).to(device)
        embed_matrix_array = torch.FloatTensor(Y).to(device)
        a_norm = target_embed / target_embed.norm(dim=1)[:, None]
        b_norm = embed_matrix_array / embed_matrix_array.norm(dim=1)[:, None]
        res = a_norm.matmul(b_norm.transpose(0, 1))
        # res = torch.ones_like(res) - res
        dist_matrix = res.cpu().detach().numpy()
        del target_embed, embed_matrix_array, a_norm, b_norm, res
        return dist_matrix

    def get_distToSeedsByEid(candidate_eids, wm, seedEids, idxByCandEidMap, data, choice):
        distToSeedsByEid = {}
        candidate_eid_list = list(candidate_eids)
        if choice == 1:  # return 250 nearest Neighbors based on Jaccard Similarity with seeds.
            for i in range(len(candidate_eid_list)):
                similarity = 0.0
                for seed in seedEids:
                    try:
                        similarity += wm[i].jaccard(wm[idxByCandEidMap[seed]])
                    except:
                        raise ValueError('seed \'' + data.eidToEntityMap[
                            seed] + '\' is not in candidate_eid_list! This means the user-input seeds does not have a good skipgram.')
                distToSeedsByEid[candidate_eid_list[i]] = similarity
        elif choice == 2:  # return 250 nearest Neighbors based on word embeddings
            # similarity_matrix = cosine_similarity([data.eidToEmbeddingMap[i] for i in candidate_eid_list],
            #                                       [data.eidToEmbeddingMap[i] for i in seedEids])
            similarity_matrix = fast_cosine_similarity([data.eidToEmbeddingMap[i] for i in candidate_eid_list],
                                                       [data.eidToEmbeddingMap[i] for i in seedEids])
            # for i in range(len(candidate_eid_list)):
            #     for j in range(len(seedEids)):
            #         assert abs(similarity_matrix[i][j] - similarity_matrix2[i][j]) < 0.00001
            # print('================= Same')
            similarity_matrix = np.sum(similarity_matrix, axis=1)
            for i in range(len(similarity_matrix)):
                distToSeedsByEid[candidate_eid_list[i]] = similarity_matrix[i]
        return distToSeedsByEid

    def get_250nearestNeighbors(seedEids, candidate_eids, word_vectors, idxByCandEidMap, data, print_info, query_id,
                                choice, wmg=None, wm=None):
        # choice == 1: return 250 nearest Neighbors based on Jaccard Similarity with seeds.
        # choice == 2: return 250 nearest Neighbors based on word embeddings.
        start = time.time()
        if wmg is None or wm is None:
            wmg = WeightedMinHashGenerator(word_vectors.shape[1], sample_size=300)
            wm = list()  # Calculating wm takes time!!!
            for i in range(len(candidate_eids)):
                wm.append(wmg.minhash(word_vectors[i]))
        distToSeedsByEid = get_distToSeedsByEid(candidate_eids, wm, seedEids, idxByCandEidMap, data, choice)
        nearestNeighbors = []
        ct = 0
        for eid in sorted(distToSeedsByEid, key=distToSeedsByEid.__getitem__, reverse=True):
            if ct >= 250:
                break
            if eid not in seedEids:
                nearestNeighbors.append(eid)
                ct += 1
        assert ct >= 250
        # print 'Nearest Neighbors are: '
        # for i in nearestNeighbors:
        #     print eidToEntityMap[i],
        # print ' '
        end = time.time()
        print('[utils.py] Done finding 250 nearest neighbors using %.1f seconds' % (end - start))
        print_info[query_id] += ('[utils.py] Done finding 250 nearest neighbors using %.1f seconds\n' % (end - start))
        return nearestNeighbors, wmg, wm

    def ego_network_construction(nearestNeighbors, word_vectors, idxByCandEidMap, seedEids, wmg, print_info, query_id):
        start = time.time()
        idxByNNEidMap = create_idxByXXXMap(nearestNeighbors)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nearestNeighbors)))
        wm = list()
        for eid in nearestNeighbors:
            wm.append(wmg.minhash(word_vectors[idxByCandEidMap[eid]]))
        # The following loop is slow, because of the jaccard function.
        for i in range(len(nearestNeighbors)):
            for j in range(i + 1, len(nearestNeighbors)):
                eid1 = nearestNeighbors[i]
                eid2 = nearestNeighbors[j]
                if eid2 != eid1 and eid1 not in seedEids and eid2 not in seedEids:
                    # graph.add_weighted_edges_from([(idxByNNEidMap[eid1], idxByNNEidMap[eid2], wm[i].jaccard(wm[j]))])
                    if wm[i].jaccard(wm[j]) > 0.05:
                        graph.add_edge(idxByNNEidMap[eid1], idxByNNEidMap[eid2])
        partition = community_louvain.best_partition(graph)
        end = time.time()
        print('[utils.py] Done Ego-network Construction & Detection using time %.1f seconds' % (end - start))
        print_info[query_id] += (
                '[utils.py] Done Ego-network Construction & Detection using time %.1f seconds\n' % (end - start))
        return partition

    def post_processing(partition, nearestNeighbors, seedEids, wm, idxByCandEidMap, print_info, query_id):
        start = time.time()
        members_list = []  # size(members_list) might not be the same as size(partition.values())

        for i in set(partition.values()):
            # print('----------------------------------------------')
            # print ('Community', i)
            members = [nearestNeighbors[nodes] for nodes in partition.keys() if partition[nodes] == i]
            if len(members) < 5:
                continue
            '''1. Lack of Sec3.4 Fusing Ontologies (Wiki List)'''
            '''2. Lack of Sec3.5 first issue -- duplicate clusters'''
            '''3. Lack of Sec3.5 second issue -- off-topic cluster'''
            members += list(seedEids)
            # Sort the members
            similarity = [0] * len(members)
            for j in range(len(members)):
                for k in seedEids:
                    similarity[j] += wm[idxByCandEidMap[members[j]]].jaccard(wm[idxByCandEidMap[k]])
            members_sorted = [x for _, x in sorted(zip(similarity, members), reverse=True)]
            # print (members_sorted)
            members_list.append(members_sorted)
            # '''There is a bug in the following code at the remove part. The remove function removes elements from the list and thus make index for the new list wrong.'''
            # '''And I don't know what these lines of code is doing.'''
            # # pruning
            # for eid1 in members:
            #     d = 0.0
            #     wm1 = wmg.minhash(word_vectors[idxByCandEidMap[eid1]])
            #     for eid2 in members:
            #         wm2 = wmg.minhash(word_vectors[idxByCandEidMap[eid2]])
            #         d += wm1.jaccard(wm2)
            #     if d / len(nearestNeighbors) < 0.5:
            #         members.remove(eid1)
            # print (members)
            # members_name = []
            # for j in members:
            #     members_name.append(eidToEntityMap[j])
            # print members_name
        end = time.time()
        print('[evaluation.py] Done Post-Processing using time %.1f seconds' % (end - start))
        print_info[query_id] += ('[evaluation.py] Done Post-Processing using time %.1f seconds\n' % (end - start))
        return members_list

    def avoid_unkown_error(candidate_eids, word_vectors):
        wmg = WeightedMinHashGenerator(word_vectors.shape[1], sample_size=300)
        wm = list()  # Calculating wm takes time!!!
        __candidate_eids = []
        for i, eid in tqdm(zip(range(len(candidate_eids)), candidate_eids), total=len(candidate_eids)):
            try:
                wm.append(wmg.minhash(word_vectors[i]))
                __candidate_eids.append(eid)
            except ValueError as e:
                pass
        return set(__candidate_eids), wmg, wm

    global print_info
    query_id, seedEids = seedEids_and_gt_result
    seedEids, flag = data.print_seeds(seedEids, query_id, print_info)
    # if flag:
    #     return print_info[query_id]
    if not seedEids:
        return print_info[query_id]
    # '''Get the most significant type feature in that query. '''
    # coreType = get_coreType(seedEids, typesByEidMap, weightByEidAndTypeMap) # The most significant type feature in that Query.
    '''1. Get good skipgrams -- Sec3.1'''
    sgs = get_good_sgs(seedEids, data, query_id, print_info, size1=5, size2=50)  # sgs contains good skipgrams (with 5-50 neighbors).
    if sgs == []:
        print('There is no sgs in this query!!!')
        print_info[query_id] += 'There is no sgs in this query!!!\n'
        return print_info[query_id]

    '''Get candidate eids and word_vectors -- Sec3.1'''
    candidate_eids = get_candidate_eids(sgs, data, query_id, print_info)  # candidate_eids have at least one common sgs (good skipgrams) with seeds.
    seedEids = check_seedEids_in_candidate_eids(seedEids, candidate_eids, query_id, print_info)
    idxBySgMap = create_idxByXXXMap(sgs)
    idxByCandEidMap = create_idxByXXXMap(candidate_eids)
    word_vectors = generate_word_vectors(candidate_eids, sgs, idxByCandEidMap, idxBySgMap, data, query_id, print_info)
    ### minhash may crash, I don't know why, try to remove those candidates
    new_candidate, wmg, wm = avoid_unkown_error(candidate_eids, word_vectors)
    if len(new_candidate) < len(candidate_eids):
        candidate_eids = new_candidate
        seedEids = check_seedEids_in_candidate_eids(seedEids, candidate_eids, query_id, print_info)
        idxBySgMap = create_idxByXXXMap(sgs)
        idxByCandEidMap = create_idxByXXXMap(candidate_eids)
        word_vectors = generate_word_vectors(candidate_eids, sgs, idxByCandEidMap, idxBySgMap, data, query_id, print_info)

    '''Find 250 nearest neighbors -- Sec3.2'''
    nearestNeighbors, wmg, wm = get_250nearestNeighbors(seedEids, candidate_eids, word_vectors,idxByCandEidMap,
                                                        data, print_info, query_id, choice=2, wmg=wmg, wm=wm)
    '''Ego-network Construction & Detection -- Sec3.2, Sec3.3'''
    partition = ego_network_construction(nearestNeighbors, word_vectors, idxByCandEidMap, seedEids, wmg, print_info, query_id)
    '''Post-processing -- Sec3.5'''
    members_list = post_processing(partition, nearestNeighbors, seedEids, wm, idxByCandEidMap, print_info, query_id)

    return members_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main.py', description='')
    parser.add_argument('-evaluation_class', type=str, default='groups/nyt/class_state.txt', required=False, help='Evaluation synonym set')
    parser.add_argument('-evaluation_query', type=str, default='groups/nyt/query_state.txt', required=False, help='Evaluation synonym set')
    parser.add_argument('-data', required=False, default='NYT', help='wiki, nyt or pubmed')
    parser.add_argument('-threads', required=False, default=1)
    parser.add_argument('-filename', required=False, default='baseline_egoset_nyt_state.txt', help='Filename to write')
    args = parser.parse_args()

    start_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if not os.path.isdir('baseline_log'):
        os.mkdir('baseline_log')
    sys.stdout = Logger("baseline_log/" + start_time_str + '_' + args.filename)

    threads = int(args.threads)  # Global Variable
    filename = args.filename  # Global Variable
    data = read_data(args.data, args.evaluation_class, args.evaluation_query).loadall()  # Global Variable -- remains unchanged inside functions.
    print_info = ['\n']*len(data.queries)  # Global Variable

    start = time.time()
    if threads == 1:
        actual_print = []
        for seedEids in tqdm(zip(range(len(data.queries)), data.queries), total=len(data.queries)):
            actual_print.append(expand_single_query_egoset(seedEids))

    else:  # Parallel Processing
        pool = Pool(processes=threads, maxtasksperchild=1)
        actual_print = pool.map(expand_single_query_egoset, zip(range(len(data.queries)), data.queries))

        pool.close()
        pool.join()

    print("**********************************************************************")
    print(actual_print)
    end = time.time()
    print('[baseline.py] Done all using %.1f seconds' % (end - start))
    if not os.path.isdir('results_baseline'):
        os.mkdir('results_baseline')
    with open('results_baseline/' + args.filename.replace('.txt', '_' + start_time_str + '.json'), 'w') as fout:
        record = []
        for seed, ranklists, gt in zip(data.queries, actual_print, data.gt_results):
            print(seed)
            result = [{'entity' : data.eidToEntityMap[i], 'score' : j} for (i, j) in merge_ranklists(ranklists, [data.EntityToeidMap[i] for i in seed])]
            eval_socres = evaluate([each['entity'] for each in result], gt)
            record.append({'groundtruth' : gt, 'seed' : seed, 'result' : result, 'eval_scores':eval_socres})
        json.dump({'epochs' : record, 'setting' : vars(args), 'time' : (end - start)}, fout)
