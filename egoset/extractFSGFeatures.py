import os
import mmap
import itertools
from tqdm import tqdm
from collections import defaultdict


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def getSkipgrams(tokens, location):
    cleaned_tokens = []
    for tok in tokens:
        if tok == "\t":
            cleaned_tokens.append("TAB")
        else:
            cleaned_tokens.append(tok)
    positions = [(-1, 1), (-2, 1), (-1, 2), (-3, 1), (-1, 3), (-2, 2)]
    skipgrams = []
    for pos in positions:
        sg = ' '.join(cleaned_tokens[location+pos[0]:location]) + ' __ ' + ' '.join(cleaned_tokens[location+1:location+1+pos[1]])

        skipgrams.append(sg)
    return skipgrams


def processSentence(sent, entity2eid):
    def getEntityMentions(tokens):
        entityMentions = []
        for i in range(len(tokens)):
            if '||' in tokens[i]:
                entityId = entity2eid.get(tokens[i][:-2], None)
                if entityId is not None:
                    entityMentions.append({'entityId': entity2eid[tokens[i][:-2]], 'location': i})
        return entityMentions

    eidSkipgrams = {}
    eidPairs = []
    tokens = sent.split(' ')
    entityMentions = getEntityMentions(tokens)
    eids = set()
    for em in entityMentions:
        eid = em['entityId']

        location = em['location']
        eids.add(eid)

        for skipgram in getSkipgrams(tokens, location):
            key = (eid, skipgram)
            if key in eidSkipgrams:
                eidSkipgrams[key] += 1
            else:
                eidSkipgrams[key] = 1

    for pair in itertools.combinations(eids, 2):
        eidPairs.append(frozenset(pair))
    return eidSkipgrams, eidPairs


def updateMapFromMap(fromMap, toMap):
    for key in fromMap:
        if key in toMap:
            toMap[key] += fromMap[key]
        else:
            toMap[key] = fromMap[key]
    return toMap


def updateMapFromList(fromList, toMap):
    for ele in fromList:
        if ele in toMap:
            toMap[ele] += 1
        else:
            toMap[ele] = 1
    return toMap


def main(dataname):
    print('%s starts!'%dataname)
    infilename = '../../data/'+dataname+'/corpus.txt'
    fo = '../../data/'+dataname+'/eidFSGCounts.txt'

    if os.path.isfile(fo):
        print('FSGCounts file of %s corpus already exists!'%dataname)
        return None

    entity2eid = dict()
    with open('../../data/'+dataname+'/vocab.txt', 'r') as fin:
        for line in fin:
            entity2eid[line.strip().split('\t')[0]] = int(line.strip().split('\t')[1])

    eidSkipgramCounts = {}
    eidPairCounts = {}  # entity sentence-level co-occurrence features
    with open(infilename, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, total=get_num_lines(infilename), desc="Generating skipgram and sentence-level co-occurrence features for %s corpus"%dataname):
            eidSkipgrams, eidPairs = processSentence(line.strip(), entity2eid)
            updateMapFromMap(eidSkipgrams, eidSkipgramCounts)
            updateMapFromList(eidPairs, eidPairCounts)

    print("Number of (eid, skipgram) pairs: {}".format(len(eidSkipgramCounts)))

    eid2skipgram2count = defaultdict(lambda: defaultdict(int))
    skipgram2eid2count = defaultdict(lambda: defaultdict(int))

    for eidSkipgram in tqdm(eidSkipgramCounts):
        eid = eidSkipgram[0]
        skipgram = eidSkipgram[1]
        count = eidSkipgramCounts[eidSkipgram]
        eid2skipgram2count[eid][skipgram] = count
        skipgram2eid2count[skipgram][eid] = count

    # convert to dict
    skipgram2eid2count = dict(skipgram2eid2count)
    skipgram2eid2count = {skipgram: dict(skipgram2eid2count[skipgram]) for skipgram in skipgram2eid2count}

    # convert to dict
    eid2skipgram2count = dict(eid2skipgram2count)
    eid2skipgram2count = {eid: dict(eid2skipgram2count[eid]) for eid in eid2skipgram2count}


    min_ent_sup = 3
    candidateSkipgramPool = [skipgram for skipgram in skipgram2eid2count if len(skipgram2eid2count[skipgram]) >= min_ent_sup]
    print("Number of skipgrams passing min_set_sup={} in pool: {}".format(min_ent_sup, len(candidateSkipgramPool)))


    # currently matched entities
    vocab = set()
    for sg in tqdm(candidateSkipgramPool):
        vocab = vocab.union(set(skipgram2eid2count[sg].keys()))

    # for each entity that is not currently matched, add its corresponding skipgrams into the pool
    for eid in eid2skipgram2count:
        if eid not in vocab:
            candidateSkipgramPool += list(eid2skipgram2count[eid].keys())
    candidateSkipgramPool = list(set(candidateSkipgramPool))
    print("Number of final selected skipgrams in pool: {}".format(len(candidateSkipgramPool)))


    with open(fo, "w", encoding='utf8') as fout:
        for sg in candidateSkipgramPool:
            for eid in skipgram2eid2count[sg]:
                fout.write("{}\t{}\t{}\n".format(eid, sg, skipgram2eid2count[sg][eid]))


if __name__ == '__main__':
    corpus = ['NYT', 'Wiki']
    list(map(main, corpus))
