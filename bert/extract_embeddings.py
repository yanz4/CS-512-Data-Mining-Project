# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:50:12 2019

@author: msq96
"""


from tqdm import tqdm
import logging
import time
import pickle
import numpy as np
import re
import mmap

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_sent_info(raw_line, intersection, sent_id, max_seq_length):

    sent_info = []
    for term in intersection:
        words_in_term = term.split('||')[0].split('_')
        words_in_sent = raw_line.split(' ')

        idx = 0
        term_len = len(words_in_term)
        while idx < len(words_in_sent) and idx < max_seq_length-2:
            if words_in_term == words_in_sent[idx: idx + term_len]:
                # loc is used to index a list, so note that the true location is idx to idx+term_len-1
                sent_info.append({'term': term, 'loc': [idx, idx+term_len]})
                idx += term_len
            else:
                idx += 1

    return sent_info


def tokenize_sent(sent_id, raw_sent, max_seq_length, tokenizer):

    tokens = tokenizer.tokenize(raw_sent)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)


    return {'unique_id': sent_id, 'tokens': tokens,
            'input_ids': input_ids, 'input_mask': input_mask}


def extract_embeddings(dataname, layer_indexes=[-1,-2,-3,-4], bert_model='bert-large-uncased', max_seq_length=128, batch_size=32, data_dir='../data/'):

    input_corpus = data_dir + dataname + '/corpus.txt'
    input_vocab = data_dir + dataname + '/vocab.txt'
    output_embedding = data_dir + dataname + '/bert_embeddings.pickle'
#    output_tokenized_corpus = data_dir + dataname + '/tokenized_corpus.pickle'

    reader = open(input_corpus, 'r', encoding='utf8')
    total_lines = get_num_lines(input_corpus)
    vocab = set([each.split('\t')[0]+'||' for each in open(input_vocab, 'r', encoding='utf8').read().split('\n')])


    tokenizer = BertTokenizer.from_pretrained(bert_model)

    model = BertModel.from_pretrained(bert_model)
    model.cuda()
    model.eval()

    batch_sentences = {}
    fout = open(output_embedding, 'wb')
    with torch.no_grad():
        for sent_id, line in enumerate(tqdm(reader, total=total_lines)):

            if len(batch_sentences) < batch_size and sent_id < total_lines-1:
                line = line.strip()
                terms = line.split(' ')
                intersection = set(terms).intersection(vocab)
                if intersection:
                    raw_line = re.sub("\|\|.+?\|\|", '', line).replace('_', ' ')
                    sent_info = get_sent_info(raw_line, intersection, sent_id, max_seq_length)
                    batch_sentences[sent_id] = {'raw_sent': raw_line, 'sent_info': sent_info}
            else:

                batch_tokenized_sents = [tokenize_sent(sent_id, values['raw_sent'], max_seq_length, tokenizer) for sent_id, values in batch_sentences.items()]

                batch_input_ids = torch.tensor([sent['input_ids'] for sent in batch_tokenized_sents], dtype=torch.long).cuda()
                batch_input_mask = torch.tensor([sent['input_mask'] for sent in batch_tokenized_sents], dtype=torch.long).cuda()


                all_encoder_layers, _ = model(batch_input_ids, attention_mask=batch_input_mask)


                ### performance bottleneck
                #s = time.time()
                emb_encoder_layers = torch.stack(all_encoder_layers)[layer_indexes]
                for idx, sent_id in enumerate(batch_sentences):
                    sent_info = batch_sentences[sent_id]['sent_info']
                    [term_info.__setitem__('embedding', emb_encoder_layers[:, idx, term_info['loc'][0]:term_info['loc'][1], :].detach().cpu().numpy().astype(np.float16))\
                     for term_info in sent_info]
                #print(time.time()-s)
                ### performance bottleneck


                pickle.dump(batch_sentences, fout)
                batch_sentences = {}

                line = line.strip()
                terms = line.split(' ')
                intersection = set(terms).intersection(vocab)
                if intersection:
                    raw_line = re.sub("\|\|.+?\|\|", '', line).replace('_', ' ')
                    sent_info = get_sent_info(raw_line, intersection, sent_id, max_seq_length)
                    batch_sentences[sent_id] = {'raw_sent': raw_line, 'sent_info': sent_info}

    reader.close()
    fout.close()


if __name__ == '__main__':
    dataname='NYT'
    layer_indexes=[-1,-2,-3,-4]
    bert_model='bert-base-uncased'
    max_seq_length=128
    batch_size=128
    data_dir='../data/'

    extract_embeddings(dataname=dataname, layer_indexes=layer_indexes, bert_model=bert_model,
                       max_seq_length=max_seq_length, batch_size=batch_size, data_dir=data_dir)
