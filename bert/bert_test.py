# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:26:39 2019

@author: msq96
"""


import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from bert_serving.client import BertClient


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text1 = "My name is BERT and what's your name ?"
text2 = "[CLS] My name is BERT and what's your name ? [SEP]"

tokenized_text1 = tokenizer.tokenize(text1)
tokenized_text2 = tokenizer.tokenize(text2)


# bert-as-service (tensorflow backend)
bc = BertClient()
vec = bc.encode([text1])
res1 = vec[0, 5]


# pytorch_pretrained_bert
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text2)
tokens_tensor = torch.tensor([indexed_tokens])
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)

res2 = encoded_layers[-2][0, 5].data.cpu().numpy()


print(np.linalg.norm(res1-res2))
