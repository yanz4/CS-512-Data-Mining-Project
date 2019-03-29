# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:26:39 2019

@author: msq96
"""

import numpy as np
import torch
import mxnet as mx
from bert_embedding import BertEmbedding
from pytorch_pretrained_bert import BertTokenizer, BertModel
from bert_serving.client import BertClient



ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


text1 = "My name is Siqi and what's your name ?"
tokenized_text1 = tokenizer.tokenize(text1)

res1 = bert(tokenized_text1)[0][1][0]






text2 = "[CLS] My name is Siqi and what's your name ? [SEP]"
tokenized_text2 = tokenizer.tokenize(text2)
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text2)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor)

res2 = encoded_layers[-2][0, 5].data.cpu().numpy()






bc = BertClient()
vec = bc.encode([text1])

res3 = vec[0, 5]


print(np.linalg.norm(res2-res3))
