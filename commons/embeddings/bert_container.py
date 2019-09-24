from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from pytorch_transformers import XLNetModel, XLNetTokenizer
import torch
import torch.nn as nn
import numpy as np


def embedding_tensor_listify(tensor):
    D1, D2, D3 = list(tensor.size())
    list_tensor = []
    for d1 in range(D1):
        one_list_tensor = []
        for d2 in range(D2):
            one_list_tensor.append(tensor[d1, d2])
        list_tensor.append(one_list_tensor)
    return list_tensor


def list_tensor_tensify(list_tensor):
    for idx, one_list_tensor in enumerate(list_tensor):
        list_tensor[idx] = torch.stack(one_list_tensor)
    list_tensor = torch.stack(list_tensor)
    return list_tensor


class BertParameterWrapper(nn.Module):
    def __init__(self):
        super(BertParameterWrapper, self).__init__()
        self.pos0 = nn.Parameter(torch.rand(1024) / 100)
        self.pos1 = nn.Parameter(torch.rand(1024) / 100)
        self.pos2 = nn.Parameter(torch.rand(1024) / 100)
        self.pos3 = nn.Parameter(torch.rand(1024) / 100)
        self.pos4 = nn.Parameter(torch.rand(1024) / 100)
        self.pos5 = nn.Parameter(torch.rand(1024) / 100)


class BertContainer:
    def __init__(self, lr):
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
        self.bert_param = BertParameterWrapper()
        if torch.cuda.is_available():
            self.main_bert.cuda(0)
        self.other_optimizer = torch.optim.Adam(self.bert_param.parameters(), lr=lr)
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=lr)

    def bert(self, inp):
        return self.main_bert(inp, output_all_encoded_layers=False)[0]

    def train(self):
        self.main_bert.train()

    def eval(self):
        self.main_bert.eval()

    def zero_grad(self):
        self.main_bert_optimizer.zero_grad()
        self.other_optimizer.zero_grad()

    def step(self):
        self.main_bert_optimizer.step()
        self.other_optimizer.step()
