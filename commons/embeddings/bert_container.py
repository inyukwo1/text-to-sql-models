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
            self.main_bert.cuda()
        self.other_optimizer = torch.optim.Adam(self.bert_param.parameters(), lr=lr)
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=lr)

    def bert(self, inp, inp_len, q_inp_len, sep_embeddings):

        # [batch_num, max_seq_len] = list(inp.size())
        # mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
        # for idx, leng in enumerate(inp_len):
        #     mask[idx, :leng] = np.ones(leng, dtype=np.float32)
        # [batch_num, max_seq_len] = list(inp.size())
        # emb_mask = np.ones((batch_num, max_seq_len), dtype=np.float32)
        # for idx, leng in enumerate(q_inp_len):
        #     emb_mask[idx, :leng] = np.zeros(leng, dtype=np.float32)
        # mask = torch.LongTensor(mask)
        # emb_mask = torch.LongTensor(emb_mask)
        # if torch.cuda.is_available():
        #     mask = mask.cuda()
        #     emb_mask = emb_mask.cuda()
        #
        # extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.main_bert.parameters()).dtype)
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # embedding_output = self.main_bert.embeddings(inp, emb_mask)
        #
        # # expand_embeddings = []
        # # for one_sep_embeddings in sep_embeddings:
        # #     embed_tensors = []
        # #     for loc_idx in range(max_seq_len):
        # #         if loc_idx >= len(one_sep_embeddings):
        # #             embed_tensor = torch.zeros_like(self.bert_param.pos0)
        # #         elif one_sep_embeddings[loc_idx] == -1:
        # #             embed_tensor = torch.zeros_like(self.bert_param.pos0)
        # #         elif one_sep_embeddings[loc_idx] == 0:
        # #             embed_tensor = self.bert_param.pos0
        # #         elif one_sep_embeddings[loc_idx] == 1:
        # #             embed_tensor = self.bert_param.pos1
        # #         elif one_sep_embeddings[loc_idx] == 2:
        # #             embed_tensor = self.bert_param.pos2
        # #         elif one_sep_embeddings[loc_idx] == 3:
        # #             embed_tensor = self.bert_param.pos3
        # #         elif one_sep_embeddings[loc_idx] == 4:
        # #             embed_tensor = self.bert_param.pos4
        # #         elif one_sep_embeddings[loc_idx] == 5:
        # #             embed_tensor = self.bert_param.pos5
        # #         embed_tensors.append(embed_tensor)
        # #     embed_tensors = torch.stack(embed_tensors)
        # #     expand_embeddings.append(embed_tensors)
        # # expand_embeddings = torch.stack(expand_embeddings)
        # # if torch.cuda.is_available():
        # #     expand_embeddings = expand_embeddings.cuda()
        # # embedding_output = embedding_output + expand_embeddings
        #
        # x = embedding_output
        # for layer_num, layer_module in enumerate(self.main_bert.encoder.layer):
        #     x = layer_module(x, extended_attention_mask)

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
