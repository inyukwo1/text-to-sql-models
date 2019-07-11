from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
import torch
import torch.nn as nn
import numpy as np


def truncated_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


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
        self.embeddings = nn.ParameterList([
            nn.Parameter(truncated_normal_(torch.rand(1024), std=0.02)) for _ in range(10)
        ])


class BertContainer:
    def __init__(self, lr):
        self.main_bert = BertModel.from_pretrained('bert-large-cased')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
        self.bert_param = BertParameterWrapper()
        if torch.cuda.is_available():
            self.main_bert.cuda()
            self.bert_param.cuda()
        self.other_optimizer = torch.optim.Adam(self.bert_param.parameters(), lr=lr)
        self.main_bert_optimizer = torch.optim.Adam(self.main_bert.parameters(), lr=lr)

    def bert(self, inp, inp_len, q_inp_len, special_tok_locs):
        [batch_num, max_seq_len] = list(inp.size())
        mask = np.zeros((batch_num, max_seq_len), dtype=np.float32)
        for idx, leng in enumerate(inp_len):
            mask[idx, :leng] = np.ones(leng, dtype=np.float32)
        [batch_num, max_seq_len] = list(inp.size())
        emb_mask = np.ones((batch_num, max_seq_len), dtype=np.float32)
        for idx, leng in enumerate(q_inp_len):
            emb_mask[idx, :leng] = np.zeros(leng, dtype=np.float32)
        mask = torch.LongTensor(mask)
        emb_mask = torch.LongTensor(emb_mask)
        if torch.cuda.is_available():
            mask = mask.cuda()
            emb_mask = emb_mask.cuda()

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.main_bert.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.main_bert.embeddings(inp, emb_mask)
        for b in range(batch_num):
            for (loc, id) in special_tok_locs[b]:
                embedding_output[b, loc] = self.bert_param.embeddings[id]

        x = embedding_output
        for layer_num, layer_module in enumerate(self.main_bert.encoder.layer):
            x = layer_module(x, extended_attention_mask)

        return x

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
