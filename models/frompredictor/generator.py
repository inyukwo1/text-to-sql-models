import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from models.frompredictor.ontology import Ontology
from commons.utils import run_lstm
from commons.embeddings.graph_utils import *
from commons.embeddings.bert_container import BertContainer
from commons.embeddings.word_embedding import WordEmbedding
from transformer.encoder import Encoder as TransformerEncoder
from commons.utils import run_lstm, col_tab_name_encode, seq_conditional_weighted_num, SIZE_CHECK


def truncated_normal_(tensor, mean=0., std=1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class Generator(nn.Module):
    def __init__(self, H_PARAM):
        super(Generator, self).__init__()
        self.acc_num = 1
        self.acc = 0

        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
        self.toy = H_PARAM['toy']
        self.gpu = H_PARAM['gpu']
        self.use_hs = H_PARAM['use_hs']
        self.B_word = H_PARAM['B_WORD']
        self.N_word = H_PARAM['N_WORD']
        self.threshold = H_PARAM['threshold'] #0.5

        self.save_dir = H_PARAM['save_dir']
        self.bert = BertContainer(H_PARAM['bert_lr'])
        self.onefrom = H_PARAM['onefrom']
        self.encoded_num = H_PARAM['encoded_num'] #1024
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        self.embed_layer = WordEmbedding(H_PARAM['glove_path'].format(self.B_word, self.N_word),
                                         self.N_word, gpu=self.gpu, SQL_TOK=self.SQL_TOK, use_bert=True, use_small=H_PARAM["toy"])

        self.entity_encoder = TransformerEncoder(3, 3, self.encoded_num, 128, 128, 0.1, 0.1, 0)
        self.entity_attention = nn.Linear(self.encoded_num, self.encoded_num + 1)
        # self.fin_attention = nn.Linear(self.N_h, self.N_h)
        self.none_entity = nn.Parameter(truncated_normal_(torch.rand(1, 1024), std=0.02))

        self.outer1 = nn.Sequential(nn.Linear(self.encoded_num * 2, self.N_h), nn.ReLU())
        self.outer2 = nn.Sequential(nn.Linear(self.N_h, 1))
        if self.onefrom:
            self.onefrom_vec = nn.Parameter(torch.zeros(self.N_h))
        if self.gpu:
            self.cuda()

    def load_model(self):
        if self.toy:
            return
        print('Loading from model...')
        dev_type = 'cuda' if self.gpu else 'cpu'
        device = torch.device(dev_type)

        self.load_state_dict(torch.load(os.path.join(self.save_dir, "gen_models.dump"), map_location=device))
        self.bert.main_bert.load_state_dict(torch.load(os.path.join(self.save_dir, "genbert_from_models.dump"), map_location=device))
        self.bert.bert_param.load_state_dict(torch.load(os.path.join(self.save_dir, "genbert_from_params.dump"), map_location=device))


    def save_model(self, acc):
        print('tot_err:{}'.format(acc))

        if acc > self.acc:
            self.acc = acc
            torch.save(self.state_dict(), os.path.join(self.save_dir, 'gen_models.dump'))
            torch.save(self.bert.main_bert.state_dict(), os.path.join(self.save_dir + "genbert_from_models.dump"))
            torch.save(self.bert.bert_param.state_dict(), os.path.join(self.save_dir + "genbert_from_params.dump"))

    def train(self, mode=True):
        super().train(mode)
        if self.bert:
            self.bert.train()

    def eval(self):
        super().train(False)
        if self.bert:
            self.bert.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()
        if self.bert:
            self.bert.zero_grad()

    def step(self):
        self.optimizer.step()
        if self.bert:
            self.bert.step()

    def _forward(self, q_embs, q_len, q_q_len, labels, entity_ranges_list, hint_tensors_list, input_qs):
        B = len(q_len)
        q_enc = self.bert.bert(q_embs, q_len, q_q_len, [])
        attentions = []
        co_attentions = []
        _, max_entities = list(labels.size())
        for b in range(B):
            question_vectors = q_enc[b, :q_q_len[b], :]
            question_vectors = torch.cat((question_vectors, hint_tensors_list[b]), dim=1)
            one_batch_entity_ranges = entity_ranges_list[b]
            entity_tensors = []
            max_entity_range_len = max([entity_range[1] - entity_range[0] - 1 for entity_range in one_batch_entity_ranges])
            for entity_range in one_batch_entity_ranges:
                entity_tensor = q_enc[b, entity_range[0] + 1:entity_range[1], :]
                if entity_range[1] - entity_range[0] - 1 < max_entity_range_len:
                    padding = torch.zeros(max_entity_range_len - (entity_range[1] - entity_range[0] - 1), self.encoded_num)
                    if self.gpu:
                        padding = padding.cuda()
                    entity_tensor = torch.cat((entity_tensor, padding), dim=0)
                entity_tensors.append(entity_tensor)
            entity_tensors = torch.stack(entity_tensors)
            SIZE_CHECK(entity_tensors, [-1, max_entity_range_len, self.encoded_num])
            # encoded_entity = self.entity_encoder(entity_tensors)
            # encoded_entity = encoded_entity[:, 0, :]
            encoded_entity = torch.sum(entity_tensors, dim=1)
            encoded_entity = torch.cat((encoded_entity, self.none_entity), dim=0)
            cur_entity_len, _ = list(encoded_entity.size())
            if max_entities + 1 > cur_entity_len:
                padding = torch.zeros((max_entities + 1 - cur_entity_len, self.encoded_num))
                if self.gpu:
                    padding = padding.cuda()
                encoded_entity = torch.cat((encoded_entity, padding), dim=0)
            co_attention = torch.mm(question_vectors, self.entity_attention(encoded_entity).transpose(0, 1))
            co_attention = torch.softmax(co_attention, dim=1)
            co_attentions.append(co_attention)
            co_attention = torch.sum(co_attention, dim=0)
            co_attention = torch.clamp(co_attention, max=1.)
            # co_attention = torch.tanh(co_attention)
            co_attention = co_attention[:-1]
            attentions.append(co_attention)
        attentions = torch.stack(attentions)
        return attentions, co_attentions

    def forward(self, input_data, single_forward=False):
        scores = self._forward(*input_data)
        return scores

    def loss(self, score, labels):
        score, co_attention = score
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, scores, gt_data, batch=None, log=False):
        # Parse Input
        anses, schemas = gt_data

        graph_correct_list = []
        selected_tbls = []
        scores, co_attentions = scores
        if self.gpu:
            scores = scores.data.cpu().numpy()
            anses = anses.data.cpu().numpy()
        else:
            scores = scores.data.numpy()
            anses = anses.data.numpy()
        for i in range(len(scores)):
            wrong = False
            co_attention = co_attentions[i]
            co_attention = co_attention.data.cpu().numpy()
            for b in range(len(scores[i])):
                if scores[i][b] < 0.5 and anses[i][b] != 0.:
                    wrong = True
                elif scores[i][b] > 0.5 and anses[i][b] != 1.:
                    wrong = True
            if True:
                print("==========================================")
                print("question: {}".format(batch[i]["question"]))
                print("sql: {}".format(batch[i]["query"]))
                for table_num, table_name in enumerate(batch[i]["tbl"]):
                    print("Table {}: {}".format(table_num, table_name))
                    for col_num, (par_num, col_name) in enumerate(batch[i]["column"]):
                        if par_num == table_num:
                            print("  {}: {}".format(col_num, col_name))
                print("ans: {}".format(anses[i]))
                print("score: {}".format(scores[i]))
                print("co_attention: {}".format(co_attention))
            graph_correct_list.append(not wrong)

        return graph_correct_list

    def evaluate(self, score, gt_data, batch=None, log=False):
        return self.check_acc(score, gt_data, batch, log).count(True)

    def preprocess(self, batch):
        q_seq = []
        ontologies = []
        schemas = []

        for item in batch:
            ontologies.append(item['ontology'])
            q_seq.append(item['question_toks'])
            schemas.append(item['schema'])
        q_embs, q_len, q_q_len, labels, entity_ranges_list, hint_tensors_list, input_qs = self.embed_layer.gen_for_generator(q_seq, schemas, ontologies)

        input_data = q_embs, q_len, q_q_len, labels, entity_ranges_list, hint_tensors_list, input_qs
        gt_data = labels if self.training else (labels, schemas)

        return input_data, gt_data