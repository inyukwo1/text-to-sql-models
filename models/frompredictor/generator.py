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

        self.table_linear = nn.Linear(self.encoded_num, self.encoded_num)
        self.col_linear = nn.Linear(self.encoded_num, self.encoded_num)

        self.topk_judger = nn.Sequential(nn.Linear(self.encoded_num, self.N_h),
                                        nn.ReLU(),
                                        nn.Linear(self.N_h, 6),
                                         nn.Sigmoid())

        self.outer1 = nn.Sequential(nn.Linear(self.encoded_num, self.N_h), nn.ReLU())
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

    def _forward(self, q_embs, q_len, q_q_len, labels, entity_ranges_list, hint_tensors_list, input_qs, schemas: List[Schema]):
        B = len(q_len)
        q_enc = self.bert.bert(q_embs, q_len, q_q_len, [])
        attentions = []
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
            cnt = 0
            col_encoded_entities = []
            for table_num in schemas[b].get_all_table_ids():
                table_encoded_entity = self.table_linear(encoded_entity[cnt, :])
                cnt += 1
                for col_id in schemas[b].get_child_col_ids(table_num):
                    col_encoded_entities.append(self.col_linear(encoded_entity[cnt, :]) + table_encoded_entity)
                    cnt += 1
            encoded_entity = torch.stack(col_encoded_entities)
            # encoded_entity = torch.cat((encoded_entity, self.none_entity), dim=0)
            cur_entity_len, _ = list(encoded_entity.size())
            if max_entities > cur_entity_len:
                padding = torch.zeros((max_entities - cur_entity_len, self.encoded_num))
                if self.gpu:
                    padding = padding.cuda()
                encoded_entity = torch.cat((encoded_entity, padding), dim=0)
            # co_attention = torch.mm(question_vectors, self.entity_attention(encoded_entity).transpose(0, 1))
            # co_attention = torch.softmax(co_attention, dim=1)
            # co_attentions.append(co_attention)
            # co_attention = torch.sum(co_attention, dim=0)
            # co_attention = torch.clamp(co_attention, max=1.)
            # co_attention = torch.tanh(co_attention)
            # co_attention = co_attention[:-1]
            attentions.append(encoded_entity)
        attentions = torch.stack(attentions)
        attentions = self.outer2(self.outer1(attentions)).squeeze(2)

        topk = self.topk_judger(q_enc[:, 0, :])
        return attentions, topk

    def forward(self, input_data, single_forward=False):
        scores = self._forward(*input_data)
        return scores

    def loss(self, score, labels):
        score, topk_score = score
        label, topk_label = labels
        loss = F.binary_cross_entropy_with_logits(score, label) + F.cross_entropy(topk_score, topk_label)
        return loss

    def check_acc(self, scores, gt_data, batch=None, log=False):
        # Parse Input
        (anses, topk_ans), schemas = gt_data

        correct_list = []
        topk_correct_list = []
        topk_total_correct_list = []
        topk_inside_correct_list = []

        scores, topk_score = scores
        if self.gpu:
            scores = scores.data.cpu().numpy()
            anses = anses.data.cpu().numpy()
            topk_score = topk_score.data.cpu().numpy()
            topk_ans = topk_ans.data.cpu().numpy()
        else:
            scores = scores.data.numpy()
            anses = anses.data.numpy()
            topk_score = topk_score.data.cpu().numpy()
            topk_ans = topk_ans.data.cpu().numpy()
        for i in range(len(scores)):
            wrong = False
            for b in range(len(scores[i])):
                if scores[i][b] < 0. and anses[i][b] != 0.:
                    wrong = True
                elif scores[i][b] > 0. and anses[i][b] != 1.:
                    wrong = True

            if np.argmax(topk_score[i]) != topk_ans[i]:
                topk_wrong = True
            else:
                topk_wrong = False

            topk_total_wrong = True
            if not topk_wrong:
                selected = np.argsort(scores[i])[:topk_ans[i]]
                for b in range(len(anses[i])):
                    if anses[i][b] == 1. and b not in selected:
                        topk_total_wrong = False
            if wrong and topk_total_wrong:
                topk_inside_wrong = True
            else:
                topk_inside_wrong = False
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
                print("topk_ans: {}".format(topk_ans[i]))
                print("topk_score: {}".format(topk_score[i]))
                print("topk_prd: {}".format(np.argmax(topk_score[i])))
                print("wrong: {}, topk_wrong: {}, cor_wrong: {}, inside_wrong: {}".format(wrong, topk_wrong, topk_total_wrong, topk_inside_wrong))
            correct_list.append(not wrong)
            topk_correct_list.append(not topk_wrong)
            topk_total_correct_list.append(not topk_total_wrong)
            topk_inside_correct_list.append(not topk_inside_wrong)

        return correct_list, topk_correct_list, topk_total_correct_list, topk_inside_correct_list

    def evaluate(self, score, gt_data, batch=None, log=False):
        correct_list, topk_correct_list, topk_total_correct_list, topk_inside_correct_list = self.check_acc(score, gt_data, batch, log)
        return correct_list.count(True), topk_correct_list.count(True), topk_total_correct_list.count(True), topk_inside_correct_list.count(True)

    def preprocess(self, batch):
        q_seq = []
        select_cols = []
        schemas = []

        for item in batch:
            select_cols.append(item['select_cols'])
            q_seq.append(item['question_toks'])
            schemas.append(item['schema'])
        q_embs, q_len, q_q_len, labels, topk_labels, entity_ranges_list, hint_tensors_list, input_qs, schemas = self.embed_layer.gen_for_generator(q_seq, schemas, select_cols)

        input_data = q_embs, q_len, q_q_len, labels, entity_ranges_list, hint_tensors_list, input_qs, schemas
        comb_labels = labels, topk_labels
        if self.training:
            gt_data = comb_labels
        else:
            gt_data = (comb_labels, schemas)

        return input_data, gt_data
