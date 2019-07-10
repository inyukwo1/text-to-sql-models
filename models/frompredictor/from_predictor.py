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


class FromPredictor(nn.Module):
    def __init__(self, H_PARAM):
        super(FromPredictor, self).__init__()
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
        self.entity_attention = nn.Linear(self.encoded_num, self.encoded_num + 2)
        # self.fin_attention = nn.Linear(self.N_h, self.N_h)

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

        self.load_state_dict(torch.load(os.path.join(self.save_dir, "from_models.dump"), map_location=device))
        self.bert.main_bert.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_models.dump"), map_location=device))
        self.bert.bert_param.load_state_dict(torch.load(os.path.join(self.save_dir, "bert_from_params.dump"), map_location=device))


    def save_model(self, acc):
        print('tot_err:{}'.format(acc))

        if acc > self.acc:
            self.acc = acc
            torch.save(self.state_dict(), os.path.join(self.save_dir, 'from_models.dump'))
            torch.save(self.bert.main_bert.state_dict(), os.path.join(self.save_dir + "bert_from_models.dump"))
            torch.save(self.bert.bert_param.state_dict(), os.path.join(self.save_dir + "bert_from_params.dump"))

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

    def _forward(self, q_emb, q_len, q_q_len, hs_emb_var, hs_len, input_qs, entity_ranges, hint_tensors):
        B = len(q_len)
        q_enc = self.bert.bert(q_emb, q_len, q_q_len, input_qs)
        weighted_attentions = []
        for b in range(B):
            question_vectors = q_enc[b, :q_q_len[b], :]
            question_vectors = torch.cat((question_vectors, hint_tensors[b]), dim=1)
            one_batch_entity_ranges = entity_ranges[b]
            entity_tensors = []
            max_entity_range_len = max([entity_range[1] - entity_range[0] for entity_range in one_batch_entity_ranges])
            for entity_range in one_batch_entity_ranges:
                entity_tensor = q_enc[b, entity_range[0]:entity_range[1], :]
                if entity_range[1] - entity_range[0] < max_entity_range_len:
                    padding = torch.zeros(max_entity_range_len - (entity_range[1] - entity_range[0]), self.encoded_num)
                    entity_tensor = torch.cat((entity_tensor, padding), dim=0)
                entity_tensors.append(entity_tensor)
            entity_tensors = torch.stack(entity_tensors)
            SIZE_CHECK(entity_tensors, [-1, max_entity_range_len, self.encoded_num])
            encoded_entity = self.entity_encoder(entity_tensors)
            encoded_entity = encoded_entity[:, 0, :]
            co_attention = torch.mm(question_vectors, self.entity_attention(encoded_entity).transpose(0, 1)) # 24 1026 302 2
            co_attention = torch.softmax(co_attention, dim=1)
            weigted_attention = torch.mm(co_attention, encoded_entity).sum(0)
            SIZE_CHECK(weigted_attention, [self.encoded_num])
            weighted_attentions.append(weigted_attention)
        weighted_attentions = torch.stack(weighted_attentions)

        q_enc = q_enc[:, 0, :]
        q_enc = torch.cat((q_enc, weighted_attentions), dim=1)

        x = self.outer1(q_enc)
        x = self.outer2(x).squeeze(1)
        return x

    def forward(self, input_data, single_forward=False):
        # Parse Input
        q_embs, q_lens, q_q_lens, hs_emb_var, hs_len, input_qs_lists, entity_ranges, hint_tensors = input_data

        max_size = 4

        if self.training or single_forward:
            scores = self._forward(*input_data)
        else:
            scores = []
            for i in range(len(q_embs)):
                length = len(q_embs[i])
                one_batch_score = []
                for j in range(0, length, max_size):
                    score = self._forward(q_embs[i][j:min(length,j+max_size)], q_lens[i][j:min(length,j+max_size)],
                        q_q_lens[i][j:min(length,j+max_size)], hs_emb_var, hs_len, input_qs_lists[i][j:min(length,j+max_size)],
                        entity_ranges[i][j:min(length,j+max_size)], hint_tensors[i][j:min(length,j+max_size)])
                    one_batch_score.append(score.data.cpu().numpy())
                one_batch_score = np.concatenate(one_batch_score)
                scores.append(one_batch_score)

        return scores

    def loss(self, score, labels):
        loss = F.binary_cross_entropy_with_logits(score, labels)
        return loss

    def check_acc(self, scores, gt_data, batch=None, log=False):
        # Parse Input
        anses, ontology_lists, schemas, input_qs = gt_data

        graph_correct_list = []
        selected_tbls = []
        for i in range(len(scores)):
            selected_ontology = ontology_lists[i][np.argmax(scores[i])]
            ans = anses[i]
            ontology_correct = selected_ontology.is_same(ans)
            if log and not ontology_correct:
                print("==========================================")
                print("question: {}".format(batch[i]["question"]))
                print("sql: {}".format(batch[i]["query"]))
                for table_num, table_name in enumerate(batch[i]["tbl"]):
                    print("Table {}: {}".format(table_num, table_name))
                    for col_num, (par_num, col_name) in enumerate(batch[i]["column"]):
                        if par_num == table_num:
                            print("  {}: {}".format(col_num, col_name))
                print("ans: {}".format(ans))
                print("selected: {}".format(selected_ontology))
                print("selected q: {}".format(input_qs[i][np.argmax(scores[i])]))
                print("cands: ")
                for cand_idx in range(len(scores[i])):
                    print(ontology_lists[i][cand_idx])
                    print("q: {}".format(input_qs[i][cand_idx]))
                    print("score: {}".format(scores[i][cand_idx]))
                    print("%%%", flush=True)
            graph_correct_list.append(ontology_correct)
            selected_tbls.append(selected_ontology.tables)

        return graph_correct_list, selected_tbls

    def evaluate(self, score, gt_data, batch=None, log=False):
        return self.check_acc(score, gt_data, batch, log)[0].count(True)

    def preprocess(self, batch):
        q_seq = []
        history = []
        labels = []
        schemas = []

        # For acc
        q_embs = []
        q_lens = []
        q_q_lens = []
        ontology_lists = []
        input_qs_lists = []
        matching_conts = []
        entity_ranges_list = []
        hint_tensors_list = []

        if self.training:
            for item in batch:
                history.append(item['history'] if self.use_hs else ['root', 'none'])
                labels.append(item['ontology'])
                q_seq.append(item['question_toks'])
                schemas.append(item['schema'])
                matching_conts.append(item['matching_conts'])
            q_embs, q_lens, q_q_lens, labels, input_qs_lists, entity_ranges_list, hint_tensors_list = self.embed_layer.gen_bert_batch_with_table(q_seq, schemas, labels, matching_conts)
        else:
            for item in batch:
                history.append(item['history'] if self.use_hs else ['root', 'none'])
                labels.append(item['ontology'])
                schemas.append(item['schema'])
                q_emb, q_len, q_q_len, ontology_list, input_qs, entity_ranges, hint_tensors = self.embed_layer.gen_bert_for_eval(
                    item['question_toks'], item['schema'], item['ontology'], item['matching_conts'])
                q_embs.append(q_emb)
                q_lens.append(q_len)
                q_q_lens.append(q_q_len)
                ontology_lists.append(ontology_list)
                input_qs_lists.append(input_qs)
                entity_ranges_list.append(entity_ranges)
                hint_tensors_list.append(hint_tensors)

        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

        input_data = q_embs, q_lens, q_q_lens, hs_emb_var, hs_len, input_qs_lists, entity_ranges_list, hint_tensors_list
        gt_data = labels if self.training else (labels, ontology_lists, schemas, input_qs_lists)

        return input_data, gt_data