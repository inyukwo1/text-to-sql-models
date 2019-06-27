import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from commons.utils import run_lstm
from commons.embeddings.graph_utils import *
from commons.embeddings.bert_container import BertContainer
from commons.embeddings.word_embedding import WordEmbedding
from commons.utils import run_lstm, col_tab_name_encode, seq_conditional_weighted_num, SIZE_CHECK
from models.syntaxsql.net_utils import to_batch_from_candidates, to_batch_seq, to_batch_tables_generator


class Generator(nn.Module):
    def __init__(self, H_PARAM):
        super(Generator, self).__init__()
        self.acc_num = 1
        self.acc = 0

        self.N_h = H_PARAM['N_h']
        self.N_depth = H_PARAM['N_depth']
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
                                         self.N_word, gpu=self.gpu, SQL_TOK=self.SQL_TOK, use_bert=True)

        self.q_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h//2,
                num_layers=self.N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.col_tab_lstm = nn.LSTM(input_size=self.N_word, hidden_size=self.N_h // 2,
                              num_layers=self.N_depth, batch_first=True,
                              dropout=0.3, bidirectional=True)

        self.att = nn.Linear(self.N_h, self.N_h)
        if self.gpu:
            self.cuda()

    def load_model(self):
        print('Loading from model...')
        dev_type = 'cuda' if self.gpu else 'cpu'
        device = torch.device(dev_type)

        self.load_state_dict(torch.load(os.path.join(self.save_dir, "gen_models.dump"), map_location=device))


    def save_model(self, acc):
        print('tot_err:{}'.format(acc))

        if acc > self.acc:
            self.acc = acc
            torch.save(self.state_dict(), os.path.join(self.save_dir, 'gen_models.dump'))

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

    def forward(self, q_emb, q_len, col_emb_var, col_name_len, col_len, tab_emb_var, tab_name_len, tab_len, schemas: List[Schema]):
        B, max_q_len, _ = list(q_emb.size())
        _, max_c_len, _ = list(col_emb_var.size())
        _, max_t_len, _ = list(tab_emb_var.size())
        q_enc = run_lstm(self.q_lstm, q_emb, q_len)
        col_enc, _ = col_tab_name_encode(col_emb_var, col_name_len, col_len, self.col_tab_lstm)
        tab_enc, _ = col_tab_name_encode(tab_emb_var, tab_name_len, tab_len, self.col_tab_lstm)
        for b in range(B):
            for tab_id in schemas[b].get_all_table_ids():
                for col_id in schemas[b].get_child_col_ids(tab_id):
                    col_enc[b][col_id] = col_enc[b][col_id] + tab_enc[b][tab_id]
        atted = self.att(q_enc)
        col_att = torch.bmm(atted, col_enc.transpose(1, 2))
        SIZE_CHECK(col_att, [B, max_q_len, max_c_len])
        tab_att = torch.bmm(atted, tab_enc.transpose(1, 2))
        SIZE_CHECK(tab_att, [B, max_q_len, max_t_len])

        return col_att, tab_att

    def loss(self, scores, labels):
        col_score, tab_score = scores
        col_label, tab_label = labels
        loss = F.binary_cross_entropy_with_logits(col_score.sum(1), col_label) + F.binary_cross_entropy_with_logits(tab_score.sum(1), tab_label)
        return loss

    def check_acc(self, scores, gt_data, batch=None, log=False):
        # Parse Input
        ans_ontology = gt_data
        col_score, tab_score = scores

        col_score = col_score.data.cpu().numpy()
        tab_score = tab_score.data.cpu().numpy()

        for i in range(len(col_score)):
            print("==========================================")
            print("question: {}".format(batch[i]["question"]))
            print("sql: {}".format(batch[i]["query"]))
            for table_num, table_name in enumerate(batch[i]["tbl"]):
                print("Table: {} {}".format(table_num, table_name))
                for col_num, (par_num, col_name) in enumerate(batch[i]["column"]):
                    if par_num == table_num:
                        print("  {}: {}".format(col_num, col_name))
            print("ans_ontology: {}".format(ans_ontology[i]))
            print("col_score: {}".format(col_score[i]))
            print("tab_score: {}".format(tab_score[i]))

        return 1

    def evaluate(self, score, gt_data, batch=None, log=False):
        return self.check_acc(score, gt_data, batch, log)

    def preprocess(self, batch):
        q_seq = []
        labels = []
        schemas = []

        # For acc
        q_embs = []
        q_lens = []
        q_q_lens = []
        table_graph_lists = []
        full_graph_lists = []
        sep_embedding_lists = []
        ans_ontologies = []

        if self.training:
            for item in batch:
                ans_ontologies.append(item["ontology"])
                schemas.append(item['schema'])
            q_seq, history, label = to_batch_seq(batch)
            q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
            col_seq, tab_seq = to_batch_tables_generator(batch)
            col_emb_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col_seq)
            tab_emb_var, tab_name_len, tab_len = self.embed_layer.gen_col_batch(tab_seq)
        else:
            for item in batch:
                ans_ontologies.append(item["ontology"])
                schemas.append(item['schema'])
            q_seq, history, label = to_batch_seq(batch)
            q_emb_var, q_len = self.embed_layer.gen_x_q_batch(q_seq)
            col_seq, tab_seq, par_tab_nums, foreign_keys = to_batch_tables(batch, self.table_type)

        hs_emb_var, hs_len = self.embed_layer.gen_x_history_batch(history)

        input_data = q_emb_var, q_len, col_emb_var, col_name_len, col_len, tab_emb_var, tab_name_len, tab_len, schemas
        gt_data = labels if self.training else (ans_ontologies)

        return input_data, gt_data